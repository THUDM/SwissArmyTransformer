import sys
import io
import os
import re
import json
import random
import tarfile
import numpy as np
from functools import partial
import torch.distributed as dist
from PIL import Image

import webdataset as wds
from webdataset import ResampledShards, DataPipeline, tarfile_to_samples
from webdataset.utils import pytorch_worker_seed
from webdataset.filters import pipelinefilter
from webdataset.tariterators import url_opener, group_by_keys
from webdataset.handlers import reraise_exception

def worker_seed_sat(group=None, seed=0):
    return pytorch_worker_seed(group=group) + seed * 23

class ConfiguredResampledShards(ResampledShards):
    def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
        from sat.mpu import get_data_parallel_group
        try:
            group = get_data_parallel_group()
        except AssertionError:
            group = None
        worker_seed_sat_this = partial(worker_seed_sat, group=group, seed=seed)
        super().__init__(urls, nshards, worker_seed_sat_this, deterministic)

class SimpleDistributedWebDataset(DataPipeline):
    def __init__(self, path, process_fn, seed, *, shuffle_buffer=1000):
        # set shuffle_buffer = 1 to disable it, model-parallel will be different due to shuffle
        try:
            from sat.mpu import get_model_parallel_world_size
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass
        super().__init__(
            ConfiguredResampledShards(path, seed), # Lots of shards are recommended, or not evenly
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn
        )

def tar_file_iterator_with_meta(fileobj, meta_names, skip_meta=r"__[^/]*__($|/)", suffix=None,handler=reraise_exception):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    :param fileobj: byte stream suitable for tarfile
    :param meta_names: key of different items in meta file
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")

    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    data_dir, filename = fileobj.name.rsplit('/', 1)
    
    meta_data = {} # {id: {meta_name: meta_value, meta_name2: meta_value2, ...}}
    for meta_name in meta_names:
        meta_file_name = filename.split('.')[0] + '.meta.jsonl'
        meta_path = os.path.join(data_dir, meta_file_name)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as meta_file:
                meta_list = []
                for lineno, line in enumerate(meta_file):
                    try:
                        meta_list.append(json.loads(line))
                    except Exception as exn:
                        from sat.helpers import print_rank0
                        print_rank0(f'Error in loading jsonl {meta_file_name}, lineno {lineno}: {line}', level='DEBUG')
                        continue
                for item in meta_list:
                    if not item['key'] in meta_data:
                        meta_data[item['key']] = {}
                    if meta_name in item:
                        meta_data[item['key']][meta_name] = item[meta_name]
    
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith("__")
                and fname.endswith("__")
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue
            if fname.endswith('.txt') and suffix is not None:
                data = (stream.extractfile(tarinfo).read().decode() + suffix).encode()
            else:
                data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            
            if fname.endswith('.id'):
                fid = fname.split('.')[0]
                meta_data_fid = meta_data.get(fid, {})
                for meta_name in meta_names:
                    meta_fname = fid + '.' + meta_name
                    meta = meta_data_fid.get(meta_name, None)
                    yield dict(fname=meta_fname, data=meta)
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream
    
def tar_file_expander_with_meta(data, meta_names, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.

    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator_with_meta(source["stream"], meta_names):
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break
            
def tarfile_samples_with_meta(src, meta_names, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander_with_meta(streams, meta_names, handler)
    samples = group_by_keys(files, handler=handler)
    return samples  
        
class MetaDistributedWebDataset(DataPipeline):
    '''WebDataset with meta information files
    Extra Format:
        in webdataset (tar), for each sample there is a '.id'; 
        for each tar file, there is a '.meta.jsonl' file with the same name;
        The '.meta.jsonl' file contains lines of json objects, each with a 'key' field to match '.id'.
    '''
    def __init__(self, path, process_fn, seed, *, meta_names=[], nshards=sys.maxsize, shuffle_buffer=1000, include_dirs=None):
        # os.environ['WDS_SHOW_SEED'] = '1'
        if include_dirs is not None: # /webdatasets/A,/webdatasets/C
            other_paths = []
            include_dirs = include_dirs.split(',')
            for include_dir in include_dirs:
                if '*' in include_dir:
                    include_dir, n = include_dir.split('*')
                    n = int(n)
                else:
                    n = 1
                for cur_dir, dirs, files in os.walk(include_dir):
                    for f in files:
                        if f.endswith('tar') and os.path.getsize(os.path.join(cur_dir,f)) > 0:
                            # other_paths.append(os.path.join(cur_dir,f))
                            other_paths.extend([os.path.join(cur_dir,f)]*n)
            # print(f'Adding dataset paths {",".join(other_paths)}')
            from braceexpand import braceexpand
            if len(path) > 0: # not "" 
                path = list(braceexpand(path)) + other_paths
            else:
                path = other_paths
        
        tarfile_samples = partial(tarfile_samples_with_meta, meta_names=meta_names)
        tarfile_to_samples = pipelinefilter(tarfile_samples)

        # if model parallel, shuffle_buffer should be 1 to disable shuffling
        try:
            from sat.mpu import get_model_parallel_world_size
            if get_model_parallel_world_size() > 1:
                shuffle_buffer = 1
        except Exception:
            pass
        
        super().__init__(
            ConfiguredResampledShards(path, seed, nshards=nshards),
            tarfile_to_samples(),
            wds.shuffle(shuffle_buffer),
            process_fn
        )