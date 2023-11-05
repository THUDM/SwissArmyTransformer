import torch
import json

from .webds import ConfiguredResampledShards, DataPipeline

import webdataset
from webdataset.handlers import reraise_exception
from webdataset.tariterators import url_opener
from webdataset.filters import pipelinefilter


class JsonlIterableDataset(DataPipeline):
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
            jsonl_samples,
            webdataset.shuffle(shuffle_buffer),
            process_fn
        )

def jsonl_expander(streams):
    for source in streams:
        for line in source['stream']:
            sample = json.loads(line)
            sample['__url__'] = source['url']
            for k,v in sample.items():
                if v is None:
                    sample[k] = ''
            yield sample

def jsonl_samples(src, handler=reraise_exception):
    streams = url_opener(src, handler=handler)
    return jsonl_expander(streams)


    