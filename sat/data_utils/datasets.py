# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import os
import sys
import math
import random

import numpy as np
import pickle

import torch
from torch.utils.data import Dataset

class LMDBDataset(Dataset):
    def __init__(self, path, process_fn):
        import lmdb
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.process_fn = process_fn
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        with self.env.begin(write=False) as txn:
            key = str(idx).encode('utf-8')
            row = pickle.loads(txn.get(key))
            return self.process_fn(row)

class BinaryDataset(Dataset):
    def __init__(self, path, process_fn, length_per_sample=64+1024+4096, dtype='int32', preload=False, **kwargs): # TODO ARGS
        assert length_per_sample is not None
        self.length_per_sample = length_per_sample
        self.dtype = np.dtype(dtype)
        self.process_fn = process_fn
        if preload:
            self.bin = np.fromfile(path, dtype=self.dtype).reshape(-1, length_per_sample)
        else:
            with open(path, 'r') as fid:
                nbytes = fid.seek(0, 2)
                flen = fid.tell() // self.dtype.itemsize
            self.bin = np.memmap(path, dtype=self.dtype, shape=(flen // length_per_sample, length_per_sample))
    
    def __len__(self):
        return self.bin.shape[0]
    
    def __getitem__(self, index):
        return self.process_fn(self.bin[index])

class TSVDataset(Dataset):
    def __init__(self, path, process_fn, with_heads=True, **kwargs):
        self.process_fn = process_fn
        with open(path, 'r') as fin:
            if with_heads:
                self.heads = fin.readline().split('\t')
            else:
                self.heads = None
            self.items = [line.split('\t') for line in fin]

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.process_fn(self.items[index])

try:
    import webdataset as wds
    from webdataset import ResampledShards, DataPipeline, tarfile_to_samples
    from webdataset.utils import pytorch_worker_seed
    def worker_seed_sat(group=None, seed=0):
        return pytorch_worker_seed(group=group) + seed * 23
    
    class ConfiguredResampledShards(ResampledShards):
        def __init__(self, urls, seed, nshards=sys.maxsize, deterministic=True):
            from sat.mpu import get_data_parallel_group
            worker_seed_sat_this = partial(worker_seed_sat, group=get_data_parallel_group(), seed=seed)
            super().__init__(urls, nshards, worker_seed_sat_this, deterministic)

    class SimpleDistributedWebDataset(DataPipeline):
        def __init__(self, path, process_fn, seed, *, shuffle_buffer=1000):
            super().__init__(
                ConfiguredResampledShards(path, seed), # Lots of shards are recommended, or not evenly
                tarfile_to_samples(),
                wds.shuffle(shuffle_buffer), # set shuffle_buffer = 1 to disable it, TODO model-parallel with different due to shuffle
                process_fn
            )
        
except ModuleNotFoundError: # webdataset not install, use pip to install
    pass