# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
import logging


import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle
from collections import namedtuple

from torch.utils.data import Dataset
import lmdb

from .unified_tokenizer import get_tokenizer
from .templates import TextCodeTemplate

logger = logging.getLogger(__name__)


class LMDBDataset(Dataset):
    def __init__(self, path, process_fn):
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

def get_dataset_by_type(dataset_type, path: str, args, DS_CLASS=LMDBDataset): 
    kwargs_to_dataset = {}      

    tokenizer = get_tokenizer()
    if args.layout[-1] > args.max_position_embeddings:
        ml = args.layout[-1]
    else:
        ml = args.max_position_embeddings

    def pad_to_len(ret):
        if len(ret) < ml: # pad
            return np.concatenate((ret, 
                np.array([tokenizer['[PAD]']] * (ml - len(ret)))),
                axis=0), len(ret)
        else:
            if len(ret) > ml:
                logger.warning('Out of max len, truncated.')
            return ret[:ml], ml

    if dataset_type == 'TokenizedDataset':
        # already tokenized when saved
        def process_fn(row):
            ret, attention_mask_sep = pad_to_len(row.flatten())
            return {'text': ret, 
                'loss_mask':  np.array([1] * attention_mask_sep + [0] * (len(ret) - attention_mask_sep))
                }

    elif dataset_type == 'TextCodeDataset':
        def process_fn(row):
            text, code = row[0], row[1].flatten()
            ret = TextCodeTemplate(text, code)
            ret, attention_mask_sep = pad_to_len(ret)
            return {'text': ret, 
                'loss_mask':  np.array([1] * attention_mask_sep + [0] * (len(ret) - attention_mask_sep))
                }

    elif dataset_type == 'CompactBinaryDataset':
        layout = args.layout
        DS_CLASS = BinaryDataset
        kwargs_to_dataset['length_per_sample'] = layout[-1]
        def process_fn(row):
            row = row.astype(np.int64)
            # THIS IS Reverse order, TODO 
            lens = list(reversed([layout[i] - layout[i-1] for i in range(1, len(layout))]))
            codes = [row[layout[0]: layout[0]+lens[0]]]
            if len(lens) > 1:
                codes.append(row[layout[0]+lens[0]: layout[0]+lens[0]+lens[1]])
            text = row[:layout[0]]
            text = text[text>0][:layout[0] - 3] # [CLS] [BASE] [ROI1]
            n_pad = layout[0]-3-len(text)
            parts = [
                np.array([tokenizer['[PAD]']] * n_pad, dtype=np.int64),
                TextCodeTemplate(text, codes[-1]),
                *reversed(codes[:-1])
            ]
            ret = np.concatenate(parts, axis=0)
            return {'text': ret, 
                'loss_mask':  np.array([0] * (n_pad+1) + [1] * (len(ret) - n_pad - 1)) # don't predict [CLS]
                }
    elif dataset_type == 'BinaryDataset':
        DS_CLASS = BinaryDataset
        def process_fn(row):
            loss_mask = (row >= 0).astype(np.int64)
            return {'text': row.astype(np.int64), 
                'loss_mask':  loss_mask
                }

    return DS_CLASS(path, process_fn, **kwargs_to_dataset)

