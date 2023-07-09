# -*- encoding: utf-8 -*-
'''
@File    :   cached_autoregressive_model.py
@Time    :   2021/10/02 01:36:24
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from .base_model import BaseModel, BaseMixin, non_conflict
from sat.model.transformer import standard_attention, split_tensor_along_last_dim


class VectorKvCache():
    def __init__(self, num_layers, head_nums, hidden_units, max_len, capacity=0, factor=2):
        """
        1. capacity: the size of the storage space currently allocated for the cache
        2. max_len: the max length of tokens
        3. self.mem_size: the number of elements in the cache, like size in c++ vector
        """
        self.factor = factor
        if self.factor <= 1.0 :
            raise ValueError("factor should be greater than 1.")
        self.max_len = max_len
        self.mem_size = 0
        self.capacity = capacity
        self.mems_k = None
        self.mems_v = None
        self.num_layers = num_layers
        self.head_nums = head_nums
        self.hidden_units = hidden_units
        self.size_per_head = int(hidden_units / head_nums)

    def append_kv(self, k, v, layer_id):
        b, nh, seq_len, hidden_size = k.shape
        mem_len = self.mem_size
        self.mems_k[layer_id][:, :, mem_len:mem_len+seq_len, :] = k
        self.mems_v[layer_id][:, :, mem_len:mem_len+seq_len, :] = v
        

    def get_kv(self, layer_id, seq_len):
        #  return key value for attention forward
        mem_k = self.mems_k[layer_id]
        mem_v = self.mems_v[layer_id]
        seq_len = self.mem_size + seq_len
        k = mem_k[:, :, :seq_len, :]
        v = mem_v[:, :, :seq_len, :]
        return k, v
    
    def get_mem_size(self):
        return self.mem_size

    def update_mem_size(self, seq_len):
        self.mem_size += seq_len

    def reMalloc(self, seq_len, batch_size, dtype, device):
        new_capacity = seq_len + self.mem_size
        if new_capacity > self.capacity:
            new_mems_size = [self.num_layers, batch_size, self.head_nums, 0, self.size_per_head] # [num_layers, batch_size, head_num, seq_len, size_per_head]
            if int(new_capacity * self.factor) <= self.max_len:
                new_mems_size[3] = int(new_capacity * self.factor)
                self.capacity = int(new_capacity * self.factor)
            else:
                new_mems_size[3] = self.max_len
                self.capacity = self.max_len
            new_mems_k = torch.empty(*new_mems_size, dtype=dtype, device=device)
            new_mems_v = torch.empty(*new_mems_size, dtype=dtype, device=device)
            if self.mems_k is not None and self.mems_v is not None :
                new_mems_k[:, :, :, :self.mem_size, :] = self.mems_k
                new_mems_v[:, :, :, :self.mem_size, :] = self.mems_v
            self.mems_k = new_mems_k
            self.mems_v = new_mems_v

class CachedAutoregressiveMixin(BaseMixin):
    def __init__(self, num_layers, head_nums, hidden_units, max_len, capacity=0, factor=2):
        super().__init__()
        self.num_layers = num_layers  
        self.mems = VectorKvCache(num_layers, head_nums, hidden_units, max_len, capacity=capacity, factor=factor)
           
    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, cross_attention=False, old_impl=standard_attention,
                     **kw_args):
        if not cross_attention:
            layer_id = kw_args['layer_id']
            b, nh, seq_len, hidden_size = k.shape
            if layer_id == 0 :
                self.mems.reMalloc(seq_len, b, k.dtype, k.device)
            self.mems.append_kv(k, v, layer_id)
            k, v = self.mems.get_kv(layer_id, seq_len)
            if layer_id == self.num_layers - 1 :
                self.mems.update_mem_size(seq_len)

        return old_impl(q, k, v, mask, dropout_fn, cross_attention=cross_attention, **kw_args)


class CachedAutoregressiveModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin('auto-regressive', CachedAutoregressiveMixin())
