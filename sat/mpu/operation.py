# -*- encoding: utf-8 -*-
'''
@File    :   operation.py
@Time    :   2023/06/21 17:05:39
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import torch
from sat.mpu import destroy_model_parallel, initialize_model_parallel, get_model_parallel_rank, get_model_parallel_world_size
from sat.mpu import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

def mp_split_checkpoint(path):
    raise NotImplementedError

def mp_merge_checkpoint(path):
    raise NotImplementedError

def mp_split_model(model, new_model_parallel_size):
    from sat.model.transformer import SelfAttention, CrossAttention

    destroy_model_parallel()
    initialize_model_parallel(new_model_parallel_size)
    def iter_repartition(module):
        for name, sub_module in module.named_children():
            if isinstance(sub_module, (ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, 
                                       SelfAttention, CrossAttention)):
                sub_module.repartition()
            iter_repartition(sub_module)
    iter_repartition(model)
    
def mp_merge_model(model):
    raise NotImplementedError