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
from SwissArmyTransformer.mpu.transformer import standard_attention, split_tensor_along_last_dim

class CachedAutoregressiveMixin(BaseMixin):
    def __init__(self):
        super().__init__()     
           
    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, mems=None, cross_attention=False, old_impl=standard_attention,
                     **kw_args):
        if not cross_attention:
            mem = mems[kw_args['layer_id']] if mems is not None else None # 2, batch, head, seqlen, hidden_size
            b, nh, seq_len, hidden_size = k.shape

            cache_kv = torch.stack((k, v)).permute(1, 3, 0, 2, 4).detach().contiguous().view(b, seq_len, nh * hidden_size * 2)
            kw_args['output_this_layer']['mem_kv'] = cache_kv

            if mem is not None: # the first time, mem is None
                # might change batch_size
                mem = mem.expand(b, -1, -1).reshape(b, mem.shape[1], 2, nh, hidden_size).permute(2, 0, 3, 1, 4)
                memk, memv = mem[0], mem[1]
                k = torch.cat((memk, k), dim=2)
                v = torch.cat((memv, v), dim=2)
        return old_impl(q, k, v, mask, dropout_fn, cross_attention=cross_attention, mems=mems, **kw_args)


class CachedAutoregressiveModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin('auto-regressive', CachedAutoregressiveMixin())
