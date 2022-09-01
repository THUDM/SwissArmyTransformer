# -*- encoding: utf-8 -*-

# here put the import lib
import os
import sys
import math
import random
import torch

from SwissArmyTransformer.mpu.transformer import standard_attention
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict


class PrefixTuningMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size_per_attention_head, num_attention_heads, prefix_len, layer_range=None):
        super().__init__()
        if layer_range is None:
            layer_range = [i for i in range(num_layers)]
        self.layer_range = layer_range
        self.prefix = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(2, num_attention_heads, prefix_len, hidden_size_per_attention_head)*0.01)
            for layer_id in range(num_layers)
        ])
        self.prefix_len = prefix_len

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
        layer_id = kw_args['layer_id']
        if layer_id in self.layer_range:
            prefix_k, prefix_v = self.prefix[kw_args['layer_id']]

            b, nh, seq_len, hidden_size = k.shape
            prefix_k = prefix_k.unsqueeze(0).expand(b, nh, -1, hidden_size)
            prefix_v = prefix_v.unsqueeze(0).expand(b, nh, -1, hidden_size)

            k = torch.cat((k, prefix_k), dim=2)
            v = torch.cat((v, prefix_v), dim=2)
            if mask.numel() > 1:
                mask_prefixed = torch.ones(self.prefix_len, device=mask.device, dtype=mask.dtype)
                mask_prefixed = mask_prefixed.expand(*(mask.size()[:-1]), -1)
                mask = torch.cat((mask, mask_prefixed), dim=-1)
        return old_impl(q, k, v, mask, dropout_fn, **kw_args)

PTuningV2Mixin = PrefixTuningMixin