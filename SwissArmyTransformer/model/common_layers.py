# -*- encoding: utf-8 -*-
'''
@File    :   components.py
@Time    :   2021/11/23 18:20:22
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from SwissArmyTransformer.mpu.utils import divide, split_tensor_along_last_dim
from SwissArmyTransformer.mpu.transformer import standard_attention, LayerNorm

class CrossAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                attention_dropout_prob, output_dropout_prob,
                init_method, enc_hidden_size=None, inner_hidden_size=None, output_layer_init_method=None):
        super(CrossAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        if inner_hidden_size is None:
            inner_hidden_size = hidden_size
        self.inner_hidden_size = inner_hidden_size
        if enc_hidden_size is None:
            enc_hidden_size = hidden_size
        self.enc_hidden_size = enc_hidden_size

        # To make user understand better, temporally not support model parallel
        world_size = 1
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)

        # To map encoder outputs
        self.kv_linear = torch.nn.Linear(
            enc_hidden_size, inner_hidden_size * 2
        )
        init_method(self.kv_linear.weight)

        # To map self
        self.q_linear = torch.nn.Linear(
            hidden_size, inner_hidden_size
        )
        init_method(self.q_linear.weight)

        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = torch.nn.Linear(
            inner_hidden_size,
            hidden_size,
        )
        output_layer_init_method(self.dense.weight)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)


    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, encoder_outputs, **kw_args):
        
        query_layer = self.q_linear(hidden_states)
        key_layer, value_layer = split_tensor_along_last_dim(self.kv_linear(encoder_outputs), 2)
        
        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(query_layer)
        key_layer = self._transpose_for_scores(key_layer)
        value_layer = self._transpose_for_scores(value_layer)
        
        context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)
        
        if self.training:
            output = self.output_dropout(output)
        
        return output
