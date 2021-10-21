# -*- encoding: utf-8 -*-
'''
@File    :   mixins.py
@Time    :   2021/10/01 17:52:40
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
from mpu import ColumnParallelLinear, RowParallelLinear
from mpu.transformer import unscaled_init_method, LayerNorm

class BaseMixin(torch.nn.Module):
    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params
    def reinit(self, transformer, *pre_mixins):
        # reload the initial params from previous trained modules
        pass

class PositionEmbeddingMixin(BaseMixin):
    def __init__(self, additional_sequence_length, hidden_size, 
                init_method_std=0.02, reinit_slice=slice(-1024, None)
        ):
        super(PositionEmbeddingMixin, self).__init__()
        self.reinit_slice = reinit_slice
        self.position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)
    def reinit(self, transformer, *pre_mixins):
        old_weights = transformer.position_embeddings.weight.data[self.reinit_slice]
        old_len, hidden_size = old_weights.shape
        assert hidden_size == self.position_embeddings.weight.shape[-1]
        self.position_embeddings.weight.data.view(-1, old_len, hidden_size).copy_(old_weights)

class AttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02)
        ):
        super(AttentionMixin, self).__init__()
        self.num_layers = num_layers # replace attention in the LAST n layers
        self.query_key_value = torch.nn.ModuleList(
            [ColumnParallelLinear(hidden_size, 3*hidden_size,stride=3,
                gather_output=False,init_method=init_method)
                for layer_id in range(num_layers)
            ])
        self.dense = torch.nn.ModuleList(
            [RowParallelLinear(hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
                for layer_id in range(num_layers)
            ])
    def reinit(self, transformer, *pre_mixins):
        start_layer = len(transformer.layers) - self.num_layers
        assert start_layer >= 0
        for layer_id in range(self.num_layers):
            old_attention = transformer.layers[start_layer + layer_id].attention
            self.query_key_value[layer_id].weight.data.copy_(old_attention.query_key_value.weight.data)
            self.query_key_value[layer_id].bias.data.copy_(old_attention.query_key_value.bias.data)
            self.dense[layer_id].weight.data.copy_(old_attention.dense.weight.data)
            self.dense[layer_id].bias.data.copy_(old_attention.dense.bias.data)

class VideoAttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                video_hidden_size,
                video_n_head,
                attention_dropout_prob,
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02)
        ):
        super(VideoAttentionMixin, self).__init__()
        self.num_layers = num_layers # replace attention in the LAST n layers
        self.hidden_size = hidden_size
        self.video_hidden_size = video_hidden_size
        self.video_n_head = video_n_head
        self.query_key_value = torch.nn.ModuleList(
            [ColumnParallelLinear(video_hidden_size, 3*video_hidden_size,stride=3,
                gather_output=False,init_method=init_method)
                for layer_id in range(num_layers)
            ])
        self.dense = torch.nn.ModuleList(
            [RowParallelLinear(video_hidden_size,
                video_hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
                for layer_id in range(num_layers)
            ])
        self.attention_dropout = torch.mm.ModuleList(
            [torch.nn.Dropout(attention_dropout_prob)
             for layer_id in range(num_layers)]
        )
        self.startmap_i2v = RowParallelLinear(hidden_size,
                video_hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
        self.keymap_i2v = torch.nn.Modulelist(
            [RowParallelLinear(hidden_size,
                video_hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
             for layer_id in range(num_layers)
             ])
        self.valmap_i2v = torch.nn.Modulelist(
            [RowParallelLinear(hidden_size,
                video_hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
             for layer_id in range(num_layers)
             ])

    def reinit(self, transformer, *pre_mixins):
        assert self.num_layers == len(transformer.layers)
        # initial with pseudo-inverse
        # dense_weight = torch.linalg.pinv(self.densemap_i2v.weight.data.type(torch.float32)).type(torch.float16)
        # self.densemap_v2i.weight.data.copy_(torch.clamp(dense_weight, min=-5, max=5))
        # for layer_id in range(self.num_layers):
        #     old_attention_weight = transformer.layers[layer_id].attention.query_key_value.weight.data
        #     # y^T = A x^T
        #     new_weight = old_attention_weight.reshape(3, self.hidden_size, self.hidden_size)
        #     new_weight = torch.matmul(torch.matmul(self.densemap_i2v.weight.data, new_weight), self.densemap_v2i.weight.data)
        #     new_weight = new_weight.reshape(3*self.video_hidden_size, self.video_hidden_size)
        #     self.query_key_value[layer_id].weight.data.copy_(new_weight)
        #     old_attention_bias = transformer.layers[layer_id].attention.query_key_value.bias.data
        #     # Q, K, V
        #     new_bias = torch.cat((torch.matmul(self.densemap_i2v.weight.data, old_attention_bias[..., :self.hidden_size]), 
        #                             torch.matmul(old_attention_bias[..., self.hidden_size:2*self.hidden_size], self.densemap_v2i.weight.data), 
        #                             torch.matmul(old_attention_bias[..., 2*self.hidden_size:], self.densemap_v2i.weight.data)), 
        #                            dim = -1
        #                            )
        #     self.query_key_value[layer_id].bias.data.copy_(new_bias)

class VideoMLPMixin(BaseMixin):
    def __init__(self, num_layers, video_hidden_size, hidden_size, output_dropout_prob, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=None):
        super(VideoMLPMixin, self).__init__()
        self.num_layers = num_layers
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = torch.nn.ModuleList(
            [ColumnParallelLinear(
            video_hidden_size,
            4*video_hidden_size,
            gather_output=False,
            init_method=init_method)
            for layer_id in range(num_layers)]
        )
        # Project back to h.
        self.dense_4h_to_h = torch.nn.ModuleList(
            [RowParallelLinear(
            4*video_hidden_size,
            video_hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
            for layer_id in range(num_layers)]
        )
        self.dropout = torch.nn.ModuleList(
            [torch.nn.Dropout(output_dropout_prob)
             for layer_id in range(num_layers)]
        )
        self.endmap_v2i = RowParallelLinear(video_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
    def reinit(self, transformer, *pre_mixins):
        assert self.num_layers == len(transformer.layers)