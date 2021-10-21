# -*- encoding: utf-8 -*-
# here put the import lib
import os
import sys
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

from .base_model import BaseModel
from .mixins import *

from mpu.transformer import split_tensor_along_last_dim, standard_attention
from mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker


class ParallelLinearMixin(BaseMixin):
    def __init__(self, input_size, output_size, bias=True,
                 init_method=torch.nn.init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ParallelLinearMixin, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.parallel_linear = ColumnParallelLinear(
            input_size, output_size, bias=bias, gather_output=True,
            init_method=init_method, stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test)

    def forward(self, input_):
        return self.parallel_linear(input_)

class ParallelDoubleLayerLinearMixin(BaseMixin):
    def __init__(self, input_size, hidden_size, output_size, bias=True,
                 init_method=torch.nn.init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False):
        super(ParallelDoubleLayerLinearMixin, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.column_parallel_linear = ColumnParallelLinear(
            input_size, hidden_size, bias=bias, gather_output=False,
            init_method=init_method, stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test)
        self.row_parallel_linear = RowParallelLinear(
            hidden_size, output_size, bias=bias,
            init_method=init_method, stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test)
    
    def forward(self, input_):
        return self.row_parallel_linear(self.column_parallel_linear(input_))

class ParameterMixin(BaseMixin):
    def __init__(self, size, init_value=0.):
        super(ParameterMixin, self).__init__()
        self.parameter = nn.Parameter(torch.ones(size) * init_value)

class RetrievalModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.layout = args.layout
        self.num_layers = args.num_layers
        self.retrieval_num_layers = args.retrieval_num_layers
        self.txt_img_split = args.txt_img_split
        self.add_pos_embed = args.retrieval_pos_embed
        
        self.txt_linear_idx = len(self.mixins)
        self.img_linear_idx = len(self.mixins) + 1
        self.mixins.extend([
            ParallelLinearMixin(
                args.hidden_size, args.retrieval_size),
            ParallelLinearMixin(
                args.hidden_size, args.retrieval_size)
            ])
    
        if args.txt_img_split:
            self.attention_addition_idx = len(self.mixins)
            self.mixins.append(AttentionMixin(
                num_layers=self.retrieval_num_layers,
                hidden_size=args.hidden_size
            ))

        self.temp_idx = len(self.mixins)
        self.mixins.append(ParameterMixin(size=1, init_value=args.retrieval_init_temp))
        self.retrieval_temp_scale = args.retrieval_temp_scale
            
        if args.retrieval_pos_embed:
            self.pos_embed_idx = len(self.mixins)
            self.mixins.append(PositionEmbeddingMixin(
                2, args.hidden_size,
                reinit_slice=None
            ))
    
    def position_embedding_forward(self, position_ids, *other_tensors):
        if self.add_pos_embed:
            position_embeddings = torch.cat(
                    (
                        self.transformer.position_embeddings(position_ids[:, :-2]),
                        self.mixins[self.pos_embed_idx].position_embeddings(position_ids[:, -2:])
                    ),
                    dim=-2
                )
        else:
            position_embeddings = self.transformer.position_embeddings(position_ids)
        return position_embeddings
    
    def attention_forward(self, hidden_states, mask, *other_tensors, layer_id=None):
        if not self.txt_img_split or layer_id < self.num_layers - self.retrieval_num_layers:
            attn_module = self.transformer.layers[layer_id].attention
            
            mixed_raw_layer = attn_module.query_key_value(hidden_states)
            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
            
            dropout_fn = attn_module.attention_dropout if self.training else None
            
            
            query_layer = attn_module._transpose_for_scores(mixed_query_layer)
            key_layer = attn_module._transpose_for_scores(mixed_key_layer)
            value_layer = attn_module._transpose_for_scores(mixed_value_layer)
            
            context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
            output = attn_module.dense(context_layer)
            
            if self.training:
                output = attn_module.output_dropout(output)
                
            return output, None
        else:
            attn_module = self.transformer.layers[layer_id].attention
            qkv_addition = self.mixins[self.attention_addition_idx].query_key_value[layer_id - self.num_layers + self.retrieval_num_layers]
            dense_addition = self.mixins[self.attention_addition_idx].dense[layer_id - self.num_layers + self.retrieval_num_layers]
            
            dropout_fn = attn_module.attention_dropout if self.training else None
            
            txt_hidden_states = torch.cat([hidden_states[:, :self.layout[0]-5], hidden_states[:, -1:]], dim=1)
            txt_mask = torch.cat([mask[..., :self.layout[0]-5], mask[..., -1:]], dim=-1)
            txt_mask = torch.cat([txt_mask[..., :self.layout[0]-5, :], txt_mask[..., -1:, :]], dim=-2)
            img_hidden_states = hidden_states[:, self.layout[0]-5:-1]
            img_mask = mask[..., self.layout[0]-5:-1, self.layout[0]-5:-1]
            
            txt_mixed_raw_layer = attn_module.query_key_value(txt_hidden_states)
            img_mixed_raw_layer = qkv_addition(img_hidden_states)
            (txt_mixed_query_layer,
             txt_mixed_key_layer,
             txt_mixed_value_layer) = split_tensor_along_last_dim(txt_mixed_raw_layer, 3)
            (img_mixed_query_layer,
             img_mixed_key_layer,
             img_mixed_value_layer) = split_tensor_along_last_dim(img_mixed_raw_layer, 3)

            txt_query_layer = attn_module._transpose_for_scores(txt_mixed_query_layer)
            txt_key_layer = attn_module._transpose_for_scores(txt_mixed_key_layer)
            txt_value_layer = attn_module._transpose_for_scores(txt_mixed_value_layer)
            img_query_layer = attn_module._transpose_for_scores(img_mixed_query_layer)
            img_key_layer = attn_module._transpose_for_scores(img_mixed_key_layer)
            img_value_layer = attn_module._transpose_for_scores(img_mixed_value_layer)

            txt_context_layer = standard_attention(txt_query_layer, txt_key_layer, txt_value_layer, txt_mask, dropout_fn)
            txt_context_layer = txt_context_layer.permute(0, 2, 1, 3).contiguous()
            new_txt_context_layer_shape = txt_context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
            txt_context_layer = txt_context_layer.view(*new_txt_context_layer_shape)
            txt_output = attn_module.dense(txt_context_layer)
            
            img_context_layer = standard_attention(img_query_layer, img_key_layer, img_value_layer, img_mask, dropout_fn)
            img_context_layer = img_context_layer.permute(0, 2, 1, 3).contiguous()
            new_img_context_layer_shape = img_context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
            img_context_layer = img_context_layer.view(*new_img_context_layer_shape)
            img_output = dense_addition(img_context_layer)

            output = torch.cat([txt_output[:, :-1], img_output, txt_output[:, -1:]], dim=1)
            
            if self.training:
                output = attn_module.output_dropout(output)
                
            return output, None
    
    def final_forward(self, logits, *other_tensors):
        txt_logits = logits[:, -1, :]
        img_logits = logits[:, -2, :]
        return (self.mixins[self.txt_linear_idx](txt_logits), self.mixins[self.img_linear_idx](img_logits), self.mixins[self.temp_idx].parameter * self.retrieval_temp_scale)
    
    def disable_untrainable_params(self):
        for i in range(self.num_layers - self.retrieval_num_layers):
            self.transformer.layers[i].requires_grad_(False)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('RetrievalModel', 'retrieval model configurations')
        group.add_argument('--txt-img-split', action='store_true')
        group.add_argument('--retrieval-init-temp', type=float, default=0.)
        group.add_argument('--retrieval-temp-scale', type=float, default=1.)
        group.add_argument('--retrieval-mode', type=str, default='txt2img',
                            choices=['txt2img', 'img2txt', 'symmetric'])
        group.add_argument('--retrieval-num-layers', type=int, default=1)
        group.add_argument('--retrieval-size', type=int, default=1024)
        group.add_argument('--retrieval-pos-embed', action='store_true')
        group.add_argument("--layout", type=str, default='64,1088')
        return parser