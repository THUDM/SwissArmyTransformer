# coding=utf-8
# -*- encoding: utf-8 -*-
'''
@File    :   transformer_defaults.py
@Time    :   2022/06/01 21:44:17
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import math
import copy
import torch
import torch.nn.functional as F

from SwissArmyTransformer import mpu
from SwissArmyTransformer.mpu.initialize import get_model_parallel_world_size
from SwissArmyTransformer.mpu.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from SwissArmyTransformer.mpu.mappings import gather_from_model_parallel_region, copy_to_model_parallel_region

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

from SwissArmyTransformer.mpu.utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim

def standard_attention(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if log_attention_weights is not None:
        attention_scores += log_attention_weights

    if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
        # if auto-regressive, skip
        attention_scores = torch.mul(attention_scores, attention_mask) - \
                           10000.0 * (1.0 - attention_mask)

    attention_probs = F.softmax(attention_scores, dim=-1)

    if attention_dropout is not None:
        if mpu.get_cuda_rng_tracker is not None:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = attention_dropout(attention_probs)
        else:
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_forward_default(self, hidden_states, mask, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].attention
    attention_fn = standard_attention
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_raw_layer = self.query_key_value(hidden_states)
    (mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

    dropout_fn = self.attention_dropout if self.training else None

    query_layer = self._transpose_for_scores(mixed_query_layer)
    key_layer = self._transpose_for_scores(mixed_key_layer)
    value_layer = self._transpose_for_scores(mixed_value_layer)

    context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = self.dense(context_layer)

    if self.training:
        output = self.output_dropout(output)
    return output

def cross_attention_forward_default(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].cross_attention
    attention_fn = standard_attention
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_query_layer = self.query(hidden_states)
    mixed_x_layer = self.key_value(encoder_outputs)
    (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)

    dropout_fn = self.attention_dropout if self.training else None
    # Reshape and transpose [b, np, s, hn]
    query_layer = self._transpose_for_scores(mixed_query_layer)
    key_layer = self._transpose_for_scores(mixed_key_layer)
    value_layer = self._transpose_for_scores(mixed_value_layer)

    context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn, **kw_args)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    # [b, s, hp]
    context_layer = context_layer.view(*new_context_layer_shape)

    # Output. [b, s, h]
    output = self.dense(context_layer)
    if self.training:
        output = self.output_dropout(output)
    return output

def mlp_forward_default(self, hidden_states, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].mlp
    intermediate_parallel = self.dense_h_to_4h(hidden_states)
    intermediate_parallel = self.activation_func(intermediate_parallel)
    output = self.dense_4h_to_h(intermediate_parallel)
    return output

def word_embedding_forward_default(self, input_ids, output_cross_layer, **kw_args):
    return self.transformer.word_embeddings(input_ids)

def position_embedding_forward_default(self, position_ids, output_cross_layer, **kw_args):
    return self.transformer.position_embeddings(position_ids)

def final_forward_default(self, logits, **kw_args):
    return F.linear(logits, self.transformer.word_embeddings.weight)

def layer_forward_default(self, hidden_states, mask, *args, **kw_args):
    '''
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
    '''
    self = self.transformer.layers[kw_args['layer_id']]
    # Layer norm at the begining of the transformer layer.
    attention_input = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output = self.attention(attention_input, mask, **kw_args)

    # Third LayerNorm
    if self.layernorm_order == 'sandwich':
        attention_output = self.third_layernorm(attention_output)
    
    # Residual connection.
    if self.layernorm_order == 'post':
        hidden_states = attention_input + attention_output
    else:
        hidden_states = hidden_states + attention_output

    
    mlp_input = self.post_attention_layernorm(hidden_states)

    if self.is_decoder:
        encoder_outputs = kw_args['encoder_outputs']
        if encoder_outputs is not None:
            assert 'cross_attention_mask' in kw_args
            # Cross attention
            attention_output = self.cross_attention(mlp_input, **kw_args)
            # Residual connection.
            hidden_states = hidden_states + attention_output
            # Layer norm post the cross attention
            mlp_input = self.post_cross_attention_layernorm(hidden_states)

    # MLP.
    mlp_output = self.mlp(mlp_input, **kw_args)

    # Fourth LayerNorm
    if self.layernorm_order == 'sandwich':
        mlp_output = self.fourth_layernorm(mlp_output)

    # Second residual connection.
    if self.layernorm_order == 'post':
        output = mlp_input + mlp_output
    else:
        output = hidden_states + mlp_output

    return output

HOOKS_DEFAULT = {
    'attention_fn': standard_attention,
    'attention_forward': attention_forward_default,
    'cross_attention_forward': cross_attention_forward_default,
    'mlp_forward': mlp_forward_default,
    'word_embedding_forward': word_embedding_forward_default,
    'position_embedding_forward': position_embedding_forward_default,
    'final_forward': final_forward_default,
    'layer_forward': layer_forward_default
}