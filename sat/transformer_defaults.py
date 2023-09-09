# coding=utf-8
# -*- encoding: utf-8 -*-
'''
@File    :   transformer_defaults.py
@Time    :   2022/06/01 21:44:17
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import math
import torch
import torch.nn.functional as F

from sat import mpu

from sat.mpu.utils import split_tensor_along_last_dim

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

def attention_fn_default(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # expand head dim to query dim, if necessary
    # only useful for multi-query attention
    batch_size, num_query_heads = query_layer.shape[:2] # [b, np, s, hn]
    num_kv_heads = key_layer.shape[1] # [b, np, s, hn]
    key_layer = key_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *key_layer.shape[2:])
    value_layer = value_layer.unsqueeze(2).expand(-1, -1, num_query_heads//num_kv_heads, -1, -1).contiguous().view(batch_size, num_query_heads, *value_layer.shape[2:])

    is_low_triangle = (attention_mask == torch.ones_like(attention_mask, dtype=torch.float).tril()).all()
    is_full = (attention_mask is None) or (attention_mask > 0).all()

    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score and (is_full or is_low_triangle):
        # Pytorch 2.0 attention uses very much memory if attention_mask is float, and has NaN bug if attention_mask is None.
        dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
        return torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, 
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=not is_full
        )
    else:
        return standard_attention(
            query_layer, key_layer, value_layer, attention_mask,
            attention_dropout=attention_dropout, log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score, **kwargs
        )

def attention_forward_default(self, hidden_states, mask, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].attention
    attention_fn = attention_fn_default
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_raw_layer = self.query_key_value(hidden_states)
    (mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

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
    attention_fn = attention_fn_default
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

    context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn, cross_attention=True, **kw_args)
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

from sat.mpu import gather_from_model_parallel_region
def final_forward_default(self, logits, **kw_args):
    logits_parallel = F.linear(logits, self.transformer.word_embeddings.weight)
    if not kw_args['parallel_output']:
        logits_parallel = gather_from_model_parallel_region(logits_parallel)
    return logits_parallel

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

    # DropPath for attention
    if self.training and self.drop_path > 0.:
        if mpu.get_cuda_rng_tracker is not None:
            # drop_path must use model parallel rng tracker
            # the tracker is initialized as seed of `seed + model_parallel_rank`
            # deepspeed act-ckpt record the model parallel tracker states
            with mpu.get_cuda_rng_tracker().fork():
                # drop_path percentage 0, others 1/(1-p)
                random_tensor = (1-self.drop_path
                                  + torch.rand((attention_output.shape[0],), dtype=attention_output.dtype, device=attention_output.device)).floor_() / (1-self.drop_path)
                attention_output = random_tensor.view(-1, 1, 1) * attention_output
    
    # Residual connection.
    if self.layernorm_order == 'post':
        hidden_states = attention_input + attention_output
        mlp_input = self.post_attention_layernorm(hidden_states)
    else:
        hidden_states = hidden_states + attention_output

    if self.is_decoder:
        encoder_outputs = kw_args['encoder_outputs']
        if encoder_outputs is not None:
            assert 'cross_attention_mask' in kw_args
            # Cross attention
            if self.layernorm_order == 'post':
                attention_output = self.cross_attention(mlp_input, **kw_args)
                # Residual connection.
                hidden_states = mlp_input + attention_output
                # Layer norm post the cross attention
                mlp_input = self.post_cross_attention_layernorm(hidden_states)
            else:
                cross_input = self.post_cross_attention_layernorm(hidden_states)
                attention_output = self.cross_attention(cross_input, **kw_args)
                hidden_states = hidden_states + attention_output

    if self.layernorm_order != 'post':
        mlp_input = self.post_attention_layernorm(hidden_states)    

    # MLP.
    mlp_output = self.mlp(mlp_input, **kw_args)

    # Fourth LayerNorm
    if self.layernorm_order == 'sandwich':
        mlp_output = self.fourth_layernorm(mlp_output)

    # DropPath for mlp
    if self.training and self.drop_path > 0.:
        if mpu.get_cuda_rng_tracker is not None:
            with mpu.get_cuda_rng_tracker().fork():
                random_tensor = (1-self.drop_path
                                  + torch.rand((mlp_output.shape[0],), dtype=mlp_output.dtype, device=mlp_output.device)).floor_() / (1-self.drop_path)
                mlp_output = random_tensor.view(-1, 1, 1) * mlp_output

    # Second residual connection.
    if self.layernorm_order == 'post':
        output = mlp_input + mlp_output
    else:
        output = hidden_states + mlp_output

    return output

HOOKS_DEFAULT = {
    'attention_fn': attention_fn_default,
    'attention_forward': attention_forward_default,
    'cross_attention_forward': cross_attention_forward_default,
    'mlp_forward': mlp_forward_default,
    'word_embedding_forward': word_embedding_forward_default,
    'position_embedding_forward': position_embedding_forward_default,
    'final_forward': final_forward_default,
    'layer_forward': layer_forward_default
}

ARGS_DEFAULT = {
    'embedding_dropout_prob': ('hidden_dropout', 0),
    'attention_dropout_prob': ('attention_dropout', 0),
    'output_dropout_prob': ('hidden_dropout', 0),
    'inner_hidden_size': ('inner_hidden_size', None),
    'hidden_size_per_attention_head': ('hidden_size_per_attention_head', None),
    'checkpoint_activations': ('checkpoint_activations', False),
    'checkpoint_num_layers': ('checkpoint_num_layers', 1),
    'checkpoint_skip_layers': ('checkpoint_skip_layers', 0),
    'is_decoder': ('is_decoder', False),
    'cross_attn_hidden_size': ('cross_attn_hidden_size', None),
    'use_final_layernorm': ('use_final_layernorm', True),
    'layernorm_epsilon': ('layernorm_epsilon', 1e-5),
    'use_bias': ('use_bias', True),
    'use_qkv_bias': ('use_qkv_bias', False),
    'num_multi_query_heads': ('num_multi_query_heads', 0),
    'drop_path': ('drop_path', 0.),
}