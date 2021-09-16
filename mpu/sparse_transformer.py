# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math
import random
import argparse

import torch
import torch.nn.init as init
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm

from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .mappings import gather_from_model_parallel_region

import deepspeed

from .random import checkpoint
from .random import get_cuda_rng_tracker

from .utils import divide, sqrt
from .utils import split_tensor_along_last_dim
import torch.distributed as dist

class LayerNorm(FusedLayerNorm):
    def __init__(self, *args, pb_relax=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pb_relax = pb_relax
    def forward(self, x):
        if not self.pb_relax:
            return super().forward(x)
        return super().forward(x / (x.abs().max().detach()/8))

class GPT2ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence length, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, layer_id, output_layer_init_method=None,sparse_config=None,
                 finetune=False):
        super(GPT2ParallelSelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)

        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

        self.sparse_config = sparse_config

        if finetune: 
            # build new branch
            self.query_key_value_plus = ColumnParallelLinear(hidden_size, 3*hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
            self.dense_plus = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)

    def init_plus_from_old(self):
        self.query_key_value_plus.weight.data.copy_(self.query_key_value.weight.data)
        if hasattr(self.query_key_value_plus, 'bias') and hasattr(self.query_key_value, 'bias'):
            self.query_key_value_plus.bias.data.copy_(self.query_key_value.bias.data)
        
        self.dense_plus.weight.data.copy_(self.dense.weight.data)
        if hasattr(self.dense_plus, 'bias') and hasattr(self.dense, 'bias'):
            self.dense_plus.bias.data.copy_(self.dense.bias.data)
    def reset_sparse_config(self, config):
        self.sparse_config = config

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)


    def forward(self, hidden_states, mask, mem=None):
        sparse_config = self.sparse_config
        layout = sparse_config.layout
        if sparse_config.sparse_type == 'cuda_2d':
            assert hidden_states.size(1) == sparse_config.layout[-1]
            # [PAD]... [ROI1] text ... [BOI1] {layout[0]} 1024 {layout[1]} [EOI1] 4095 {layout[2]}
            hidden_states_plus = hidden_states[:, layout[1]:]
            hidden_states = hidden_states[:, :layout[1]]

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
        if mem is not None and len(mem) > 0:
            memk, memv = split_tensor_along_last_dim(mem, 2)
            mixed_key_layer = torch.cat((memk, mixed_key_layer), dim=1)
            mixed_value_layer = torch.cat((memv, mixed_value_layer), dim=1)
        
        if sparse_config.sparse_type == 'cuda_2d':
            mixed_raw_layer_plus = self.query_key_value_plus(hidden_states_plus)
            q1, k1, v1 = split_tensor_along_last_dim(mixed_raw_layer_plus, 3)

        dropout_fn = self.attention_dropout if self.training else None

        if sparse_config.sparse_type == 'standard':
            query_layer = self._transpose_for_scores(mixed_query_layer)
            
            key_layer = self._transpose_for_scores(mixed_key_layer)
            value_layer = self._transpose_for_scores(mixed_value_layer)
            
            context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn, layer_id=self.layer_id, txt_len=layout[0] if not self.training else -1)
            
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + \
                                    (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
            
        elif sparse_config.sparse_type == 'cuda_2d':
            context_layer0, context_layer1 = sparse_attention_2d_light(
                mixed_query_layer, mixed_key_layer, mixed_value_layer,
                q1, k1, v1,
                mask,
                n_head=self.num_attention_heads_per_partition,
                text_len=sparse_config.layout[0],
                kernel_size=sparse_config.kernel_size,
                kernel_size2=sparse_config.kernel_size2,
                attention_dropout=dropout_fn,
                text_start=(1-mask[...,-1,:]).sum().long().item()+1 if not self.training else -1,
                layer_id=self.layer_id
            )

        if sparse_config.sparse_type == 'cuda_2d':
            output_0 = self.dense(context_layer0)
            output_1 = self.dense_plus(context_layer1)
            output = torch.cat((output_0, output_1), dim=1)
        else:
            output = self.dense(context_layer)
            
        if self.training:
            output = self.output_dropout(output)
        
        if mem is not None:
            new_mem = mixed_raw_layer.detach()[..., -(mixed_raw_layer.shape[-1] // 3 * 2):].contiguous()
        else:
            new_mem = None
        return output, new_mem


@torch.jit.script
def gelu_impl(x):
     """OpenAI's gelu implementation."""
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def gelu(x): 
    return gelu_impl(x)


class GPT2ParallelMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(GPT2ParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4*hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]

        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        if self.training:
            output = self.dropout(output)
        return output


class GPT2ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 layer_id,
                 output_layer_init_method=None,
                 sandwich_ln=True,
                 sparse_config=argparse.Namespace(sparse_type='standard'),
                 finetune=False
                 ):
        super(GPT2ParallelTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = GPT2ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            output_layer_init_method=output_layer_init_method,
            sparse_config=sparse_config,
            finetune=finetune
            )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)
        self.sandwich_ln = sandwich_ln
        if sandwich_ln:
            self.third_layernorm = LayerNorm(hidden_size,
                                                    eps=layernorm_epsilon)
            self.fourth_layernorm = LayerNorm(hidden_size,
                                                    eps=layernorm_epsilon)

        # MLP
        self.mlp = GPT2ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

        self.sparse_config = sparse_config

    def reset_sparse_config(self, config):
            self.sparse_config = config
            self.attention.reset_sparse_config(config)
    
    def forward(self, hidden_states, ltor_mask, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output1 = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, qkv = self.attention(layernorm_output1, ltor_mask, mem)

        # Third LayerNorm
        if self.sandwich_ln:
            attention_output = self.third_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Fourth LayerNorm
        if self.sandwich_ln:
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output, qkv

def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT2ParallelTransformer(torch.nn.Module):
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """
    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 sandwich_ln=True,
                 sparse_config=argparse.Namespace(sparse_type='standard'),
                 finetune=False
                 ):
        super(GPT2ParallelTransformer, self).__init__()
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_memory_length = max_memory_length
        self.max_sequence_length = max_sequence_length

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                      num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                        hidden_size)
        # Initialize the position embeddings.
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        if finetune:
            self.position_embeddings_plus = torch.nn.Embedding(4096, # FIXME
                                                            hidden_size)
            # Initialize the position embeddings.
            torch.nn.init.normal_(self.position_embeddings_plus.weight, mean=0.0, std=init_method_std)

        def get_layer(layer_id):
            return GPT2ParallelTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                layer_id,
                output_layer_init_method=output_layer_init_method,
                sandwich_ln=sandwich_ln,
                sparse_config=sparse_config,
                finetune=finetune
                )

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint
        self.sparse_config = sparse_config

    def init_plus_from_old(self):
        self.position_embeddings_plus.weight.data.view(4, 1024, -1).copy_(self.position_embeddings.weight.data[-1024:]) # FIXME
        for layer in self.layers:
            layer.attention.init_plus_from_old()

    def reset_sparse_config(self, config):
            self.sparse_config = config
            for layer in self.layers:
                layer.reset_sparse_config(config)

    def forward(self, hidden_states, position_ids, attention_mask, *mems):

        batch_size, query_length = hidden_states.size()[:2]
        memory_length = mems[0].size(1) if mems else 0
        key_length = query_length + memory_length

        # legacy
        if isinstance(attention_mask, int) or attention_mask.numel() == 1:
            # if given a int "sep", means the seperation of full attention part and single direction part
            # attention mask is the beginning postion of B region, \in [0, query_len)
            sep = attention_mask
            # conventional transformer
            def build_mask_matrix(query_length, key_length, sep):
                m = torch.ones((1, query_length, key_length), device=hidden_states.device, dtype=hidden_states.dtype)
                assert query_length <= key_length
                m[0, :, -query_length:] = torch.tril(m[0, :, -query_length:])
                m[0, :, :sep + (key_length - query_length)] = 1
                m = m.unsqueeze(1)
                return m
            attention_mask = build_mask_matrix(query_length, key_length, sep)

    
        if self.sparse_config.sparse_type == 'cuda_2d':
            position = position_ids[..., :self.sparse_config.layout[1]]
            position_plus = position_ids[..., self.sparse_config.layout[1]:]
            position_embeddings = torch.cat(
                (self.position_embeddings(position), self.position_embeddings_plus(position_plus)), dim=-2)
        else:
            position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        mem_layers = []
        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, mask, mems_ = inputs[0], inputs[1], inputs[2:]
            
                for i, layer in enumerate(layers_):
                    if mems_:
                        mem_i_ = mems_[i]  
                    elif self.max_memory_length > 0:
                        mem_i_ = []
                    else:
                        mem_i_ = None
                    x_, qkv = layer(x_, mask, mem=mem_i_)
                    if self.max_memory_length > 0:
                        mem_layers.append(qkv)
                return x_
            return custom_forward

        attention_mask_saved = attention_mask
        
        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask_saved]

                if mems:
                    args += mems[l: l + chunk_length]

                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask_saved]
                if mems:
                    mem_i = mems[i]  
                elif self.max_memory_length > 0:
                    mem_i = []
                else:
                    mem_i = None
                hidden_states, qkv = layer(*args, mem=mem_i)
                if self.max_memory_length > 0:
                    mem_layers.append(qkv) 

        # Final layer norm.
        output = self.final_layernorm(hidden_states)

        return (output, *mem_layers)
        

def _chunk(x, w, times):
    '''convert into overlapping chunkings. Chunk size = times * w, overlap size = w
    Args:
        x: [b, np, s, hn]
        ...
    '''
    s = x.size(2)
    # x pad to [b, np, s+xx to k*w + w*(times-1), hn]
    assert s % w == 0
    npad = (times-1) * w
    x = torch.nn.functional.pad(x, (0, 0, npad, 0), value=0)

    x = x.view(x.size(0), x.size(1),  x.size(2) // w, w, x.size(3))

    chunk_size = list(x.size())
    chunk_stride = list(x.stride())

    chunk_size[2] = chunk_size[2] - times + 1

    chunk_size[3] = w * times

    return x.as_strided(size=chunk_size, stride=chunk_stride)

def standard_attention(query_layer, key_layer, value_layer, attention_mask, attention_dropout=None, layer_id = -1, txt_len=-1):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    if len(attention_mask.shape) == 3:
        attention_mask = attention_mask.unsqueeze(1)
    # Raw attention scores. [b, np, s, s]
    attention_scores = torch.matmul(query_layer / math.sqrt(query_layer.shape[-1]), key_layer.transpose(-1, -2))
    
    # Apply the left to right attention mask.
    if attention_mask.shape[2] > 1:
        attention_scores = torch.mul(attention_scores, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    
    # Attention probabilities [b, np, s, s]
    attention_probs = F.softmax(attention_scores, dim=-1)

    if txt_len > 0:
        t = key_layer.shape[-2] - txt_len - 1
        if t // 32 <= 32:
            # line = attention_probs[..., :, 1:txt_len].max(dim=-1, keepdim=True)[0]
            # tmask = attention_probs[..., :, 1:txt_len] >= line
            attention_probs[..., :, 1:txt_len] *= 6 if txt_len <= 10 else 4
            attention_probs /= attention_probs.sum(dim=-1, keepdim=True)[0]

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)
    # Context layer.
    # [b, np, s, hn]

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer


def sparse_attention_2d_light(q0, k0, v0, q1, k1, v1, attention_mask, n_head, text_len=64, kernel_size=9, kernel_size2=7, attention_dropout=None, text_start = -1, layer_id=-1, **kwargs):
    '''
    q0, k0, v0: [batch_size, 1088, hidden_size]
    q1, k1, v1: [batch_size, 4096, h2]
    n_head: int
    attention_mask: [batch_size, 1088, 1088]
    '''
    from .local_attention_function import f_similar, f_weighting
    b, s0, h0 = q0.shape
    b, s1, h1 = q1.shape
    assert v1.shape[-1] == h0, 'q1, k1 can be smaller, but v1 cannot.'
    h = h0 // n_head
    l0, l1 = int(math.sqrt(s0-text_len)+0.0001), int(math.sqrt(s1)+0.0001)

    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.reshape(b, s0, n_head, h).permute(0, 2, 3, 1)
    # standard attention for level 0
    attention_scores = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T)
    attention_scores = torch.mul(attention_scores, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    attention_probs0 = F.softmax(attention_scores, dim=-1)
    if text_start > 0:
        attention_probs0[..., :, text_start:text_len-2] *= 1
        attention_probs0 /= attention_probs0.sum(dim=-1, keepdim=True)[0]
    # local attention for level 1
    q1 = (q1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1) / math.sqrt(h1//n_head)).contiguous().view(b*n_head, h1//n_head, l1, l1)
    k1 = k1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    v1 = v1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    scores_1_to_1 = f_similar(q1, k1, kernel_size*2-1, kernel_size, True)    
    # attention_probs1 = F.softmax(scores_1_to_1, dim=-1)

    # cross attention
    k0T = k0T[..., -l0**2:].reshape(b*n_head, h, l0, l0).contiguous()
    scores_1_to_0 = f_similar(q1, k0T, kernel_size2, kernel_size2, False) # [b*n_head, l1, l1, field]
    scores_1 = torch.cat(
        (
            scores_1_to_0.view(b*n_head, -1, scores_1_to_0.shape[3]),
            scores_1_to_1.view(b*n_head, -1, scores_1_to_1.shape[3])
        ),
        dim=-1)
    attention_probs1 = F.softmax(scores_1, dim=-1)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs0 = attention_dropout(attention_probs0)
            attention_probs1 = attention_dropout(attention_probs1)
        
    # weighting for level 0
    context0 = torch.matmul(attention_probs0, v0) # [b, n_head, s0, h]
    # weighting for level 1
    probs_1_to_1 = attention_probs1[:, :, -scores_1_to_1.shape[3]:].view_as(scores_1_to_1)
    context1_to_1 = f_weighting(v1, probs_1_to_1.contiguous(), kernel_size*2-1, kernel_size, True)
    context1 = context1_to_1.view(b, n_head * h, l1**2)
    # weighting for cross attention
    probs_1_to_0 = attention_probs1[:, :, :scores_1_to_0.shape[3]].view_as(scores_1_to_0)
    v0_part = v0[:, :, -l0**2:].transpose(-1, -2).contiguous().view(b*n_head, h, l0, l0)
    context1_to_0 = f_weighting(v0_part, probs_1_to_0.contiguous(), kernel_size2, kernel_size2, False)
    context1_to_0 = context1_to_0.view(b, n_head * h, l1**2)
    context1 = context1 + context1_to_0
    return context0.transpose(1, 2).reshape(b, s0, h0), context1.transpose(-1, -2)