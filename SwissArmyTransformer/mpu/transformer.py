# coding=utf-8
# rewritten, Copyright (c) 2021, Ming Ding.  All rights reserved.
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
import copy
import torch
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm

from SwissArmyTransformer import mpu
from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from .mappings import gather_from_model_parallel_region, copy_to_model_parallel_region

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

from .utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from .utils import split_tensor_along_last_dim


class LayerNorm(FusedLayerNorm):
    def __init__(self, *args, pb_relax=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pb_relax = pb_relax

    def forward(self, x):
        if not self.pb_relax:
            return super().forward(x)
        return super().forward(x / (x.abs().max().detach() / 8))


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


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True,
                 hooks={}):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition

        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3 * self.inner_hidden_size,
            stride=3,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            module=self,
            name="query_key_value"
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            module=self,
            name="dense"
        )
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

    def forward(self, hidden_states, mask, *args, **kw_args):
        if 'attention_forward' in self.hooks:
            return self.hooks['attention_forward'](hidden_states, mask, **kw_args)
        else:
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


class CrossAttention(torch.nn.Module):
    """Parallel cross-attention layer for Transformer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method,
                 layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True, hooks={}):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        # Strided linear layer.
        self.query = ColumnParallelLinear(hidden_size, self.inner_hidden_size,
                                          gather_output=False,
                                          init_method=init_method, bias=bias, module=self, name="query")
        self.key_value = ColumnParallelLinear(hidden_size, 2 * self.inner_hidden_size,
                                              stride=2,
                                              gather_output=False,
                                              init_method=init_method, bias=bias, module=self, name="key_value")
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(
            self.inner_hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method, bias=bias, module=self, name="dense")
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

    def forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        # hidden_states: [b, s, h]
        if 'cross_attention_forward' in self.hooks:
            return self.hooks['cross_attention_forward'](hidden_states, cross_attention_mask, encoder_outputs, **kw_args)
        else:
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

            context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn,
                                         cross_attention=True, **kw_args)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            # [b, s, hp]
            context_layer = context_layer.view(*new_context_layer_shape)

            # Output. [b, s, h]
            output = self.dense(context_layer)
            if self.training:
                output = self.output_dropout(output)

            return output


class MLP(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method, inner_hidden_size=None,
                 output_layer_init_method=None, layer_id=None, hooks={}, bias=True, activation_func=gelu):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            self.hidden_size,
            self.inner_hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=bias,
            module=self,
            name="dense_h_to_4h"
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            self.inner_hidden_size,
            self.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            bias=bias,
            module=self,
            name="dense_4h_to_h"
        )
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args)
        else:
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = self.activation_func(intermediate_parallel)
            output = self.dense_4h_to_h(intermediate_parallel)

        if self.training:
            output = self.dropout(output)
        return output


class BaseTransformerLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            layernorm_epsilon,
            init_method,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            output_layer_init_method=None,
            sandwich_ln=True,
            layernorm=LayerNorm,
            is_decoder=False,
            use_bias=True,
            activation_func=gelu,
            hooks={}
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.is_decoder = is_decoder
        self.hooks = hooks

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            hooks=hooks
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        self.sandwich_ln = sandwich_ln
        if sandwich_ln:
            self.third_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Cross attention.
        if self.is_decoder:
            self.cross_attention = CrossAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                layer_id,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=output_layer_init_method,
                bias=use_bias,
                hooks=hooks
            )
            self.post_cross_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            inner_hidden_size=inner_hidden_size,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            hooks=hooks
        )

    def forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        # Layer norm at the begining of the transformer layer.
        layernorm_output1 = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(layernorm_output1, mask, **kw_args)

        # Third LayerNorm
        if self.sandwich_ln:
            attention_output = self.third_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        if self.is_decoder:
            encoder_outputs = kw_args['encoder_outputs']
            if encoder_outputs is not None:
                assert 'cross_attention_mask' in kw_args
                # Cross attention
                attention_output = self.cross_attention(layernorm_output, **kw_args)
                # Residual connection.
                layernorm_input = layernorm_input + attention_output
                # Layer norm post the cross attention
                layernorm_output = self.post_cross_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output, **kw_args)

        # Fourth LayerNorm
        if self.sandwich_ln:
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output, kw_args['output_this_layer'], kw_args['output_cross_layer']


class BaseTransformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 sandwich_ln=True,
                 parallel_output=True,
                 is_decoder=False,
                 use_bias=True,
                 activation_func=gelu,
                 layernorm=LayerNorm,
                 init_method=None,
                 hooks={}
                 ):
        super(BaseTransformer, self).__init__()

        # recording parameters
        self.is_decoder = is_decoder
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.hooks = copy.copy(hooks)  # hooks will be updated each forward

        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=unscaled_init_method(0.02))

        self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        # create all layers
        if init_method is None:
            self.output_layer_init_method = scaled_init_method(init_method_std, num_layers)
            self.init_method = unscaled_init_method(init_method_std)
        else:
            self.output_layer_init_method = init_method
            self.init_method = init_method

        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                inner_hidden_size=inner_hidden_size,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=self.output_layer_init_method,
                is_decoder=self.is_decoder,
                sandwich_ln=sandwich_ln,
                layernorm=layernorm,
                use_bias=use_bias,
                activation_func=activation_func,
                hooks=self.hooks
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, input_ids, position_ids, attention_mask, *,
                output_hidden_states=False, **kw_args):
        # sanity check
        assert len(input_ids.shape) == 2
        batch_size, query_length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(1, 1, device=input_ids.device).type_as(
                next(self.parameters())
            )  # None means full attention
        assert len(attention_mask.shape) == 2 or \
               len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}

        # embedding part
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](input_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:  # default
            hidden_states = self.word_embeddings(input_ids)

        if 'position_embedding_forward' in self.hooks:
            position_embeddings = self.hooks['position_embedding_forward'](position_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == query_length
            position_embeddings = self.position_embeddings(position_ids)
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        output_per_layers = []
        if self.checkpoint_activations:
            # define custom_forward for checkpointing
            def custom(start, end, kw_args_index, cross_layer_index):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask = inputs[0], inputs[1]

                    # recover kw_args and output_cross_layer
                    flat_inputs = inputs[2:]
                    kw_args, output_cross_layer = {}, {}
                    for k, idx in kw_args_index.items():
                        kw_args[k] = flat_inputs[idx]
                    for k, idx in cross_layer_index.items():
                        output_cross_layer[k] = flat_inputs[idx]
                    # -----------------

                    output_per_layers_part = []
                    for i, layer in enumerate(layers_):
                        if 'layer_forward' in self.hooks:
                            layer_ret = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id,
                                **kw_args, **output_cross_layer,
                                output_this_layer={}, output_cross_layer={}
                            )
                        else:
                            layer_ret = layer(
                                x_, mask, layer_id=layer.layer_id,
                                **kw_args, **output_cross_layer,
                                output_this_layer={}, output_cross_layer={}
                            )
                        if torch.is_tensor(layer_ret): # only hidden_states
                            x_, output_this_layer, output_cross_layer = layer_ret, {}, {}
                        elif len(layer_ret) == 2: # hidden_states & output_this_layer
                            x_, output_this_layer = layer_ret
                            output_cross_layer = {}
                        elif len(layer_ret) == 3:
                            x_, output_this_layer, output_cross_layer = layer_ret
                        assert isinstance(output_this_layer, dict)
                        assert isinstance(output_cross_layer, dict)
                        if output_hidden_states:
                            output_this_layer['hidden_states'] = x_
                        output_per_layers_part.append(output_this_layer)

                    # flatten for re-aggregate keywords outputs
                    flat_outputs = []
                    for output_this_layer in output_per_layers_part:
                        for k in output_this_layer:
                            # TODO add warning for depth>=2 grad tensors
                            flat_outputs.append(output_this_layer[k])
                            output_this_layer[k] = len(flat_outputs) - 1
                    for k in output_cross_layer:
                        flat_outputs.append(output_cross_layer[k])
                        output_cross_layer[k] = len(flat_outputs) - 1
                    # --------------------

                    return x_, output_per_layers_part, output_cross_layer, flat_outputs
                return custom_forward

            # prevent to lose requires_grad in checkpointing.
            # To save memory when only finetuning the final layers, don't use checkpointing.
            if self.training:
                hidden_states.requires_grad_(True)

            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            output_this_layer = []
            while l < num_layers:
                args = [hidden_states, attention_mask]
                # flatten kw_args and output_cross_layer
                flat_inputs, kw_args_index, cross_layer_index = [], {}, {}
                for k, v in kw_args.items():
                    flat_inputs.append(v)
                    kw_args_index[k] = len(flat_inputs) - 1
                for k, v in output_cross_layer.items():
                    flat_inputs.append(v)
                    cross_layer_index[k] = len(flat_inputs) - 1
                # --------------------

                hidden_states, output_per_layers_part, output_cross_layer, flat_outputs = \
                    checkpoint(custom(l, l + chunk_length, kw_args_index, cross_layer_index), *args, *flat_inputs)

                # recover output_per_layers_part, output_cross_layer
                for output_this_layer in output_per_layers_part:
                    for k in output_this_layer:
                        output_this_layer[k] = flat_outputs[output_this_layer[k]]
                for k in output_cross_layer:
                    output_cross_layer[k] = flat_outputs[output_cross_layer[k]]
                # --------------------

                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            output_this_layer = []
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]

                if 'layer_forward' in self.hooks: # customized layer_forward
                    layer_ret = self.hooks['layer_forward'](*args, layer_id=torch.tensor(i),
                        **kw_args,
                        **output_cross_layer,
                        output_this_layer={}, output_cross_layer={}
                    )
                else:
                    layer_ret = layer(*args, layer_id=torch.tensor(i), **kw_args, **output_cross_layer,
                        output_this_layer={}, output_cross_layer={})
                if torch.is_tensor(layer_ret): # only hidden_states
                    hidden_states, output_this_layer, output_cross_layer = layer_ret, {}, {}
                elif len(layer_ret) == 2: # hidden_states & output_this_layer
                    hidden_states, output_this_layer = layer_ret
                    output_cross_layer = {}
                elif len(layer_ret) == 3:
                    hidden_states, output_this_layer, output_cross_layer = layer_ret

                if output_hidden_states:
                    output_this_layer['hidden_states'] = hidden_states
                output_per_layers.append(output_this_layer)

        # Final layer norm.
        logits = self.final_layernorm(hidden_states)

        if 'final_forward' in self.hooks:
            logits_parallel = self.hooks['final_forward'](logits, **kw_args)
        else:
            logits_parallel = copy_to_model_parallel_region(logits)
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)

        if not self.parallel_output:
            logits_parallel = gather_from_model_parallel_region(logits_parallel)

        outputs = [logits_parallel]
        outputs.extend(output_per_layers)

        return outputs
