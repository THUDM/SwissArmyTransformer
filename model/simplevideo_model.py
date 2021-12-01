# -*- encoding: utf-8 -*-
'''
@File    :   cuda2d_model.py
@Time    :   2021/10/02 01:36:32
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F


from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, AttentionMixin

from mpu.transformer import split_tensor_along_last_dim
from mpu.local_attention_function import f_similar, f_weighting
from mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker
from fused_kernel.fused_softmax import FusedScaleMaskSoftmax
from fused_kernel.enums import AttnMaskType


class SimpleVideoModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        # additional_seqlen = args.new_sequence_length - args.max_sequence_length
        # self.mixins.append(PositionEmbeddingMixin(
        #     additional_seqlen, args.hidden_size
        # )) 

        self.layout = args.layout
        # {frame1} {layout[0] 256} {frame2} ... {frame8} {layout[1] 2048}
        self.log_attention_weights = None
        
        if args.mode != 'inference':
            from fused_kernel.initialize import initialize_fused_kernel
            initialize_fused_kernel(args)
            self.mask_softmax = FusedScaleMaskSoftmax(
                                    input_in_fp16=True, 
                                    input_in_bf16=False,
                                    attn_mask_type=AttnMaskType.causal,
                                    scaled_masked_softmax_fusion=True,
                                    mask_func=attention_mask_func,
                                    softmax_in_fp32=False,
                                    scale=None)
        else:
            self.mask_softmax = None #Use Cached Inference
    
    # def position_embedding_forward(self, position_ids, **kw_tensors):
    #     position = position_ids[..., :self.layout[1]]
    #     position_plus = position_ids[..., self.layout[1]:]
    #     position_embeddings = torch.cat(
    #             (
    #                 self.transformer.position_embeddings(position),
    #                 self.mixins[0].position_embeddings(position_plus)
    #             ),
    #             dim=-2
    #         )
    #     return position_embeddings
    
    def attention_forward(self, hidden_states, mask, layer_id=None, mems=None, **kw_tensors):
        attn_module = self.transformer.layers[layer_id].attention
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = attn_module.attention_dropout if self.training else None

        query_layer = attn_module._transpose_for_scores(mixed_query_layer)
        key_layer = attn_module._transpose_for_scores(mixed_key_layer)
        value_layer = attn_module._transpose_for_scores(mixed_value_layer)
        
        context_layer = fused_attention(query_layer, key_layer, value_layer, mask_softmax=self.mask_softmax,
                                        mask=mask, attention_dropout=dropout_fn)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = attn_module.dense(context_layer)
        
        if self.training:
            output = attn_module.output_dropout(output)
        
        return output, None
        
    
    def disable_untrainable_params(self):
        pass
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('SimpleVideoModel', 'SimpleVideo model configurations')
        group.add_argument("--layout", type=str, default='256, 2048')
        group.add_argument("--masked-softmax-fusion", action='store_true', help="enable mask&softmax fusion")
        return parser


def fused_attention(query_layer, key_layer, value_layer, mask_softmax, mask,
                    attention_dropout=None, log_attention_weights=None):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    attention_scores = torch.matmul(
        query_layer / math.sqrt(query_layer.shape[-1]),
        key_layer.transpose(-1, -2)
    )
    if log_attention_weights is not None:
        attention_scores += log_attention_weights
    
    # if attention_mask.shape[-2] > 1: # if auto-regressive, skip
    # attention_scores = torch.mul(attention_scores, attention_mask) - \
    #             10000.0 * (1.0 - attention_mask)

    # attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs = mask_softmax(attention_scores, mask=mask)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores