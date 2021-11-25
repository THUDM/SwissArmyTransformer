# -*- encoding: utf-8 -*-
'''
@File    :   encoder_decoder_model.py
@Time    :   2021/11/22 23:35:28
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import argparse
from .base_model import BaseModel, BaseMixin
from .common_layers import CrossAttention, LayerNorm


class CrossAttentionMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size, num_attention_heads,
                attention_dropout_prob, output_dropout_prob,
                init_method, enc_hidden_size=None, inner_hidden_size=None, output_layer_init_method=None):
        super().__init__()
            
        self.cross_attentions = torch.nn.ModuleList(
            [CrossAttention(
                hidden_size, num_attention_heads,
                attention_dropout_prob, output_dropout_prob,
                init_method, enc_hidden_size=enc_hidden_size, inner_hidden_size=inner_hidden_size, 
                output_layer_init_method=output_layer_init_method
            ) for layer_id in range(num_layers)]
        ) # Just copy args
        self.cross_lns = torch.nn.ModuleList(
            [LayerNorm(hidden_size, 1e-5)
            for layer_id in range(num_layers)]
        )
        

    def layer_forward(self, hidden_states, mask, layer_id, **kw_args):
        layer = self.transformer.layers[layer_id]
        encoder_outputs = kw_args['encoder_outputs']
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
            encoder_outputs: [batch, enc_seq_len, enc_hidden_size]
        '''
        # Layer norm at the begining of the transformer layer.
        layernorm_output = layer.input_layernorm(hidden_states)
        attention_output, output_this_layer = layer.attention(layernorm_output, mask, **kw_args)
        # Third LayerNorm
        if layer.sandwich_ln:
            attention_output = layer.third_layernorm(attention_output)
        # Residual connection.
        hidden_states = hidden_states + attention_output

        # Cross attention.
        layernorm_output = self.cross_lns[layer_id](hidden_states)
        cross_attn_output = self.cross_attentions[layer_id](
            layernorm_output, 
            torch.ones(1, 1, device=hidden_states.device, dtype=hidden_states.dtype), 
            encoder_outputs
            )
        hidden_states = hidden_states + cross_attn_output

        # Layer norm post the layer attention.
        layernorm_output = layer.post_attention_layernorm(hidden_states)
        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)

        # Fourth LayerNorm
        if layer.sandwich_ln:
            mlp_output = layer.fourth_layernorm(mlp_output)
        output = hidden_states + mlp_output

        return output, output_this_layer

    
class DecoderModel(BaseModel):
    def __init__(self, args, transformer=None):
        dec_args = argparse.Namespace(**vars(args))
        dec_args.enc_hidden_size = dec_args.hidden_size # used for cross attn
        override_attrs = ['num_layers', 'vocab_size', 
            'hidden_size', 'num_attention_heads', 
            'max_sequence_length', 'sandwich_ln' # TODO
            ]
        for name in override_attrs:
            dec_attr = getattr(dec_args, 'dec_' + name, None)
            if dec_attr is not None: # else use encoder-config
                setattr(dec_args, name, dec_attr)

        super().__init__(dec_args, transformer=transformer)
        self.add_mixin('cross_attention',
            CrossAttentionMixin(
                dec_args.num_layers,
                dec_args.hidden_size, dec_args.num_attention_heads,
                dec_args.attention_dropout, dec_args.hidden_dropout,
                self.transformer.init_method, 
                enc_hidden_size=dec_args.enc_hidden_size, 
                inner_hidden_size=getattr(dec_args, 'dec_inner_hidden_size', None), 
                output_layer_init_method=self.transformer.output_layer_init_method
            )
        )

class EncoderDecoderModel(torch.nn.Module):
    def __init__(self, args, encoder=None, decoder=None):
        super(EncoderDecoderModel, self).__init__()
        if encoder is not None:
            assert isinstance(encoder, BaseModel)
            self.encoder = encoder
        else:
            self.encoder = BaseModel(args)
        
        if decoder is not None:
            assert isinstance(decoder, BaseModel)
            self.decoder = decoder
        else:
            self.decoder = DecoderModel(args)

    def reinit(self):
        self.encoder.reinit()
        self.decoder.reinit()
    
    def disable_untrainable_params(self):
        self.encoder.disable_untrainable_params()
        self.decoder.disable_untrainable_params()
    
    def forward(self, enc_input_ids, enc_position_ids, dec_input_ids, dec_position_ids, dec_attention_mask, *, branch_input=None, **kw_args):
        mask_one = torch.ones(1, 1, device=enc_input_ids.device, dtype=dec_attention_mask.dtype)
        enc_outputs, *_dumps = self.encoder(enc_input_ids, enc_position_ids, mask_one, branch_input=branch_input, **kw_args)
        dec_outputs, *dec_mems = self.decoder(dec_input_ids, dec_position_ids, dec_attention_mask, encoder_outputs=enc_outputs, branch_input=branch_input, **kw_args)
        return enc_outputs, dec_outputs, *dec_mems

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EncoderDecoderModel', 'T5 or Bart')
        group.add_argument("--dec_num_layers", type=int, default=None)
        group.add_argument("--dec_vocab_size", type=int, default=None)
        group.add_argument("--dec_hidden_size", type=int, default=None)
        group.add_argument("--dec_num_attention_heads", type=int, default=None)
        group.add_argument("--dec_max_sequence_length", type=int, default=None)
        group.add_argument("--dec_sandwich_ln", action='store_true')
        group.add_argument("--dec_inner_hidden_size", type=int, default=None)
        return parser