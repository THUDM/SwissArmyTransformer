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
from SwissArmyTransformer.mpu.mappings import copy_to_model_parallel_region
from .common_layers import LayerNorm


def get_extended_attention_mask(attention_mask, input_shape, device, dtype=torch.float32, is_decoder=False):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.
        dtype:
        is_decoder:

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask is None or attention_mask.dim() == 2:
        batch_size, seq_length = input_shape
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(dtype)
            if attention_mask is not None:
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                         causal_mask], axis=-1)

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = causal_mask[:, None, :, :]
        else:
            if attention_mask is None:
                extended_attention_mask = torch.ones(1, 1, 1, seq_length, device=device, dtype=dtype)
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    return extended_attention_mask


class EncoderFinalMixin(BaseMixin):
    def final_forward(self, logits, **kwargs):
        logits = copy_to_model_parallel_region(logits)
        return logits


class EncoderDecoderModel(torch.nn.Module):
    def __init__(self, args, encoder=None, decoder=None, parallel_output=False, **kwargs):
        super(EncoderDecoderModel, self).__init__()
        if encoder is not None:
            assert isinstance(encoder, BaseModel)
            self.encoder = encoder
        else:
            self.encoder = BaseModel(args, **kwargs)
        self.encoder.add_mixin("final", EncoderFinalMixin())
        if decoder is not None:
            assert isinstance(decoder, BaseModel)
            self.decoder = decoder
        else:
            dec_args = argparse.Namespace(**vars(args))
            dec_args.enc_hidden_size = dec_args.hidden_size  # used for cross attn
            override_attrs = ['num_layers', 'hidden_size', 'num_attention_heads',
                              'max_sequence_length', 'inner_hidden_size', 'hidden_size_per_attention_head']
            for name in override_attrs:
                dec_attr = getattr(dec_args, 'dec_' + name, None)
                if dec_attr is not None:  # else use encoder-config
                    setattr(dec_args, name, dec_attr)
            self.decoder = BaseModel(args, is_decoder=True, parallel_output=parallel_output, **kwargs)

    def reinit(self):
        self.encoder.reinit()
        self.decoder.reinit()

    def disable_untrainable_params(self):
        self.encoder.disable_untrainable_params()
        self.decoder.disable_untrainable_params()

    def forward(self, input_ids=None, input_position_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_position_ids=None, decoder_attention_mask=None, encoder_outputs=None,
                **kw_args):
        dtype = self.encoder.transformer.word_embeddings.weight.dtype
        if encoder_outputs is None:
            batch_size, encoder_seq_length = input_ids.size()[:2]
        else:
            batch_size, encoder_seq_length = encoder_outputs.size()[:2]
        encoder_attention_mask = get_extended_attention_mask(attention_mask, (batch_size, encoder_seq_length),
                                                             device=input_ids.device, dtype=dtype)
        decoder_seq_length = decoder_input_ids.size(1)
        if encoder_outputs is None:
            encoder_outputs, *_dumps = self.encoder(input_ids, input_position_ids, encoder_attention_mask, **kw_args)
        decoder_attention_mask = get_extended_attention_mask(decoder_attention_mask, (batch_size, decoder_seq_length),
                                                             device=input_ids.device, dtype=dtype, is_decoder=True)
        decoder_outputs, *decoder_mems = self.decoder(decoder_input_ids, decoder_position_ids, decoder_attention_mask,
                                                      encoder_outputs=encoder_outputs,
                                                      cross_attention_mask=encoder_attention_mask, **kw_args)
        return encoder_outputs, decoder_outputs, *decoder_mems

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('EncoderDecoderModel', 'T5 or Bart')
        group.add_argument("--dec-num-layers", type=int, default=None)
        group.add_argument("--dec-hidden-size", type=int, default=None)
        group.add_argument("--dec-num-attention-heads", type=int, default=None)
        group.add_argument("--dec-max-sequence-length", type=int, default=None)
        group.add_argument("--dec-inner-hidden-size", type=int, default=None)
        group.add_argument("--dec-hidden-size-per-attention-head", type=int, default=None)
        return parser
