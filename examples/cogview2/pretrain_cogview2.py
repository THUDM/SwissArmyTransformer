# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_cogview2.py
@Time    :   2021/10/06 00:58:32
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
import numpy as np

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model import Cuda2dModel
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import BinaryDataset
from SwissArmyTransformer.tokenization.cogview import TextCodeTemplate

def get_masks_and_position_ids(data,
                            loss_mask=None,
                            attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        assert loss_mask is not None
        # loss_mask has n_pad(+1 CLS and [1:] then) zeros, so it is the same as attention_mask, reuse.
        attention_mask = loss_mask[:, :args.layout[1]].unsqueeze(-2).expand(batch_size, args.layout[1], args.layout[1]).tril()
        for i in range(batch_size):
            attention_mask[i].fill_diagonal_(1)
        attention_mask = attention_mask.unsqueeze(1)
        
    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    # Position ids.
    assert loss_mask is not None
    layout = args.layout
    assert seq_length == layout[-1]
    n_pads = seq_length - loss_mask.sum(dim=-1).long()
    position_ids = torch.zeros(batch_size, seq_length, dtype=torch.long,
                                device=data.device)
    for i in range(batch_size):
        torch.arange(layout[1] - n_pads[i], out=position_ids[i, n_pads[i]:layout[1]], 
            dtype=torch.long, device=data.device)
        torch.arange(layout[2] - layout[1], 
            out=position_ids[i, layout[1]:],
            dtype=torch.long, device=data.device)

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()

    labels = tokens_[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    attention_mask = None        

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        args=args
        )
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    # split img & txt positions, [PAD] not included # TODO check enough
    tokenizer = get_tokenizer()
    img_txt_sep = tokenizer.img_tokenizer.num_tokens
    img_indices_bool = (tokens < img_txt_sep) & (loss_mask > 0)
    txt_indices_bool = (~img_indices_bool) & (loss_mask > 0)
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask[txt_indices_bool] *= args.txt_loss_scale
    loss_mask = loss_mask.view(-1)  

    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()
    # =====================   Log partial losses   ======================== #
    img_indices_bool2 = img_indices_bool.clone()
    img_indices_bool2[:, :args.layout[1]] = False
    img_loss2 = losses[img_indices_bool2.view(-1)].detach().sum() / max(img_indices_bool2.sum(), 1)
    
    img_indices_bool = img_indices_bool.view(-1)
    txt_indices_bool = txt_indices_bool.view(-1)
    img_loss = losses[img_indices_bool].detach().sum() / max(img_indices_bool.sum(), 1)
    txt_loss = losses[txt_indices_bool].detach().sum() / max(txt_indices_bool.sum(), 1) / args.txt_loss_scale
    # ===================== END OF BLOCK ======================= #
    return loss, {'img_loss': img_loss, 'txt_loss': txt_loss, 'img_loss2': img_loss2}
    
def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    layout = [64, 64+16**2, 64+16**2+32**2, 64+64**2+16**2+32**2] # FIXME
    def process_fn(row):
        row = row.astype(np.int64)
        codes = [row[layout[i-1]:layout[i]] for i in range(1, len(layout))]
        
        text = row[:layout[0]]
        text = text[text>0][:layout[0] - 3] # [CLS] [BASE] [ROI1]
        n_pad = layout[0]-3-len(text)
        parts = [
            np.array([tokenizer['[PAD]']] * n_pad, dtype=np.int64),
            TextCodeTemplate(text, codes[1], tokenizer),
            *codes[2:]
        ]
        ret = np.concatenate(parts, axis=0)
        return {'text': ret, 
            'loss_mask':  np.array([0] * (n_pad+1) + [1] * (len(ret) - n_pad - 1)) # don't predict [CLS]
            }
    return BinaryDataset(path, process_fn, length_per_sample=layout[-1])

if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    
    py_parser.add_argument('--txt-loss-scale', type=float, default=1)
    
    Cuda2dModel.add_model_specific_args(py_parser)
    
    known, args_list = py_parser.parse_known_args()
    
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    args.layout = [int(x) for x in args.layout.split(',')]
    
    training_main(args, model_cls=Cuda2dModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
