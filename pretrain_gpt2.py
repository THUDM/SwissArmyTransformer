# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_gpt2.py
@Time    :   2021/10/06 00:58:32
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

import mpu
from arguments import get_args
from model.base_model import BaseModel
from training.deepspeed_training import main

def get_masks_and_position_ids(data,
                            loss_mask=None,
                            attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=data.device)
        attention_mask.tril_()
        
    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

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
    
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.view(-1)  

    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()
    
    return loss, {}

if __name__ == '__main__':
    args = get_args()
    main(args, model_cls=BaseModel, forward_step=forward_step)
