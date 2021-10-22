# -*- encoding: utf-8 -*-
'''
@File    :   autoregressive_sampling.py
@Time    :   2021/10/08 15:43:59
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from .sampling_strategies import BaseStrategy

def get_masks_and_position_ids(seq):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    position_ids = torch.arange(len(seq), dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

def update_mems(hiddens, mems, max_memory_length):
    '''
        hiddens: list (num_layers) of [batch, query_length, 2d]
        mems: None or [num_layers, batch, memory_length, 2d]
    '''
    if hiddens is None:
        return None
    hiddens = torch.stack(hiddens)
    memory_length = mems.shape[2] if mems is not None else 0
    query_length = hiddens.shape[2]
    new_memory_length = min(max_memory_length, memory_length + query_length)
    new_mems = []
    with torch.no_grad():
        if new_memory_length <= query_length:
            return hiddens[:, :, -new_memory_length:]
        else:
            if mems.shape[1] < hiddens.shape[1]:
                mems = mems.expand(-1, hiddens.shape[1], -1, -1)
            return torch.cat(
                (mems[:, :, -new_memory_length+query_length:], hiddens),
                dim=2
            )


def filling_sequence(
        model, 
        seq, 
        batch_size,
        strategy=BaseStrategy(),
        max_memory_length=100000,
        log_attention_weights=None
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
    '''
    assert len(seq.shape) == 1

    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1 # [0, context_length-1] are given
    assert context_length > 0
    tokens, attention_mask, position_ids = get_masks_and_position_ids(seq)
    tokens = tokens[..., :context_length]
    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter'' 
    index = 0 # Next forward starting index, also the length of cache.
    mems = None # mems are the first-level citizens here, but we don't assume what is memorized.
        
    # step-by-step generation
    while counter < len(seq) - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.

        if seq[counter + 1] >= 0: # provided
            tokens = torch.cat(
                (
                    tokens, 
                    seq[counter+1: counter+2].expand(tokens.shape[0], 1)
                ), dim=1
            )
            counter += 1
            continue

        # forward
        if log_attention_weights is not None:
            model.log_attention_weights = log_attention_weights[..., index: counter+1, :counter+1] # TODO memlen
        kw_tensors = {'mems': mems} if mems is not None else {}
        logits, *mem_kv = model(
            tokens[:, index:], 
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            **kw_tensors # if no mems, cannot pass
        )
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        counter += 1
        index = counter
        # sampling
        logits = logits[:, -1].expand(batch_size, -1) # [batch size, vocab size]
        tokens = tokens.expand(batch_size, -1)
        tokens, mems = strategy.forward(logits, tokens, mems)
        
    model.log_attention_weights = None
    return tokens