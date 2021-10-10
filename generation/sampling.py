# -*- encoding: utf-8 -*-
'''
@File    :   sampling.py
@Time    :   2021/01/13 19:52:12
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from pretrain_gpt2 import get_masks_and_position_ids
from data_utils import get_tokenizer
from copy import deepcopy

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # s1 = (logits-logits.max()).exp().sum()
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value      
        # s2 = (logits-logits.max()).exp().sum()
        # with open('lion.txt', 'a') as fout:
        #     fout.write(f'{s1} {s2}\n')

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits

def get_batch(context_tokens, device, args):
    tokens = context_tokens
    if len(tokens.shape) == 1:
        tokens = tokens.unsqueeze(0).contiguous()
    else:
        tokens = tokens.view(tokens.shape[0], -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens, args=args)
    return tokens, attention_mask, position_ids

def update_mems(hiddens, mems, max_memory_length=10000):
    memory_length = mems[0].size(1) if mems else 0
    query_length = hiddens[0].size(1)
    new_memory_length = min(max_memory_length, memory_length + query_length)
    new_mems = []
    with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length+query_length:], hiddens[i]), dim=1))
    return new_mems

def filling_sequence(
        model, 
        seq, 
        args, 
        mems=None, 
        invalid_slices=[], 
        **kwargs):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -N (N beams), -1]
        context_length: first non(-1)s
    '''
    tokenizer = get_tokenizer()
    device = seq.device
    assert len(seq.shape) == 1
    out_seq_length = len(seq)
    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    offset = 100000

    invalid_slices = [slice(0, tokenizer.img_tokenizer.num_tokens)]

    while seq[context_length] >= 0:
        # change what to generate
        if seq[context_length] in [tokenizer['[BOI1]'], tokenizer['[BOI2]']]:
            invalid_slices = [slice(tokenizer.img_tokenizer.num_tokens, None)]
        elif seq[context_length] in [tokenizer['[EOI1]'], tokenizer['[EOI2]']]:
            invalid_slices = [
                slice(0, tokenizer.img_tokenizer.num_tokens),
                slice(tokenizer.img_tokenizer.num_tokens + tokenizer.txt_tokenizer.num_tokens, None)]

        if seq[context_length] == tokenizer['[ROI2]']:
            offset = context_length
        context_length += 1
    tokens, attention_mask, position_ids = get_batch(seq[:context_length], device, args)
    txt_len = seq.tolist().index(tokenizer['[BASE]'])
    print('txt_len:', txt_len)
    config = deepcopy(model.module.transformer.sparse_config)
    ori_config = model.module.transformer.sparse_config
    config.layout[0] = txt_len
    model.module.transformer.reset_sparse_config(config)

    counter = context_length - 1 # == len(tokens) - 1
    index = 0 # len(mems)
    if mems is None:
        mems = []
    score = [0] # sum log likelihood for beams
    
    while counter < (out_seq_length - 1):
        # Now, we want to generate seq[counter + 1]
        # token[:, index: counter+1] are just added.

        if seq[counter + 1] in [tokenizer['[BOI1]'], tokenizer['[BOI2]']]:
            invalid_slices = [slice(tokenizer.img_tokenizer.num_tokens, None)]
        elif seq[counter + 1] in [tokenizer['[EOI1]'], tokenizer['[EOI2]']]:
            invalid_slices = [
                slice(0, tokenizer.img_tokenizer.num_tokens),
                slice(tokenizer.img_tokenizer.num_tokens + tokenizer.txt_tokenizer.num_tokens, None)]

        if index == 0: # first 
            position_ids[position_ids > offset] -= offset
            logits, *qkv = model(tokens, position_ids, attention_mask, *mems)
            mems = update_mems(qkv, mems)

            # tmp = -F.log_softmax(logits, dim=-1)
            # tmp = tmp[0,:-1].gather(dim=-1,index=tokens[0,1:].unsqueeze(-1))[4:,0]
            # for i in range(1,len(tmp)):
            #     print(i, tmp[i].item())
            index = counter
            # print(tmp[1:].mean(), file=sys.stderr)
        elif seq[counter + 1] >= 0: # provided
            if seq[counter + 1] == tokenizer['[ROI2]']:
                offset = counter + 1
            tokens, mems, score = shrink_beams(tokens, mems, 1, score)
            nb = 1
            counter += 1
            tokens = torch.cat((tokens, seq[counter: counter+1].expand(tokens.shape[0], 1)), dim=1)
            continue
        else:
            assert tokens.shape[1] == counter + 1 
            position_ids = torch.arange(index, counter + 1, dtype=torch.long, device=tokens.device).unsqueeze(0)
            position_ids[position_ids > offset] -= offset
            # TODO each time, the feed input cannot be too long (window size), or it will have a discrepcy from sparse training, but this is not very important. 
            tokens, mems, score = shrink_beams(tokens, mems, -seq[counter + 1], score)
            logits, *qkv = model(tokens[:, index: ], 
                position_ids,
                0, # rebuild in transformers (sep version)
                *mems)
            mems = update_mems(qkv, mems)

            index = counter
        nb = -seq[counter + 1]
        counter += 1
        index += 1


        logits = logits[:, -1] # [batch size, vocab size]

        temp = args.temperature
        real_topk = args.top_k
        if counter <= context_length + 32:
            real_topk = 80
        # else:
            # real_topk = args.top_k
        # if counter == context_length + 32 + 12:
        #     import pdb;pdb.set_trace()
        # TODO since the temperature is crucial, how can we find a good setting?
        logits /= temp
        for invalid_slice in invalid_slices: #   to generate other tokens
            logits[..., invalid_slice] = -float('Inf')
        # logits = top_k_logits(logits, top_k=real_topk, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        tk_value, tk_idx = torch.topk(probs, real_topk, dim=-1)

        # expand beams
        if nb > 1 and tokens.shape[0] == 1: # 1->nb
            tokens = tokens.expand(nb, -1).contiguous()
            mems = [mem.expand(nb, -1, -1) for mem in mems]
            prev = torch.multinomial(probs, num_samples=nb, replacement=True)
            score = torch.log(torch.gather(probs, dim=1, index=prev)[0]).tolist()
        else: # nb -> nb
            assert tokens.shape[0] == nb
            prev = torch.multinomial(probs, num_samples=1)
            for j in range(0, prev.shape[0]):
                if probs[j, prev[j,-1]] < tk_value[j, -1]:
                    prev[j, -1] = tk_idx[j,torch.randint(tk_idx.shape[-1]-100, tk_idx.shape[-1], (1,))]
                    # prev[j, -1] = tk_idx[j,torch.randint(0, tk_idx.shape[-1], (1,))]

            score_plus = torch.log(torch.gather(probs, dim=1, index=prev)[:, 0])
            for idx in range(nb):
                score[idx] += score_plus[idx]
        
        tokens = torch.cat((tokens, prev.view(tokens.shape[0], 1)), dim=1)

    output_tokens_list = tokens.view(tokens.shape[0], -1).contiguous()
    model.module.transformer.reset_sparse_config(ori_config)
    return output_tokens_list

def shrink_beams(tokens, mems, nb, score):
    # beam search is a failed attempt, will be removed soon...
    if tokens.shape[0] == nb:
        return tokens, mems, score
    # shrink
    maximum = max(score)
    max_idx = score.index(maximum)
    tokens = tokens[max_idx].unsqueeze(0)
    score = [0]
    new_mems = [mem[max_idx: max_idx + 1] for mem in mems]
    return tokens, new_mems, score

def add_interlacing_beam_marks(seq, nb=12, period=30000):
    assert isinstance(seq, list) or len(seq.shape) == 1
    blk_cnt = 0
    for i in range(len(seq)):
        if seq[i] == -1:
            blk_cnt += 1
            seq[i] = -nb
            if blk_cnt == period:
                nb += (nb % 2) * 2 - 1
                blk_cnt = 0
        else:
            blk_cnt = 0


def inverse_prompt_score(model, seq, args):
    tokenizer = get_tokenizer()
    device = seq.device
    assert len(seq.shape) == 2

    botext = 2 + 1024 + 1
    assert tokenizer['[ROI1]'] == seq[0][botext]

    tokens, attention_mask, position_ids = get_batch(seq, device, args)
    logits, *qkv = model(tokens, position_ids, attention_mask)
    mems = update_mems(qkv, mems)

    logits[..., :tokenizer.img_tokenizer.num_tokens] = -float('Inf')
    log_probs = torch.log(F.softmax(logits, dim=-1))

    pred = log_probs[:, botext:-1, :] 
    target = tokens[:, botext+1:].unsqueeze(-1) 
    scores = torch.gather(pred, dim=2, index=target).squeeze(-1).sum(dim=-1)
    return scores
            