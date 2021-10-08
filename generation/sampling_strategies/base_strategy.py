# -*- encoding: utf-8 -*-
'''
@File    :   base_strategy.py
@Time    :   2021/10/08 22:22:42
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

def top_k_logits_(logits, top_k=0, filter_value=-float('Inf')):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value     
    return logits

class BaseStrategy:
    def __init__(self, invalid_slices=[], temperature=1., topk=200, debias=False):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = topk
        self.debias = debias
    def forward(self, logits, tokens, mems, temperature=None):
        if temperature is None:
            temperature = self.temperature 
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -float('Inf')
        if self.debias:
            probs = F.softmax(logits, dim=-1)
            tk_value, tk_idx = torch.topk(probs, self.topk, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
            for j in range(0, pred.shape[0]):
                if probs[j, pred[j,-1]] < tk_value[j, -1]:
                    pred[j, -1] = tk_idx[j, torch.randint(tk_idx.shape[-1]-100, tk_idx.shape[-1], (1,))] # 100 is the last N as outlier, which is chosen casually
        else:
            logits = top_k_logits_(logits)
            probs = F.softmax(logits, dim=-1)
            pred = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, pred.view(tokens.shape[0], 1)), dim=1)
        return tokens, mems
