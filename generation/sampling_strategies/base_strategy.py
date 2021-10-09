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
    def __init__(self, invalid_slices=[], temperature=1., topk=200, eps=1e-4):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = topk
        self.eps = eps
    def forward(self, logits, tokens, mems, temperature=None):
        if temperature is None:
            temperature = self.temperature 
        logits = logits / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -65504
            
        logits = top_k_logits_(logits, self.topk)
        probs = F.softmax(logits.float(), dim=-1) # float is essetial, due to a bug in Pytorch
        pred = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, pred.view(tokens.shape[0], 1)), dim=1)
        return tokens, mems
