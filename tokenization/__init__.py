# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2021/10/06 17:58:04
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

def get_tokenizer(args=None):
    if not hasattr(get_tokenizer, 'tokenizer'):
        # the first time to load the tokenizer
        if args.tokenizer_type == 'cogview':
            from .cogview import UnifiedTokenizer
            get_tokenizer.tokenizer = UnifiedTokenizer(
                args.img_tokenizer_path, 
                device=torch.cuda.current_device()
                )
        elif args.tokenizer_type == 'glm':
            pass
        else:
            assert args.vocab_size > 0
            get_tokenizer.tokenizer = FakeTokenizer(args.vocab_size)
    return get_tokenizer.tokenizer

class FakeTokenizer(object):
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens
    def __len__(self):
        return self.num_tokens