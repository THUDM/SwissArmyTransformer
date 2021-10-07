# -*- encoding: utf-8 -*-
'''
@File    :   base_model.py
@Time    :   2021/10/01 22:40:33
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from mpu import BaseTransformer

class BaseModel(torch.nn.Module):
    def __init__(self, args, transformer=None):
        super(BaseModel, self).__init__()
        self.hooks = self.collect_hooks()
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = BaseTransformer(
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                max_sequence_length=args.max_sequence_length,
                embedding_dropout_prob=args.hidden_dropout,
                attention_dropout_prob=args.attention_dropout,
                output_dropout_prob=args.hidden_dropout,
                checkpoint_activations=args.checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                sandwich_ln=args.sandwich_ln,
                parallel_output=True,
                hooks=self.hooks
            )
        self.mixins = torch.nn.ModuleList()
        
    def reinit(self):
        # if some mixins are loaded, overrides this function
        for m in self.mixins: 
            m.reinit(self.transformer)
    
    def forward(self, *args, **kwargs):
        # update hooks as the current model (overrided forwards)
        # Attention! the transformer might be shared by multiple models
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)
        
    def collect_hooks(self):
        names = ['word_embedding_forward', 'position_embedding_forward',
                    'attention_forward', 'mlp_forward', 'final_forward']
        hooks = {}
        for name in names:
            if hasattr(self, name):
                hooks[name] = getattr(self, name)
        return hooks

    def disable_untrainable_params(self):
        pass