# -*- encoding: utf-8 -*-
'''
@File    :   base_model.py
@Time    :   2021/10/01 22:40:33
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
import os
import sys
import math
import random
import torch

from SwissArmyTransformer.mpu import BaseTransformer
from SwissArmyTransformer.mpu.transformer import standard_attention
from SwissArmyTransformer import update_args_with_file
from SwissArmyTransformer.training.deepspeed_training import load_checkpoint, get_model

def non_conflict(func):
    func.non_conflict = True
    return func

class BaseMixin(torch.nn.Module):
    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    # can define hook-functions here
    # ...

    # If the hook is just a pre- or post- transformation,
    # You can use @non_conflict to mark it,
    # and run `old_impl` to make it compatible with other mixins.
    # Eg., 
    # 
    # @non_conflict
    # def attention_fn(q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
    #     new_q, new_k, new_v = pre_hack(q, k, v)
    #     attn_result = old_impl(q, k, v, mask, dropout_fn, **kw_args)
    #     attn_result = post_hack(attn_result)
    #     return attn_result


class BaseModel(torch.nn.Module):
    def __init__(self, args, transformer=None, **kwargs):
        super(BaseModel, self).__init__()
        self.mixins = torch.nn.ModuleDict()
        self.collect_hooks_()
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
                inner_hidden_size=args.inner_hidden_size,
                hidden_size_per_attention_head=args.hidden_size_per_attention_head,
                layernorm_epsilon=args.layernorm_epsilon,
                checkpoint_activations=args.checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                sandwich_ln=args.sandwich_ln,
                hooks=self.hooks,
                **kwargs
            )

    def reinit(self, mixin_names=None):  # will be called when loading model, None means all
        # if some mixins are loaded, overrides this function
        for k, m in self.mixins.items():
            if k in mixin_names or mixin_names is None:
                m.reinit(self)

    def add_mixin(self, name, new_mixin, reinit=False):
        assert name not in self.mixins
        assert isinstance(new_mixin, BaseMixin)

        self.mixins[name] = new_mixin  # will auto-register parameters
        object.__setattr__(new_mixin, 'transformer', self.transformer)  # cannot use pytorch set_attr

        self.collect_hooks_()
        if reinit:
            new_mixin.reinit(self)  # also pass current mixins

    def del_mixin(self, name):
        assert name in self.mixins
        del self.mixins[name]
        self.collect_hooks_()

    def get_mixin(self, name):
        return self.mixins[name]

    def forward(self, *args, **kwargs):
        # update hooks as the current model (overrided forwards)
        # Attention! the transformer might be shared by multiple models
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)

    def collect_hooks_(self):
        names = ['word_embedding_forward', 'position_embedding_forward',
                 'attention_forward', 'cross_attention_forward', 'mlp_forward', 'final_forward', 'layer_forward',
                 'attention_fn'
                 ]
        hooks = {}
        hook_origins = {}
        for name in names:
            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if name in hooks: # if this hook name is already registered
                        if hasattr(getattr(m, name), 'non_conflict'):
                            hooks[name] = partial(getattr(m, name), old_impl=hooks[name])
                            hook_origins[name] = mixin_name + ' -> ' + hook_origins[name]
                        else: # conflict
                            raise ValueError(f'Hook {name} conflicts at {mixin_name} and {hook_origins[name]}.')
                    else: # new hook
                        hooks[name] = getattr(m, name)
                        hook_origins[name] = mixin_name

            if hasattr(self, name):
                # if name in hooks: # defined in mixins, can override
                #     print(f'Override {name} in {hook_origins[name]}...')
                hooks[name] = getattr(self, name)
                hook_origins[name] = 'model'
        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks

    def disable_untrainable_params(self):
        pass

    @classmethod
    def from_pretrained(cls, args):
        args = update_args_with_file(args)
        model = get_model(args, cls)
        if args.load:
            load_checkpoint(model, args)
        return model, args