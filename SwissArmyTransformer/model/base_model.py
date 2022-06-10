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

from SwissArmyTransformer.model.transformer import BaseTransformer, standard_attention
from SwissArmyTransformer import update_args_with_file
from SwissArmyTransformer.training.deepspeed_training import load_checkpoint, get_model

from SwissArmyTransformer.transformer_defaults import HOOKS_DEFAULT
from SwissArmyTransformer.resources import auto_create

def non_conflict(func):
    func.non_conflict = True
    return func

class BaseMixin(torch.nn.Module):
    non_conflict = non_conflict
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
                checkpoint_activations=args.checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                layernorm_order=args.layernorm_order,
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
        names = list(HOOKS_DEFAULT.keys())
        hooks = {}
        hook_origins = {}
        for name in names:
            if hasattr(self, name):
                hooks[name] = getattr(self, name)
                hook_origins[name] = 'model'

            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if hasattr(getattr(m, name), 'non_conflict'):
                        if name in hooks:
                            old_impl = hooks[name]
                        elif name == 'attention_fn': # the only hook without self
                            old_impl = HOOKS_DEFAULT[name]
                        else:
                            old_impl = partial(HOOKS_DEFAULT[name], self)
                        old_origin = hook_origins.get(name, 'default')
                        hooks[name] = partial(getattr(m, name), old_impl=old_impl)
                        hook_origins[name] = mixin_name + ' -> ' + old_origin
                    elif name in hooks: # if this hook name is already registered
                        raise ValueError(f'Hook {name} conflicts at {mixin_name} and {hook_origins[name]}.')
                    else: # new hook
                        hooks[name] = getattr(m, name)
                        hook_origins[name] = mixin_name

        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks

    def disable_untrainable_params(self):
        pass

    @classmethod
    def from_pretrained(cls, args, name, *, home_path=None, url=None):
        model_path = auto_create(name, path=home_path, url=url)
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        model = get_model(args, cls)
        load_checkpoint(model, args, load_path=model_path)
        return model, args

class AutoModel():
    @classmethod
    def from_pretrained(cls, args, name, *, home_path=None, url=None):
        '''Automatically find the class and instantiate it. Auto-download.
            Args:
                args: NameSpace. will add the loaded args into it.
                name: The identifier of the pretrained model.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: manually specified url for the `name` model.
        '''
        model_path = auto_create(name, path=home_path, url=url)
        args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
        if not hasattr(args, 'model_class'):
            raise ValueError('model_config.json must have key "model_class" for AutoModel.from_pretrained.')
        import SwissArmyTransformer.model
        if not hasattr(SwissArmyTransformer.model, args.model_class): 
            # TODO dynamic loading
            raise ValueError(f'model_class {args.model_class} not found.')
        else:
            model_cls = getattr(SwissArmyTransformer.model, args.model_class)
        model = get_model(args, model_cls)
        load_checkpoint(model, args, load_path=model_path)
        return model, args


