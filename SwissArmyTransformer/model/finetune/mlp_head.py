
# -*- encoding: utf-8 -*-

# here put the import lib
import os
import sys
import math
import random

import torch
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict


class MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005):
        super().__init__()
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for sz in output_sizes:
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            self.layers.append(this_layer)

    def final_forward(self, logits, **kw_args):
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class NEW_MLPHeadMixin(BaseMixin):
    def __init__(self, args, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005, old_model=None):
        super().__init__()
        # init_std = 0.1
        self.args = args
        self.cls_number = args.cls_number
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i, sz in enumerate(output_sizes):
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            if old_model is None:
                torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            else:
                print("****************************load head weight***********************************")
                old_weights = old_model.mixins["classification_head"].layers[i].weight.data
                this_layer.weight.data.copy_(old_weights)
            self.layers.append(this_layer)

    def reset_parameter(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_parameters()
            torch.nn.init.normal_(self.layers[i].weight, mean=0, std=0.005)

    def final_forward(self, logits, **kw_args):
        cls_logits = logits[:,:self.cls_number].sum(1)
        if 'pos' in kw_args.keys():
            word = kw_args['word'].unsqueeze(-1)
            pos1_embedding = torch.sum(logits * word, dim=1)
            # breakpoint()
            # logits = logits.reshape([-1, logits.shape[-1]])

            # pos1_embedding = logits[kw_args['pos1']] #32 * hidden
            #
            # pos2_embedding = logits[kw_args['pos2']]
            #
            # cls_logits = pos1_embedding - pos2_embedding
            # cls_logits = torch.abs(pos1_embedding - pos2_embedding)
            cls_logits = torch.cat([cls_logits, pos1_embedding], dim=-1)
        if 'pos1' in kw_args.keys():
            word1 = kw_args['word1'].unsqueeze(-1)
            pos1_embedding = torch.sum(logits * word1, dim=1)
            word2 = kw_args['word2'].unsqueeze(-1)
            pos2_embedding = torch.sum(logits * word2, dim=1)
            # breakpoint()
            # logits = logits.reshape([-1, logits.shape[-1]])

            # pos1_embedding = logits[kw_args['pos1']] #32 * hidden
            #
            # pos2_embedding = logits[kw_args['pos2']]
            #
            # cls_logits = pos1_embedding - pos2_embedding
            # cls_logits = torch.abs(pos1_embedding - pos2_embedding)
            cls_logits = torch.cat([cls_logits, pos1_embedding, pos2_embedding, torch.abs(pos1_embedding-pos2_embedding), pos1_embedding * pos2_embedding], dim=-1)
        logits = cls_logits
        if len(kw_args['attention_output']) > 0:
            attention_output = kw_args['attention_output']
            logits += torch.cat(attention_output, dim=1).sum(1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class NER_MLPHeadMixin(BaseMixin):
    def __init__(self, args, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005, old_model=None):
        super().__init__()
        # init_std = 0.1
        self.args = args
        self.cls_number = args.cls_number
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i, sz in enumerate(output_sizes):
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            if old_model is None:
                torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            else:
                print("****************************load head weight***********************************")
                old_weights = old_model.mixins["classification_head"].layers[i].weight.data
                this_layer.weight.data.copy_(old_weights)
            self.layers.append(this_layer)

    def reset_parameter(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_parameters()
            torch.nn.init.normal_(self.layers[i].weight, mean=0, std=0.005)

    def final_forward(self, logits, **kw_args):
        if len(kw_args['attention_output']) > 0:
            attention_output = kw_args['attention_output']
            logits += torch.cat(attention_output, dim=1).sum(1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class QA_MLPHeadMixin(BaseMixin):
    def __init__(self, args, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005, old_model=None):
        super().__init__()
        # init_std = 0.1
        self.args = args
        self.cls_number = args.cls_number
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i, sz in enumerate(output_sizes):
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            if old_model is None:
                torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            else:
                print("****************************load head weight***********************************")
                old_weights = old_model.mixins["classification_head"].layers[i].weight.data
                this_layer.weight.data.copy_(old_weights)
            self.layers.append(this_layer)
        self.S = torch.nn.Parameter(torch.zeros(hidden_size))
        self.E = torch.nn.Parameter(torch.zeros(hidden_size))
        if old_model is None:
            torch.nn.init.normal_(self.S, mean=init_mean, std=init_std)
            torch.nn.init.normal_(self.E, mean=init_mean, std=init_std)
        else:
            self.S.data.copy_(old_model.mixins["classification_head"].S.data)
            self.E.data.copy_(old_model.mixins["classification_head"].E.data)
    def reset_parameter(self):
        for i in range(len(self.layers)):
            self.layers[i].reset_parameters()
            torch.nn.init.normal_(self.layers[i].weight, mean=0, std=0.005)
        torch.nn.init.normal_(self.S, mean=0, std=0.005)
        torch.nn.init.normal_(self.E, mean=0, std=0.005)
    def final_forward(self, logits, **kw_args):
        #Start
        start_logits = logits @ self.S
        end_logits = logits @ self.E
        cls_logits = logits[:,:self.cls_number].sum(1)
        logits = cls_logits
        if len(kw_args['attention_output']) > 0:
            attention_output = kw_args['attention_output']
            logits += torch.cat(attention_output, dim=1).sum(1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return start_logits, end_logits, logits