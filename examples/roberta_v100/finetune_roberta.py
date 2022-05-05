import math
import os

import torch
import argparse
import numpy as np
import copy
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main, initialize_distributed, load_checkpoint
from roberta_model import RobertaModel, LoRAMixin, CLSMixin, CollectorMixin, PrefixTuningMixin, FFADDMixin, LoRAM2Mixin
from SwissArmyTransformer.model.mixins import BaseMixin
from functools import partial
from utils import *

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
            self.S.data.copy_(old_model.mixins["classification_head"].S.weight.data)
            self.E.data.copy_(old_model.mixins["classification_head"].E.weight.data)
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

class WIC_MLPHeadMixin(BaseMixin):
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
        cls_embedding = logits[:,0] #32 * hidden
        logits = logits.reshape([-1, logits.shape[-1]])
        pos1_embedding = logits[kw_args['pos1']] #32 * hidden
        pos2_embedding = logits[kw_args['pos2']]
        logits = torch.cat([cls_embedding, pos1_embedding, pos2_embedding], dim=-1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class MLPHeadMixin(BaseMixin):
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
        if 'pos1' in kw_args.keys():
            logits = logits.reshape([-1, logits.shape[-1]])
            pos1_embedding = logits[kw_args['pos1']] #32 * hidden
            pos2_embedding = logits[kw_args['pos2']]
            cls_logits = torch.cat([cls_logits, pos1_embedding, pos2_embedding], dim=-1)
        logits = cls_logits
        if len(kw_args['attention_output']) > 0:
            attention_output = kw_args['attention_output']
            logits += torch.cat(attention_output, dim=1).sum(1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        num_input = args.final_input
        num_output = 1 if args.class_num == 2 else args.class_num
        self.add_mixin('classification_head', MLPHeadMixin(args, num_input, 2048, num_output, old_model=args.old_model))
        self.finetune_type = args.finetune_type
        if 'coll' in self.finetune_type:
            print('Add collector')
            self.add_mixin('collector', CollectorMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.collect_len))
        if 'pt' in self.finetune_type:
            print('Add prefix tuning mixin')
            self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
        if 'lora' in self.finetune_type:
            print('Add lora mixin')
            if 'lora_m2' in self.finetune_type:
                self.add_mixin('loraM2', LoRAM2Mixin(args.hidden_size, args.num_layers, args.lora_r, args.lora_alpha))
            else:
                self.add_mixin('lora', LoRAMixin(args.hidden_size, args.num_layers, args.lora_r, args.lora_alpha, args.lora_dropout))
        if 'cls' in self.finetune_type:
            print('Add CLS mixin')
            self.add_mixin('cls', CLSMixin(args))
        if 'ffadd' in self.finetune_type:
            print('Add FFADD mixin')
            self.add_mixin('ffadd', FFADDMixin(args.hidden_size, args.num_layers, args.ffadd_r))
            
    def disable_untrainable_params(self):
        if not 'all' in self.finetune_type:
            print('froze model parameter')
            self.transformer.requires_grad_(False)

        if 'NO_Bitfit' in self.finetune_type:
            print('froze bitfit')
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(False) #b_m2
                self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(False) #b_qkv

        if 'bitfit' in self.finetune_type:
            print('Use bitfit')
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(True) #b_m2
                self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(True) #b_qkv
        if 'Wqkv' in self.finetune_type:
            print('Use Wqkv and bias')
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].attention.query_key_value.requires_grad_(True) #qkv
        if 'Wm1' in self.finetune_type:
            print("Use Wm1 and bias")
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].attention.dense.requires_grad_(True) #Wm1
        if 'Wm2' in self.finetune_type:
            print('Use Wm2')
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].mlp.dense_h_to_4h.requires_grad_(True) #Wm2
        if 'Wm3' in self.finetune_type:
            print('Use Wm3')
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].mlp.dense_4h_to_h.requires_grad_(True) #Wm3



    def get_optimizer(self, args, train_data):
        optimizer_kwargs = {
            "betas": (0.9, 0.98),
            "eps": 1e-6,
        }
        optimizer_kwargs["lr"] = args.lr
        optimizer = partial(ChildTuningAdamW, reserve_p=args.reserve_p, mode=args.child_type, **optimizer_kwargs)
        return optimizer

def handle_metrics(metrics):
    acc = sum(metrics['acc'].split(1,0))/len(metrics['acc'])
    if 'tp' in metrics.keys():
        TP = sum(metrics['tp'].split(1,0))
        TN = sum(metrics['tn'].split(1,0))
        FP = sum(metrics['fp'].split(1,0))
        FN = sum(metrics['fn'].split(1,0))
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Precision*Recall)/(Precision+Recall)
        MC = (TP*TN-FP*FN)/torch.sqrt((TP+FP)*(FN+TP)*(FN+TN)*(FP+TN))
        return {'acc': acc, 'f1': F1, 'mc':MC}
    else:
        return {'acc': acc}

def get_loss_metrics(logits, labels, dataset_name, **extra_data):
    if dataset_name in ['rte', 'boolq', 'wic', 'mrpc', 'qnli', 'qqp', 'cola', 'wnli']:
        pred = logits.contiguous().float().squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            labels.float()
        )
        true_pos = ((pred > 0.).long() * labels).sum() * 1.0
        false_pos = ((1-(pred > 0.).long()) * labels).sum() * 1.0
        true_neg = ((1-(pred > 0.).long()) * (1-labels)).sum() * 1.0
        false_neg = ((pred > 0.).long() * (1-labels)).sum() * 1.0
        acc = ((pred > 0.).long() == labels).sum() / labels.numel()
        eval_acc = ((pred > 0.).long() == labels).float()

        return loss, {'acc': acc, 'tp': true_pos, 'fp': false_pos, 'tn': true_neg, 'fn': false_neg, 'eval_acc': eval_acc}
    elif dataset_name=="copa":
        bz = logits.shape[0] // 2
        logits = logits.squeeze(-1).reshape(2, bz).permute(1, 0)
        pred = logits.contiguous().float()
        loss = torch.nn.functional.cross_entropy(
            pred,
            labels
        )
        acc = (torch.argmax(pred, dim=1).long() == labels).sum() / labels.numel()
        return loss, {'acc': acc}
    elif dataset_name=='cb':
        pred = logits.contiguous().float()
        loss = torch.nn.functional.cross_entropy(pred, labels)
        acc = ((pred > 0.).long() == labels).sum() / labels.numel()
        return loss, {'acc': acc}
    elif dataset_name in ['squad', 'squad_v2']:
        # For negative examples, abstaining receives a score of 1,
        # and any other response gets 0, for both exact match and F1.
        start_logits, end_logits, cls_logits = logits
        start_logits = start_logits.contiguous().float()
        end_logits = end_logits.contiguous().float()
        cls_logits = cls_logits.contiguous().float()
        start_list = extra_data['start_list']
        end_list = extra_data['end_list']
        pred = logits.contiguous().float().squeeze(-1)
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            labels.float()
        )
        loss = loss1 + 0
        return loss, {'em': 0, 'f1':0}

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    # timers('batch generator').start()
    get_batch = get_batch_function(args.dataset_name)
    tokens, labels, attention_mask, position_ids, loss_mask, *extra_data = get_batch(
        data_iterator, args, timers)
    # timers('batch generator').stop()
    if len(extra_data) >= 1:
        extra_data = extra_data[0]
    else:
        extra_data = {}
    attention_output = []
    logits, *mems = model(tokens, position_ids, attention_mask, attention_output = attention_output, **extra_data)

    return get_loss_metrics(logits, labels, args.dataset_name, **extra_data)

#模型并行会有问题！！
if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512)

    #type
    py_parser.add_argument('--finetune-type', type=str, default="all")

    #pt
    py_parser.add_argument('--prefix_len', type=int, default=17)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--dataset-name', type=str, required=True)

    #lora
    py_parser.add_argument('--lora-r', type=int, default=8)
    py_parser.add_argument('--lora-alpha', type=float, default=16)
    py_parser.add_argument('--lora-dropout', type=str, default=None)

    #child
    py_parser.add_argument('--child-type', type=str, default="ChildTuning-D")
    py_parser.add_argument('--reserve-p', type=float, default=0.3)
    py_parser.add_argument('--max-grad-norm', type=float, default=1.0)
    py_parser.add_argument('--child-load', type=str, default=None)

    #old_model
    py_parser.add_argument('--head-load', action="store_true")
    py_parser.add_argument('--head-path', type=str, default=None)
    py_parser.add_argument('--body-path', type=str, default=None)

    #cls
    py_parser.add_argument('--cls-number', type=int, default=4)

    #collector
    py_parser.add_argument('--collect-len', type=int, default=2)

    #2step
    py_parser.add_argument('--step1-lr', type=float, default=5e-5)
    py_parser.add_argument('--step1-iters', type=int, default=None)
    py_parser.add_argument('--step1-epochs', type=int, default=None)

    #ffadd
    py_parser.add_argument('--ffadd-r', type=int, default=32)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    #print information

    print(f"*******************Experiment Name is {args.experiment_name}****************************")
    print(f"*******************Finetune Type is {args.finetune_type}****************************")
    print(f"*******************Learning Rate is {args.lr}****************************")

    if args.dataset_name == 'copa':
        args.sample_length = args.sample_length // 2

    if 'pt' in args.finetune_type:
        args.sample_length -= args.prefix_len
    if 'cls' in args.finetune_type:
        args.sample_length -= args.cls_number - 1
    if 'coll' in args.finetune_type:
        args.sample_length -= args.collect_len

    print(f"*******************True Sample length is {args.sample_length}****************************")

    args.class_num = get_class_num(args.dataset_name)
    args.final_input = args.hidden_size
    if args.dataset_name == 'wic':
        args.final_input = 3 * args.hidden_size
    args.get_optimizer_group = None
    args.old_model = None
    


    if '2step' in args.finetune_type:
        step1_lr = args.step1_lr
        step2_lr = args.lr
        step1_epochs = args.step1_epochs
        step2_epochs = args.epochs - args.step1_epochs
        # step1_iters = args.step1_iters
        # step2_iters = args.train_iters - step1_iters


        pre_args = copy.deepcopy(args)
        #step1
        args.lr = step1_lr
        args.epochs = step1_epochs
        # args.train_iters = step1_iters
        training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, handle_metrics=handle_metrics)
        #step2
        pre_args.load = args.save
        del args

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        args = pre_args
        args.lr = step2_lr
        args.experiment_name += f'pretype-{args.finetune_type}-'
        args.finetune_type = 'all'
        args.epochs = step2_epochs
        # args.train_iters = step2_iters
        training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, already_init=True, handle_metrics=handle_metrics)

    elif 'child' in args.finetune_type:
        if args.child_load is not None:
            args.load = args.child_load
        training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, get_optimizer_from_model=True, set_optimizer_mask=set_optimizer_mask, handle_metrics=handle_metrics)
    elif args.head_load:
        if args.body_path:
            args.load = args.body_path
            training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, reset_part="head", handle_metrics=handle_metrics)
        else:
            load = args.load
            args.load = args.head_path
            initialize_distributed(args)
            old_model = ClassificationModel(args)
            args.do_train=True
            _ = load_checkpoint(old_model, args)
            old_model.requires_grad_(False)
            if args.fp16:
                old_model.half()
            elif args.bf16:
                old_model.bfloat16()
            old_model.cuda(torch.cuda.current_device())
            args.old_model = old_model
            training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, already_init=True, handle_metrics=handle_metrics)
    else:
        training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, handle_metrics=handle_metrics)

    # args.load = "/workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/finetune-roberta-large-boolq-lora-1e-4-03-18-12-27"
    # args.load = "/workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/finetune-roberta-large-boolq-bitfit-1e-3-03-08-13-15"
    # args.load = "/workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/finetune-roberta-large-boolq-pt-7e-3-nowarmup-03-08-10-58"