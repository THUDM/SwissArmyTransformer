import math
import os

import torch
import argparse
import numpy as np
import copy
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main, initialize_distributed, load_checkpoint
from SwissArmyTransformer.model.finetune import *
from deberta_model import DebertaModel
from SwissArmyTransformer.model.mixins import BaseMixin
from functools import partial

class ClassificationModel(DebertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('deberta-final')
        num_input = args.final_input
        num_output = 1 if args.class_num == 2 else args.class_num
        layer_range = None
        if args.layer_range is not None:
            if '-' in args.layer_range:
                i,j = args.layer_range.split('-')
                i,j = int(i), int(j)
                layer_range = [i for i in range(i, j+1)]
            else:
                layer_range = [int(i) for i in args.layer_range.split('.')]
        # layer_range = [0,1,2,21,22,23]
        print("layer_range!!!", layer_range)
        self.layer_range = layer_range
        if 'squad' in args.dataset_name:
            self.add_mixin('classification_head', QA_MLPHeadMixin(args, num_input, 2048, num_output, old_model=args.old_model))
        elif 'conll' in args.dataset_name:
            self.add_mixin('classification_head', NER_MLPHeadMixin(args, num_input, 2048, num_output, old_model=args.old_model))
        else:
            self.add_mixin('classification_head', NEW_MLPHeadMixin(args, num_input, 2048, num_output, old_model=args.old_model))
        self.finetune_type = args.finetune_type
        if 'geglue' in self.finetune_type:
            print('Add GEGLUE')
            self.add_mixin('geglue', GEGLUMixin(args.hidden_size, args.num_layers, layer_range))
        if 'coll' in self.finetune_type:
            print('Add collector')
            self.add_mixin('collector', CollectorMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.collect_len))
        if 'pt' in self.finetune_type:
            print('Add prefix tuning mixin')
            self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len, layer_range))
        if 'lora' in self.finetune_type:
            print('Add lora mixin')
            if 'lora_m2' in self.finetune_type:
                self.add_mixin('loraM2', LoRAM2Mixin(args.hidden_size, args.num_layers, args.lora_r, args.lora_alpha))
            else:
                self.add_mixin('lora', LoRAMixin(args.hidden_size, args.num_layers, args.lora_r, args.lora_alpha, args.lora_dropout, layer_range))
        if 'cls' in self.finetune_type:
            print('Add CLS mixin')
            self.add_mixin('cls', CLSMixin(args))
        if 'ffadd' in self.finetune_type:
            print('Add FFADD mixin')
            self.add_mixin('ffadd', FFADDMixin(args.hidden_size, args.num_layers, args.ffadd_r, layer_range))
            
    def disable_untrainable_params(self):
        if not 'all' in self.finetune_type:
            print('froze model parameter')
            self.transformer.requires_grad_(False)
            self.mixins["deberta-attention"].requires_grad_(False)

        if 'all' in self.finetune_type and self.layer_range is not None:
            print('froze part parameter')
            self.transformer.requires_grad_(False)
            for i in self.layer_range:
                self.transformer.layers[i].requires_grad_(True)

        if 'NO_Bitfit' in self.finetune_type:
            print('froze bitfit')
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(False) #b_m2
                self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(False) #b_qkv

        if 'bitfit' in self.finetune_type:
            print('Use bitfit')
            for layer_id in range(len(self.transformer.layers)):
                if self.layer_range is None or layer_id in self.layer_range:
                    self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(True) #b_m2
                    self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(True) #b_qkv
                    self.transformer.layers[layer_id].mlp.dense_4h_to_h.bias.requires_grad_(True) #b_m3
                    self.transformer.layers[layer_id].attention.dense.bias.requires_grad_(True)
                    self.transformer.layers[layer_id].input_layernorm.bias.requires_grad_(True)
                    self.transformer.layers[layer_id].post_attention_layernorm.bias.requires_grad_(True)
                    #debert
                    self.mixins["deberta-attention"].para.requires_grad_(True)
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
    py_parser.add_argument('--max-relative-positions', type=int, default=-1)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--name-model', type=str, default=None)
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

    if '[' in args.finetune_type:
        finetune_type, layer_range = args.finetune_type.split('[')
        args.finetune_type = finetune_type
        args.layer_range = layer_range[:-1]
    else:
        args.layer_range = None
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
        if 'bitfit' not in args.finetune_type:
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
        else:
            #old_model
            # breakpoint()
            pre_load = args.load
            args.load = args.save
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

            import gc
            gc.collect()
            torch.cuda.empty_cache()

            args.load = pre_load
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