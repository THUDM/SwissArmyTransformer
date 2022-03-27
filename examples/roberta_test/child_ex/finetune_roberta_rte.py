# -*- encoding: utf-8 -*-
# @File    :   finetune_bert_boolq.py
# @Time    :   2022/3/5
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import os

import torch
import argparse
import numpy as np
from functools import partial
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main, initialize_distributed, load_checkpoint
from roberta_model import RobertaModel, LoRAMixin
from SwissArmyTransformer.model.mixins import PrefixTuningMixin, BaseMixin
from typing import Callable, Iterable, Tuple
from torch.optim import Optimizer
import math
from torch.distributions.bernoulli import Bernoulli
from SwissArmyTransformer.training.utils import Timers
from tqdm import tqdm


class ChildTuningAdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            reserve_p = 1.0,
            mode = None
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.gradient_mask = None
        self.reserve_p = reserve_p
        self.mode = mode

    def set_gradient_mask(self, gradient_mask):
        self.gradient_mask = gradient_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================
                if self.mode is not None:
                    if self.mode == 'ChildTuning-D':
                        if self.gradient_mask and p in self.gradient_mask:
                            grad *= self.gradient_mask[p]
                    else:
                        # ChildTuning-F
                        grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

#140551568434816
class MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005, old_model=None):
        super().__init__()
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i, sz in enumerate(output_sizes):
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            if old_model is None:
                torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            else:
                old_weights = old_model.mixins["classification_head"].layers[i].weight.data
                this_layer.weight.data.copy_(old_weights)
            self.layers.append(this_layer)


    def final_forward(self, logits, **kw_args):
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
            logits = layer(logits)
        return logits

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1, init_std=0.02, old_model=args.old_model))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))

        # self.add_mixin('lora', LoRAMixin(args.hidden_size, args.num_layers, args.lora_r, args.lora_alpha, args.lora_dropout))
    def disable_untrainable_params(self):
        pass
        # self.transformer.requires_grad_(False)
        # self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        # self.transformer.layers[layer_id].mlp.dense_h_to_4h.requires_grad_(True) #Wm2
        # self.transformer.layers[layer_id].attention.dense.requires_grad_(True) #Wm1
        # self.transformer.layers[layer_id].attention.query_key_value.requires_grad_(True) #QKV
        # self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(True) #m2
        # self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(True) #bqk



    def get_optimizer(self, args, train_data):
        optimizer_kwargs = {
            "betas": (0.9, 0.98),
            "eps": 1e-6,
        }
        optimizer_kwargs["lr"] = args.lr
        optimizer = partial(ChildTuningAdamW, reserve_p=args.reserve_p, mode=args.child_type, **optimizer_kwargs)
        return optimizer

def calc_mask(model, args, train_data):
    timers = Timers()
    N = len(train_data)//100
    if N > 200:
        N = 200
    print(f"{N} samples to calc mask")

    model.train()
    gradient_mask = dict()
    for name, params in model.named_parameters():
        if 'transformer.layers' in name:
            gradient_mask[params] = params.new_zeros(params.size())
    for _ in tqdm(range(N)):
        loss, _ = forward_step(train_data, model, args, timers)
        loss.backward()

        for name, params in model.named_parameters():
            if 'transformer.layers' in name:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                gradient_mask[params] += (params.grad ** 2)
        model.zero_grad()
    print('Calculate Fisher Information')

    # Numpy
    r = None
    for k, v in gradient_mask.items():
        v = v.view(-1).cpu().numpy()
        if r is None:
            r = v
        else:
            r = np.append(r, v)
    polar = np.percentile(r, (1-args.reserve_p)*100)
    for k in gradient_mask:
        gradient_mask[k] = gradient_mask[k] >= polar

    print('Polar => {}'.format(polar))

    for name, params in model.named_parameters():
        if 'transformer.layers' in name:
            cnt = gradient_mask[params].sum()
            sz = gradient_mask[params].size()
            cnt2 = 1
            for szz in sz:
                cnt2 *= szz
            print(name[18:], f"{cnt}/{cnt2}", (cnt/cnt2).cpu().numpy().tolist())
    return gradient_mask

def set_optimizer_mask(model, args, train_data, optimizer):
    train_data = iter(train_data)
    if args.child_type == "ChildTuning-D":
        grad_mask = calc_mask(model, args, train_data)
        optimizer.set_gradient_mask(grad_mask)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'attention_mask', 'label']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['label'].long()
    position_ids = data_b['position_ids'].long()
    attention_mask = data_b['attention_mask'][:, None, None, :].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, attention_mask, position_ids, (tokens!=1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""
    # Get the batch.

    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, loss_mask = get_batch(
        data_iterator, args, timers)

    #boolq 每行最大有400个左右
    timers('batch generator').stop()
    logits, *mems = model(tokens, position_ids, attention_mask)
    # pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    pred = logits.contiguous().float().squeeze(-1)[..., 0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

pretrain_path = ''
from transformers import RobertaTokenizer
tokenizer =  RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

def _encode(text, text_pair):
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation='only_first')
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

from SwissArmyTransformer.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        pack, label = _encode(row['passage'], row['question']), int(row['label'])
        return {
            'input_ids': np.array(pack['input_ids'], dtype=np.int64),
            'position_ids': np.array(pack['position_ids'], dtype=np.int64),
            'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
            'label': label
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "attention_mask", "label"], cache_dir='/dataset/fd5061f6/SwissArmyTransformerDatasets', offline=True, transformer_name="rte_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--child-type', type=str, default="ChildTuning-D")
    py_parser.add_argument('--reserve-p', type=float, default=0.3)
    py_parser.add_argument('--max-grad-norm', type=float, default=1.0)
    py_parser.add_argument('--child-load', type=str, default=None)

    py_parser.add_argument('--lora-r', type=int, default=8)
    py_parser.add_argument('--lora-alpha', type=float, default=16)
    py_parser.add_argument('--lora-dropout', type=str, default=None)

    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    if args.child_load is not None:
        args.load = args.child_load


    args.old_model = None
    load = args.load
    args.load = "/workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/finetune-roberta-large-boolq-lora-1e-4-03-18-12-27"
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



    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function, get_optimizer_from_model=True, set_optimizer_mask=set_optimizer_mask, already_init=True)
