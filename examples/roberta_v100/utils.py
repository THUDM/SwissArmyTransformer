# -*- encoding: utf-8 -*-
# @File    :   utils.py
# @Time    :   2022/3/22
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

import numpy as np
import os
import torch
import math
from torch.optim import Optimizer
from torch.distributions.bernoulli import Bernoulli
from typing import Callable, Iterable, Tuple
from tqdm import tqdm
from SwissArmyTransformer.training.utils import Timers

pretrain_path = ''
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from SwissArmyTransformer.data_utils import load_hf_dataset

def _encode_double_text(text, text_pair, args):
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation='only_first')
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

def create_dataset_function(path, args):
    dataset_name = args.dataset_name
    if dataset_name == "rte":
        def process_fn(row):
            pack, label = _encode_double_text(row['premise'], row['hypothesis'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
        return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "attention_mask", "label"], cache_dir='/workspace/yzy/SwissArmyTransformerDatasets', offline=True, transformer_name="rte_transformer")
    elif dataset_name == "boolq":
        def process_fn(row):
            pack, label = _encode_double_text(row['passage'], row['question'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
        return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "attention_mask", "label"], cache_dir='/workspace/yzy/SwissArmyTransformerDatasets', offline=True, transformer_name="boolq_transformer")


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
                        if p in self.gradient_mask:
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

def calc_mask(model, args, train_data, forward_step):
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

def set_optimizer_mask(model, args, train_data, optimizer, forward_step):
    train_data = iter(train_data)
    if args.child_type == "ChildTuning-D":
        grad_mask = calc_mask(model, args, train_data, forward_step)
        optimizer.set_gradient_mask(grad_mask)