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
from SwissArmyTransformer import mpu

pretrain_path = ''
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'), local_files_only=True)
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from SwissArmyTransformer.data_utils import load_hf_dataset

def get_dataset_keys(dataset_name):
    dataset_keys = {
        #superglue
        'boolq':["input_ids", "position_ids", "attention_mask", "label"],
        'rte':["input_ids", "position_ids", "attention_mask", "label"],
        'copa':["input_ids_1", "position_ids_1", "attention_mask_1", "input_ids_2", "position_ids_2", "attention_mask_2", "label"],
        'cb':["input_ids", "position_ids", "attention_mask", "label"],
        'wic':["input_ids", "position_ids", "attention_mask", "label", "pos1", "pos2"],
        #glue
        'mrpc':["input_ids", "position_ids", "attention_mask", "label"],
        'qnli':["input_ids", "position_ids", "attention_mask", "label"],
        'qqp':["input_ids", "position_ids", "attention_mask", "label"],
        'cola':["input_ids", "position_ids", "attention_mask", "label"],
        'wnli':["input_ids", "position_ids", "attention_mask", "label"],
        'sst2':["input_ids", "position_ids", "attention_mask", "label"],
        'stsb':["input_ids", "position_ids", "attention_mask", "label"],
        #squad
        'squad':["input_ids", "position_ids", "attention_mask", "label", "start_list", "end_list"],
        'squad_v2':["input_ids", "position_ids", "attention_mask", "label", "start_list", "end_list"],
        #conll
        'conll2003':["input_ids", "position_ids", "attention_mask", "label"],
    }
    return dataset_keys[dataset_name]

def get_class_num(dataset_name):
    dataset_class_num = {
        'boolq':2,
        'rte':2,
        'copa':2,
        'cb':3,
        'wic':2,
        'mrpc':2,
        'qnli':2,
        'qqp':2,
        'cola':2,
        'wnli':2,
        'sst2':2,
        'stsb':2,
        'squad':2,
        'squad_v2':2,
        'conll2003':9,
    }
    return dataset_class_num[dataset_name]

def get_batch_function(dataset_name):
    if dataset_name in ["boolq", "rte", "cb", "mrpc", "qnli", "qqp", "cola", 'wnli', 'conll2003', 'sst2', 'stsb'] :
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
    elif dataset_name == "copa":
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ["input_ids_1", "position_ids_1", "attention_mask_1", "input_ids_2", "position_ids_2", "attention_mask_2", "label"]
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
            tokens_1 = data_b['input_ids_1'].long()
            tokens_2 = data_b['input_ids_2'].long()
            tokens = torch.cat([tokens_1, tokens_2], dim=0)
            labels = data_b['label'].long()
            position_ids_1 = data_b['position_ids_1'].long()
            position_ids_2 = data_b['position_ids_2'].long()
            position_ids = torch.cat([position_ids_1, position_ids_2], dim=0)

            attention_mask_1 = data_b['attention_mask_1'][:, None, None, :].float()
            attention_mask_2 = data_b['attention_mask_2'][:, None, None, :].float()
            attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=0)

            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1)
    elif dataset_name == 'wic':
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'label', 'pos1', 'pos2']
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
            pos1 = data_b['pos1'].long()
            pos2 = data_b['pos2'].long()
            bz = tokens.shape[0]
            for i in range(bz):
                pos1[i] += i * tokens.shape[1]
                pos2[i] += i * tokens.shape[1]

            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1), {'pos1':pos1, 'pos2':pos2}
    elif dataset_name in ['squad', 'squad_v2']:
        def get_batch(data_iterator, args, timers):
            # Items and their type.
            keys = ['input_ids', 'position_ids', 'attention_mask', 'label', 'start_list', 'end_list']
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
            start_list = data_b['start_list'].long()
            end_list = data_b['end_list'].long()
            # Convert
            if args.fp16:
                attention_mask = attention_mask.half()

            return tokens, labels, attention_mask, position_ids, (tokens!=1), {'start_list':start_list, 'end_list':end_list}
    else:
        raise Exception('dataset name is wrong')
    return get_batch

def _encode_single_text(text, args, is_split_into_words=False):
    encoded_input = tokenizer(text, max_length=args.sample_length, padding='max_length', truncation='only_first', is_split_into_words=is_split_into_words)
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

def _encode_double_text(text, text_pair, args, truncation='only_first'):
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation=truncation)
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

def create_dataset_function(path, args):
    dataset_name = args.dataset_name
    if os.getenv('PLATFORM') == "jinan":
        cache_dir = '/thudm/workspace/SwissArmyTransformerDatasets'
    elif os.getenv('PLATFORM') == "wudao":
        cache_dir = '/sharefs/cogview-new/yzy/SwissArmyTransformerDatasets'
    else:
        raise Exception("no PLATFORM")
    offline = True
    transformer_name = f"{dataset_name}_transformer_{args.sample_length}"
    process_fn = None
    filter_fn = None
    if dataset_name == "sst2":
        def process_fn(row):
            pack, label = _encode_single_text(row['sentence'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "stsb":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence1'], row['sentence2'], args), float(row['label'])
            label = int(label * 1000)
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "wic":
        def process_fn(row):
            pack = _encode_double_text(row['sentence1'], row['sentence2'], args)
            label = int(row['label'])
            start1, end1 = int(row['start1']), int(row['end1'])
            start2, end2 = int(row['start2']), int(row['end2'])
            pos1 = tokenizer(row['sentence1'][:start1])['input_ids'].__len__() - 2
            pos2 = tokenizer(row['sentence2'][:start2])['input_ids'].__len__() - 2 + tokenizer(row['sentence1'])['input_ids'].__len__()
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label,
                'pos1': pos1,
                'pos2': pos2
            }
    elif dataset_name == 'cb':
        def process_fn(row):
            pack, label = _encode_double_text(row['premise'], row['hypothesis'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "rte":
        def process_fn(row):
            pack, label = _encode_double_text(row['premise'], row['hypothesis'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "boolq":
        def process_fn(row):
            pack, label = _encode_double_text(row['passage'], row['question'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "copa":
        def process_fn(row):
            type = row['question']
            premise, choice1, choice2 = row['premise'], row['choice1'], row['choice2']
            premise = premise[:-1]
            choice1 = choice1[0].lower() + choice1[1:]
            choice2 = choice2[0].lower() + choice2[1:]
            if type=='cause':
                sentence1 = premise + ' because ' + choice1
                sentence2 = premise + ' because ' + choice2
            else:
                sentence1 = premise + ' so ' + choice1
                sentence2 = premise + ' so ' + choice2
                pass
            pack_1 = _encode_single_text(sentence1, args)
            pack_2 = _encode_single_text(sentence2, args)
            label = int(row['label'])
            return {
                'input_ids_1': np.array(pack_1['input_ids'], dtype=np.int64),
                'input_ids_2': np.array(pack_2['input_ids'], dtype=np.int64),
                'position_ids_1': np.array(pack_1['position_ids'], dtype=np.int64),
                'position_ids_2': np.array(pack_2['position_ids'], dtype=np.int64),
                'attention_mask_1': np.array(pack_1['attention_mask'], dtype=np.int64),
                'attention_mask_2': np.array(pack_2['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "mrpc":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence1'], row['sentence2'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "qnli":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence'], row['question'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "qqp":
        def process_fn(row):
            pack, label = _encode_double_text(row['question1'], row['question2'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "cola":
        def process_fn(row):
            pack, label = _encode_single_text(row['sentence'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name == "wnli":
        def process_fn(row):
            pack, label = _encode_double_text(row['sentence1'], row['sentence2'], args), int(row['label'])
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label
            }
    elif dataset_name in ['squad', 'squad_v2']:
        def process_fn(row):
            question = row['question'].strip()
            pack = _encode_double_text(question, row['context'], args, truncation='only_second')
            answers = row['answers']
            label = 0 if len(answers["text"]) == 0 else 1
            assert len(answers) <= 4

            start_list = []
            end_list = []
            for i in range(len(answers["text"])):
                texts = answers["text"][i]
                start = answers["answer_start"][i]

                start_pos = tokenizer(row['context'][:start])['input_ids'].__len__() - 2
                if start_pos == 0:
                    start_pos += 1
                end_pos = tokenizer(row['context'][:(start+len(texts))])['input_ids'].__len__() - 2

                start_pos += tokenizer(question)['input_ids'].__len__()
                end_pos += tokenizer(question)['input_ids'].__len__()
                if end_pos > 511:
                    print('exceed 512')
                    start_pos = 0
                    end_pos = 0
                # else:
                #     print(texts, tokenizer.decode(pack['input_ids'][start_pos:end_pos+1]))
                #     print(len(texts), len(tokenizer.decode(pack['input_ids'][start_pos:end_pos+1])))
                start_list.append(start_pos)
                end_list.append(end_pos)

            assert len(start_list)<=7
            while len(start_list) < 7:
                start_list.append(-1)
                end_list.append(-1)
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': label,
                'start_list': np.array(start_list, dtype=np.int64),
                'end_list': np.array(end_list, dtype=np.int64)
            }
    elif dataset_name in ['conll2003']:
        def process_fn(row):
            labels = row['ner_tags']
            pack = _encode_single_text(row['tokens'], args, is_split_into_words=True)
            word_ids = [None]
            for j, word in enumerate(row['tokens']):
                token = tokenizer.encode([word], add_special_tokens=False, is_split_into_words=True)
                word_ids += [j] * len(token)
            word_ids += [None]
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                    # label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            label_ids = label_ids + ([-100] * (len(pack['input_ids']) - len(label_ids)))
            return {
                'input_ids': np.array(pack['input_ids'], dtype=np.int64),
                'position_ids': np.array(pack['position_ids'], dtype=np.int64),
                'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
                'label': np.array(label_ids, dtype=np.int64),
            }
        def filter_fn(row):
            if len(row['tokens']) == 0:
                return False
            else:
                return True
    else:
        raise Exception('Dataset name is wrong.')
    return load_hf_dataset(path, process_fn, filter_fn = filter_fn, columns = get_dataset_keys(dataset_name), cache_dir=cache_dir, offline=offline, transformer_name=transformer_name)

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