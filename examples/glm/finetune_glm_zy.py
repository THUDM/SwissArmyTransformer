# -*- encoding: utf-8 -*-
'''
@File    :   finetune_glm_sst2.py
@Time    :   2021/12/12 20:53:28
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
sys.path.append("../..")

from SwissArmyTransformer.data_utils.datasets import TSVDataset
import torch
import argparse
import numpy as np
import logging
from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import TSVDataset
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.mpu.transformer import standard_attention
from SwissArmyTransformer.model.mixins import MLPHeadMixin, PrefixTuningMixin

class ClassificationModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, args.num_categories))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
        self.tuning_mode = args.tuning_mode
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        if self.tuning_mode == "ptuning":
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].requires_grad_(False)
    
def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['sentence', 'label']
    datatype = torch.int64

    # Broadcast data.
    try:
        timers('data loader').start()
    except AssertionError:
        pass
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    try:
        timers('data loader').stop()
    except AssertionError:
        pass
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['sentence'].long()
    labels = data_b['label'].long()
    batch_size, seq_length = tokens.size()
    
    position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    position_ids = position_ids.unsqueeze(0)
    
    attention_mask = torch.ones((batch_size, 1, seq_length, seq_length), device=tokens.device)

    attention_mask[...,:seq_length] -= (tokens==-1).view(batch_size, 1, 1, seq_length).float()
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    return tokens, labels, attention_mask, position_ids, (tokens!=-1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    try:
        timers('batch generator').start()
    except AssertionError:
        pass
    tokens, labels, attention_mask, position_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    try:
        timers('batch generator').stop()
    except AssertionError:
        pass

    logits, *mems = model(tokens, position_ids, attention_mask)
    loss_mask = loss_mask.unsqueeze(-1).repeat(1, 1, args.num_categories)

    pred = ((logits.contiguous().float()) * loss_mask).sum(dim=-2) / torch.sum(loss_mask)
    m = torch.nn.LogSoftmax(dim=1)
    #loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(m(pred), labels)
    acc = torch.sum((pred.argmax(dim=-1).eq(labels)).float()) / labels.numel()
    return loss, {'acc': acc}, pred

def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    def process_fn(row):
        sentence, label = tokenizer._encode(row[0]), int(row[1].strip())
        sentence = [tokenizer.get_command('ENC').Id] + sentence + [tokenizer.get_command('eos').Id]
        if len(sentence) >= args.sample_length:
            sentence = sentence[:args.sample_length]
        else:
            sentence.extend([-1] * (args.sample_length-len(sentence)))
        return {'sentence': np.array(sentence, dtype=np.int64), 'label': label}
    return TSVDataset(path, process_fn, with_heads=True)

if __name__ == '__main__':
    import logging
    logging.getLogger('DeepSpeed').setLevel(logging.WARN)
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=80)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--num_categories', type=int, default=3)
    py_parser.add_argument('--tuning_mode', type=str, default="ptuning")
    py_parser.add_argument('--visible_devices', type=str, default="7")
    GLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
