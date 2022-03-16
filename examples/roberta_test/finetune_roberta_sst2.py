# -*- encoding: utf-8 -*-
# @File    :   finetune_roberta_sst2.py
# @Time    :   2022/3/8
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
# -*- encoding: utf-8 -*-

# here put the import lib
import os
import sys
import math
import random

from SwissArmyTransformer.data_utils.datasets import TSVDataset
import torch
import argparse
import numpy as np

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import TSVDataset
from roberta_model import RobertaModel
from SwissArmyTransformer.mpu.transformer import standard_attention
from SwissArmyTransformer.model.mixins import MLPHeadMixin, PrefixTuningMixin

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
        # self.add_mixin('CLS', CLSMixin())

    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
            # self.transformer.layers[layer_id].mlp.dense_h_to_4h.requires_grad_(True) #Wm2
            # self.transformer.layers[layer_id].attention.dense.requires_grad_(True) #Wm1
            # self.transformer.layers[layer_id].attention.query_key_value.requires_grad_(True) #QKV
            # self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(True) #m2
            # self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(True) #bqk


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'label', 'position_ids', 'attention_mask']
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
    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask)
    pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
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

def _encode(text):
    encoded_input = tokenizer(text, max_length=args.sample_length, padding='max_length', truncation='only_first')
    position_ids = create_position_ids_from_input_ids(torch.tensor([encoded_input['input_ids']]), 1, 0)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids[0].numpy(), attention_mask=encoded_input['attention_mask'])

from SwissArmyTransformer.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        pack, label = _encode(row['sentence']), int(row['label'])
        return {
            'input_ids': np.array(pack['input_ids'], dtype=np.int64),
            'position_ids': np.array(pack['position_ids'], dtype=np.int64),
            'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
            'label': label
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "attention_mask", "label"], cache_dir='/dataset/fd5061f6/SwissArmyTransformerDatasets', offline=True, transformer_name="sst2_transformer")


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
