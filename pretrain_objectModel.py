# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_cogview2.py
@Time    :   2021/10/06 00:58:32
@Author  :   Ming Ding
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import argparse
import numpy as np

import mpu
from arguments import get_args
from model.ObjectModel import ObjectModel
from training.deepspeed_training import training_main
from data_utils import BinaryDataset
from tokenization import get_tokenizer
from tokenization.cogview import TextCodeTemplate

def get_names():
    return ["背景",
            "人","自行车","汽车","摩托车","飞机","公交车","火车","卡车","船","红绿灯",
            "消防栓","空", "停车牌","停车收费表","长椅","鸟","猫","狗","马","羊",
            "牛","大象","熊","斑马","长颈鹿","空","背包","伞","空","空",
            "手提包","领带","手提箱","飞盘","滑雪板","滑雪板","球","风筝","棒球棒","棒球手套",
            "滑板","冲浪板","网球拍","瓶子","空", "酒杯","杯子","叉子","刀子","勺子",
            "碗","香蕉","苹果","三明治","橘子","花椰菜","胡萝卜","热狗","比萨饼","甜甜圈",
            "蛋糕","椅子","沙发","盆栽植物","床","空", "餐桌","空","空","厕所",
            "空","电视","笔记本","鼠标","遥控器","键盘","手机","微波炉","烤箱","烤面包机",
            "水槽","冰箱","空","书","钟","花瓶","剪刀","泰迪熊","吹风机","牙刷"]


def get_masks_and_position_ids(data,
                               n_pads,
                               object_pads,
                               loss_mask=None,
                               attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        assert loss_mask is not None
        # loss_mask has n_pad(+1 CLS and [1:] then) zeros, so it is the same as attention_mask, reuse.
        attention_mask = loss_mask[:, :seq_length].unsqueeze(-2).expand(batch_size, seq_length, seq_length).tril()
        for i in range(batch_size):
            attention_mask[i].fill_diagonal_(1)
        attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    # Position ids.
    #1270
    position_ids = torch.zeros(batch_size, seq_length, dtype=torch.long,
                                device=data.device)

    for i in range(batch_size):
        torch.arange(64 - n_pads[i], out=position_ids[i, n_pads[i]:64],
                     dtype=torch.long, device=data.device)
        torch.arange(180-object_pads[i], out=position_ids[i, 64+object_pads:64+180])
        # breakpoint()
        torch.arange(64 - n_pads[i],  64 - n_pads[i] + seq_length - (64+180),
                     out=position_ids[i, 64+180:],
                     dtype=torch.long, device=data.device)
    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'loss_mask', 'object_pad', 'n_pad']
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
    # breakpoint()
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()
    n_pads = data_b['n_pad'].long()
    object_pads = data_b['object_pad'].long()

    labels = tokens_[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    attention_mask = None

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        n_pads,
        object_pads,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        args=args
    )
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.view(-1)

    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()

    return loss, {}

def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    layout = [100,164,1188]
    names = get_names()
    tokens = []
    for name in names:
        tokens.append(tokenizer.EncodeAsIds(name))
    # 4 + 4 + 1 一个object需要9个token 20个需要 20*9 = 180个
    def process_fn(row):
        row = row.astype(np.int64)
        codes = [row[layout[1]:layout[2]]]
        text = row[layout[0]:layout[1]]
        text = text[text > 0][:63]  # [ROI]
        object_tokens = []
        # print(row[:100])
        for i in range(20):
            object = row[i * 5: (i+1) * 5]
            if object[0] == -1:
                break
            # print("object", object)
            object[2] += object[0]
            object[3] += object[1]
            object_tokens.append(tokenizer['[POS0]'])
            object_tokens.extend([object[j] + args.old_token_num for j in range(4)])
            object_tokens.extend(tokens[object[4]])
        object_tokens = np.array(object_tokens)
        # print(object_tokens)
        object_pad = 180 - object_tokens.shape[-1]
        object_tokens = np.concatenate([
            np.array([tokenizer['[PAD]']] * object_pad, dtype=np.int64),
            object_tokens
        ], axis = 0)
        # 180
        text_object = np.concatenate([text, np.array(object_tokens, dtype=np.int64)], axis=0)
        # print(len(text), len(text_object))
        # print(len(codes[0]))
        merged = TextCodeTemplate(text_object, codes[0], tokenizer)
        # print(len(merged), len(text))
        n_pad = args.new_sequence_length - len(merged)
        parts = [
            np.array([tokenizer['[PAD]']] * n_pad, dtype=np.int64),
            merged
        ]
        ret = np.concatenate(parts, axis=0)
        return {'text': ret,
                'loss_mask': np.array([0] * n_pad + [1] * (len(text) + 1) + [0] * object_pad + [1] * (182 - object_pad + 1025)),
                'object_pad':object_pad,
                'n_pad':n_pad
                }

    return BinaryDataset(path, process_fn, length_per_sample=layout[-1])



if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)

    py_parser.add_argument('--txt-loss-scale', type=float, default=1)

    ObjectModel.add_model_specific_args(py_parser)

    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    args.layout = [int(x) for x in args.layout.split(',')]

    training_main(args, model_cls=ObjectModel, forward_step_function=forward_step,
                  create_dataset_function=create_dataset_function)
