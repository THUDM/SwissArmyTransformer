# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_video.py
@Time    :   2021/10/13 00:58:32
@Author  :   wenyihong
@Contact :   hongwy18@mails.tsinghua.edu.cn
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
from model.video_model import VideoModel
from training.deepspeed_training import training_main
from data_utils import BinaryDataset
from tokenization import get_tokenizer
from tokenization.cogview import TextCodeTemplate


def get_masks_and_position_ids(data,
                            loss_mask=None,
                            attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if attention_mask is None:
        assert loss_mask is not None
        # loss_mask has n_pad(+1 CLS and [1:] then) zeros, so it is the same as attention_mask, reuse.
        attention_mask = loss_mask[:, :args.layout[1]].unsqueeze(-2).expand(batch_size, args.layout[1], args.layout[1]).tril()
        for i in range(batch_size):
            attention_mask[i].fill_diagonal_(1)
        attention_mask_f2i = loss_mask[:, :args.layout[1]].unsqueeze(-2).expand(batch_size, args.layout[1]-args.layout[0], args.layout[1]).tril()
        attention_mask = torch.cat((attention_mask, attention_mask_f2i), dim=1)
        attention_mask = attention_mask.unsqueeze(1)
        
    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    # Position ids.
    assert loss_mask is not None
    layout = args.layout
    assert seq_length == layout[-1]
    n_pads = seq_length - loss_mask.sum(dim=-1).long()
    position_ids = torch.zeros(batch_size, layout[1], dtype=torch.long,
                                device=data.device)
    for i in range(batch_size):
        torch.arange(layout[1] - n_pads[i], out=position_ids[i, n_pads[i]:layout[1]], 
            dtype=torch.long, device=data.device)
    video_position_ids = torch.arange(layout[2]-layout[1], dtype=torch.long, device=data.device).unsqueeze(0) # but dont share same pos embedding with [0, layout[1])
    # video_position_ids = torch.arange(layout[1]-layout[0], dtype=torch.long, device=data.device).repeat(15).unsqueeze(0)
    video_position_ids = video_position_ids.repeat(batch_size, 1)
    position_ids = torch.cat((position_ids, video_position_ids), dim=1)

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'loss_mask']
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
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()

    labels = tokens_[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    attention_mask = None        

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
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
    
    # split img & txt positions, [PAD] not included # TODO check enough
    tokenizer = get_tokenizer()
    img_txt_sep = tokenizer.img_tokenizer.num_tokens
    img_indices_bool = (tokens < img_txt_sep) & (loss_mask > 0)
    txt_indices_bool = (~img_indices_bool) & (loss_mask > 0)
    # Forward model.
    
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    logits, *mems = model(tokens, position_ids, attention_mask)
    # print(prof.table())
    # prof.export_chrome_trace('trace/pretrain_video_320_2_profile.json')
    
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask[txt_indices_bool] *= args.txt_loss_scale
    loss_mask = loss_mask.view(-1)  

    losses_1d = losses.view(-1) * loss_mask
    loss = torch.sum(losses_1d) / loss_mask.sum()
    # =====================   Log partial losses   ======================== #
    log_loss_dict = {}
    perframe_len = args.layout[1]-args.layout[0]
    bs = losses.shape[0]
    for i in range(16):
        log_loss_dict[f'f{i}_loss'] = losses[:, args.layout[0]+i*perframe_len-1:args.layout[0]+(i+1)*perframe_len-1].contiguous().view(-1).detach().sum() / (perframe_len*bs)
    
    return loss, log_loss_dict

def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    layout = [64, 64+1024, 64+1024*16]
    def process_fn(row):
        row = row.astype(np.int64)
        codes = [row[64+1024*i:64+1024*(i+1)] for i in range(16)]
        begin_codes = ['[BOI1]', '[BOI2]', '[BOI3]', '[EOI2]', '[EOI3]', '[ROI2]', '[ROI3]', '[POS0]', 
                    '[POS1]', '[POS2]', '[POS3]', '[POS4]', '[POS5]', '[POS6]', '[POS7]', '[POS8]']
        text = row[:64]
        text = text[text>0][:64 - 2] # [ROI1][BASE]
        n_pad = 64-2-len(text)
        parts = [
            np.array([tokenizer['[PAD]']] * n_pad, dtype=np.int64),
            np.array([tokenizer['[ROI1]']], dtype=np.int64),
            text,
            np.array([tokenizer['[BASE]']], dtype=np.int64),
        ]
        for i in range(16):
            # parts += [np.array([tokenizer[begin_codes[i]]], dtype=np.int64), codes[i]]
            parts.append(np.array(codes[i])) # 每层结束需要view，最好让frame的总长是2的倍数(m*2^k，k越大越好)
        parts.append(np.array([tokenizer['[EOI1]']], dtype=np.int64))
        
        ret = np.concatenate(parts, axis=0)
        return {'text': ret, 
            'loss_mask':  np.array([0] * (n_pad+1) + [1] * (len(ret) - n_pad - 1)) # don't predict [ROI1]
            }
    return BinaryDataset(path, process_fn, length_per_sample=64+16*1024)


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    
    py_parser.add_argument('--txt-loss-scale', type=float, default=1)
    
    VideoModel.add_model_specific_args(py_parser)
    
    known, args_list = py_parser.parse_known_args()
    
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    args.layout = [int(x) for x in args.layout.split(',')]
    
    training_main(args, model_cls=VideoModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
