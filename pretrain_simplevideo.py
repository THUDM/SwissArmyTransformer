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
from model.simplevideo_model import SimpleVideoModel
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
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=data.device)
        attention_mask.tril_()
        attention_mask.unsqueeze_(1)
        
    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

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
    # tokens_: [PAD]{frame0}{frame1}...{frame7}
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
    # Forward model.
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
    logits, *mems = model(tokens, position_ids, attention_mask)
    # print(prof.table())
    # prof.export_chrome_trace('trace/cogview2_profile.json')
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.view(-1)  

    losses_1d = losses.view(-1) * loss_mask
    loss = torch.sum(losses_1d) / loss_mask.sum()
    # =====================   Log partial losses   ======================== #
    log_loss_dict = {}
    perframe_len = 256
    bs = losses.shape[0]
    for i in range(8):
        log_loss_dict[f'f{i}_loss'] = losses[:, i*perframe_len:(i+1)*perframe_len].contiguous().view(-1).detach().sum() / (perframe_len*bs)
    # ===================== END OF BLOCK ======================= #
    return loss, log_loss_dict
    
def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    layout = [256, 2048] # FIXME
    def process_fn(row):
        row = row.astype(np.int64)
        parts = [
            np.array([tokenizer['[BOI1]']], dtype=np.int64),
            row
        ]
        ret = np.concatenate(parts, axis=0)
        return {'text': ret, 
            'loss_mask':  np.array([0] * 1 + [1] * (len(ret) - 1)) # don't predict [CLS]
            }
    return BinaryDataset(path, process_fn, length_per_sample=layout[-1])

if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    
    py_parser.add_argument('--txt-loss-scale', type=float, default=1)
    
    SimpleVideoModel.add_model_specific_args(py_parser)
    
    known, args_list = py_parser.parse_known_args()
    
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    args.layout = [int(x) for x in args.layout.split(',')]
    
    training_main(args, model_cls=SimpleVideoModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
