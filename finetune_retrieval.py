# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F
import argparse
import numpy as np

import mpu
from arguments import get_args
from model.retrieval_model import RetrievalModel
from training.deepspeed_training import training_main
from data_utils import BinaryDataset
from tokenization import get_tokenizer
from tokenization.cogview import TextCodeTemplate

def get_masks_and_position_ids(data,
                               n_pads,
                               attention_mask=None, args=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()
    
    layout = args.layout
    
    # Attention mask
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=data.device)
        attention_mask.tril_()
        
        attention_mask[:, :args.layout[0] - 5, args.layout[0] - 5:] = 0 # 0 attention for txt to img
        attention_mask[:, args.layout[0] - 5:-2, :args.layout[0] - 5] = 0 # 0 attention for img to txt
        
        attention_mask[:, -1, args.layout[0] - 5:-1] = 0 # attention for txt retrieval
        attention_mask[:, -2, :args.layout[0] - 5] = 0 # attention for img retrieval
        
        for i in range(batch_size): # 0 attention for padding
            attention_mask[i, :n_pads[i], :] = 0
            attention_mask[i, :, :n_pads[i]] = 0
        
        attention_mask.unsqueeze_(1)
    
    # Position ids.
    position_ids = torch.zeros(batch_size, seq_length, dtype=torch.long,
                               device=data.device)
    if args.retrieval_pos_embed:
        for i in range(batch_size): # all start from beginning.
            torch.arange(layout[0] - 5 - n_pads[i], out=position_ids[i, n_pads[i]:layout[0]-5],
                dtype=torch.long, device=data.device)
            torch.arange(layout[0] - 3, layout[1], out=position_ids[i, layout[0]-5:layout[1]-2],
                dtype=torch.long, device=data.device)
        position_ids[:, -1] = 0
        position_ids[:, -2] = 1
    else:
        for i in range(batch_size): # all start from beginning.
            torch.arange(layout[0] - 5 - n_pads[i], out=position_ids[i, n_pads[i]:layout[0]-5],
                dtype=torch.long, device=data.device)
            torch.arange(layout[0] - 5, layout[1], out=position_ids[i, layout[0]-5:layout[1]],
                dtype=torch.long, device=data.device)

    return attention_mask, position_ids


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['text', 'n_pads']
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
    tokens = data_b['text'].long()
    n_pads = data_b['n_pads'].long()
    
    attention_mask = None
    
    # Get the masks and postition ids.
    attention_mask, position_ids = get_masks_and_position_ids(
        tokens,
        n_pads,
        attention_mask=attention_mask,
        args=args
        )
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
        
    return tokens, attention_mask, position_ids, n_pads

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def parallel_contrastive_loss(lvec, rvec, temp, args):
    device = args.device
    lvec = lvec * temp.exp()
    
    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()
    
    batch_size_per_partition, hidden_size = lvec.shape
    batch_size = batch_size_per_partition * world_size
    split_start, split_end = rank * batch_size_per_partition, (rank + 1) * batch_size_per_partition
    arange_1d = torch.arange(0, batch_size, device=lvec.device)
    
    # Broadcast vecs
    broadcast_lvec, broadcast_rvec = lvec.detach(), rvec.detach()
    parallel_lvec = [torch.zeros(batch_size_per_partition, hidden_size, device=torch.device(device)).half() for _ in range(world_size)]
    parallel_rvec = [torch.zeros(batch_size_per_partition, hidden_size, device=torch.device(device)).half() for _ in range(world_size)]
    
    torch.distributed.all_gather(parallel_lvec, broadcast_lvec, group=mpu.get_data_parallel_group())
    torch.distributed.all_gather(parallel_rvec, broadcast_rvec, group=mpu.get_data_parallel_group())
    
    parallel_lvec = torch.cat(parallel_lvec, dim=0)
    parallel_rvec = torch.cat(parallel_rvec, dim=0)
    
    # if mpu.get_data_parallel_rank() == 0:
    #     print('parallel vec', parallel_lvec, parallel_rvec)
    
    # Calculate logits
    local_local_logits = lvec @ rvec.permute(1, 0)
    local_dist_logits = lvec @ parallel_rvec.permute(1, 0)
    dist_local_logits = parallel_lvec @ rvec.permute(1, 0)
    
    # if mpu.get_data_parallel_rank() == 0:
    #     print('local_dist_logits', local_dist_logits)
    
    # Broadcast logits
    broadcast_local_dist_logits = local_dist_logits.detach()
    parallel_logits = [torch.zeros(batch_size_per_partition, batch_size, device=torch.device(device)).half() for _ in range(world_size)]
    torch.distributed.all_gather(parallel_logits, broadcast_local_dist_logits, group=mpu.get_data_parallel_group())
    parallel_logits = torch.cat(parallel_logits, dim=0)
    
    # Fill in local logits for backward
    cat_ll_ld_logits = torch.cat([local_dist_logits[:, :split_start], local_local_logits, local_dist_logits[:, split_end:]], dim=1)
    parallel_logits = torch.cat([parallel_logits[:, :split_start], dist_local_logits, parallel_logits[:, split_end:]], dim=1)
    parallel_logits = torch.cat([parallel_logits[:split_start, :], cat_ll_ld_logits, parallel_logits[split_end:, :]], dim=0)
    # parallel_logits[split_start:split_end, :] = local_dist_logits
    # parallel_logits[:, split_start:split_end] = dist_local_logits
    # parallel_logits[split_start:split_end, split_start:split_end] = local_local_logits    
    
    # predicted_logits = parallel_logits[arange_1d, arange_1d]
    
    # # Calculate left2right loss
    # left_logits_max = torch.max(parallel_logits, dim=1)[0]
    # left_logits = parallel_logits.sub(left_logits_max.unsqueeze(dim=1))
    # left_exp_logits = left_logits.exp()
    # left_sum_exp_logits = left_exp_logits.sum(dim=1)
    # left_loss = torch.log(left_sum_exp_logits) - predicted_logits # Loss = log(sum(exp(logits))) - predicted-logit.
    # left_loss = left_loss.sum()
    
    # # Calculate right2left loss
    # parallel_logits_t = parallel_logits.permute(1, 0)
    # right_logits_max = torch.max(parallel_logits_t, dim=1)[0]
    # right_logits = parallel_logits_t.sub(right_logits_max.unsqueeze(dim=1))
    # right_exp_logits = right_logits.exp()
    # right_sum_exp_logits = right_exp_logits.sum(dim=1)
    # right_loss = torch.log(right_sum_exp_logits) - predicted_logits # Loss = log(sum(exp(logits))) - predicted-logit.
    # right_loss = right_loss.sum()

    left_loss = F.cross_entropy(parallel_logits, arange_1d)
    right_loss = F.cross_entropy(parallel_logits.permute(1, 0), arange_1d)
    
    total_loss = (left_loss + right_loss) / 2
    
    # if mpu.get_data_parallel_rank() == 0 and args.iteration % 200 == 0:
    #     import pdb
    #     pdb.set_trace()
    
    return total_loss, left_loss, right_loss

def forward_step(data_iterator, model, args, timers):
    """Forward step."""
    
    # Get the batch.
    timers('batch generator').start()
    tokens, attention_mask, position_ids, n_pads = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    
    # Forward model.
    (txt_vecs, img_vecs, temp), *mems = model(tokens, position_ids, attention_mask)
    
    # L2 Normalize
    txt_vecs = F.normalize(txt_vecs)
    img_vecs = F.normalize(img_vecs)
    # txt_vecs = txt_vecs / txt_vecs.pow(2).sum(dim=1).sqrt().unsqueeze(1)
    # img_vecs = img_vecs / img_vecs.pow(2).sum(dim=1).sqrt().unsqueeze(1)
    
    loss, txt2img_loss, img2txt_loss = parallel_contrastive_loss(txt_vecs, img_vecs, temp, args)
    # loss, txt2img_loss, img2txt_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
    
    return loss, {'txt2img_loss': txt2img_loss, 'img2txt_loss': img2txt_loss}
    
def create_dataset_function(path, args):
    tokenizer = get_tokenizer(args)
    # layout = args.layout
    layout = [64, 1088]
    def process_fn(row):
        row = row.astype(np.int64)
        codes = [row[layout[i-1]:layout[i]] for i in range(1, len(layout))]
        
        text = row[:layout[0]]
        text = text[text > 0][:layout[0] - 6]
        n_pad = layout[0] - 6 - len(text)
        merged = TextCodeTemplate(text, codes[0], tokenizer)
        parts = [
            np.array([tokenizer['[PAD]']] * n_pad, dtype=np.int64),
            merged, 
            np.array([tokenizer['[POS0]'], tokenizer['[POS1]']], dtype=np.int64)
        ]
        ret = np.concatenate(parts, axis=0)
        return {'text': ret,
                'n_pads': n_pad}
    return BinaryDataset(path, process_fn, length_per_sample=layout[-1])
    

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    
    RetrievalModel.add_model_specific_args(py_parser)
    
    known, args_list = py_parser.parse_known_args()
    
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    args.layout = [int(x) for x in args.layout.split(',')]
    
    training_main(args, model_cls=RetrievalModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)