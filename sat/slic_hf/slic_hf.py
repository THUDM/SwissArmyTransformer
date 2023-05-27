import os
import sys
import math
import random
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from functools import partial
from typing import Dict

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import dataset

from datasets import load_dataset, Dataset

from sat import AutoModel, mpu, get_args, get_tokenizer
from sat.model.base_model import BaseModel
from sat.training.deepspeed_training import training_main
from sat.data_utils import BinaryDataset

from transformers import AutoTokenizer


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
    keys = ['prompt', 'chosen', 'rejected', 'reference', 'prompt_loss_mask', 'chosen_loss_mask', 'rejected_loss_mask', 'reference_loss_mask']
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
    """
    tokens.shape[0] = 3 * batch_size:
        prompt + chosen    (1 * batch_size)
        prompt + rejected  (1 * batch_size)
        prompt + reference (1 * batch_size)
    """
    tokens_ = torch.cat(
        [
            data_b['prompt'].long().repeat((3, 1)),
            torch.cat([data_b['chosen'].long(), data_b['rejected'].long(), data_b['reference'].long()])
        ],
        dim=-1
    )[..., :args.max_sequence_length]  # TODO

    loss_mask = torch.cat(
        [
            data_b['prompt_loss_mask'].repeat((3, 1)),
            torch.cat([data_b['chosen_loss_mask'], data_b['rejected_loss_mask'], data_b['reference_loss_mask']])
        ],
        dim=-1
    ).float()[..., :args.max_sequence_length]

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
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)

    # scaling loss mask
    loss_mask = loss_mask.view(-1)
    
    para_lambda = args.para_lambda
    para_delta = args.para_delta
    
    pos_loss_mask, neg_loss_mask, ref_loss_mask = loss_mask.chunk(3, dim=0)
    pos_losses, neg_losses, ref_losses = losses.chunk(3, dim=0)
    pos_losses = pos_losses.view(-1) * pos_loss_mask
    neg_losses = neg_losses.view(-1) * neg_loss_mask
    ref_losses = ref_losses.view(-1) * ref_loss_mask
    
    pos_loss = torch.sum(pos_losses) / pos_loss_mask.sum()
    neg_loss = torch.sum(neg_losses) / neg_loss_mask.sum()
    ref_loss = torch.sum(ref_losses) / ref_loss_mask.sum()
    
    # we want lower pos_loss, and higher neg_loss. 
    # ref_loss plays a role similar to kl.
    loss = (1 - para_lambda) * max(para_delta + pos_loss - neg_loss, torch.tensor(0, device=losses.device)) + para_lambda * ref_loss
    return loss, {}


class TestDataSet2(dataset.Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = data
        self.eos_token_id = torch.tensor([tokenizer.eos_token_id])

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = int(index)
        sample = {
            "prompt": self.tokenizer(self.dataset[index]["prompt"], return_tensors='pt', truncation=True)['input_ids'][0],
            "chosen": self.tokenizer(self.dataset[index]["chosen"], return_tensors='pt', truncation=True)['input_ids'][0],
            "rejected": self.tokenizer(self.dataset[index]["rejected"], return_tensors='pt', truncation=True)['input_ids'][0],
            "reference": self.tokenizer(self.dataset[index]["response"], return_tensors='pt', truncation=True)['input_ids'][0],
        }
        for k in sample:
            if not k == "prompt":
                sample[k] = torch.cat([sample[k], self.eos_token_id])
        return sample

    def __len__(self) -> int:
        return self.dataset.__len__()


def create_slic_dataset_function(path, args, tokenizer):
    dataset = load_dataset('parquet', data_files="/zhangpai21/sxx/data/rm-static/data/train-00000-of-00001-2a1df75c6bce91ab.parquet")["train"]
    return TestDataSet2(dataset, tokenizer)


def collate_fn(samples):
    pad_token_id = 0
    max_length  = {
        "prompt": 0,
        "response": 0  # chosen, rejected, reference
    }
    keys = ["prompt", "chosen", "rejected", "reference"]

    for sample in samples:
        for k in keys:
            if k == "prompt":
                max_length["prompt"] = max(max_length[k], sample[k].shape[0])
            else:
                max_length["response"] = max(max_length["response"], sample[k].shape[0])

    batch = {}

    for k in keys:
        if k == "prompt":
            batch["prompt"] = torch.stack([F.pad(sample["prompt"], pad=(max_length["prompt"] - sample["prompt"].shape[0], 0), mode='constant', value=pad_token_id) for sample in samples])
            batch["prompt_loss_mask"] = torch.zeros(batch["prompt"].shape, dtype=torch.int64)
        else:
            batch[k + "_loss_mask"] = torch.stack([torch.cat([torch.ones(sample[k].shape[0]), torch.zeros(max_length["response"] - sample[k].shape[0])]) for sample in samples]).to(torch.int64)
            batch[k] = torch.stack([F.pad(sample[k], pad=(0, max_length["response"] - sample[k].shape[0]), mode='constant', value=pad_token_id) for sample in samples])

    return batch


if __name__ == '__main__':    
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--train-ds-config', type=str, default="ds_config.json")
    py_parser.add_argument('--para_delta', type=float, default=1.0)
    py_parser.add_argument('--para_lambda', type=float, default=0.9)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list + '--train-data fake --num-workers 0'.split())
    args = argparse.Namespace(**vars(args), **vars(known))

    actor_model, actor_model_args = AutoModel.from_pretrained(name="gpt2", args=args)

    tokenizer = AutoTokenizer.from_pretrained("/zhangpai21/sxx/models/gpt2")

    training_main(actor_model_args, model_cls=actor_model, forward_step_function=forward_step, create_dataset_function=partial(create_slic_dataset_function, tokenizer=tokenizer), collate_fn=collate_fn)
