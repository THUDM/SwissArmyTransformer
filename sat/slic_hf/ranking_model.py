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
    tokens_ = data_b['text'].long()[..., :args.max_sequence_length]
    loss_mask = data_b['loss_mask'].float()[..., :args.max_sequence_length]

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


def ranking_forward_step(data_iterator, model, args, timers):
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
    loss_mask_sum = loss_mask.sum()
    if loss_mask_sum == 0:
        loss = torch.sum(losses)
    else:
        loss = torch.sum(losses) / loss_mask_sum
    return loss, {}


class RankingDataSet(dataset.Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = data
        self.eos_token_id = torch.tensor([tokenizer.eos_token_id])

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = int(index)
        if index % 2 == 0:
            text = "[PROMPT]{prompt}[RESPONSE A]{response_a}[RESPONSE B]{response_b}[Better]".format(
                prompt=self.dataset[index // 2]["prompt"],
                response_a=self.dataset[index // 2]["chosen"],
                response_b=self.dataset[index // 2]["rejected"]
            )
            choice = "A"
        else:
            text = "[PROMPT]{prompt}[RESPONSE A]{response_a}[RESPONSE B]{response_b}[Better]".format(
                prompt=self.dataset[index // 2]["prompt"],
                response_a=self.dataset[index // 2]["rejected"],
                response_b=self.dataset[index // 2]["chosen"]
            )
            choice = "B"
        text = self.tokenizer(text, return_tensors='pt', truncation=True)['input_ids'][0]
        choice = self.tokenizer(choice, return_tensors='pt', truncation=True)['input_ids'][0]
        sample = {
            "text": text,
            "choice": torch.cat([choice, self.eos_token_id]),
        }
        return sample

    def __len__(self) -> int:
        return self.dataset.__len__() * 2


def create_ranking_dataset_function(path, args, tokenizer):
    dataset = load_dataset('parquet', data_files=path)["train"]
    print(dataset)
    return RankingDataSet(dataset, tokenizer)


def ranking_collate_fn(samples):
    pad_token_id = 0
    max_length = {
        "text": 0,
        "choice": 0
    }

    for sample in samples:
        max_length["text"] = max(max_length["text"], sample["text"].shape[0])
        max_length["choice"] = max(max_length["choice"], sample["choice"].shape[0])

    batch = {}

    texts = torch.stack([F.pad(sample["text"], pad=(0, max_length["text"] - sample["text"].shape[0]), mode='constant', value=pad_token_id) for sample in samples])

    choices = torch.stack([F.pad(sample["choice"], pad=(max_length["choice"] - sample["choice"].shape[0], 0), mode='constant', value=pad_token_id) for sample in samples])

    batch["text"] = torch.cat([texts, choices], dim=-1)
    batch["loss_mask"] = torch.stack([torch.cat([torch.zeros(max_length["text"]), torch.ones(sample["choice"].shape[0]), torch.zeros(max_length["choice"] - sample["choice"].shape[0])]) for sample in samples]).to(torch.int64)

    return batch


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--train-ds-config', type=str, default="ds_config.json")
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    ranking_model, ranking_model_args = AutoModel.from_pretrained(name="gpt2", args=args)

    tokenizer = AutoTokenizer.from_pretrained("/zhangpai21/sxx/models/gpt2")

    training_main(ranking_model_args, model_cls=ranking_model, forward_step_function=ranking_forward_step, create_dataset_function=partial(create_ranking_dataset_function, tokenizer=tokenizer), collate_fn=ranking_collate_fn)
