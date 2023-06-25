import os
import random
import math
import numpy as np
import torch

from collections import defaultdict
from datetime import datetime
from contextlib import ExitStack
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import default_data_collator

import torch.distributed as dist
import torch.nn.functional as F
import deepspeed

from sat.training.learning_rates import AnnealingLR
from sat.training.model_io import load_checkpoint, save_checkpoint

from sat.training.utils import Timers
from sat.training.utils import report_memory
from sat.training.utils import print_args
# from sat.training.utils import print_rank_0
from sat.training.utils import get_sample_writer

from sat import AutoModel
from sat import mpu
from sat.data_utils import make_loaders
from sat.ops import LayerNorm
from sat.arguments import set_random_seed, initialize_distributed
from sat.helpers import print_rank0 as print_rank_0

from .ppo_trainer import DeepSpeedPPOTrainer  # , DeepSpeedPPOTrainerUnsupervised
from .rlhf_engine import DeepSpeedRLHFEngine
from .reward_model import RewardModel
from .utils import MiniDataset, to_device, get_all_reduce_mean


class TestDataSet(dataset.Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        # [{'prompt': '\n\nHuman: Can you teach me to make deviled eggs?\n\nAssistant:', 'response': ' Sure! I love deviled eggs, so I’m happy to do so! First, we need to get the egg yolks separated from the egg whites. You can do this with this separator you see in the image below.\n\n[]', 'chosen': ' Sure! I love deviled eggs, so I’m happy to do so! First, we need to get the egg yolks separated from the egg whites. You can do this with this separator you see in the image below.\n\n[]', 'rejected': ' I would be happy to help you to make deviled eggs.  Can you tell me more about what deviled eggs are and what you’d like to do with them?'}]
        tokenized_data = [tokenizer(sample["prompt"], return_tensors='pt') for sample in data]
        self.data = tokenized_data
        print(self.data[0])

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size, pad_token_id=0):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}
        print("data", data)
        pad_token_id = self.pad_token_id  # TODO
        pad_token_id = 220

        prompt = pad_sequence([f["input_ids"][0].flip(0) for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)

        # maybe no use?
        token_mask = pad_sequence([torch.ones_like(f["input_ids"][0].flip(0)) for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)

        prompt_mask = pad_sequence([f["attention_mask"][0].flip(0) for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)

            batch["token_mask"] = F.pad(token_mask,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)

            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["token_mask"] = token_mask
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["token_mask"] = batch["token_mask"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        
        # batch["prompt"] = F.pad(  # TODO
        #     batch["prompt"],
        #     pad=(0, 64),
        #     mode='constant',
        #     value=-1
        # )
        
        # print("batch:", batch)
        
        return batch


def create_datasets(args, tokenizer):
    prompt_train_dataset = TestDataSet(load_dataset("Dahoas/rm-static")["train"], tokenizer)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=DataCollatorRLHF(
            args.max_prompt_seq_len,
            args.inference_tp_size,
            pad_token_id=tokenizer.pad_token_id
        )
    )
    return prompt_train_dataloader


def rlhf_training_main(args, unsupervised_training_enabled=False, device=torch.device("cuda")):  # , reward_model_cls, actor_model_cls, forward_step_function, create_dataset_function, handle_metrics_function=None, init_function=None, collate_fn=None, forward_step_eval=None
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path="gpt2",  # args.actor_model_name_or_path,
        critic_model_name_or_path="gpt2",  # args.critic_model_name_or_path,
        num_total_iters=args.train_iters,
        args=args
    )

    global_rank = torch.distributed.get_rank()

    tokenizer = rlhf_engine.tokenizer
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # TODO: move to engine
    
    prompt_train_dataloader = create_datasets(args, tokenizer)
    unsupervised_train_dataloader = [None] * len(prompt_train_dataloader)

    trainer = DeepSpeedPPOTrainer(rlhf_engine, args)

    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    print_rank_0("***** Running training *****")

    for epoch in range(args.num_train_epochs):
        print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}")
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])

            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['token_mask'])
            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:  # TODO:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                print_rank_0(
                    f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}|act_loss: {actor_loss_sum/inner_iter}|cri_loss: {critic_loss_sum/inner_iter}|unsuper_loss: {unsup_loss_sum/inner_iter}',
                    torch.distributed.get_rank())
                average_reward = get_all_reduce_mean(average_reward).item()
                print_rank_0(
                    f"average reward score: {average_reward/inner_iter}",
                    torch.distributed.get_rank())
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    torch.distributed.get_rank())

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()


