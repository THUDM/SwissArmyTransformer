# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from packaging import version

from sat.model.mixins import CachedAutoregressiveMixin
from sat import AutoModel, get_tokenizer
from sat import mpu
from sat.training.deepspeed_training import get_optimizer_param_groups, get_learning_rate_scheduler

from .reward_model import RewardModel
# from utils.ds_utils import get_train_ds_config, get_eval_ds_config
# from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
# from utils.model.model_utils import create_hf_model, create_critic_model
# from utils.utils import get_optimizer_grouped_parameters
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedRLHFEngine():

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = None

        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        self.actor_ema = None
        if self.args.enable_ema:
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)

        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)
        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    def _init_actor(self, actor_model_name_or_path):
        stime = log_init("Actor")

        # Model
        actor_model, self.actor_model_args = AutoModel.from_pretrained(name=actor_model_name_or_path)
        actor_model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        # self.args.disable_actor_dropout
        self.tokenizer = get_tokenizer(self.actor_model_args)
        self.tokenizer.pad_token = 'Ġ'
        self.tokenizer.pad_token_id = 220  # TODO
        

        # TODO:
        # self.args.disable_actor_dropout
        # LoRA
        # move enable_hybrid_engine and pin_parameters to ds_config
        # zero

        param_groups = get_optimizer_param_groups(actor_model)

        actor_engine, optimizer, _, _ = deepspeed.initialize(
            model=actor_model,
            model_parameters=param_groups,
            args=self.args,
            mpu=mpu,
            dist_init_required=False,
            # config=self.args.train_ds_config
        )

        lr_scheduler = get_learning_rate_scheduler(optimizer, self.args.train_iters, self.args)
        actor_engine._configure_lr_scheduler(lr_scheduler)

        log_init("Actor", stime=stime)

        return actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        
        ref_model, self.ref_model_args = AutoModel.from_pretrained(name=actor_model_name_or_path)
        ref_model.add_mixin('auto-regressive', CachedAutoregressiveMixin())  # TODO: remove this when trainning
        ref_engine, *_ = deepspeed.initialize(
            model=ref_model,
            args=self.args,
            # config=self.args.eval_ds_config
        )

        log_init("Ref", stime=stime)
        return ref_engine

    def _init_ema(self, actor_model_name_or_path):
        stime = log_init("EMA")

        # TODO: LoRA
        
        actor_model_ema, self.actor_model_ema_args = AutoModel.from_pretrained(name=actor_model_name_or_path)

        ema_engine, *_ = deepspeed.initialize(
            model=actor_model_ema,
            args=self.args,
            # config=self.args.eval_ds_config
        )

        log_init("EMA", stime=stime)
        return ema_engine

    def _init_critic(self, critic_model_name_or_path):
        stime = log_init("Critic")

        critic_model, self.critic_model_args = AutoModel.from_pretrained(name=critic_model_name_or_path)
        critic_model = RewardModel(critic_model, self.critic_model_args, pad_token_id=self.tokenizer.pad_token_id, num_padding_at_beginning=self.tokenizer.pad_token_id)
        # self.args.num_padding_at_beginning
        # self.args.disable_critic_dropout

        # TODO:
        # LoRA

        # DeepSpeed Engine
        param_groups = get_optimizer_param_groups(critic_model)

        critic_engine, optimizer, _, _ = deepspeed.initialize(
            model=critic_model,
            model_parameters=param_groups,
            args=self.args,
            mpu=mpu,
            dist_init_required=False,
            # config=self.args.train_ds_config
        )

        lr_scheduler = get_learning_rate_scheduler(optimizer, self.args.train_iters, self.args)
        critic_engine._configure_lr_scheduler(lr_scheduler)

        log_init("Critic", stime=stime)
        return critic_engine

    def _init_reward(self, critic_model_name_or_path):
        stime = log_init("Reward")

        #TODO: ZeRO
        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3

        reward_model, self.reward_model_args = AutoModel.from_pretrained(name=critic_model_name_or_path)
        reward_model = RewardModel(reward_model, self.critic_model_args, pad_token_id=self.tokenizer.pad_token_id, num_padding_at_beginning=self.tokenizer.pad_token_id)
        # self.args.num_padding_at_beginning

        reward_engine, *_ = deepspeed.initialize(
            model=reward_model,
            args=self.args,
            # config=self.args.eval_ds_config
        )

        log_init("Reward", stime=stime)
        return reward_engine
