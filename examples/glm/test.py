import os
import sys
sys.path.append("../..")
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import argparse
import stat
from functools import partial
import json
from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import TSVDataset
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.mpu.transformer import standard_attention
from SwissArmyTransformer.model.mixins import MLPHeadMixin, PrefixTuningMixin


def get_model(args, model_cls):
    """Build the model."""
    model = model_cls(args)

    if args.fp16:
        model.half()
    elif args.bf16:
        model.bfloat16()
    model.cuda(torch.cuda.current_device())

    return model


def setup_model_and_optimizer(args, model_cls, config_params=None):
    """Setup model and optimizer."""

    model = get_model(args, model_cls)

    model.disable_untrainable_params()  # mark trainable params
    optimizer = None

    return model, optimizer

class ClassificationModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=False):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, args.num_categories))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
        self.tuning_mode = args.tuning_mode
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        if self.tuning_mode == "ptuning":
            for layer_id in range(len(self.transformer.layers)):
                self.transformer.layers[layer_id].requires_grad_(False)


def get_batch(data, args):

    tokens = torch.LongTensor(data).unsqueeze(0).cuda()
    batch_size, seq_length = tokens.size()

    position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    position_ids = position_ids.unsqueeze(0)

    attention_mask = torch.ones((batch_size, 1, seq_length, seq_length), device=tokens.device)

    attention_mask[..., :seq_length] -= (tokens == -1).view(batch_size, 1, 1, seq_length).float()
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    return tokens, attention_mask, position_ids, (tokens != -1)

def forward_step(data, model, args):
    """Forward step."""

    # Get the batch.
    tokens, attention_mask, position_ids, loss_mask = get_batch(
        data, args)
    # print(type(tokens), type(attention_mask), type(position_ids), type(loss_mask))
    logits, *mems = model(tokens, position_ids, attention_mask)
    loss_mask = loss_mask.unsqueeze(-1).repeat(1, 1, args.num_categories)

    pred = ((logits.contiguous().float()) * loss_mask).sum(dim=-2) / torch.sum(loss_mask)
    # m = torch.nn.LogSoftmax(dim=1)
    # #loss_fn = torch.nn.NLLLoss()
    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss = loss_fn(m(pred), labels)
    # acc = torch.sum((pred.argmax(dim=-1).eq(labels)).float()) / labels.numel()
    return pred.argmax(dim=-1)


def load_hyperparam(default_args, config_path="hyperparams/bert/base_config.json"):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)
    default_args_dict = vars(default_args)
    default_args_dict.update(config_args_dict)
    args = argparse.Namespace(**default_args_dict)

    return args


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=80)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--num_categories', type=int, default=3)
    py_parser.add_argument('--tuning_mode', type=str, default="ptuning")
    py_parser.add_argument('--visible_devices', type=str, default="0")
    py_parser.add_argument('--checkpoint_path', type=str)
    GLMModel.add_model_specific_args(py_parser)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args = load_hyperparam(args, config_path="config/base_config.json")

    # Load Tokenizer
    tokenizer = get_tokenizer(args)

    # sentence = "救救孩子吧"
    # sentence = tokenizer._encode(sentence)
    # sentence = [tokenizer.get_command('ENC').Id] + sentence + [tokenizer.get_command('eos').Id]

    # Load pretrained model
    sd = torch.load(args.checkpoint_path, map_location='cpu')
    model, optimizer = setup_model_and_optimizer(args, ClassificationModel)
    missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)

    with open(args.input_source) as f:
        lines = f.readlines()
    predictions = []
    for line in lines:
        line = tokenizer._encode(line)
        line = [tokenizer.get_command('ENC').Id] + line + [tokenizer.get_command('eos').Id]
        pred = forward_step(line, model, args)
        predictions.append(pred.tolist()[0])

    output = ' '.join(list(map(str, predictions)))
    print("---------Final Prediction---------")
    for i in range(len(lines)):
        print("Input: %s"%lines[i].strip())
        print("Output: %d"%predictions[i])
        print(' ')

