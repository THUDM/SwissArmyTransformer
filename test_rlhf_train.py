import torch
import argparse

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from transformers import default_data_collator

from sat import get_args, get_tokenizer, AutoModel, training_main
from sat.rlhf.reward_model import RewardModel
from sat.rlhf.rlhf_main import rlhf_training_main
from sat.rlhf.utils import batch_filling_sequence

class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        batch = {}
        print("data", data)
        # [{'prompt': '\n\nHuman: Can you teach me to make deviled eggs?\n\nAssistant:', 'response': ' Sure! I love deviled eggs, so I’m happy to do so! First, we need to get the egg yolks separated from the egg whites. You can do this with this separator you see in the image below.\n\n[]', 'chosen': ' Sure! I love deviled eggs, so I’m happy to do so! First, we need to get the egg yolks separated from the egg whites. You can do this with this separator you see in the image below.\n\n[]', 'rejected': ' I would be happy to help you to make deviled eggs.  Can you tell me more about what deviled eggs are and what you’d like to do with them?'}]
        pad_token_id = 0

        # prompt = pad_sequence([f[0] for f in data],
        #                       padding_value=pad_token_id,
        #                       batch_first=True)
        # prompt_mask = pad_sequence([f[1] for f in data],
        #                            padding_value=0,
        #                            batch_first=True)

        prompt = torch.cat([torch.arange(self.max_token_len - 1), torch.tensor([-1])]).repeat(len(data)).reshape((len(data), self.max_token_len))
        # prompt_mask = torch.ones(self.max_token_len).repeat(len(data)).reshape((len(data), self.max_token_len))
        prompt_mask = torch.ones((len(data), 1, self.max_token_len, self.max_token_len))
        prompt_mask.tril_()

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


if __name__ == '__main__':    
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Parse args, initialize the environment. This is necessary.

    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--train-ds-config', type=str, default="ds_config.json")
    # py_parser.add_argument('--deepspeed-config', type=str, default="ds_config.json")
    py_parser.add_argument('--eval-ds-config', type=str, default="ds_config.json")
    py_parser.add_argument('--critic-gradient-checkpointing', type=bool, default=False)
    py_parser.add_argument('--actor-gradient-checkpointing', type=bool, default=False)
    py_parser.add_argument('--enable-ema', type=bool, default=False)
    py_parser.add_argument('--max-answer-seq_len', type=int, default=128)
    py_parser.add_argument('--end-of-conversation-token', type=str, default="<|endoftext|>")
    py_parser.add_argument('--num-train-epochs', type=int, default=1)
    py_parser.add_argument('--generation-batch-numbers', type=int, default=2)
    py_parser.add_argument('--per-device-mini-train-batch-size', type=int, default=2)
    py_parser.add_argument('--per-device-train-batch-size', type=int, default=2)
    py_parser.add_argument('--max-prompt-seq-len', type=int, default=64)
    py_parser.add_argument('--inference-tp-size', type=int, default=1)
    py_parser.add_argument('--ppo-epochs', type=int, default=1)

    known, args_list = py_parser.parse_known_args()
    # args_list.append({"here": 1})
    print("known", known)
    args_list.extend(['--mode', 'finetune'])
    print("args_list", args_list)  # ['--old_hyperparam', 'no']
    args = get_args(args_list + '--train-data fake --num-workers 0'.split())
    args = argparse.Namespace(**vars(args), **vars(known))

    # known, args_list = args.parse_known_args()
    # print(args)
    # print(getattr(args, 'loss_scale_window'))
    # exit()

    rlhf_training_main(args)
    
