# -*- encoding: utf-8 -*-
# @File    :   analyis_head
# @Time    :   2022/4/6
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import numpy
import torch
import argparse
import numpy as np
import copy
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main, initialize_distributed, load_checkpoint
from SwissArmyTransformer.model.finetune import *
from SwissArmyTransformer.model.mixins import BaseMixin
from functools import partial
from utils import create_dataset_function, ChildTuningAdamW, set_optimizer_mask
import os
from roberta_model import RobertaModel
from utils import *
from tqdm import tqdm

from transformers import RobertaTokenizer
pretrain_path = ''
tokenizer =  RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'), local_files_only=True)

class MLPHeadMixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu):
        super().__init__()
        # init_std = 0.1
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i, sz in enumerate(output_sizes):
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            self.layers.append(this_layer)

    def final_forward(self, logits, **kw_args):
        logits = logits[:,:1].sum(1)
        for i, layer in enumerate(self.layers):
            if i > 0:
                logits = self.activation_func(logits)
                return logits
            logits = layer(logits)
        return logits

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        self.add_mixin('ffadd', FFADDMixin(args.hidden_size, args.num_layers, args.ffadd_r))
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))


from SwissArmyTransformer.data_utils import make_loaders
from SwissArmyTransformer.training.utils import Timers
from utils import create_dataset_function
def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'attention_mask', 'label']
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
    tokens = data_b['input_ids'].long()
    labels = data_b['label'].long()
    position_ids = data_b['position_ids'].long()
    attention_mask = data_b['attention_mask'][:, None, None, :].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, attention_mask, position_ids, (tokens!=1)

def solve(model, args):
    good_mixin = model.mixins["ffadd"]

    dataset_name = args.dataset_name

    args.train_data=None
    args.valid_data=[f"hf://super_glue/{dataset_name}/validation"]

    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    # print("hahahahh")
    timers = Timers()

    data_num = len(val_data)
    val_data = iter(val_data)
    ffadd_r = 32
    sentences = []
    words = []
    positive = [[] for i in range(24)]
    thre = 2
    for i in range(24):
        for j in range(ffadd_r):
            positive[i].append([])
    for i in tqdm(range(data_num)):
        tokens, labels, attention_mask, position_ids, loss_mask = get_batch(val_data, args, timers)
        attention_output = []
        output_good = model(tokens, position_ids, attention_mask, attention_output = attention_output)
        for k in range(24):
            attention_output.append(output_good[k+1]["0"])
        now_pos = len(sentences)
        now_word_pos = len(words)
        for j in range(tokens.shape[0]):
            sentences.append(tokenizer.decode(tokens[j]))
            for k in range(len(tokens[j])):
                words.append(tokenizer.decode(tokens[j][k]))
        # for j in range(24):
        #     for k in range(attention_output[j].shape[1]):
        #         for l in range(ffadd_r):
        #             value = attention_output[j][0,k,l]
        #             if value > thre:
        #                 #pos is j,l, sentence is now_pos, k
        #                 positive[j][l].append((now_pos, now_word_pos+k))
    import json
    with open(f"rte_thr{thre}.json", "r") as f:
        positive = json.load(f)

    # with open(f"rte_thr{thre}.json", "w") as f:
    #     json.dump(positive, f)

    lens = []
    ll = 5
    rr = 20
    answers = []
    for i in range(24):
        for j in range(32):
            if len(positive[i][j])>5 and len(positive[i][j])<20:
                answers.append(positive[i][j])

    for array in answers:
        print("let!!!!!")
        concat = ""
        for pos_s, pos_w in array:
            concat += words[pos_w]
        print(concat)
        breakpoint()




if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16-1)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--ssl_load2', type=str, default=None)
    py_parser.add_argument('--dataset-name', type=str, default=None, required=True)

    #ffadd
    py_parser.add_argument('--ffadd-r', type=int, default=32)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    initialize_distributed(args)
    model = ClassificationModel(args)

    args.do_train = False

    args.load = '/thudm/workspace/yzy/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-rte-ffadd-lr0.0005-seed944257842-05-16-14-17'
    _ = load_checkpoint(model, args)
    model.to('cuda:0')
    solve(model, args)