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
from roberta_model import RobertaModel, LoRAMixin, CLSMixin, CollectorMixin, PrefixTuningMixin
from SwissArmyTransformer.model.mixins import BaseMixin
from functools import partial
from utils import create_dataset_function, ChildTuningAdamW, set_optimizer_mask
import os

from transformers import RobertaTokenizer
pretrain_path = ''
tokenizer =  RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))

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

def solve(model_good, model_bad, args):
    good_mixin = model_good.mixins["classification_head"]
    bad_mixin = model_bad.mixins["classification_head"]

    dataset_name = args.dataset_name

    args.train_data=None
    args.valid_data=[f"hf://super_glue/{dataset_name}/validation"]

    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    # print("hahahahh")
    timers = Timers()

    data_num = len(val_data)
    val_data = iter(val_data)

    good_list = []
    bad_list = []
    label_list = []
    for i in enumerate(range(data_num)):
        tokens, labels, attention_mask, position_ids, loss_mask = get_batch(val_data, args, timers)
        output_good = model_good(tokens, position_ids, attention_mask, attention_output = [])[0].data.cpu().numpy()
        output_bad = model_bad(tokens, position_ids, attention_mask, attention_output = [])[0].data.cpu().numpy()
        good_list.append(output_good[0])
        bad_list.append(output_bad[0])
        label_list.append(labels[0].data.cpu().numpy())
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    good_list = numpy.stack(good_list)
    bad_list = numpy.stack(bad_list)
    good_embedding = TSNE(n_components=2).fit_transform(good_list)
    bad_embedding = TSNE(n_components=2).fit_transform(bad_list)

    plt.scatter(good_embedding[:, 0], good_embedding[:, 1], s=2, c=label_list)
    plt.savefig(f'images/after_act_rte_good_s2.jpg')
    plt.clf()
    plt.scatter(bad_embedding[:, 0], bad_embedding[:, 1], s=2, c=label_list)
    plt.savefig(f'images/after_act_rte_bad_s2.jpg')
    plt.clf()

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--ssl_load2', type=str, default=None)
    py_parser.add_argument('--dataset-name', type=str, default=None, required=True)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    initialize_distributed(args)
    model_good = ClassificationModel(args)
    model_bad = ClassificationModel(args)

    args.do_train = False

    args.load = '/sharefs/cognitive/dataset/fd5061f60d4dd7a8e055690bd68d1e2c/yzy/roberta_v100/checkpoints/finetune-roberta-large-rte-all-seed356117729-03-24-16-10'
    _ = load_checkpoint(model_good, args)
    args.load = '/sharefs/cognitive/dataset/fd5061f60d4dd7a8e055690bd68d1e2c/yzy/roberta_v100/checkpoints/finetune-roberta-large-rte-all-seed776273981-03-24-16-10'
    _ = load_checkpoint(model_bad, args)

    model_good.to('cuda:0')
    model_bad.to('cuda:0')
    solve(model_good, model_bad, args)