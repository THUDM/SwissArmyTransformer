# -*- encoding: utf-8 -*-
# @File    :   draw_corr_bias.py
# @Time    :   2022/3/3
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import torch, os
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main
from roberta_model import RobertaModel
from SwissArmyTransformer.training.deepspeed_training import get_model, initialize_distributed, set_random_seed
from SwissArmyTransformer.model.mixins import PrefixTuningMixin, MLPHeadMixin
from SwissArmyTransformer.training.model_io import load_checkpoint
import argparse
from transformers import RobertaTokenizer
pretrain_path = ''
tokenizer =  RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-large'))

class ClassificationModel(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)
        # self.transformer.word_embeddings.requires_grad_(False)
        for layer_id in range(len(self.transformer.layers)):
            # self.transformer.layers[layer_id].mlp.dense_h_to_4h.requires_grad_(True) #Wm2
            # self.transformer.layers[layer_id].attention.dense.requires_grad_(True) #Wm1
            # self.transformer.layers[layer_id].attention.query_key_value.requires_grad_(True) #QKV
            self.transformer.layers[layer_id].mlp.dense_h_to_4h.bias.requires_grad_(True) #m2
            # self.transformer.layers[layer_id].attention.query_key_value.bias.requires_grad_(True) #bqk
import torch.nn.functional as F
def solve(mode_new, model_old):
    num_layer = len(mode_new.transformer.layers)
    diff = []
    word_dict = model_old.transformer.word_embeddings.weight #[50265, 1024]
    word_dict = F.normalize(word_dict)
    tokenizer_dict = tokenizer.get_vocab()
    tokenizer_dict = {v: k for k, v in tokenizer_dict.items()}
    for i in range(num_layer):
        print(f"now layer is {i}")
        m2_old = model_old.transformer.layers[i].mlp.dense_h_to_4h.bias #[4096]
        m2_new = model_new.transformer.layers[i].mlp.dense_h_to_4h.bias
        m2_diff = m2_new - m2_old
        m3 = model_old.transformer.layers[i].mlp.dense_4h_to_h.weight #[1024,4096]
        p = torch.matmul(word_dict, m3).permute([1,0]) #[4096, 50265]
        to_word = torch.max(p, dim=1)[1].numpy() #[4096]
        to_word = [tokenizer_dict[j] for j in to_word]
        sorted, indices = torch.sort(m2_diff, descending=True)
        #看一下前十个
        for j in range(10):
            p_weight = sorted[j]
            word = to_word[indices[j]]
            print(word, p_weight)
            # word =

        #看一下合起来的向量代表的词语
        m2_diff = torch.clamp(m2_diff, min=0)
        m = torch.matmul(m3, m2_diff) #[1024]
        p = torch.matmul(word_dict, m) #[50265]
        value, indices = torch.sort(p, dim=0, descending=True)
        for k in range(10):
            print(f"all is {tokenizer_dict[indices[k].numpy().tolist()]}")
        diff.append(m2_diff)


    breakpoint()
    pass

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    initialize_distributed(args)
    set_random_seed(args.seed)
    model_new = ClassificationModel(args)
    model_old = ClassificationModel(args)


    args.do_train = True
    _ = load_checkpoint(model_old, args)
    args.load = "/workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/"
    args.load = args.load + "finetune-roberta-large-boolq-onlym2-1e-4-02-26-05-17"
    _ = load_checkpoint(model_new, args)

    solve(model_new, model_old)






# if __name__ == "__main__":
#     py_parser = argparse.ArgumentParser(add_help=False)
#     py_parser.add_argument('--new_hyperparam', type=str, default=None)
#     py_parser.add_argument('--sample_length', type=int, default=512-16)
#     py_parser.add_argument('--prefix_len', type=int, default=16)
#     py_parser.add_argument('--old_checkpoint', action="store_true")
#     known, args_list = py_parser.parse_known_args()
#     args = get_args(args_list)
#     args = argparse.Namespace(**vars(args), **vars(known))
#     args.seed = 1023
#     args.num_layers = 24
#     args.vocab_size = 50265
#     args.hidden_size = 1024
#     args.num_attention_heads = 16
#     args.max_sequence_length = 514
#
#     args.load = "/workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/"
#     args.load = args.load + "finetune-roberta-large-boolq-onlym2-1e-4-02-26-05-17"
#
#
#     #pre work
#     init_method = 'tcp://'
#     master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
#     master_port = os.getenv('MASTER_PORT', '16666')
#     init_method += master_ip + ':' + master_port
#     torch.distributed.init_process_group(
#         backend='nccl',
#         world_size=args.world_size, rank=args.rank, init_method=init_method)
#
#     import SwissArmyTransformer.mpu as mpu
#     mpu.initialize_model_parallel(args.model_parallel_size)
#     model = ClassificationModel(args)
#     args.iteration = load_checkpoint(model, args)
#
#     #do work
#     breakpoint()
#     pass