
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
class ClassificationModelWithPT(RobertaModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.del_mixin('roberta-final')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))

import torch.nn.functional as F
def solve(model_new, model_old):
    para_1 = {}
    total_diff = {}
    for name, params in model_old.named_parameters():
        if 'transformer.layers' in name:
            para_1[name] = params.data
    for name, params in model_new.named_parameters():
        if 'transformer.layers' in name and name in para_1:
            diff = para_1[name] - params.data
            diff_ab = diff.abs().sum()/torch.ones_like(diff).sum()
            eps = 1e-7
            diff_re = (diff.abs()/(para_1[name].abs()+eps)).sum()/torch.ones_like(diff).sum()
            print(name[18:], diff_re.cpu().numpy(), diff_ab.cpu().numpy())


    # for name, diff in total_diff.items():
    #     print(name, diff)

    # breakpoint()
    pass

if __name__ == '__main__':

    #checkpoints/finetune-roberta-large-rte-all-lr1e-05-seed504467621-05-30-16-53
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--ssl_load2', type=str, default=None)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    initialize_distributed(args)
    set_random_seed(args.seed)
    model_new = ClassificationModel(args)
    model_old = ClassificationModel(args)

    args.do_train = False
    _ = load_checkpoint(model_old, args)
    args.load = args.ssl_load2
    _ = load_checkpoint(model_new, args)


    solve(model_new, model_old)
