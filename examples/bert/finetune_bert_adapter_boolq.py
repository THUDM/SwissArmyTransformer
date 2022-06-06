import os

import torch
import argparse
import numpy as np

from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.model.official.bert_model import BertModel
from SwissArmyTransformer.model.mixins import MLPHeadMixin
from SwissArmyTransformer.model.base_model import BaseMixin
import torch.nn as nn

class AdapterMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size, adapter_hidden):
        super().__init__()
        self.ff1 = nn.ModuleList([
            nn.Linear(hidden_size, adapter_hidden) for _ in range(num_layers)
        ])
        self.ff2 = nn.ModuleList([
            nn.Linear(adapter_hidden, hidden_size) for _ in range(num_layers)
        ])
        self.ff3 = nn.ModuleList([
            nn.Linear(hidden_size, adapter_hidden) for _ in range(num_layers)
        ])
        self.ff4 = nn.ModuleList([
            nn.Linear(adapter_hidden, hidden_size) for _ in range(num_layers)
        ])

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        attention_output = attention_output + self.ff2[kw_args['layer_id']](nn.functional.gelu(self.ff1[kw_args['layer_id']](attention_output)))

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)
        mlp_output = mlp_output + self.ff4[kw_args['layer_id']](nn.functional.gelu(self.ff3[kw_args['layer_id']](mlp_output)))

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output
    
    def reinit(self, parent_model=None):
        # refer to https://github.com/google-research/adapter-bert/blob/1a31fc6e92b1b89a6530f48eb0f9e1f04cc4b750/modeling.py#L321
        for ly in self.ff1:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        for ly in self.ff2:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        for ly in self.ff3:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        for ly in self.ff4:
            nn.init.trunc_normal_(ly.weight, std=1e-3)
            nn.init.zeros_(ly.bias)
        

class AdapterModel(BertModel):
    def __init__(self, args, transformer=None, parallel_output=True, layernorm_epsilon=1e-12, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, layernorm_epsilon=layernorm_epsilon, **kwargs)
        self.del_mixin('bert-final')
        self.del_mixin('bert-forward')
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('adapter', AdapterMixin(args.num_layers, args.hidden_size, args.adapter_hidden))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    
    def disable_untrainable_params(self):
        enable = ['layernorm', 'adapter', 'classification_head']
        for n, p in self.named_parameters():
            flag = False
            for e in enable:
                if e in n.lower():
                    flag = True
                    break
            if not flag:
                p.requires_grad_(False)

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT-finetune', 'BERT finetune Configurations')
        # group.add_argument('--prefix_len', type=int, default=16)
        group.add_argument('--adapter_hidden', type=int, default=64)
        return super().add_model_specific_args(parser)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'position_ids', 'token_type_ids', 'attention_mask', 'label']
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
    token_type_ids = data_b['token_type_ids'].long()
    attention_mask = data_b['attention_mask'][:, None, None, :].float()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    
    return tokens, labels, attention_mask, position_ids, token_type_ids, (tokens!=1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, token_type_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    # pred = ((logits.contiguous().float().squeeze(-1)) * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    pred = logits.contiguous().float().squeeze(-1)[..., 0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

pretrain_path = ''
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrain_path, 'bert-base-uncased'))

def _encode(text, text_pair):
    encoded_input = tokenizer(text, text_pair, max_length=args.sample_length, padding='max_length', truncation='only_first')
    seq_len = len(encoded_input['input_ids'])
    position_ids = torch.arange(seq_len)
    return dict(input_ids=encoded_input['input_ids'], position_ids=position_ids.numpy(), token_type_ids=encoded_input['token_type_ids'], attention_mask=encoded_input['attention_mask'])

from SwissArmyTransformer.data_utils import load_hf_dataset
def create_dataset_function(path, args):
    def process_fn(row):
        pack, label = _encode(row['passage'], row['question']), int(row['label'])
        return {
            'input_ids': np.array(pack['input_ids'], dtype=np.int64),
            'position_ids': np.array(pack['position_ids'], dtype=np.int64),
            'attention_mask': np.array(pack['attention_mask'], dtype=np.int64),
            'token_type_ids': np.array(pack['token_type_ids'], dtype=np.int64),
            'label': label
        }
    return load_hf_dataset(path, process_fn, columns = ["input_ids", "position_ids", "token_type_ids", "attention_mask", "label"], cache_dir=args.data_root, offline=False, transformer_name="boolq_transformer")

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--sample_length', type=int, default=512-16)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--data_root', type=str)
    py_parser.add_argument('--md_type', type=str)
    py_parser = AdapterModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed
    initialize_distributed(args)
    set_random_seed(args.seed)
    model, args = AdapterModel.from_pretrained(args, args.md_type)
    # from cogdata.utils.ice_tokenizer import get_tokenizer as get_ice
    # tokenizer = get_tokenizer(args=args, outer_tokenizer=get_ice())
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
