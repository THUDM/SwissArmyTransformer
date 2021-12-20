import argparse
import os
args = argparse.Namespace(
    num_layers=12,
    vocab_size=50265,
    hidden_size=768,
    num_attention_heads=12,
    max_sequence_length=514,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    inner_hidden_size=None,
    hidden_size_per_attention_head=None,
    checkpoint_activations=True,
    checkpoint_num_layers=1,
    sandwich_ln=False,
    post_ln=True,
    model_parallel_size=1,
    world_size=1,
    rank=0
    )

import torch
init_method = 'tcp://'
master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
master_port = os.getenv('MASTER_PORT', '16666')
init_method += master_ip + ':' + master_port
torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.rank, init_method=init_method)

import SwissArmyTransformer.mpu as mpu
mpu.initialize_model_parallel(args.model_parallel_size)

from robert_model import RobertaModel
model = RobertaModel(args)

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('/data/qingsong/pretrain/roberta-base')
roberta = RobertaModel.from_pretrained('/data/qingsong/pretrain/roberta-base')

def copy_layer_param(src, dst):
    """
    in-place copy from src to dst
    src and dst should be the same layer type, e.g., both are LayerNorm or both are Linear.
        Or at least, both have same named_parameters name and shape.
    """
    src_dic = dict(src.named_parameters())
    dst_dic = dict(dst.named_parameters())
    for k in dst_dic:
        dst_dic[k].data = src_dic[k].data

def copy_layer_norm(src, dst):
    src_ln = []
    for k, v in src.named_parameters():
        if 'layernorm' in k.lower():
            src_ln.append((k, v))
    dst_ln = []
    for k, v in dst.named_parameters():
        if 'layernorm' in k.lower():
            dst_ln.append((k, v))
    assert len(src_ln) == len(dst_ln)
    for kvs, kvd in zip(src_ln, dst_ln):
        kvd[1].data = kvs[1].data

def copy_transformer_layer_wo_ln(src, dst):
    dst.attention.query_key_value.weight.data = torch.cat([src.attention.self.query.weight.data, src.attention.self.key.weight.data, src.attention.self.value.weight.data], 0)
    dst.attention.query_key_value.bias.data = torch.cat([src.attention.self.query.bias.data, src.attention.self.key.bias.data, src.attention.self.value.bias.data], 0)
    copy_layer_param(src.attention.output.dense, dst.attention.dense)
    copy_layer_param(src.intermediate.dense, dst.mlp.dense_h_to_4h)
    copy_layer_param(src.output.dense, dst.mlp.dense_4h_to_h)

def transform_weight(hugging_model, swiss_model):
    copy_layer_param(hugging_model.embeddings.word_embeddings, swiss_model.transformer.word_embeddings)
    copy_layer_param(hugging_model.embeddings.position_embeddings, swiss_model.transformer.position_embeddings)
    copy_layer_norm(hugging_model, swiss_model)
    for src_l, dst_l in zip(hugging_model.encoder.layer, swiss_model.transformer.layers):
        copy_transformer_layer_wo_ln(src_l, dst_l)

with torch.no_grad():
    transform_weight(roberta, model)

# TODO:
# Some parameters such as pooler and lm_head are not copyed for now.

breakpoint()