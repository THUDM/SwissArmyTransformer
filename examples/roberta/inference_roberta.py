import os
pretrain_path = '/data/qingsong/pretrain'

import argparse
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
    model_parallel_size=1,
    world_size=1,
    rank=0
    )

import os
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

from roberta_model import RobertaModel
model = RobertaModel(args)
model.load_state_dict(torch.load(os.path.join(pretrain_path, 'roberta-base.pt')))

from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids
from transformers import RobertaTokenizer, RobertaForMaskedLM
tokenizer = RobertaTokenizer.from_pretrained(os.path.join(pretrain_path, 'roberta-base'))
roberta = RobertaForMaskedLM.from_pretrained(os.path.join(pretrain_path, 'roberta-base'), output_hidden_states=True)

model.eval()
with torch.no_grad():
    text = ["This is a piece of text.", "Another piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    hugging_output = roberta(**encoded_input)[0]
    position_ids = create_position_ids_from_input_ids(encoded_input['input_ids'], 1, 0)
    print(position_ids)
    model.to('cuda:0')
    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    # Since we don't use padding_idx for Embedding layers, pad output is largely different between hugging and swiss.
    # You will find it if you calculate error for hugging_output[1] and swiss_output[1].
    # However, pad output is usually not used, it doesn't matter too much.
    print("max error:", (hugging_output[0] - swiss_output[0]).abs().max())
    print("max relative error:", ((hugging_output[0] - swiss_output[0]).abs() / torch.max(swiss_output[0].abs(), hugging_output[0].abs())).max())

# breakpoint()