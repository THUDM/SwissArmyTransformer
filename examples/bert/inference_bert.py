import os
import argparse
from SwissArmyTransformer import get_args
py_parser = argparse.ArgumentParser(add_help=False)
py_parser.add_argument('--pretrain_path', type=str, default=None)
py_parser.add_argument('--old_checkpoint', action="store_true")
py_parser.add_argument('--num-types', type=int)
known, args_list = py_parser.parse_known_args()
args = get_args(args_list)
args = argparse.Namespace(**vars(args), **vars(known))
pretrain_path = args.pretrain_path
model_type = '-'.join(args.load.split('/')[-1].split('-')[1:])
print(model_type)

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

from bert_model import BertModel
from SwissArmyTransformer.training.deepspeed_training import load_checkpoint
model = BertModel(args)
load_checkpoint(model, args)

from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrain_path, model_type))
bert = BertForMaskedLM.from_pretrained(os.path.join(pretrain_path, model_type), output_hidden_states=True)

model.eval()
with torch.no_grad():
    text = [["This is a piece of text.", "Another piece of text."]]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    seq_len = encoded_input['input_ids'].size(1)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand_as(encoded_input['input_ids'])
    hugging_output = bert(**encoded_input)[0]
    print(position_ids)
    model.to('cuda:0')
    swiss_output = model(input_ids=encoded_input['input_ids'].cuda(), position_ids=position_ids.cuda(), token_type_ids=encoded_input['token_type_ids'].cuda(), attention_mask=encoded_input['attention_mask'][:, None, None, :].cuda())[0].cpu()
    # Since we don't use padding_idx for Embedding layers, pad output is largely different between hugging and swiss.
    # You will find it if you calculate error for hugging_output[1] and swiss_output[1].
    # However, pad output is usually not used, it doesn't matter too much.
    print("max error:", (hugging_output[:,0] - swiss_output[:,0]).abs().max())
    print("max relative error:", ((hugging_output[:,0] - swiss_output[:,0]).abs() / torch.max(swiss_output[:,0].abs(), hugging_output[:,0].abs())).max())

# breakpoint()