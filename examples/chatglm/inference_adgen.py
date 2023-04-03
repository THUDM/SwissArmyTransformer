from dotenv import load_dotenv
load_dotenv()

import os
import torch
import argparse
from SwissArmyTransformer import get_args

py_parser = argparse.ArgumentParser(add_help=False)
py_parser.add_argument('--ckpt_path', type=str)
known, args_list = py_parser.parse_known_args()
args = get_args(args_list)
args = argparse.Namespace(**vars(args), **vars(known))

model_path = args.ckpt_path
from finetune_chatglm import PTModel
from chat_model import ChatModel
model, args = ChatModel.from_pretrained(args, model_path, PTModel, prefix='model.')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = model.eval().cuda()
while True:
    summary = input("输入提示语：")
    tokens = tokenizer(summary, return_tensors='pt')['input_ids'].cuda()
    gen_kwargs = {"max_length": 512, "num_beams": 1, "do_sample": True, "top_p": 0.7,
                    "temperature": 0.95}
    outputs = model.generate(input_ids=tokens, **gen_kwargs)
    preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(preds)