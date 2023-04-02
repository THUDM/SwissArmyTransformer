from dotenv import load_dotenv
load_dotenv()

import os
import torch
import argparse
from SwissArmyTransformer import get_args
from SwissArmyTransformer import update_args_with_file
from SwissArmyTransformer.training.deepspeed_training import load_checkpoint

args = get_args()

model_path = 'checkpoints/finetune-chatglm-6b-adgen-04-02-04-08/'
args = update_args_with_file(args, path=os.path.join(model_path, 'model_config.json'))
from finetune_chatglm import PTModel
model = PTModel(args)
from chat_model import ChatModel
model = ChatModel(args, model)
load_checkpoint(model, args, load_path=model_path)

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