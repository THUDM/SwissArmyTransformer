import os
import torch
import argparse
from SwissArmyTransformer import get_args, AutoModel

args = get_args()

model_type = 'chatglm-6b'
model, args = AutoModel.from_pretrained(args, model_type)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

model = model.eval()
with torch.no_grad():
    text = ["This is a piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    encoded_input['input_ids'] = encoded_input['input_ids'].cuda(0)
    attention_mask, position_ids = model.get_inputs(encoded_input['input_ids'])
    dst_output = model(input_ids=encoded_input['input_ids'], position_ids=position_ids, attention_mask=attention_mask)

breakpoint()