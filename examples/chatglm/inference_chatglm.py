import os
import torch
import argparse
from sat import get_args, AutoModel

args = get_args()

model_type = 'chatglm-6b'
model, args = AutoModel.from_pretrained(args, model_type)
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
chatglm = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True) 

chatglm = chatglm.eval().half().cuda(0)
model = model.eval()
with torch.no_grad():
    text = ["This is a piece of text."]
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)
    encoded_input['input_ids'] = encoded_input['input_ids'].cuda(0)
    hugging_output = chatglm(**encoded_input).logits.cpu()
    encoded_input['input_ids'] = encoded_input['input_ids'].cuda(1)
    attention_mask, position_ids = model.get_inputs(encoded_input['input_ids'])
    dst_output = model(input_ids=encoded_input['input_ids'], position_ids=position_ids, attention_mask=attention_mask)
    swiss_output = dst_output[0].cpu()
    print("max error:", (hugging_output - swiss_output).abs().max())
    print("max relative error:", ((hugging_output - swiss_output).abs() / torch.max(swiss_output.abs(), hugging_output.abs())).max())

breakpoint()