import os
import torch
import argparse
from sat import get_args, AutoModel

args = get_args()

model_type = 'llama-7b'
model, args = AutoModel.from_pretrained(model_type, args)
device = model.parameters().__next__().device
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

model = model.eval()
with torch.no_grad():
    batch = tokenizer(
        "This is a piece of text.",
        return_tensors="pt", 
        add_special_tokens=False
    )
    batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).unsqueeze(0)
    batch = {k: v.cuda() for k, v in batch.items()}
    batch['attention_mask'] = batch['attention_mask'][:, None, None, :]
    output = model(**batch)[0]
    print(output)

breakpoint()