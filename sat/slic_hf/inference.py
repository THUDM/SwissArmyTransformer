# -*- encoding: utf-8 -*-

import os
import sys
import torch
import argparse
import json
import torch.nn.functional as F

from functools import partial
from tqdm import tqdm

from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize
from sat.model import AutoModel
from sat.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from sat.generation.autoregressive_sampling import filling_sequence

from datasets import load_dataset, Dataset
from torch.utils.data.dataloader import DataLoader

from utils import batch_filling_sequence, get_masks_and_position_ids_gpt2
from slic_hf import SLiCDataSet

def main():
    # CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 16666 inference.py --from_pretrained /zhangpai21/sxx/workspace/slic-hf/checkpoints/finetune-gpt-2-05-27-11-32 --top_k 1
    # CUDA_VISIBLE_DEVICES=0 deepspeed --master_port 16666 inference.py --from_pretrained gpt2 --top_k 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=128, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=40, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.7, help='temperature for sampling')
    parser.add_argument("--from_pretrained", type=str, default="/zhangpai21/sxx/workspace/slic-hf/checkpoints/finetune-gpt-2-05-27-11-32", help='pretrained ckpt')
    parser.add_argument("--local_rank", type=int, default=0, help='deepspeed local rank')
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
            fp16=True,
            skip_init=True,
            use_gpu_initialization=True,
            device='cuda'
        )
    )
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = AutoTokenizer.from_pretrained("/zhangpai21/sxx/models/gpt2")

    strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, end_tokens=[tokenizer.eos_token_id])
    
    # seq = tokenizer("\n\nHuman: How to install windows?\n\nAssistant:", return_tensors="pt")["input_ids"].long().cuda()

    # output = batch_filling_sequence(model, seq, context_lengths=torch.tensor([seq.shape[1]] * seq.shape[0]), strategy=strategy, get_masks_and_position_ids=partial(get_masks_and_position_ids_gpt2, max_answer_seq_len=args.max_length))[0]

    # print(tokenizer.decode(output[0].long().tolist()))

    dataset = load_dataset('parquet', data_files="/zhangpai21/sxx/data/rm-static/data/test-00000-of-00001-8c7c51afc6d45980.parquet")
    print(dataset)
    slic_dataset = SLiCDataSet(dataset["train"], tokenizer)

    results = []
    cnt = 0

    for sample in tqdm(slic_dataset):  # TODO: batched generation
        # print(sample)
        seq = sample["prompt"].unsqueeze_(0).cuda()
        output = batch_filling_sequence(model, seq, context_lengths=torch.tensor([seq.shape[1]] * seq.shape[0]), strategy=strategy, get_masks_and_position_ids=partial(get_masks_and_position_ids_gpt2, max_answer_seq_len=args.max_length))[0]
        prompt = tokenizer.decode(seq[0].long().tolist())
        output_text = tokenizer.decode(output[0][seq.shape[1]:].long().tolist())
        ref_text = tokenizer.decode(sample["reference"].long().tolist())
        results.append(
            {
                "prompt": prompt,
                "output": output_text,
                "reference": ref_text
            }
        )
        # print("================================================================")
        # print(prompt)
        # print(output_text)
        # print(ref_text)
        # input()
        cnt += 1
        if cnt == 100:
            break
    with open("results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

if __name__ == "__main__":
    main()
