# -*- encoding: utf-8 -*-
'''
@File    :   inference_cogview.py
@Time    :   2021/10/09 19:41:58
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import argparse
import numpy as np

from arguments import get_args
from model.cached_autoregressive_model import CachedAutoregressiveModel
from model.simplevideo_model import SimpleVideoModel
from training import load_checkpoint, initialize_distributed, set_random_seed, prepare_tokenizer
from data_utils import BinaryDataset, make_loaders
from tokenization import get_tokenizer
from generation.sampling_strategies import BaseStrategy
from generation.autoregressive_sampling import filling_sequence
from generation.utils import timed_name, save_multiple_images, generate_continually


def create_dataset_function(path, args):
    tokenizer = get_tokenizer()
    layout = [256, 2048] # FIXME
    def process_fn(row):
        row = row.astype(np.int64).tolist()
        return [tokenizer['[BOI1]']] + row
    return BinaryDataset(path, process_fn, length_per_sample=layout[-1])


def main(args):
    initialize_distributed(args)
    tokenizer = prepare_tokenizer(args)
    
    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    val_data_iterator = iter(val_data)
    args.do_train = False
    # build model 
    model = SimpleVideoModel(args)
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_checkpoint(model, args)
    model0 = CachedAutoregressiveModel(args, transformer=model.transformer)
    set_random_seed(args.seed)
    model.eval()
    model0.eval()
    # define function for each query
    invalid_slices = [slice(tokenizer.img_tokenizer.num_tokens, None)]
    strategy = BaseStrategy(invalid_slices, 
        temperature=args.temperature, topk=args.top_k)
    
    def process(gt_seq, query_id):
        given_frames = 2
        seq = gt_seq[:given_frames*256+1] + [-1 for i in range((8-given_frames)*256)]
        # generation
        seq = torch.cuda.LongTensor(seq, device=args.device)
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = []
        for tim in range(max(args.batch_size // mbz, 1)):
            output_list.append(
                filling_sequence(model0, seq.clone(),
                    batch_size=min(args.batch_size, mbz),
                    strategy=strategy,
                    )
                )
        output_tokens = torch.cat(output_list, dim=0)
        # decoding
        imgs, txts = [], []
        for seq in output_tokens:
            decoded_imgs = [tokenizer.DecodeIds(seq.tolist()[i*256+1: (i+1)*256+1])[-1][0] for i in range(8)]
            imgs.append(decoded_imgs) # only the last image (target)
        # save
        full_path = os.path.join(args.output_path, str(query_id)+"_gt.jpg")
        # os.makedirs(full_path, exist_ok=True)
        save_multiple_images([tokenizer.DecodeIds(gt_seq[i*256+1: (i+1)*256+1])[-1][0] for i in range(8)], full_path, debug=True)
        for clip_i in range(len(imgs)):
            full_path = os.path.join(args.output_path, str(query_id)+'_'+str(clip_i)+'.jpg')
            # os.makedirs(full_path, exist_ok=True)
            save_multiple_images(imgs[clip_i], full_path, debug=True)
    
    os.makedirs(args.output_path, exist_ok=True)
    start_qi = 8
    end_qi = 32
    for q_i in range(end_qi):
        data = next(val_data_iterator)
        if q_i < start_qi:
            continue
        process(data, q_i)

if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--full-query', action='store_true')
    py_parser.add_argument('--img-size', type=int, default=256)
    
    SimpleVideoModel.add_model_specific_args(py_parser)

    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    args.layout = [int(x) for x in args.layout.split(',')]
    
    with torch.no_grad():
        main(args)