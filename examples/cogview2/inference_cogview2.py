# -*- encoding: utf-8 -*-
'''
@File    :   inference_cogview2.py
@Time    :   2021/10/10 16:31:34
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
from torchvision import transforms

from SwissArmyTransformer import get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed
from SwissArmyTransformer.model import CachedAutoregressiveModel, Cuda2dModel
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy, IterativeEntfilterStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.cuda2d_sampling import filling_sequence_cuda2d
from SwissArmyTransformer.generation.utils import timed_name, save_multiple_images, generate_continually

def main(args):
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # build model 
    model = Cuda2dModel(args)
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_checkpoint(model, args)
    model0 = CachedAutoregressiveModel(args, transformer=model.transformer)
    set_random_seed(args.seed)
    model.eval()
    model0.eval()
    # define function for each query
    query_template = '[ROI1] {} [BASE] [BOI1] [MASK]*1024 [EOI1]' if not args.full_query else '{}'
    invalid_slices = [slice(tokenizer.img_tokenizer.num_tokens, None)]
    strategy0 = BaseStrategy(invalid_slices,
                             temperature=args.temperature, top_k=args.top_k)
    strategy1 = IterativeEntfilterStrategy(invalid_slices,
        temperature=args.temperature, topk=10) # temperature not used
    tr = transforms.Compose([
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR), 
            ])
    
    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        print('raw text: ', raw_text)
        text = query_template.format(raw_text)
        seq = tokenizer.parse_query(text, img_size=args.img_size)
        if len(seq) > 1088:  
            raise ValueError('text too long.')
        # calibrate text length
        txt_len = seq.index(tokenizer['[BASE]'])
        log_attention_weights = torch.zeros(len(seq), len(seq), 
            device=args.device, dtype=torch.half if args.fp16 else torch.float32)
        log_attention_weights[txt_len+2:, 1:txt_len] = 1.8 if txt_len <= 10 else 1.6 # TODO args

        # generation
        seq = torch.cuda.LongTensor(seq, device=args.device)
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = []
        for tim in range(max(args.batch_size // mbz, 1)):
            output0 = filling_sequence(model0, seq.clone(),
                    batch_size=min(args.batch_size, mbz),
                    strategy=strategy0,
                    log_attention_weights=log_attention_weights
                    )[0]
             # auto del mems to save CUDA memory as possible
            imgs = [tr(tokenizer.img_tokenizer.DecodeIds(x[-1025:-1].tolist())) for x in output0]
            blur64 = tokenizer.img_tokenizer.EncodeAsIds(torch.cat(imgs, dim=0).to(args.device), add_normalization=True) # [batch_size, 4096]
            len_tim = output0.shape[0]
            for tim2 in range(0, len_tim, 4):
                output1 = filling_sequence_cuda2d(model, output0[tim2: tim2+4], blur64[tim2: tim2+4], 
                    warmup_steps=3, block_hw=(4, 4),
                    strategy=strategy1
                    )
                output_list.append(output1)
        output_tokens = torch.cat(output_list, dim=0)
        # decoding
        imgs, txts = [], []
        for seq in output_tokens:
            decoded_txts, decoded_imgs = tokenizer.DecodeIds(seq.tolist())
            for i in range(len(decoded_imgs)):
                if decoded_imgs[i].shape[-1] < 512:
                    decoded_imgs[i] = torch.nn.functional.interpolate(decoded_imgs[i], size=(512, 512))
            if args.with_id:
                imgs.append(decoded_imgs[-1]) # only the last image (target)
            else:
                imgs.extend(decoded_imgs)
        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id)
            os.makedirs(full_path, exist_ok=True)
            save_multiple_images(imgs, full_path, False)
        else:
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.jpg', args.output_path)
            save_multiple_images(imgs, full_path, True)
    
    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source)

if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--full-query', action='store_true')
    py_parser.add_argument('--img-size', type=int, default=256)
    
    Cuda2dModel.add_model_specific_args(py_parser)
    
    known, args_list = py_parser.parse_known_args()
    
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    
    args.layout = [int(x) for x in args.layout.split(',')]
    
    with torch.no_grad():
        main(args)