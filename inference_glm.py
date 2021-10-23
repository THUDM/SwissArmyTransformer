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
import random
import time
from datetime import datetime
import torch
import torch.nn.functional as F

import mpu
from arguments import get_args
from model.glm_model import GLMModel
from training import load_checkpoint, initialize_distributed, set_random_seed, prepare_tokenizer
from generation.glm_sampling import filling_sequence_glm
from generation.sampling_strategies import BeamSearchStrategy, BaseStrategy


def read_context(tokenizer, args, output=None):
    terminate_runs, skip_run = 0, 0
    if mpu.get_model_parallel_rank() == 0:
        while True:
            raw_text = input("\nContext prompt (stop to exit) >>> ")
            if not raw_text:
                print('Prompt should not be empty!')
                continue
            if raw_text == "stop":
                terminate_runs = 1
                break
            generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
            if args.block_lm and 'MASK]' not in raw_text:
                raw_text += ' ' + generation_mask
            if output is not None:
                output.write(raw_text)
            context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization
            if args.block_lm:
                context_tokens = [tokenizer.get_command('ENC').Id] + context_tokens
                if not raw_text.endswith('MASK]'):
                    context_tokens = context_tokens + [tokenizer.get_command('eos').Id]
            context_length = len(context_tokens)

            if context_length >= args.max_sequence_length:
                print("\nContext length", context_length,
                      "\nPlease give smaller context than the window length!")
                continue
            break
    else:
        context_length = 0

    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    terminate_runs = terminate_runs_tensor[0].item()

    if terminate_runs == 1:
        return terminate_runs, None, None, None

    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item()
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    if mpu.get_model_parallel_rank() != 0:
        raw_text = tokenizer.DecodeIds(context_tokens_tensor.tolist())
    return terminate_runs, raw_text, context_tokens_tensor, context_length


def get_batch(context_tokens, args):
    tokens = context_tokens
    tokens = tokens.view(1, -1).contiguous()
    tokens = tokens.to('cuda')

    # Get the masks and postition ids.
    if args.block_lm:
        attention_mask = torch.ones(tokens.size(1), tokens.size(1), device='cuda', dtype=torch.long)
        if args.fp16:
            attention_mask = attention_mask.half()
        position_ids = torch.arange(tokens.size(1), device='cuda', dtype=torch.long)
        if not args.no_block_position:
            block_position_ids = torch.zeros(tokens.size(1), device='cuda', dtype=torch.long)
            position_ids = torch.stack((position_ids, block_position_ids), dim=0)
        position_ids = position_ids.unsqueeze(0)
    else:
        raise NotImplementedError

    return tokens, attention_mask, position_ids


def generate_samples(model, tokenizer, args):
    model.eval()
    output_path = "./samples"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M')}.txt")
    with torch.no_grad(), open(output_path, "w") as output:
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs, raw_text, context_tokens_tensor, context_length = read_context(tokenizer, args, output)
            if terminate_runs == 1:
                return
            start_time = time.time()
            if args.block_lm:
                mems = []
                tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, args)
                mask_tokens = ['MASK', 'sMASK', 'gMASK'] if args.task_mask else ['MASK']
                mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
                end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
                mask_positions = []
                for token in mask_tokens:
                    mask_positions += (context_tokens_tensor == token).nonzero(as_tuple=True)[0].tolist()
                mask_positions.sort()
                if args.no_block_position:
                    for mask_position in mask_positions:
                        position_ids[0, mask_position + 1:] += args.out_seq_length
                _, *mems = model(tokens, position_ids, attention_mask, *mems)
                for mask_position in mask_positions:
                    if args.no_block_position:
                        position = position_ids[0, mask_position].item()
                    else:
                        position = mask_position
                    if args.num_beams > 1:
                        strategy = BeamSearchStrategy(num_beams=args.num_beams, max_length=args.out_seq_length,
                                                      length_penalty=args.length_penalty, end_tokens=end_tokens)
                    else:
                        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                                                end_tokens=end_tokens)
                    new_tokens, mems = filling_sequence_glm(model, tokenizer, position, strategy, args, mems=mems,
                                                            end_tokens=end_tokens)
                    tokens = torch.cat((tokens, new_tokens), dim=1)
            output_tokens_list = tokens.view(-1).contiguous()
            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())
                trim_decode_tokens = decode_tokens
                print("\nGLM:", trim_decode_tokens, flush=True)
                output.write(trim_decode_tokens + "\n")

            torch.distributed.barrier(group=mpu.get_model_parallel_group())


def main(args):
    initialize_distributed(args)
    tokenizer = prepare_tokenizer(args)
    # build model
    model = GLMModel(args)
    if args.fp16:
        model = model.half()
    model = model.to(args.device)
    load_checkpoint(model, args)
    set_random_seed(args.seed)
    model.eval()
    generate_samples(model, tokenizer, args)


if __name__ == "__main__":
    args = get_args()

    with torch.no_grad():
        main(args)
