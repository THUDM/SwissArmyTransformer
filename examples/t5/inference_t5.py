# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
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
import argparse
import stat
from functools import partial


from SwissArmyTransformer import mpu, get_args, get_tokenizer, initialize_distributed, set_random_seed, load_checkpoint
from SwissArmyTransformer.model import T5Model
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually


def get_masks_and_position_ids_t5(seq):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(1, len(seq), device=tokens.device, dtype=torch.long)

    return tokens, attention_mask, position_ids


def main(args):
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # build model
    model = T5Model(args)
    model.decoder.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    if args.fp16:
        model = model.half()
    elif args.bf16:
        model = model.bfloat16()
    model = model.to(args.device)
    load_checkpoint(model, args)
    set_random_seed(args.seed)
    model.eval()

    end_tokens = [tokenizer.get_command('eos').Id]
    # define function for each query
    if args.num_beams == 1:
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    else:
        strategy = BeamSearchStrategy(args.num_beams, length_penalty=args.length_penalty, consider_end=True,
                                      end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                      min_tgt_length=args.min_tgt_length)

    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        # add MASK
        if '<extra_id' not in raw_text:
            raw_text += ' ' + "<extra_id_0>"
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        seq = seq + [tokenizer.get_command('eos').Id]
        print('raw text: {}\n'.format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = [seq]
        # continually detect the first mark position
        seq = output_list[0]  # TODO find the best one
        seq = torch.cuda.LongTensor(seq).unsqueeze(0)
        enc_attention_mask = torch.ones(1, 1, 1, seq.size(1), device=args.device)
        if args.fp16:
            enc_attention_mask = enc_attention_mask.half()
        if args.bf16:
            enc_attention_mask = enc_attention_mask.bfloat16()
        # detect

        output_list = []
        for tim in range(max(args.batch_size // mbz, 1)):
            encoder_outputs = model.encode(seq, enc_attention_mask)
            input_seq = torch.cuda.LongTensor(
                [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - 1),
                device=args.device)
            output = filling_sequence(model.decoder, input_seq,
                                      batch_size=args.num_beams,
                                      strategy=strategy,
                                      log_attention_weights=None,
                                      get_masks_and_position_ids=get_masks_and_position_ids_t5,
                                      encoder_outputs=encoder_outputs,
                                      cross_attention_mask=enc_attention_mask
                                      )[0]  # we don't use mems, fill back
            if isinstance(output, torch.Tensor):  # different strategies
                output = list(output)

            output_list.extend(output)

        # decoding
        txts = []
        for seq in output_list:
            decode_tokens = tokenizer.DecodeIds(seq)
            txts.append(decode_tokens)

        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id + '.txt')
        else:
            prefix = raw_text.replace('/', '')[:20]
            full_path = timed_name(prefix, '.txt', args.output_path)
            print(txts[0])  # print the first.
        with open(full_path, 'w') as fout:
            for txt in txts:
                fout.write(txt + '\n')
        os.chmod(full_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source)


if __name__ == "__main__":
    py_parser = argparse.ArgumentParser(add_help=False)
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    with torch.no_grad():
        main(args)
