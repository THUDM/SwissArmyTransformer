# -*- encoding: utf-8 -*-
'''
@File    :   inference_glm.py
@Time    :   2021/10/22 19:41:58
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
from functools import partial
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

from SwissArmyTransformer import mpu, get_args, get_tokenizer, load_checkpoint, initialize_distributed, set_random_seed

from SwissArmyTransformer.model import T5Model
from SwissArmyTransformer.model.mixins import CachedAutoregressiveMixin
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence, evaluate_perplexity
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from SwissArmyTransformer.training.deepspeed_training import setup_model_and_optimizer


def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids


def main(args):
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)
    # load_checkpoint(model, args)
    set_random_seed(args.seed)

    # Model, optimizer, and learning rate.
    model_cls = T5Model
    model, optimizer = setup_model_and_optimizer(args, model_cls=model_cls)

    missing_keys, unexpected_keys = model.module.load_state_dict(
        torch.load("/dataset/fd5061f6/yanan/huggingface_models/t5-large/model_states.pt")["module"])
    optimizer.refresh_fp32_params()
    model.eval()
    input_ids = tokenizer.EncodeAsIds("The <extra_id_0> walks in <extra_id_1> park").tokenization
    input_ids = input_ids + [tokenizer.get_command("eos").Id]
    input_ids = torch.LongTensor([input_ids])
    decoder_input_ids = tokenizer.EncodeAsIds('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>').tokenization
    decoder_input_ids = decoder_input_ids + [tokenizer.get_command("eos").Id]
    decoder_input_ids = torch.LongTensor([decoder_input_ids])
    data = {'text': input_ids, 'loss_mask': input_ids.new_ones(input_ids.shape), 'target': decoder_input_ids,
            'attention_mask': input_ids.new_ones(input_ids.shape)}
    tokens, decoder_tokens, labels, loss_mask, attention_mask = get_batch(data, args)
    encoder_outputs, logits, *_ = model(enc_input_ids=tokens, dec_input_ids=decoder_tokens,
                                        enc_attention_mask=attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask)
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()
    loss.backward()

    breakpoint()

    end_tokens = [tokenizer.get_command('eop').Id, tokenizer.get_command('eos').Id]
    # define function for each query
    if args.sampling_strategy == 'BaseStrategy':
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    elif args.sampling_strategy == 'BeamSearchStrategy':
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True,
                                      end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
                                      min_tgt_length=args.min_tgt_length)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')

    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split('\t')
        # add MASK
        generation_mask = '[gMASK]' if args.task_mask else '[MASK]'
        if 'MASK]' not in raw_text:
            raw_text += ' ' + generation_mask
        seq = tokenizer.EncodeAsIds(raw_text).tokenization
        seq = [tokenizer.get_command('ENC').Id] + seq
        if not raw_text.endswith('MASK]'):
            seq = seq + [tokenizer.get_command('eos').Id]
        print('raw text: {}\n'.format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError('text too long.')

        # generation
        mbz = args.max_inference_batch_size
        assert args.batch_size < mbz or args.batch_size % mbz == 0
        output_list = [seq]
        # continually detect the first mark position
        while True:
            seq = output_list[0]  # TODO find the best one
            # detect
            mask_tokens = ['MASK', 'sMASK', 'gMASK'] if args.task_mask else ['MASK']
            mask_tokens = [tokenizer.get_command(token).Id for token in mask_tokens]
            mask_position = len(seq)
            for token in mask_tokens:
                try:
                    mask_position = min(mask_position, seq.index(token))
                except ValueError:
                    pass
            if mask_position == len(seq):
                break

            get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=len(seq))
            output_list = []
            for tim in range(max(args.batch_size // mbz, 1)):
                input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command('sop').Id] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device)
                output = filling_sequence(model, input_seq,
                                          batch_size=min(args.batch_size, mbz),
                                          strategy=strategy,
                                          log_attention_weights=None,
                                          get_masks_and_position_ids=get_func
                                          )[0]  # we don't use mems, fill back
                if isinstance(output, torch.Tensor):  # different strategies
                    output = list(output)

                output_list.extend(output)

            # clip -1s and fill back generated things into seq
            for i in range(len(output_list)):
                output = output_list[i].tolist()
                try:
                    unfinished = output.index(-1)
                except ValueError:
                    unfinished = len(output)
                if output[unfinished - 1] in end_tokens:
                    unfinished -= 1
                bog = output.index(tokenizer.get_command('sop').Id)
                output_list[i] = output[:mask_position] + output[bog + 1:unfinished] + output[mask_position + 1:bog]

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
    py_parser.add_argument('--sampling-strategy', type=str, default='BaseStrategy',
                           help='type name of sampling strategy')
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    with torch.no_grad():
        main(args)
