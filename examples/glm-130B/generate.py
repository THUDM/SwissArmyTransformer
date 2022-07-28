import os
import torch
import stat
import re

from functools import partial

from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy, BaseStrategy
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from initialize import initialize, initialize_model_and_tokenizer


def add_generation_specific_args(parser):
    parser.add_argument("--sampling-strategy", type=str, default="BaseStrategy", help="type name of sampling strategy")


def get_masks_and_position_ids(seq, mask_position, context_length, gmask=False):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(len(seq), dtype=torch.long, device=tokens.device)
    if not gmask:
        position_ids[context_length - 1 :] = mask_position

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def main(args):
    model, tokenizer = initialize_model_and_tokenizer(args)

    end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]
    # define function for each query

    if args.sampling_strategy == "BaseStrategy":
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k, end_tokens=end_tokens)
    elif args.sampling_strategy == "BeamSearchStrategy":
        strategy = BeamSearchStrategy(
            args.num_beams,
            length_penalty=args.length_penalty,
            consider_end=True,
            end_tokens=end_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            min_tgt_length=args.min_tgt_length,
        )
    else:
        raise ValueError(f"unknown strategy {args.sampling_strategy}")

    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split("\t")

        # add MASK
        generation_mask = "[MASK]" if "[MASK]" in raw_text else "[gMASK]"

        mask_pattern = r"\[g?MASK\]"
        text_list = re.split(mask_pattern, raw_text)
        pattern_list = re.compile(mask_pattern).findall(raw_text)
        seq = []
        for i in range(len(pattern_list)):
            pattern = pattern_list[i]
            sub_text = text_list[i]
            seq.extend(tokenizer.tokenize(sub_text))
            seq.append(tokenizer.get_command(pattern))

        seq.extend(tokenizer.tokenize(text_list[-1]))

        if "MASK]" not in raw_text:
            seq += [tokenizer.get_command(generation_mask)]
            raw_text += " " + generation_mask
        if not raw_text.endswith("MASK]"):
            seq = seq + [tokenizer.get_command("eos")]
        if mpu.get_model_parallel_rank() == 0:
            print("raw text: {}\n".format(raw_text))
        if len(seq) > args.max_sequence_length:
            raise ValueError("text too long.")

        # generation
        args.batch_size = 1
        mbz = 1
        output_list = [seq]
        # continually detect the first mark position
        while True:
            seq = output_list[0]  # TODO find the best one
            # detect
            mask_tokens = tokenizer.get_command(generation_mask)
            mask_position = len(seq)
            try:
                mask_position = min(mask_position, seq.index(mask_tokens))
            except ValueError:
                pass
            if mask_position == len(seq):
                break

            get_func = partial(
                get_masks_and_position_ids, mask_position=mask_position, context_length=len(seq) + 1,
                gmask=generation_mask == "[gMASK]"
            )

            output_list = []

            for tim in range(max(args.batch_size // mbz, 1)):
                input_seq = torch.cuda.LongTensor(
                    seq + [tokenizer.get_command("sop")] + [-1] * (args.out_seq_length - len(seq) - 1),
                    device=args.device,
                )
                output = filling_sequence(
                    model,
                    input_seq,
                    batch_size=args.num_beams if args.sampling_strategy == "BeamSearchStrategy" else 1,
                    strategy=strategy,
                    log_attention_weights=None,
                    get_masks_and_position_ids=get_func,
                )[
                    0
                ]  # we don't use mems, fill back
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
                bog = output.index(tokenizer.get_command("sop"))
                output_list[i] = output[:mask_position] + output[bog + 1 : unfinished] + output[mask_position + 1 : bog]

        # decoding
        txts = []
        for seq in output_list:
            decode_tokens = tokenizer.detokenize(seq)
            txts.append(decode_tokens)

        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id + ".txt")
        else:
            prefix = raw_text.replace("/", "")[:20]
            full_path = timed_name(prefix, ".txt", args.output_path)
            if mpu.get_model_parallel_rank() == 0:
                print("answer", txts)  # print the first.
        with open(full_path, "w", encoding="utf-8") as fout:
            for txt in txts:
                fout.write(txt + "\n")
        os.chmod(full_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source)


if __name__ == "__main__":
    args = initialize(extra_args_provider=add_generation_specific_args)

    with torch.no_grad():
        main(args)
