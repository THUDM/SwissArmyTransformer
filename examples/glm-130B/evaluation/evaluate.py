"""GLM zero-shot evaluation."""

import os
import glob
import time
import itertools
import torch
import torch.distributed as dist
import numpy as np

from collections import defaultdict
from dataset import build_dataset

from utils import build_data_loader, cond_log_prob, generate_text
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer import mpu, get_tokenizer
from SwissArmyTransformer.training import load_checkpoint
from SwissArmyTransformer.model import GLM130B
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

def evaluate(dataset, data_loader, model, strategy, batch_size, max_length):
    model.eval()

    prediction = []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            if dataset.task_type == "mul":
                output = cond_log_prob(batch, model)
                prediction.append(np.argmax(output))
            elif dataset.task_type == "gen":
                blank = []
                blank = generate_text(model, batch, strategy, batch_size, max_length=max_length)
                prediction.append(blank)

    result = None

    world_size = mpu.get_data_parallel_world_size()
    prediction_gathered = [None for _ in range(world_size)]
    dist.all_gather_object(prediction_gathered, prediction, group=mpu.get_data_parallel_group())
    prediction = list(itertools.chain(*zip(*prediction_gathered)))[: len(dataset)]
    result = {key: metric(prediction, dataset.data) for key, metric in dataset.metrics}

    return result, prediction


def main(args):
    """Main program."""
    args.do_train = False
    initialize_distributed(args)
    tokenizer = get_tokenizer(args)

    # build model 
    model = GLM130B(args)

    if args.fp16:
        model = model.half()
    model = model.to(args.device)

    load_checkpoint(model, args)
    model.eval()

    end_tokens = [tokenizer.get_command('eop'), tokenizer.get_command('eos')]

    if args.sampling_strategy == 'BaseStrategy':
        strategy = BaseStrategy(temperature=args.temperature, top_k=args.top_k,end_tokens=end_tokens)
    elif args.sampling_strategy == 'BeamSearchStrategy':
        strategy = BeamSearchStrategy(args.batch_size, length_penalty=args.length_penalty, consider_end=True, end_tokens=end_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size, min_tgt_length=args.min_tgt_length)
    else:
        raise ValueError(f'unknown strategy {args.sampling_strategy}')

    start = time.time()

    for task in args.task:
        datasets, dataloaders, filenames = [], [], []

        if torch.distributed.get_rank() == 0:
            print(f"Evaluating task {task}")
        for file_name in sorted(glob.glob(os.path.join(args.eval_data_path, task, "**/*.json*"), recursive=True)):
            if file_name.endswith("_predict.json"):
                continue
            dataset = build_dataset(file_name, args.unified_multitask_encoding)
            dataloader = build_data_loader(dataset, args.micro_batch_size, args.num_workers, drop_last=False)
            datasets.append(dataset)
            dataloaders.append(dataloader)
            filenames.append(file_name)

        if len(datasets) == 0:
            continue
        result_dict_all = defaultdict(lambda: [])
        weight = []
        for dataset, dataloader, filename in zip(datasets, dataloaders, filenames):
            result_dict, _ = evaluate(dataset, dataloader, model, strategy, args.batch_size, args.out_seq_length)
            if torch.distributed.get_rank() == 0:
                output_str = f"    Finish {filename}"
                for key, value in result_dict.items():
                    result_dict_all[key].append(value)
                    output_str += f", {key} = {value:.3f}%"
                print(output_str)
                weight.append(len(dataset))
        if torch.distributed.get_rank() == 0:
            print(f"Task {task}:")
        for key, value in result_dict_all.items():
            idx = np.argmax(value)
            if torch.distributed.get_rank() == 0:
                print(
                    f"    Metric {key}: max({'/'.join(result_dict_all.keys())}) = "
                    f"{'/'.join(map(lambda x: str(x[idx]), result_dict_all.values()))}"
                    f" | median = {np.median(value)}, average = {(np.array(value) * np.array(weight) / np.sum(weight)).sum()}"
                )

    dist.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"done :-), total time: {time.time() - start}")
