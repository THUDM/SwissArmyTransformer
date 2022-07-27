import torch
import torch.distributed as dist
from itertools import chain
from typing import List

from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence


def print_rank_0(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def process_data(batch):
    return (
        batch["tokens"].to(device=torch.cuda.current_device()).long(),
        batch["targets"].to(device=torch.cuda.current_device()).long(),
        batch["position_ids"].to(device=torch.cuda.current_device()).long(),
        batch["attention_mask"].to(device=torch.cuda.current_device()).bool().unsqueeze(1),
    )


def build_data_loader(dataset, micro_batch_size, num_workers, drop_last):
    # Sampler.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    return data_loader


def gather_result(prediction, total_length):
    torch.cuda.empty_cache()
    world_size = mpu.get_data_parallel_world_size()
    prediction_gathered = [None for _ in range(world_size)]
    dist.all_gather_object(prediction_gathered, prediction, group=mpu.get_data_parallel_group())
    return list(chain(*zip(*prediction_gathered)))[:total_length]


def cond_log_prob(model, batch):
    # Get the batch.
    tokens, targets, position_ids, attention_mask = process_data(batch)

    # Forward pass through the model.
    logits, *output_per_layers = model(tokens, position_ids, attention_mask, log_attention_weights=None)  # TODO memlen

    # output: [b, sq, vocab]
    output = torch.nn.functional.log_softmax(logits, dim=-1)
    batch_ids = torch.arange(tokens.size(0), dtype=tokens.dtype, device=tokens.device).unsqueeze(1)

    choice_logits = []

    # Single token
    if batch["is_single_token"].any():
        target_ids = batch["choice_target_ids"][0]
        logits = output[batch_ids, target_ids, batch["choices"]]
        choice_logits = logits.squeeze(0).tolist()
    # Multi token
    else:
        for target_ids in batch["choice_target_ids"]:
            logits = output[batch_ids, target_ids, targets[batch_ids, target_ids]]
            choice_logits.append(logits.squeeze(0).sum(dim=-1).tolist())

    return choice_logits


def generate_text(model, batch, strategy, max_length) -> List[List[int]]:
    seq = torch.squeeze(batch["tokens"].to(device=torch.cuda.current_device()).long())[:max_length]
    context_length = batch["context_length"].to(device=torch.cuda.current_device()).long()
    seq[context_length:] = -1

    def get_masks_and_position_ids(seq):
        tokens = seq.unsqueeze(0)
        attention_mask = batch["attention_mask"].to(device=torch.cuda.current_device()).bool().unsqueeze(1)
        position_ids = batch["position_ids"].to(device=torch.cuda.current_device()).long()
        return tokens, attention_mask, position_ids

    output = filling_sequence(
        model,
        seq,
        get_masks_and_position_ids=get_masks_and_position_ids,
        batch_size=strategy.num_beams if hasattr(strategy, "num_beams") else 1,
        strategy=strategy,
        attention_dtype=torch.bool
    )[0]

    if isinstance(output, torch.Tensor):  # different strategies
        output = list(output)

    output_targets = []

    for line in output:
        line = line.tolist()
        unfinished = line.index(-1) if -1 in line else len(line)
        if line[unfinished - 1] in strategy.end_tokens:
            unfinished -= 1
        # bog = line.index(tokenizer.get_command("sop"))
        line = line[context_length:unfinished]
        output_targets.append(line)

    return output_targets
