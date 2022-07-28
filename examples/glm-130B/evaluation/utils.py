import torch
import torch.distributed as dist
from itertools import chain
from typing import List

from SwissArmyTransformer import mpu


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
