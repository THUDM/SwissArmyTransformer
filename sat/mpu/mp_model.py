from .initialize import get_node_rank, get_node_world_size, destroy_model_parallel, initialize_model_parallel
from sat import AutoModel
import torch
from .operation import mp_split_model_rank0, mp_split_model_receive

def get_mp_split_model(model_name, new_model_parallel_size, args, **kwargs):
    model, model_args = AutoModel.from_pretrained(model_name, args=args, overwrite_args={'model_parallel_size': new_model_parallel_size}, build_only=True, **kwargs)
    local_rank = get_node_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size % new_model_parallel_size == 0, "world size should be a multiplier of new_model_parallel_size."
    destroy_model_parallel()
    initialize_model_parallel(1)
    if local_rank == 0:
        args.use_gpu_initialization = False
        args.device = 'cpu'
        model_full, args_ = AutoModel.from_pretrained(model_name, args=args, **kwargs)
    torch.distributed.barrier()
    destroy_model_parallel()
    initialize_model_parallel(new_model_parallel_size)
    if local_rank == 0:
        mp_split_model_rank0(model, model_full)
        del model_full
    else:
        mp_split_model_receive(model)
    return model, model_args