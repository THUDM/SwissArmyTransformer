import torch
from SwissArmyTransformer import mpu, get_tokenizer
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence 

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


def cond_log_prob(batch, model):
    # Get the batch.
    tokens, targets, position_ids, attention_mask = process_data(batch)

    attention_mask = attention_mask.type_as(next(model.parameters()))

    # Forward pass through the model.
    logits, *output_per_layers = model(
            tokens, 
            position_ids,
            attention_mask, # TODO memlen
            log_attention_weights=None
        )

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

def generate_text(model, batch, strategy, batch_size, max_length=1024):
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
        batch_size=batch_size,
        strategy=strategy
    )[0]

    tokenizer = get_tokenizer(tokenizer_type='icetk-glm-130B')
    end_tokens = [tokenizer.get_command('eop'), tokenizer.get_command('eos')]

    if isinstance(output, torch.Tensor): # different strategies
        output = list(output)

    # clip -1s and fill back generated things into seq
    output = output[0].tolist()
    try:
        unfinished = output.index(-1)
    except ValueError:
        unfinished = len(output)
    if output[unfinished - 1] in end_tokens:
        unfinished -= 1
    bog = output.index(tokenizer.get_command('sop'))
    output = output[bog + 1:unfinished]

    return output