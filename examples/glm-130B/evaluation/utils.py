import torch
from SwissArmyTransformer import mpu, get_tokenizer
from SwissArmyTransformer.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy

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


def cond_log_prob(batch, model, args):
    # Get the batch.
    tokens, targets, position_ids, attention_mask = process_data(batch)

    # Tell the model what our actual batch size will be
    # args.micro_batch_size, args.seq_length = tokens.shape[:2]

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

def update_mems(hiddens, mems, max_memory_length):
    '''
        hiddens: list (num_layers) of [batch, query_length, 2d]
        mems: None or [num_layers, batch, memory_length, 2d]
    '''
    if hiddens is None:
        return None
    hiddens = torch.stack(hiddens)
    memory_length = mems.shape[2] if mems is not None else 0
    query_length = hiddens.shape[2]
    new_memory_length = min(max_memory_length, memory_length + query_length)
    with torch.no_grad():
        if new_memory_length <= query_length:
            return hiddens[:, :, -new_memory_length:]
        else:
            if mems.shape[1] < hiddens.shape[1]:
                mems = mems.expand(-1, hiddens.shape[1], -1, -1)
            return torch.cat(
                (mems[:, :, -new_memory_length+query_length:], hiddens),
                dim=2
            )

def filling_sequence(
        model, 
        context_tokens,
        context_length,
        attention_mask,
        position_ids,
        max_length,
        batch_size,
        strategy=BaseStrategy(),
        max_memory_length=100000,
        mems=None,
        **kw_args
        ):
    assert context_length > 0

    tokens = context_tokens[..., :context_length]

    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter'' 
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    # step-by-step generation

    while counter < max_length - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.

        if context_tokens[0, counter + 1] > 0: # provided
            tokens = torch.cat(
                (
                    tokens, 
                    context_tokens[0, counter+1: counter+2].expand(tokens.shape[0], 1)
                ), dim=1
            )
            counter += 1
            continue

        logits, *output_per_layers = model(
            tokens[:, index:], 
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=mems,
            log_attention_weights=None,
            **kw_args
        )
        mem_kv = [o['mem_kv'] for o in output_per_layers]
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        counter += 1
        index = counter
        # sampling
        logits = logits[:, -1].expand(batch_size, -1) # [batch size, vocab size]
        tokens = tokens.expand(batch_size, -1)
        tokens, mems = strategy.forward(logits, tokens, mems)

        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)[0]


def generate_text(model, batch, strategy, batch_size, max_length=1024):
    output = filling_sequence(
        model,
        batch["tokens"].to(device=torch.cuda.current_device()).long(),
        batch["context_length"].to(device=torch.cuda.current_device()).long(),
        batch["attention_mask"].to(device=torch.cuda.current_device()).bool().unsqueeze(1),
        batch["position_ids"].to(device=torch.cuda.current_device()).long(),
        max_length=max_length,
        batch_size=batch_size,
        strategy=strategy
    )

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