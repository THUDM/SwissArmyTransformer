import torch
import torch.nn.functional as F
import mpu
from .autoregressive_sampling import update_mems
from .sampling_strategies.beam_search_strategy import BeamSearchScorer


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits


def filling_sequence_glm(model, tokenizer, mask_position, strategy, args, mems=None, end_tokens=None, device='cuda'):
    tokens = torch.full((1, 1), tokenizer.get_command('sop').Id, device=device, dtype=torch.long)
    counter = 0
    if mems is None:
        mems = []
    # if end_tokens is None:
    #     end_tokens = [tokenizer.get_command('eos').Id]
    while counter < args.out_seq_length:
        last_beam_num = tokens.size(0)
        if args.block_lm:
            if args.no_block_position:
                position_ids = torch.full((last_beam_num, 1), mask_position + counter, device=device, dtype=torch.long)
            else:
                position_ids = torch.ones(last_beam_num, 2, 1, device=device, dtype=torch.long)
                position_ids[:, 0] = mask_position
                position_ids[:, 1] = counter + 1
            attention_mask = torch.ones(1, 1, device=device, dtype=torch.float)
        else:
            position_ids = torch.full((last_beam_num, 1), mask_position + counter - 1, device=device, dtype=torch.long)
            attention_mask = torch.ones(last_beam_num, 1, 1, args.mem_length + 1, device=device, dtype=torch.float)
        if args.fp16:
            attention_mask = attention_mask.half()
        last_token = tokens[:, -1:]
        logits, *mem_kvs = model(last_token, position_ids, attention_mask, *mems)
        mems = update_mems(mem_kvs, mems, max_memory_length=1000000)
        next_token_logits = logits[:, -1]
        tokens, mems = strategy.forward(next_token_logits, tokens, mems)
        if strategy.is_done:
            break
        # else:
        #     next_token_logits /= args.temperature
        #     next_token_logits = top_k_logits(next_token_logits, top_k=args.top_k, top_p=args.top_p)
        #     log_probs = F.softmax(next_token_logits, dim=-1)
        #     prev = torch.multinomial(log_probs, num_samples=1)[0]
        #     is_end = prev.item() in end_tokens
        #     if is_end:
        #         break
        #     prev = prev.view(1, 1)
        #     tokens = prev if tokens is None else torch.cat((tokens, prev), dim=1)
        counter += 1
    strategy.finalize(tokens, mems)
    return tokens, mems
