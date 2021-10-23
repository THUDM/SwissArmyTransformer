import torch
import torch.nn.functional as F
import mpu
from .autoregressive_sampling import update_mems
from .sampling_strategies.beam_search_strategy import BeamSearchScorer


def filling_sequence_glm(model, tokenizer, mask_position, strategy, args, mems=None, end_tokens=None, device='cuda'):
    tokens = torch.full((1, 1), tokenizer.get_command('sop').Id, device=device, dtype=torch.long)
    counter = 0
    if mems is None:
        mems = []
    # if end_tokens is None:
    #     end_tokens = [tokenizer.get_command('eos').Id]
    while counter < args.out_seq_length - 1:
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
    tokens, mems = strategy.finalize(tokens, mems)
    return tokens, mems
