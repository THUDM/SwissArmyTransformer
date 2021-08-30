from .sampling import *
import math
import sys
from copy import deepcopy

def make_local_mask(sparse_config):
    layout = sparse_config.layout
    k1, k2 = sparse_config.kernel_size, sparse_config.kernel_size2
    k1h = k1*2-1
    h = w = int(math.sqrt(layout[2] - layout[1]) + 1e-3)
    m = torch.zeros(layout[-1]+1, layout[-1], dtype=torch.bool, device='cuda')
    for i in range(layout[1]):
        m[i, :i] = True
    m[layout[1]:, :layout[0]] = True
    for i in tqdm(range(layout[1], layout[2])):
        # m[i, layout[1]:i] = True
        x = (i - layout[1]) // w
        y = (i - layout[1]) % w
        lx = max(0, x - k1h // 2)
        ly = max(0, y - k1 // 2)
        rx = min(h-1, x + k1h // 2)
        ry = min(w-1, y + k1 // 2)
        m[i, layout[1]:layout[2]].view(h, w)[lx:x, ly:ry+1] = True
        m[i, layout[1]:layout[2]].view(h, w)[x, ly:y+1] = True
        m[i, i] = False
    for i in tqdm(range(layout[2], layout[3])):
        x = (i - layout[2]) // (2*w)
        y = (i - layout[2]) % (2*w)
        lx = max(0, x - k1h // 2)
        ly = max(0, y - k1 // 2)
        rx = min(2*h-1, x + k1h // 2)
        ry = min(2*w-1, y + k1 // 2)
        m[i, layout[2]:layout[3]].view(h*2, w*2)[lx:x, ly:ry+1] = True
        m[i, layout[2]:layout[3]].view(h*2, w*2)[x, ly:y+1] = True
        x = x // 2
        y = y // 2
        lx = max(0, x - k2 // 2)
        ly = max(0, y - k2 // 2)
        rx = min(h-1, x + k2 // 2)
        ry = min(w-1, y + k2 // 2)
        m[i, layout[1]:layout[2]].view(h, w)[lx:rx+1, ly:ry+1] = True
        m[i, i] = False
    return m.unsqueeze(-1)

def update_mems_local(hiddens, mems, start, end, mem_bag, mask):
    if isinstance(hiddens, list):
        hiddens = torch.stack(hiddens)
        mem_bag[:, :, start:end] = hiddens.to('cuda')
     # first level
    del mems
    mems = mem_bag.masked_select(mask[end]).view(*mem_bag.shape[:2], -1, mem_bag.shape[3]).to(hiddens.device)
    return mems


def filling_sequence_local(
        model, 
        seq, 
        args, 
        mems=None, 
        invalid_slices=[], 
        **kwargs):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -N (N beams), -1]
        context_length: first non(-1)s
    '''
    loss_sum, loss_n = 0, 0.0001
    tokenizer = get_tokenizer()
    device = seq.device
    assert len(seq.shape) == 1
    assert args.sparse_config.sparse_type == 'standard'
    sparse_config = deepcopy(args.sparse_config)
    # with open('tmp_save.bin', 'rb') as fout:
    #     seq = torch.load(fout)[1]
    #     seq = torch.cat((seq, torch.tensor([-1], device=seq.device)))
    # import pdb; pdb.set_trace()
    # sparse_config.layout[0] = seq.tolist().index(-1)
    sparse_config.layout[0] = seq.tolist().index(tokenizer['[BASE]'])
    n_pad = sparse_config.layout[1] - sparse_config.layout[0]
    assert n_pad > 0 # TODO long trunc
    seq = torch.cat((seq[:sparse_config.layout[0]], torch.tensor([tokenizer['[POS8]']]* n_pad, device=seq.device, dtype=seq.dtype), seq[sparse_config.layout[0]:]))
    out_seq_length = len(seq)
    # building the initial tokens, attention_mask, and position_ids
    context_length = sparse_config.layout[1] + 1

    tokens, attention_mask, position_ids = get_batch(seq[:context_length], device, args)
    tokens = tokens.expand(-min(seq), *tokens.shape[1:])

    counter = context_length - 1 # == len(tokens) - 1
    index = 0 # len(mems)
    if mems is None:
        mems = []
    mem_bag = torch.zeros(args.num_layers, tokens.shape[0], out_seq_length-1, 2*args.hidden_size, device='cuda')
    local_mask = make_local_mask(sparse_config)
    
    while counter < (out_seq_length - 1):
        if counter % 100 == 0:
            print(counter, loss_sum / loss_n, file=sys.stderr)
        # Now, we want to generate seq[counter + 1]
        # token[:, index: counter+1] are just added.
        # import pdb;pdb.set_trace()

        if index == 0: # first
            logits, *qkv = model(tokens, position_ids, attention_mask, *mems)
            mems = update_mems_local(qkv, mems, index, counter+1, mem_bag, local_mask)
            index = counter
        else:
            assert tokens.shape[1] == counter + 1 
            position_ids = torch.arange(index, counter + 1, dtype=torch.long, device=tokens.device).unsqueeze(0)
            logits, *qkv = model(tokens[:, index: ], 
                position_ids,
                0, # rebuild in transformers (sep version)
                *mems
                )
            mems = update_mems_local(qkv, mems, index, counter+1, mem_bag, local_mask)
            index = counter
        counter += 1
        index += 1

        if seq[counter] >= 0: # provided
            tokens = torch.cat((tokens, seq[counter: counter+1].expand(tokens.shape[0], 1)), dim=1)
            loss_n +=1
            loss_this = F.log_softmax(logits, dim=-1)[:, -1, seq[counter]].mean()
            print(counter-64, loss_this.item())
            loss_sum -= loss_this
            continue

        logits = logits[:, -1] # [batch size, vocab size]
        temp = args.temperature
        logits /= temp
        for invalid_slice in invalid_slices: # forbide to generate other tokens
            logits[..., invalid_slice] = -float('Inf')
        logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
        log_probs = F.softmax(logits, dim=-1)

        # expand beams
        prev = torch.multinomial(log_probs, num_samples=1)
        tokens = torch.cat((tokens, prev.view(tokens.shape[0], 1)), dim=1)

    output_tokens_list = tokens.view(tokens.shape[0], -1).contiguous()
    output_tokens_list = torch.cat(
        (
            output_tokens_list[:, :sparse_config.layout[0]],
            output_tokens_list[:, sparse_config.layout[1]+1:sparse_config.layout[2]+1],
            torch.tensor([[tokenizer['[EOI1]']]], dtype=tokens.dtype, device=tokens.device).expand(output_tokens_list.shape[0], 1),
            output_tokens_list[:, sparse_config.layout[2]+1:]
        ), dim=1)
    return output_tokens_list