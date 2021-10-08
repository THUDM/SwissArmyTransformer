# -*- encoding: utf-8 -*-
'''
@File    :   cuda2d_sampling.py
@Time    :   2021/10/09 00:46:04
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from .sampling_strategies import BaseStrategy

def filling_sequence(
        model, 
        seq0,
        seq1, 
        warmup_steps=3,
        block_hw=(4, 4),
        strategy=BaseStrategy(topk=10)
        ):
    '''
        seq: [PAD]... [ROI1] text ... [BOI1] {layout[0]} 1024 {layout[1]} [EOI1]
            4095 {layout[2]} final_token
    '''
    assert hasattr(model, 'layout')
    layout = model.layout
    assert len(seq0.shape) == 2 and len(seq1.shape) == 2 \
        and seq0.shape[0] == seq1.shape[0]
    assert len(layout) == 3
    assert seq1.shape[1] == layout[-1] - layout[-2]
    assert (seq1 >= 0).all() and (seq0 >= 0).all()
    device = seq0.device

    # concat and pad sequences
    batch_size = seq0.shape[0]
    n_pad = layout[1] + 1 - len(seq0) # +1 for [EOI1]
    assert n_pad > 0, "You should truncate long input before filling."
    seq = torch.cat((
        torch.tensor([0]*n_pad, device=device, dtype=seq0.dtype)
            .unsqueeze(0).expand(batch_size, n_pad),
        seq0, seq1), dim=1) # [b, layout[-1]+1]
    assert seq.shape[1] == layout[-1] + 1

    # build initial tokens, attention_mask, and position_ids
    tokens = seq[:, :-1].clone()
    attention_mask = torch.ones(layout[1], layout[1]).tril().to(device)
    attention_mask[n_pad:, :n_pad] = 0
    position_ids = torch.cat((
        torch.zeros(n_pad, dtype=torch.long),
        torch.arange(0, layout[1] - n_pad), 
        torch.arange(0, layout[2]-layout[1]))).to(device)

    # iterative refining
    ll, rr = block_hw
    num_steps = warmup_steps + ll + rr - 2
    for step_cnt in range(num_steps):
        logits, *_dump = model(tokens, position_ids, attention_mask)

        # warmup 
        real_topk = 10
        warmup_steps = 3
        iterative_step= warmup_steps + 6
        if step_cnt <= warmup_steps:
            real_temp = 0.1
        elif step_cnt == warmup_steps + 1:
            real_temp = 0.55
        elif step_cnt > warmup_steps + 1:
            real_temp = 0.45
        # if  5 < step_cnt:
        #     real_topk = 200
        # sampling
        for invalid_slice in invalid_slices: # forbide to generate other tokens
            logits[..., invalid_slice] = -float('Inf')
        assert args.top_k > 0
        
        # probs0 = F.softmax(logits/real_temp, dim=-1)
        topraw = (torch.topk(logits, 5, dim=-1)[0]).softmax(dim=-1)
        ent = -(topraw * topraw.log()).sum(dim=-1)
        # topsum = topraw.sum(dim=-1)
        if step_cnt > warmup_steps:
            # import pdb;pdb.set_trace()
            real_temp2 = torch.tensor([[[real_temp]]], device=logits.device).expand(*logits.shape[:2], 1) * (ent > 1.3).unsqueeze(-1) + 0.6
            # import pdb;pdb.set_trace()
        else:
            real_temp2 = real_temp
        # import pdb;pdb.set_trace()
        probs = F.softmax(logits/real_temp2, dim=-1)
        tk_value, tk_idx = torch.topk(probs, real_topk, dim=-1)
        prev = torch.multinomial(probs.view(-1, logits.shape[-1]), num_samples=1).view(*logits.shape[:2], 1)
        edge_idx = tk_idx[:, :, -1:]
        edge_value = tk_value[:, :, -1:]
        edge_mask = probs.gather(dim=-1, index=prev) < edge_value
        prev[edge_mask] = edge_idx[edge_mask]
        prev.squeeze_(-1)
        # tk_probs = (tk_value / real_temp).softmax(dim=-1).view(-1, tk_value.shape[-1])
        # prev = torch.multinomial(tk_probs, num_samples=1).view(*(tk_value.shape[:2]),1)
        # prev = torch.gather(tk_idx, dim=-1, index=prev).squeeze(-1)
        # update unfixed
        choice = 1
        if choice == 0 and warmup_steps < step_cnt:
            mprob = probs.max(dim=-1)[0].view(*(tk_value.shape[:2]))
            # import pdb;pdb.set_trace()
            dprob = mprob[:, 1:] < mprob[:, args.layout[1]:].topk(300, dim=-1, largest=False)[0][:,-1].unsqueeze(-1).expand_as(mprob[:, 1:])

            new_fixed = unfixed.clone()
            moved_new_fixed = new_fixed[:, 2:]
            moved_new_fixed &= dprob
            moved_new_fixed[:, 1:] &= dprob[:, :-1].logical_not() | unfixed[:, 2:-1].logical_not()
            moved_new_fixed[:, 2:] &= dprob[:, :-2].logical_not() | unfixed[:, 2:-2].logical_not()
            # moved_new_fixed[:, 3:] &= dprob[:, :-3].logical_not() | unfixed[:, 2:-3].logical_not()
            moved_new_fixed[:, 64:] &= dprob[:, :-64].logical_not() | unfixed[:, 2:-64].logical_not()
            moved_new_fixed[:, 65:] &= dprob[:, :-65].logical_not() | unfixed[:, 2:-65].logical_not()
            # moved_new_fixed[:, 66:] &= dprob[:, :-66].logical_not() | unfixed[:, 2:-66].logical_not()
        elif choice == 1 and warmup_steps < step_cnt:
            new_fixed = unfixed & False
            ll, rr = 4, 4
            for x in range(min(ll, step_cnt - warmup_steps)):
                y = step_cnt - warmup_steps - x - 1
                if y < rr:
                    print(x,y)
                    new_fixed[..., -4096:].view(batch_size, 64//ll, ll, 64//rr, rr)[:, :, x, :, y] = True
            new_fixed &= unfixed
        else:
            new_fixed = unfixed & False # TODO
        new_fixed[:, -1] = True

        # with open(f'bed{step_cnt}.txt', 'w') as fout:
        #     for i, prob in enumerate(topraw[0, -4096:]):
        #         s = ' '.join([str(x) for x in prob.tolist()])
        #         fout.write(f'{i} {s}\n')

        unfixed &= new_fixed.logical_not()
        # update seq and tokens
        seq[new_fixed] = prev[new_fixed[:, 1:]]
        tokens = seq[:, :-1].clone()
        tokens[:,1:][unfixed[:, 1:-1]] = prev[:, :-1][unfixed[:, 1:-1]]

        if step_cnt == iterative_step: 
            seq[:, :-1][unfixed[:, :-1]] = tokens[unfixed[:, :-1]] # if reach iterative_step
            n_unfixed = unfixed.sum(dim=-1).tolist()
            print(f'Exit with {n_unfixed} unfixed tokens.')
            break
        if args.debug:
            from torchvision.utils import save_image
            seqt = seq.clone()
            seqt[:, :-1][unfixed[:, :-1]] = tokens[unfixed[:, :-1]] # if reach iterative_step
            imgs.extend([tokenizer.img_tokenizer.DecodeIds(s[-4096:]) for s in seqt])
    if args.debug:
        imgs = torch.cat(imgs, dim=0)
        save_image(imgs, f'steps{device}.jpg', normalize=True)
    model.module.transformer.max_memory_length = args.max_memory_length

    return seq