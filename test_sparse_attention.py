import math
import random
from tqdm import tqdm

import torch
import numpy as np
from mpu.sparse_transformer import standard_attention, sparse_attention_1d, sparse_attention_2d, sparse_attention_2dfull

def test_sparse_attention_1d():       
    s, w, times = 4096 + 128, 128, 2
    num_pivot = 768
    b = 2
    g = s // w

    q, k, v = raw = torch.rand(3, b, 16, s, 64, dtype=torch.float, device='cuda', requires_grad=True)
    q1, k1, v1 = raw1 = torch.tensor(raw.cpu().detach().numpy(), dtype=torch.float, device='cuda', requires_grad=True)
    txt_indices = [torch.arange(0, 128, dtype=torch.long, device='cuda'), torch.arange(0, 22, dtype=torch.long, device='cuda')]
    img_indices = [torch.arange(128, s, dtype=torch.long, device='cuda'), torch.arange(22, s, dtype=torch.long, device='cuda')]

    pivot_idx = torch.stack([
        torch.cat((
            text_idx,
            img_indices[i][
                torch.tensor(random.sample(range(len(img_indices[i]) - times*w),  k=num_pivot - len(text_idx)), dtype=torch.long, device=text_idx.device)
            ]
        ), dim=0)
        for i, text_idx in enumerate(txt_indices)
    ]) # -times * w to verify inference

    tmp = torch.ones((g-times+1, w , w), device='cuda', dtype=torch.long)
    tmp = torch.tril(1 - torch.block_diag(*tmp))
    rmask = torch.nn.functional.pad(tmp, (0, (times-1)*w, (times-1)*w, 0)) # pad (left, right, top, bottom)

    pivot_attention_mask = rmask.expand(b, s, s).gather(dim=-1, index=pivot_idx.unsqueeze(1).expand(b, s, num_pivot))

    real_mask = torch.ones((b, s, s), device='cuda', dtype=torch.long) - rmask
    for i in range(b):
        real_mask[i][:, pivot_idx[i]] = 1
        real_mask[i].tril_()

    # test inference

    # q_part = q[..., -1:, :]
    # r0 = standard_attention(q, k, v, real_mask)
    # r0 = r0[..., -1:, :]
    # pw_idx = torch.cat((pivot_idx, torch.arange(s-times*w, s, device='cuda', dtype=torch.long).expand(b, -1)), dim=-1)

    # r1 = sparse_attention_inference(q_part, k, v, pw_idx)
    # print(( (r1-r0).abs() / (r1.abs()+r0.abs())).max())

    import time

    r0 = standard_attention(q1, k1, v1, real_mask)
    torch.cuda.synchronize()
    t0 = time.time()
    r1 = standard_attention(q1, k1, v1, real_mask)
    torch.cuda.synchronize()
    t1 = time.time()
    r2 = sparse_attention(q, k, v, pivot_idx, pivot_attention_mask, w, times)
    torch.cuda.synchronize()
    t2 = time.time()
    print('times: standard ', t1-t0, ' sparse ', t2-t1)

    print(( (r1-r2).abs() / (r1.abs()+r2.abs())).max())

    raw.retain_grad()
    l2 = r2.mean()
    l1 = r1.mean()
    l2.backward()
    l1.backward()

    g1 = raw1.grad
    g2 = raw.grad
    print( (g1-g2).abs().max())

    # import pdb; pdb.set_trace()

def test_sparse_attention_2d():
    dtype = torch.float
    device = 'cuda'
    b, n_head, hn = 2, 16, 1024
    h = w = 32
    layout = [10, 64, 64+h*w, 64+h*w*5]
    k1 = 9
    k2 = 7
    k1h = k1*2-1

    qkv = torch.rand(3, b, layout[-1], hn, dtype=dtype, device=device)
    qkv2 = qkv.clone()
    qkv.requires_grad_()
    qkv2.requires_grad_()
    mask = torch.zeros(b, layout[-1], layout[-1], dtype=dtype, device=device)
    
    m = mask[0]
    for i in range(layout[1]):
        m[i, :i+1] = 1
    m[layout[1]:, :layout[0]] = 1
    for i in tqdm(range(layout[1], layout[2])):
        m[i, layout[1]:i+1] = 1
    # for i in tqdm(range(layout[1], layout[2])):
    #     x = (i - layout[1]) // w
    #     y = (i - layout[1]) % w
    #     lx = max(0, x - k1 // 2)
    #     ly = max(0, y - k1 // 2)
    #     rx = min(h-1, x + k1 // 2)
    #     ry = min(w-1, y + k1 // 2)
    #     m[i, layout[1]:layout[2]].view(h, w)[lx:x, ly:ry+1] = 1
    #     m[i, layout[1]:layout[2]].view(h, w)[x, ly:y+1] = 1
    for i in tqdm(range(layout[2], layout[3])):
        x = (i - layout[2]) // (2*w)
        y = (i - layout[2]) % (2*w)
        lx = max(0, x - k1h // 2)
        ly = max(0, y - k1 // 2)
        rx = min(2*h-1, x + k1h // 2)
        ry = min(2*w-1, y + k1 // 2)
        m[i, layout[2]:layout[3]].view(h*2, w*2)[lx:x, ly:ry+1] = 1
        m[i, layout[2]:layout[3]].view(h*2, w*2)[x, ly:y+1] = 1

        x = x // 2
        y = y // 2
        lx = max(0, x - k2 // 2)
        ly = max(0, y - k2 // 2)
        rx = min(h-1, x + k2 // 2)
        ry = min(w-1, y + k2 // 2)
        m[i, layout[1]:layout[2]].view(h, w)[lx:rx+1, ly:ry+1] = 1
    
    mask[1:] = mask[0]
    # mask[1][layout[1]:, layout[0]-1] = 0

    print('finish making mask...')

    import time
    torch.cuda.synchronize()
    t0 = time.time()
    qkv_tmp = qkv.view(3, b, layout[-1], n_head, hn//n_head).permute(0, 1, 3, 2, 4).contiguous()
    r1 = standard_attention(*qkv_tmp, mask.unsqueeze(1)).transpose(1, 2).reshape(b, layout[3], hn)
    
    torch.cuda.synchronize()
    t1 = time.time()
    r2 = sparse_attention_2dfull(*qkv2, n_head, layout, mask[...,:layout[0]].unsqueeze(1), kernel_size=k1, kernel_size2=k2)
    torch.cuda.synchronize()
    t2 = time.time()
    print('times: standard ', t1-t0, ' sparse ', t2-t1)
    print(( (r1[:,:layout[0]]-r2[:,:layout[0]]).abs() / (r1[:,:layout[0]].abs()+r2[:,:layout[0]].abs())).max())
    print(( (r1[:,layout[1]:]-r2[:,layout[1]:]).abs() / (r1[:,layout[1]:].abs()+r2[:,layout[1]:].abs())).max())
    qkv.retain_grad()
    l2 = r2[:,layout[1]:].sum()
    l1 = r1[:,layout[1]:].sum()

    l2.backward()
    l1.backward()

    g1 = qkv.grad
    g2 = qkv2.grad
    print( (g1-g2).abs().max())
    print( ((g1-g2).abs() / (g1.abs()+g2.abs()+1e-5)).max())

    import pdb;pdb.set_trace()
    

def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    seed_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    test_sparse_attention_2d()
    