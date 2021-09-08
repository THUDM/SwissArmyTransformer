
def sparse_attention_1d(q, k, v, pivot_idx, pivot_attention_mask, query_window=128, key_window_times=6, attention_dropout=None):
    ''' Sparse Attention
    Args:
        q, k, v: inputs, [b, num_heads, s, hn], k is padded to n * query_window
        pivot_idx: [b, num_pivots]
        pivot_attention_mask: [b, s, num_pivots]
        query_window: .
        key_window_times: key_window = query_window * key_window_times
    '''

    b, n_head, s, hn = q.shape
    b, n_piv = pivot_idx.shape
    w = query_window

    pivot_idx_dummy = pivot_idx.view(b, 1, n_piv, 1).expand(b, n_head, n_piv, hn)
    # =====================   Pivot Attention   ======================== #
    pivot_k, pivot_v = torch.gather(k, 2, pivot_idx_dummy), torch.gather(v, 2, pivot_idx_dummy)
    attention_scores = torch.matmul(q, pivot_k.transpose(-1, -2))
    pivot_attention_mask = pivot_attention_mask.unsqueeze(1)

    attention_scores_pivot = torch.mul(attention_scores, pivot_attention_mask / math.sqrt(hn)) - 10000.0 * (1.0 - pivot_attention_mask)

    attention_scores_pivot = attention_scores_pivot + math.log(s // n_piv)
    # =====================   Window Attention   ======================= #
    window_k = _chunk(k, query_window, key_window_times)
    window_v = _chunk(v, query_window, key_window_times)
    # window_k [b, n_head, s // w up int, w*times, hn]

    if s % w == 0: # training # TODO args check
        assert k.shape[2] == s
        assert window_k.shape[2] == s // w
        window_q = q.view(b, n_head, s // w, w, hn)        
        attention_scores = torch.matmul(window_q, window_k.transpose(-1, -2))
        window_attention_mask = torch.ones((w, w * key_window_times), dtype=attention_scores.dtype, device=q.device).tril_(diagonal=w * (key_window_times - 1))
        attention_scores_window = torch.mul(attention_scores, window_attention_mask / math.sqrt(hn)) - 10000.0 * (1.0 - window_attention_mask)
        for t in range(1, key_window_times):
            attention_scores_window[:, :, t - 1, :, :w * key_window_times - w * t] -= 10000.0
    else: 
        raise ValueError('The seq_len must be exactly divided by window_size.')
    # =====================   Joint Softmax   ======================= #
    attention_scores_window = attention_scores_window.view(b, n_head, s, w * key_window_times)
    attention_scores = torch.cat((attention_scores_pivot, attention_scores_window), dim=-1)
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs[..., :-w * key_window_times], pivot_v) + torch.einsum('bcgwk,bcgkh->bcgwh', attention_probs[..., -w * key_window_times:].view(b, n_head, s // w, w, w * key_window_times), window_v).view(b, n_head, s, hn)

    return context_layer


def transpose_and_split(x, layout, n_head):
    x = x.transpose(1, 2)
    x = x.reshape(x.shape[0]*n_head, x.shape[1] // n_head, x.shape[2])
    x_text = x[..., :layout[0]]
    x0 = x[...,layout[1]:layout[2]].view(x.shape[0], x.shape[1], sqrt(layout[2] - layout[1]), sqrt(layout[2] - layout[1])).contiguous()
    x1 = x[...,layout[2]:layout[3]].view(x.shape[0], x.shape[1], sqrt(layout[3] - layout[2]), sqrt(layout[3] - layout[2])).contiguous()
    return x, x_text, x0, x1

def sparse_attention_2d(q, k, v, n_head, layout, attention_mask_text2d, kernel_size=9, kernel_size2=7, attention_dropout=None, **kwargs):
    '''
    q, k, v: [batch_size, 64+1024+4096, hidden_size]
    n_head: int
    layout: [endoftext/startofpad, startof0, startof1, endofall]
    attention_mask_text2d: [batch_size, sq_len, endoftext]
    '''
    from .local_attention_function import f_similar, f_weighting
    b, sq_len, hn = q.shape
    alpha = sqrt((layout[3] - layout[2]) // (layout[2] - layout[1]))

    q = q / math.sqrt(hn // n_head) # normalization

    q_all, q_text, q0, q1 = transpose_and_split(q, layout, n_head) # 0, 1 [batch * n_head, hn_per_head, h, w] text [batch * n_head, hn_per_head, endoftext]
    k_all, k_text, k0, k1 = transpose_and_split(k, layout, n_head)
    v_all, v_text, v0, v1 = transpose_and_split(v, layout, n_head)
    # import pdb; pdb.set_trace()
    # all to text
    scores_all_to_text = torch.einsum('bhi,bhj->bij', q_all, k_text).view(b, n_head, layout[3], layout[0]) * attention_mask_text2d - 10000.0 * (1.0 - attention_mask_text2d)
    scores_all_to_text = scores_all_to_text.view(b*n_head, layout[3], layout[0])
    # 0 to 0
    scores_0_to_0 = f_similar(q0, k0, kernel_size*2-1, kernel_size, True)
    # 1 to 1
    scores_1_to_1 = f_similar(q1, k1, kernel_size*2-1, kernel_size, True)    
    # 1 to 0
    scores_1_to_0 = f_similar(q1, k0, kernel_size2, kernel_size2, False) # [batch * n_head, 2h, 2w, kernel_size2**2]
    # softmax
    # if 'offset_bias' in kwargs:
    #     p1, p2 = kernel_size**2//2 + 1, kernel_size2**2
    #     offset_bias = kwargs['offset_bias'].expand(b, n_head, p1+p2).view(b*n_head, 1, p1+p2)
    #     scores_0_to_0 = scores_0_to_0 * offset_bias[...,:p1]
    #     scores_1_to_1 = scores_1_to_1 * offset_bias[...,:p1]
    #     scores_1_to_0 = scores_1_to_0 * offset_bias[...,-p2:]

    scores_0 = torch.cat(
        (scores_all_to_text[:, layout[1]:layout[2]], 
        scores_0_to_0.view(b * n_head, layout[2]-layout[1], scores_0_to_0.shape[-1])), 
        dim=-1)
    scores_1 = torch.cat(
        (scores_all_to_text[:, layout[2]:layout[3]],
         scores_1_to_0.view(scores_1_to_0.shape[0], -1, scores_1_to_0.shape[3]),
         scores_1_to_1.view(scores_1_to_1.shape[0], -1, scores_1_to_1.shape[3])),
         dim=-1)
    probs_text = F.softmax(scores_all_to_text[:, :layout[0]], dim=-1) # [batch * n_head, seq_text, seq_text]
    probs_0 = F.softmax(scores_0, dim=-1) # 
    probs_1 = F.softmax(scores_1, dim=-1)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            probs_0 = attention_dropout(probs_0)
            probs_1 = attention_dropout(probs_1)
    # weighting
    pad = torch.zeros(layout[1], device=q.device, dtype=q.dtype)
    probs_all_to_text = torch.cat((
        probs_text,
        pad[-layout[0]:].expand(b*n_head, layout[1]-layout[0], layout[0]),
        probs_0[:, :, :layout[0]],
        probs_1[:, :, :layout[0]]
    ), dim=1)

    context_all_to_text = torch.einsum('bhij,bhcj->bihc', 
        probs_all_to_text.view(b, n_head, probs_all_to_text.shape[1], probs_all_to_text.shape[2]), 
        v_text.view(b, n_head, v_text.shape[1], v_text.shape[2])).reshape(b, -1, hn)
    
    context_0_to_0 = f_weighting(v0, probs_0[..., layout[0]:].view_as(scores_0_to_0).contiguous(), kernel_size*2-1, kernel_size, True)

    context_1_to_0 = f_weighting(v0, probs_1[:, :, layout[0]:layout[0]+scores_1_to_0.shape[-1]].view_as(scores_1_to_0).contiguous(), kernel_size2, kernel_size2, False)

    context_1_to_1 = f_weighting(v1, probs_1[:, :, -scores_1_to_1.shape[-1]:].view_as(scores_1_to_1).contiguous(), kernel_size*2-1, kernel_size, True)
    
    context_all_to_01 =torch.cat(
        (
            pad.expand(b*n_head, hn//n_head, layout[1]),
            context_0_to_0.view(b*n_head, hn//n_head, layout[2]-layout[1]),
            (context_1_to_0 + context_1_to_1).view(b*n_head, hn//n_head, layout[3]-layout[2])
        ), dim=-1).view(b, hn, -1).transpose(1, 2)
    return context_all_to_text + context_all_to_01 


def sparse_attention_2dfull(q, k, v, n_head, layout, attention_mask_text2d, kernel_size=9, kernel_size2=7, attention_dropout=None, **kwargs):
    '''
    q, k, v: [batch_size, 64+1024+4096, hidden_size]
    n_head: int
    layout: [endoftext/startofpad, startof0, startof1, endofall]
    attention_mask_text2d: [batch_size, sq_len, endoftext]
    '''
    from .local_attention_function import f_similar, f_weighting
    b, sq_len, hn = q.shape
    alpha = sqrt((layout[3] - layout[2]) // (layout[2] - layout[1]))

    q = q / math.sqrt(hn // n_head) # normalization

    q_all, q_text, q0, q1 = transpose_and_split(q, layout, n_head) # 0, 1 [batch * n_head, hn_per_head, h, w] text [batch * n_head, hn_per_head, endoftext]
    k_all, k_text, k0, k1 = transpose_and_split(k, layout, n_head)
    v_all, v_text, v0, v1 = transpose_and_split(v, layout, n_head)
    # import pdb; pdb.set_trace()
    # all to text
    scores_all_to_text = torch.einsum('bhi,bhj->bij', q_all, k_text).view(b, n_head, layout[3], layout[0]) * attention_mask_text2d - 10000.0 * (1.0 - attention_mask_text2d)
    scores_all_to_text = scores_all_to_text.view(b*n_head, layout[3], layout[0])
    # 0 to 0
    if not hasattr(sparse_attention_2dfull, 'attention_mask0'):
        sparse_attention_2dfull.attention_mask0 = torch.ones((layout[2] - layout[1], layout[2] - layout[1]), device=q.device, dtype=q.dtype).tril_()
    attention_mask0 = sparse_attention_2dfull.attention_mask0
    scores_0_to_0 = torch.einsum('bhi,bhj->bij', q0.view(*q0.shape[:2], -1), k0.view(*k0.shape[:2], -1)) * attention_mask0 - 10000.0 * (1.0 - attention_mask0)
    # 1 to 1
    scores_1_to_1 = f_similar(q1, k1, kernel_size*2-1, kernel_size, True)    
    # 1 to 0
    scores_1_to_0 = f_similar(q1, k0, kernel_size2, kernel_size2, False) # [batch * n_head, 2h, 2w, kernel_size2**2]
    # softmax

    scores_0 = torch.cat(
        (scores_all_to_text[:, layout[1]:layout[2]], 
        scores_0_to_0.view(b * n_head, layout[2]-layout[1], scores_0_to_0.shape[-1])), 
        dim=-1)
    scores_1 = torch.cat(
        (scores_all_to_text[:, layout[2]:layout[3]],
         scores_1_to_0.view(scores_1_to_0.shape[0], -1, scores_1_to_0.shape[3]),
         scores_1_to_1.view(scores_1_to_1.shape[0], -1, scores_1_to_1.shape[3])),
         dim=-1)
    probs_text = F.softmax(scores_all_to_text[:, :layout[0]], dim=-1) # [batch * n_head, seq_text, seq_text]
    probs_0 = F.softmax(scores_0, dim=-1) # 
    probs_1 = F.softmax(scores_1, dim=-1)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            probs_0 = attention_dropout(probs_0)
            probs_1 = attention_dropout(probs_1)
    # weighting
    pad = torch.zeros(layout[1], device=q.device, dtype=q.dtype)
    probs_all_to_text = torch.cat((
        probs_text,
        pad[-layout[0]:].expand(b*n_head, layout[1]-layout[0], layout[0]),
        probs_0[:, :, :layout[0]],
        probs_1[:, :, :layout[0]]
    ), dim=1)

    context_all_to_text = torch.einsum('bhij,bhcj->bihc', 
        probs_all_to_text.view(b, n_head, probs_all_to_text.shape[1], probs_all_to_text.shape[2]), 
        v_text.view(b, n_head, v_text.shape[1], v_text.shape[2])).reshape(b, -1, hn)
    
    context_0_to_0 = torch.einsum('bcj,bij->bci', v0.view(*v0.shape[:2], -1), probs_0[..., layout[0]:].view_as(scores_0_to_0))

    context_1_to_0 = f_weighting(v0, probs_1[:, :, layout[0]:layout[0]+scores_1_to_0.shape[-1]].view_as(scores_1_to_0).contiguous(), kernel_size2, kernel_size2, False)

    context_1_to_1 = f_weighting(v1, probs_1[:, :, -scores_1_to_1.shape[-1]:].view_as(scores_1_to_1).contiguous(), kernel_size*2-1, kernel_size, True)
    
    context_all_to_01 =torch.cat(
        (
            pad.expand(b*n_head, hn//n_head, layout[1]),
            context_0_to_0.view(b*n_head, hn//n_head, layout[2]-layout[1]),
            (context_1_to_0 + context_1_to_1).view(b*n_head, hn//n_head, layout[3]-layout[2])
        ), dim=-1).view(b, hn, -1).transpose(1, 2)
    return context_all_to_text + context_all_to_01 


if args.sparse_config.sparse_type == 'cuda_2d':
            layout = args.sparse_config.layout
            unpad_indices = (data[:, :layout[1]+1] >= 0) * 10000.
            unpad_indices[:, -1] = 9000.
            starts = (torch.arange(layout[1]+1, device=data.device).expand_as(unpad_indices) + unpad_indices).min(dim=-1)[1]
            layout[0] = starts.max().item()
            attention_mask = torch.ones((batch_size, seq_length, layout[0]), device=data.device)
            for i in range(batch_size):
                attention_mask[i, :, starts[i]:layout[1]] = 0
            attention_mask[:, :layout[0]].tril_()
            attention_mask = attention_mask.unsqueeze(1)
elif args.sparse_config.sparse_type == 'standard':
            attention_mask = torch.ones((batch_size, seq_length, seq_length), device=data.device)
            attention_mask.tril_()
            # attention_mask = torch.zeros((seq_length, seq_length), device=data.device)
            # h = w = 32
            # k1=9
            # layout = [10, 64, 64+h*w, 64+h*w*5]
            # for i in range(layout[1]):
            #     attention_mask[i, :i+1] = 1
            # for i in range(layout[1], layout[2]):
            #     x = (i - layout[1]) // w
            #     y = (i - layout[1]) % w
            #     lx = max(0, x - k1 // 2)
            #     ly = max(0, y - k1 // 2)
            #     rx = min(h-1, x + k1 // 2)
            #     ry = min(w-1, y + k1 // 2)
            #     attention_mask[i, layout[1]:layout[2]].view(h, w)[lx:x, ly:ry+1] = 1
            #     attention_mask[i, layout[1]:layout[2]].view(h, w)[x, ly:y+1] = 1
            # attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)