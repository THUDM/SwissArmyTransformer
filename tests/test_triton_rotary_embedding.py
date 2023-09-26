import math
import os
import torch
import torch.nn.functional as F
from einops import rearrange
from sat.model.position_embedding.triton_rotary_embeddings import FastRotaryEmbedding
import time
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding as RotaryEmbeddingNeoX
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.random.manual_seed(0)
def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id):
    # batch_size, num_head, seq_len, hidden_size
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x.ndim-1)  # dim=-1 triggers a bug in earlier torch versions


class SatRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False, device=torch.device('cpu')):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        # inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            cos_cached = cos_cached.to(x.dtype)
            sin_cached = sin_cached.to(x.dtype)
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


if __name__ == "__main__":
    headdim = 128
    batch_size = 12
    seqlen = 384
    nheads = 32
    device = torch.device('cuda')
    dtype = torch.bfloat16
    position_ids = torch.arange(seqlen, device=device)
    position_ids = position_ids.repeat(batch_size,1)
    for i in range(batch_size):
        position_ids[i, i+1] = 100
    print(position_ids.size())
    query = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
    key = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
    value = torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype)
    query_neox = query.clone()
    key_neox = key.clone()
    value_neox = value.clone()

    sat_rotary_emb = SatRotaryEmbedding(
            128,
            base=10000,
            precision=torch.bfloat16,
            learnable=False,
            device=device
    )
    for i in range(10):
        cos, sin = sat_rotary_emb(value, seq_len=seqlen)
        qu, ke = apply_rotary_pos_emb_index_bhs(query, key, cos, sin, position_ids)
    torch.cuda.synchronize()
    st = time.time()
    for i in range(1000):
        cos, sin = sat_rotary_emb(value, seq_len=seqlen)
        qu, ke = apply_rotary_pos_emb_index_bhs(query, key, cos, sin, position_ids)
    torch.cuda.synchronize()
    tt = time.time() - st
    print("ref time is ", tt)
    
    
    rotary_emb = FastRotaryEmbedding(dim=128, device=device)
    for i in range(1000):
        q_triton,k_triton = rotary_emb(query_neox,key_neox, position_ids, max_seqlen=seqlen)
    torch.cuda.synchronize()
    st = time.time()
    for i in range(1000):
        q_triton,k_triton = rotary_emb(query_neox,key_neox, position_ids, max_seqlen=seqlen)
    torch.cuda.synchronize()
    tt = time.time() - st
    print("now time is ", tt)
    

    print("max ", (q_triton - qu).abs().max().item())
    print("mean ", (q_triton - qu).abs().mean().item())
    print("max ", (k_triton - ke).abs().max().item())
    print("mean ", (k_triton - ke).abs().mean().item())
