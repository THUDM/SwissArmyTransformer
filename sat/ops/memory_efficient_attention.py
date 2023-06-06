from xformers.ops import memory_efficient_attention as mea
from xformers.ops import LowerTriangularMask

def memory_efficient_attention(q, k, v, attn_bias=None, dropout_prob=0., causal_mask=False):
    if causal_mask:
        assert attn_bias is None, "Causal mask and attention bias are mutually exclusive"
        attn_bias = LowerTriangularMask()
    return mea(q, k, v, attn_bias=attn_bias, p=dropout_prob)
