import torch
import torch.nn.functional as F
from sat.ops import swiglu

def native_swiglu(x, w1, w2, w3):
    x1 = F.linear(x, w1)
    x2 = F.linear(x, w2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3)

def test_swiglu(seqlen, hidden_size, batch_size=8, multiplier=6):
    w1 = torch.randn(hidden_size * multiplier // 2, hidden_size).half().cuda()
    w2 = torch.randn(hidden_size * multiplier // 2, hidden_size).half().cuda()
    w3 = torch.randn(hidden_size, hidden_size * multiplier//2).half().cuda()
    x = torch.randn(batch_size, seqlen, hidden_size).half().cuda()
    # prof time and meomry of swiglu
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        out = swiglu(x, w1, w2, w3)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    del out
    torch.cuda.empty_cache()

    # prof time and meomry of native_swiglu
    with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True) as prof:
        out_native = native_swiglu(x, w1, w2, w3)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    test_swiglu(seqlen=4096, hidden_size=768, batch_size=8, multiplier=6)

