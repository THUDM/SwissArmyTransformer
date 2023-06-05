from swin_model import SwinModel
import argparse
import torch

swin = SwinModel(args=argparse.Namespace(
    num_layers=2,
    vocab_size=0,
    hidden_size=128,
    num_attention_heads=4,
    max_sequence_length=0,
    layernorm_order='pre',
    fp16=True,
    skip_init=False,
    use_gpu_initialization=True,
    model_parallel_size=1,
    window_size=10,
    input_resolution=[320, 240],
    shift_sizes=None,
    device='cuda'
))

swin = swin.eval().cuda()
input_states = torch.randn(2, 320*240, 128).cuda()
outputs = swin(input_ids=input_states, position_ids=None, attention_mask=None)
breakpoint()