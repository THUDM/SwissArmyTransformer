try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))
except ModuleNotFoundError:
    from sat.helpers import print_rank0
    print_rank0('Please install apex to use fused_layer_norm, fall back to torch.nn.LayerNorm', level='DEBUG')
    import torch
    class LayerNorm(torch.nn.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def forward(self, x):
            # if cpu and float16, calculate in float32 for both x and weight, bias.
            if str(x.device) == 'cpu' and x.dtype in[torch.float16, torch.bfloat16]:
                return torch.nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float()).to(x.dtype)
            else:
                return super().forward(x)    