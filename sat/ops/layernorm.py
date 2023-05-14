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
    from  torch.nn import LayerNorm