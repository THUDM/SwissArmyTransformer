from apex.normalization.fused_layer_norm import FusedLayerNorm
class LayerNorm(FusedLayerNorm):
    def __init__(self, *args, pb_relax=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pb_relax = pb_relax

    def forward(self, x):
        if not self.pb_relax:
            return super().forward(x)
        return super().forward(x / (x.abs().max().detach() / 8))
# from  torch.nn import LayerNorm