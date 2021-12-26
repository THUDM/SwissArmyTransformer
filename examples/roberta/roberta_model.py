import torch
import torch.nn as nn
from SwissArmyTransformer.mpu.transformer import LayerNorm
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel

roberta_gelu = nn.functional.gelu

class roberta_lm_head(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, layernorm_epsilon=1.0e-5):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.decoder = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.dense(x)
        x = roberta_gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class RobertaFinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lm_head = roberta_lm_head(vocab_size, hidden_size)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)


class RobertaModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(RobertaModel, self).__init__(args, transformer=transformer, activation_func=roberta_gelu, **kwargs)
        self.add_mixin("roberta-final", RobertaFinalMixin(args.vocab_size, args.hidden_size))
