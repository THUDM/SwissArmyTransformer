from lib2to3.pytree import Base
import torch
import torch.nn as nn
import math
from SwissArmyTransformer.mpu.transformer import LayerNorm, standard_attention
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel, non_conflict
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim
import torch.nn.functional as F
from SwissArmyTransformer import mpu
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

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output, kw_args['output_this_layer'], kw_args['output_cross_layer']