from sat.model import BaseMixin, BaseModel
import torch
import torch.nn as nn

from sat.model.official.gpt2_model import GPT2AttnMixin
from sat.transformer_defaults import standard_attention
from sat.mpu.utils import split_tensor_along_last_dim
from sat.model.position_embedding.rotary_embeddings import RotaryEmbedding, rotate_half
import torch.nn.functional as F

def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id):
    # batch_size, num_head, seq_len, hidden_size
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k

class RotaryMixin(BaseMixin):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(
            hidden_size // num_heads,
            base=10000,
            precision=torch.half,
            learnable=False,
        )

    def attention_forward(self, hidden_states, mask, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = standard_attention
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        cos, sin = origin.rotary_emb(value_layer, seq_len=value_layer.shape[2])
        query_layer, key_layer = apply_rotary_pos_emb_index_bhs(query_layer, key_layer, cos, sin, kw_args['position_ids'])

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)

class LLaMAMlpMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features):
        super().__init__()
        hidden_features = 4 * in_features if hidden_features is None else hidden_features
        self.gate_proj = nn.ModuleList([nn.Linear(in_features, hidden_features, bias=False) for i in range(num_layers)])

    def mlp_forward(self, hidden_states, **kw_args):
        origin = self.transformer.layers[kw_args['layer_id']].mlp
        hidden_states = origin.activation_func(self.gate_proj[kw_args['layer_id']](hidden_states)) * origin.dense_h_to_4h(hidden_states)
        hidden_states = origin.dense_4h_to_h(hidden_states)
        return hidden_states

class LMMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)

class LLaMAModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, layernorm=LlamaRMSNorm, activation_func=nn.functional.silu, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, layernorm=layernorm, activation_func=activation_func, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin("rotary", RotaryMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin("lm", LMMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("mlp", LLaMAMlpMixin(args.num_layers, args.hidden_size, args.inner_hidden_size))
        self.add_mixin("causal", GPT2AttnMixin(args.max_sequence_length))
    
    def position_embedding_forward(self, *args, **kwargs):
        return None
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('LLaMA', 'LLaMA Configurations')
        group.add_argument('--bos-token-id', type=int, default=0)
        group.add_argument('--eos-token-id', type=int, default=1)
        group.add_argument('--pad-token-id', type=int, default=-1)
        return parser
    
