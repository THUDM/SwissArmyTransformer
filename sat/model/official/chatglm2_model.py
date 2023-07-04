import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel
from sat.mpu.utils import split_tensor_along_last_dim

def swiglu(x):
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]

from sat.model.normalization import RMSNorm
from sat.transformer_defaults import standard_attention
from sat.model.position_embedding.rotary_embeddings_original import RotaryEmbedding, apply_rotary_pos_emb

class ChatGLM2AttnMixin(BaseMixin):
    def __init__(self, hidden_size, num_heads, max_seq_len):
        super().__init__()
        rotary_dim = hidden_size // num_heads
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=True)
        self.max_seq_len = max_seq_len

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        # expand head dim to query dim, if necessary
        # only useful for multi-query attention
        batch_size, num_query_heads = query_layer.shape[:2] # [b, np, s, hn]
        num_kv_heads = key_layer.shape[1] # [b, np, s, hn]
        key_layer = key_layer.unsqueeze(1).expand(-1, num_query_heads//num_kv_heads, -1, -1, -1).contiguous().view(batch_size, num_query_heads, *key_layer.shape[2:])
        value_layer = value_layer.unsqueeze(1).expand(-1, num_query_heads//num_kv_heads, -1, -1, -1).contiguous().view(batch_size, num_query_heads, *value_layer.shape[2:])

        if int(torch.__version__.split('.')[0]) >= 2:
            assert scaling_attention_score == True
            dropout_p = 0. if attention_dropout is None or not attention_dropout.training else attention_dropout.p
            if attention_mask.all() and query_layer.shape[2] == key_layer.shape[2]:
                return torch.nn.functional.scaled_dot_product_attention(
                    query_layer, key_layer, value_layer,
                    dropout_p=dropout_p,
                    is_causal=True
                )
            else:
                return torch.nn.functional.scaled_dot_product_attention(
                    query_layer, key_layer, value_layer, 
                    attention_mask,
                    dropout_p
                )
        else:
            assert attention_mask.shape[1] == 1 and query_layer.shape[1] == attention_mask.shape[2] and key_layer.shape[1] == attention_mask.shape[3]
            return standard_attention(
                query_layer, key_layer, value_layer, attention_mask,
                attention_dropout=attention_dropout, log_attention_weights=log_attention_weights,
                scaling_attention_score=scaling_attention_score, **kwargs
            )
        
    def attention_forward(self, hidden_states, mask, **kw_args):
        rotary_pos_emb = self.rotary_pos_emb(self.max_seq_len)
        if kw_args['position_ids'] is not None:
            rotary_pos_emb = rotary_pos_emb[kw_args['position_ids']]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :hidden_states.size(1)]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = standard_attention
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, self.stride)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        query_layer = apply_rotary_pos_emb(query_layer.permute(2, 0, 1, 3), rotary_pos_emb).permute(1, 2, 0, 3)
        key_layer = apply_rotary_pos_emb(key_layer.permute(2, 0, 1, 3), rotary_pos_emb).permute(1, 2, 0, 3)

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)
        breakpoint()

        if self.training:
            output = self.output_dropout(output)
        return output

class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, act_func=F.silu, bias=False):
        super().__init__()
        self.w2 = nn.ModuleList([nn.Linear(in_features, hidden_features, bias=bias) for i in range(num_layers)])
        self.act = act_func

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args['layer_id']].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args['layer_id']](x)
        hidden = self.act(x1) * x2
        x = origin.dense_4h_to_h(hidden)
        return x

from .chatglm_model import ChatGLMFinalMixin

class ChatGLM2Model(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLM2Model, self).__init__(args, transformer=transformer, activation_func=swiglu, layernorm=RMSNorm, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin("chatglm-final", ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("attn", ChatGLM2AttnMixin(args.hidden_size, args.num_attention_heads, args.max_sequence_length))
        self.add_mixin("mlp", SwiGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size, bias=args.use_bias))

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM2', 'ChatGLM2 Configurations')
        return super().add_model_specific_args(parser)