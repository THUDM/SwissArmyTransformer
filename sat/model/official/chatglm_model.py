import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel
import math
from sat import mpu
from sat.mpu.utils import split_tensor_along_last_dim

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))

def gelu(x):
    return gelu_impl(x)

from sat.model.position_embedding.rotary_embeddings import RotaryEmbedding, apply_rotary_pos_emb_index

class ChatGLMFinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)

class ChatGLMAttnMixin(BaseMixin):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(
            hidden_size // (num_heads * 2),
            base=10000,
            precision=torch.half,
            learnable=False,
        )

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                        attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
        # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 
        query_key_layer_scaling_coeff = float(kwargs['layer_id'] + 1)
        if scaling_attention_score:
            query_layer = query_layer / (math.sqrt(query_layer.shape[-1]) * query_key_layer_scaling_coeff)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        dtype = attention_scores.dtype
        if log_attention_weights is not None:
            attention_scores += log_attention_weights

        if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
            # if auto-regressive, skip
            attention_scores = torch.mul(attention_scores, attention_mask) - \
                            10000.0 * (1.0 - attention_mask)
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type(dtype)

        if attention_dropout is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = attention_dropout(attention_probs)
            else:
                attention_probs = attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

    def attention_forward(self, hidden_states, mask, **kw_args):
        mixin_self = self
        position_ids = kw_args['position_ids']
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = self.hooks['attention_fn']

        hidden_states = hidden_states.transpose(0, 1)
        mixed_raw_layer = self.query_key_value(hidden_states)

        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
        k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
        cos, sin = mixin_self.rotary_emb(q1, seq_len=position_ids.max() + 1)
        position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
            position_ids[:, 1, :].transpose(0, 1).contiguous()
        q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
        key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))

        if kw_args.get('past_key_values', None) is not None:
            pack = kw_args['past_key_values'][kw_args['layer_id']]
            if pack is not None:
                past_key, past_value = pack
                key_layer = torch.cat((past_key, key_layer), dim=0)
                value_layer = torch.cat((past_value, value_layer), dim=0)
        kw_args['output_this_layer']['past_key_values'] = (key_layer, value_layer)

        query_layer = query_layer.permute(1, 2, 0, 3)
        key_layer = key_layer.permute(1, 2, 0, 3)
        value_layer = value_layer.permute(1, 2, 0, 3)

        dropout_fn = self.attention_dropout if self.training else None

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

class ChatGLMLayerMixin(BaseMixin):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        num_layers = self.num_layers
        self = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        attention_input = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(attention_input, mask, **kw_args)
        alpha = (2 * num_layers) ** 0.5

        # Third LayerNorm
        if self.layernorm_order == 'sandwich':
            attention_output = self.third_layernorm(attention_output)
        
        # Residual connection.
        if self.layernorm_order == 'post':
            hidden_states = attention_input * alpha + attention_output
        else:
            hidden_states = hidden_states + attention_output

        
        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input, **kw_args)

        # Fourth LayerNorm
        if self.layernorm_order == 'sandwich':
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        if self.layernorm_order == 'post':
            output = mlp_input * alpha + mlp_output
        else:
            output = hidden_states + mlp_output

        return output
    
class ChatGLMModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(ChatGLMModel, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        self.add_mixin("chatglm-final", ChatGLMFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("chatglm-attn", ChatGLMAttnMixin(args.hidden_size, args.num_attention_heads))
        self.add_mixin("chatglm-layer", ChatGLMLayerMixin(args.num_layers))
        self.bos_token_id = args.bos_token_id
        self.mask_token_id = args.mask_token_id
        self.gmask_token_id = args.gmask_token_id
        self.pad_token_id = 3

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        attention_mask, position_ids = self.get_inputs(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[..., -1:]
            if input_ids.size(0) != 1:
                attention_mask = attention_mask[:, :, -1:]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
    
    def get_inputs(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        if attention_mask is None:
            if past_key_values is not None and input_ids.size(0) == 1:
                attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
            else:
                attention_mask = self.get_masks(
                    input_ids=input_ids,
                    device=input_ids.device, **kwargs
                )
        elif attention_mask.dtype is torch.bool:
            attention_mask = (~attention_mask).long()
        if position_ids is None:
            MASK, gMASK = self.mask_token_id, self.gmask_token_id
            mask_token = gMASK if gMASK in input_ids else MASK
            use_gmask = True if gMASK in input_ids else False

            mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
            position_ids = self.get_position_ids(
                input_ids=input_ids,
                mask_positions=mask_positions,
                device=input_ids.device,
                gmask=use_gmask, **kwargs
            )
        return attention_mask, position_ids
    
    def get_pad_length(self, seq):
        l = 0
        while l < len(seq) and seq[l] == self.pad_token_id:
            l += 1
        return l
    
    def get_masks(self, input_ids, device, **kwargs):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        for i, pad_length in enumerate(pad_lengths):
            attention_mask[i, :, :pad_length] = 0
            attention_mask[i, :pad_length, :] = 0
        attention_mask.unsqueeze_(1)
        # attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, device, gmask=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        context_lengths = [seq.tolist().index(self.bos_token_id) for seq in input_ids]
        position_ids = [torch.arange(seq_length-pad_length, dtype=torch.long, device=device) for pad_length in pad_lengths]
        for i, (context_length, pad_length) in enumerate(zip(context_lengths, pad_lengths)):
            position_ids[i][context_length-pad_length:] = mask_positions[i] - pad_length
        block_position_ids = [torch.cat((
            torch.zeros(context_length, dtype=torch.long, device=device),
            torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
        )) for context_length in context_lengths]
        block_position_ids = torch.stack(block_position_ids, dim=0)
        position_ids = [torch.cat((
            torch.zeros(pad_length, dtype=torch.long, device=device),
            range_pos
        )) for pad_length, range_pos in zip(pad_lengths, position_ids)]
        position_ids = torch.stack(position_ids, dim=0)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)

        return position_ids
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ChatGLM', 'ChatGLM Configurations')
        group.add_argument('--bos-token-id', type=int)
        group.add_argument('--mask-token-id', type=int)
        group.add_argument('--gmask-token-id', type=int)
        return super().add_model_specific_args(parser)