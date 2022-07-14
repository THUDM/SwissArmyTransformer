import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin

from SwissArmyTransformer.mpu.layers import ColumnParallelLinear

from SwissArmyTransformer.model.position_embedding import RotaryEmbedding
from SwissArmyTransformer.model.position_embedding import \
apply_rotary_pos_emb_torch, apply_rotary_pos_emb, apply_rotary_pos_emb_fused, \
    apply_rotary_pos_emb_index_torch, apply_rotary_pos_emb_index, apply_rotary_pos_emb_index_fused

from SwissArmyTransformer.transformer_defaults import standard_attention
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim, divide

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)



class RotaryEmbeddingMixin(BaseMixin):
    def __init__(self, fp16, apply_rotary_positional_embedding_kernel, bf16, hidden_size, num_attention_heads, model_parallel_size, learnable_rotary_embedding=False):
        super().__init__()
        hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, model_parallel_size)
        self.rotary_emb = RotaryEmbedding(
            hidden_size_per_attention_head,
            base=10000,
            precision=torch.half if fp16 else torch.float,
            learnable=False,
            device=torch.cuda.current_device())

        self.apply_rotary_fn = (
                apply_rotary_pos_emb_index_fused
                if apply_rotary_positional_embedding_kernel
                else apply_rotary_pos_emb_index_torch
                if bf16
                else apply_rotary_pos_emb_index
            )

           
    def attention_forward(self, hidden_states, mask, **kw_args):
        attn = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = standard_attention
        if 'attention_fn' in attn.hooks:
            attention_fn = attn.hooks['attention_fn']

        # [seq, b, 3 * hn * np]
        mixed_raw_layer = attn.query_key_value(hidden_states)

        # [seq, b, (np * 3 * hn)] --> [seq, b, np, 3 * hn]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head)
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)
        
        # [sq, b, np, hn]
        (query_layer,
            key_layer,
            value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = attn.attention_dropout if attn.training else None

        kw_args['position_ids'] = kw_args['position_ids'].transpose(0, 1)

        cos, sin = self.rotary_emb(value_layer, seq_len=kw_args['position_ids'].max() + 1)

        query_layer, key_layer = self.apply_rotary_fn(query_layer, key_layer, cos, sin, kw_args['position_ids'])

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        output = attn.dense(context_layer)

        if attn.training:
            output = attn.output_dropout(output)


        return output


class _GLUBaseModule(torch.nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)

class GEGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.gelu)


class DeepNormWithGLUMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size, inner_hidden_size=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size * 2 // 3
        self.inner_hidden_size = inner_hidden_size

    def reinit(self):
        del self.transformer.position_embeddings
        for layer in self.transformer.layers:
            del layer.mlp.dense_h_to_4h
            layer.mlp.dense_h_to_4h = ColumnParallelLinear(
                self.hidden_size,
                2 * self.inner_hidden_size,
                gather_output=False,
                bias=True,
                params_dtype=torch.half,
                module=self,
                name="dense_h_to_4h",
                skip_init=True,
                device=torch.cuda.current_device()
            )
            del layer.mlp.activation_func
            layer.mlp.activation_func = GEGLU()

           
    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [seq_len, batch, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        layer = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.

        attention_input = layer.input_layernorm(hidden_states)

        # Self attention.
        attention_output = layer.attention(attention_input, mask, **kw_args)

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = layer.post_attention_layernorm(hidden_states)

        if layer.is_decoder:
            encoder_outputs = kw_args['encoder_outputs']
            if encoder_outputs is not None:
                assert 'cross_attention_mask' in kw_args
                # Cross attention
                attention_output = layer.cross_attention(mlp_input, **kw_args)
                # Residual connection.
                hidden_states = hidden_states + attention_output
                # Layer norm post the cross attention
                mlp_input = layer.post_cross_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = layer.mlp(mlp_input, **kw_args)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output

class SelfAttentionWithFP32SoftmaxMixin(BaseMixin):
    def __init__(self, hidden_size, num_attention_heads, model_parallel_size):
        super().__init__()  
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.hidden_size_per_partition = divide(hidden_size, model_parallel_size) 

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask, attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, mems=None, **kwargs):

        mem = mems[kwargs['layer_id']] if mems is not None else None
        
        # seqlen, batch, head, hidden_size
        seq_len, b, nh, hidden_size = key_layer.shape

        # b, seqlen, stack, head, hidden
        cache_kv = torch.stack((key_layer, value_layer)).permute(2, 1, 0, 3, 4).detach().contiguous().view(b, seq_len, nh * hidden_size * 2)
        kwargs['output_this_layer']['mem_kv'] = cache_kv

        if mem is not None: # the first time, mem is None
            # might change batch_size
            # b, seqlen, stack, head, hidden -> stack, seqlen, b, head, hidden
            mem = mem.expand(b, -1, -1).reshape(b, mem.shape[1], 2, nh, hidden_size).permute(2, 1, 0, 3, 4)
            memk, memv = mem[0], mem[1]
            key_layer = torch.cat((memk, key_layer), dim=0)
            value_layer = torch.cat((memv, value_layer), dim=0)

        # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
        # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 
        query_key_layer_scaling_coeff = float(kwargs['layer_id'] + 1)
        if scaling_attention_score:
            query_layer = query_layer / (math.sqrt(self.hidden_size_per_attention_head) * query_key_layer_scaling_coeff)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        matmul_result = torch.empty(
                output_size[0]*output_size[1],
                output_size[2],
                output_size[3],
                dtype=query_layer.dtype,
                device=torch.cuda.current_device())
        
        matmul_result = torch.baddbmm(
                matmul_result,
                query_layer.transpose(0, 1),   # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0, alpha=1.0)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)


        # if log_attention_weights is not None:
        #     attention_scores += log_attention_weights

        if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
            # if auto-regressive, skip
            attention_scores = torch.mul(attention_scores, attention_mask) - \
                            10000.0 * (1.0 - attention_mask)

    

        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = attention_probs.half()


        if attention_dropout is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = attention_dropout(attention_probs)
            else:
                attention_probs = attention_dropout(attention_probs)
                
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # context_layer = torch.matmul(attention_probs, value_layer)
        
        return context_layer

class FinalForwardMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    def final_forward(self, logits, **kw_args):
        return F.linear(logits, self.transformer.word_embeddings.weight).transpose(0, 1).contiguous()

class NonePositionEmbedding(BaseMixin):
    def __init__(self):
        super().__init__()

    def position_embedding_forward(self, position_ids, output_cross_layer, **kw_args):
        return None

class WordEmbedding(BaseMixin):
    def __init__(self):
        super().__init__()

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        return self.transformer.word_embeddings(input_ids).transpose(0, 1)


class GLM130B(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=False):
        super().__init__(args, params_dtype=torch.half if args.fp16 else torch.float, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('glu-deepnorm',
            DeepNormWithGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size)
        )
        self.add_mixin('fp32-softmax', 
            SelfAttentionWithFP32SoftmaxMixin(args.hidden_size, args.num_attention_heads, args.model_parallel_size)
        )
        self.add_mixin('final-forward', 
            FinalForwardMixin()
        )
        self.add_mixin('non-position-embedding', 
            NonePositionEmbedding()
        )
        self.add_mixin('word-embedding', 
            WordEmbedding()
        )
        self.add_mixin('rotary-embedding', 
            RotaryEmbeddingMixin(args.fp16, args.apply_rotary_positional_embedding_kernel, args.bf16, args.hidden_size, args.num_attention_heads, args.model_parallel_size)
        )
        self.get_mixin("glu-deepnorm").reinit()

    @classmethod
    def add_model_specific_args(cls, parser):
        """Arguments for GLM"""
        group = parser.add_argument_group('GLM', 'GLM Configurations')
        
        group.add_argument('--task-mask', action='store_true', help="Use different mask for generation and blank filling")

        group.add_argument('--apply-rotary-positional-embedding-kernel', action='store_true', help="Apply rotary positional embedding kernel")

        return parser