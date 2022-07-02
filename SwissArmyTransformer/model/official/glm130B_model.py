import torch
import torch.nn as nn
from torch.nn import functional as F

from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin

from SwissArmyTransformer.mpu.layers import ColumnParallelLinear

from SwissArmyTransformer.model.positional_embeddings import RotaryEmbedding
from SwissArmyTransformer.model.positional_embeddings import \
apply_rotary_pos_emb_torch, apply_rotary_pos_emb, apply_rotary_pos_emb_fused, \
    apply_rotary_pos_emb_index_torch, apply_rotary_pos_emb_index, apply_rotary_pos_emb_index_fused

from SwissArmyTransformer.model.transformer import BaseTransformer

from SwissArmyTransformer.transformer_defaults import standard_attention
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim, divide

class BlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
        super(BlockPositionEmbeddingMixin, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
    
    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings


class RotaryEmbeddingMixin(BaseMixin):
    def __init__(self, fp16, learnable_rotary_embedding, apply_rotary_positional_embedding_kernel, bf16, hidden_size, num_attention_heads):
        super().__init__()
        hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.rotary_emb = RotaryEmbedding(
            hidden_size_per_attention_head,
            base=10000,
            precision=torch.half if fp16 else torch.float,
            learnable=learnable_rotary_embedding)

        self.apply_rotary_fn = (
                apply_rotary_pos_emb_index_fused
                if apply_rotary_positional_embedding_kernel
                else apply_rotary_pos_emb_index_torch
                if bf16
                else apply_rotary_pos_emb_index
            )

           
    def attention_forward(self, hidden_states, mask, **kw_args):
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

        # Rotary embeddings
        # [b, sq] -> [sq, b]
        kw_args['position_ids'] = kw_args['position_ids'].transpose(0, 1)
        cos, sin = self.rotary_emb(value_layer, seq_len=kw_args['position_ids'].max() + 1)
        query_layer, key_layer = self.apply_rotary_fn(query_layer, key_layer, cos, sin, kw_args['position_ids'])
            

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)

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
                name="dense_h_to_4h"
            )
            del layer.mlp.activation_func
            layer.mlp.activation_func = GEGLU()

           
    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        self = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        attention_input = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(attention_input, mask, **kw_args)
        
        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        if self.is_decoder:
            encoder_outputs = kw_args['encoder_outputs']
            if encoder_outputs is not None:
                assert 'cross_attention_mask' in kw_args
                # Cross attention
                attention_output = self.cross_attention(mlp_input, **kw_args)
                # Residual connection.
                hidden_states = hidden_states + attention_output
                # Layer norm post the cross attention
                mlp_input = self.post_cross_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input, **kw_args)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output

class FP32SoftmaxMixin(BaseMixin):
    def __init__(self):
        super().__init__()  


    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):

        # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
        # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 
        query_key_layer_scaling_coeff = kw_args['layer_id']
        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1]) / query_key_layer_scaling_coeff
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if log_attention_weights is not None:
            attention_scores += log_attention_weights

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

        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

def empty_init(master_weight, module, name):
    pass

class GLM130B(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, empty_init, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('glu-deep-norm',
            DeepNormWithGLUMixin(args.num_layers, args.hidden_size, args.inner_hidden_size)
        )
        self.add_mixin('fp32-softmax', 
            FP32SoftmaxMixin()
        )

    @classmethod
    def add_model_specific_args(cls, parser):
        """Arguments for GLM"""
        group = parser.add_argument_group('GLM', 'GLM Configurations')
        group.add_argument('--block-lm', action='store_true', help="whether use the BlockLM pre-training")
        group.add_argument('--masked-lm', action='store_true', help='whether to use the mlm objective')
        group.add_argument('--bert-prob', type=float, default=0.5)
        group.add_argument('--gpt-infill-prob', type=float, default=0.5)
        group.add_argument('--gpt-min-ratio', type=float, default=0.5)
        group.add_argument('--gap-sentence-prob', type=float, default=0.0)
        group.add_argument('--gap-sentence-ratio', type=float, default=0.15)
        group.add_argument('--avg-block-length', type=int, default=3)
        group.add_argument('--short-seq-prob', type=float, default=0.0)
        group.add_argument('--single-span-prob', type=float, default=0.0)
        group.add_argument('--task-mask', action='store_true', help="Use different mask for generation and blank filling")
        group.add_argument('--no-shuffle-block', action='store_true', help="not shuffle the blocks when filling the blank")
        group.add_argument('--no-block-position', action='store_true',
                        help='Use (rough) absolute positions instead of block positions')
        group.add_argument('--sentinel-token', action='store_true',
                        help="Use sentinel (mask) tokens to replace 2d position encoding")
        group.add_argument('--block-mask-prob', type=float, default=0.0)
        group.add_argument('--context-mask-ratio', type=float, default=0.0)
        group.add_argument('--random-position', action='store_true',
                        help="Use random start position to cover all the position embeddings")
        group.add_argument('--cloze-eval', action='store_true', help='Evaluation dataset with cloze task')
        group.add_argument('--old-checkpoint', action='store_true', help="Loading the checkpoint from old libraray")

        group.add_argument('--tokenizer-model-type', type=str,
                       default=None,
                       help="Model type to use for sentencepiece tokenization \
                           (one of ['bpe', 'char', 'unigram', 'word']) or \
                           bert vocab to use for BertWordPieceTokenizer (one of \
                           ['bert-large-uncased', 'bert-large-cased', etc.])")

        group.add_argument('--apply-rotary-positional-embedding-kernel', action='store_true', help="Apply rotary positional embedding kernel")

        group.add_argument('--learnable-rotary-embedding', action='store_true', help="Makes rotary embedding learnable")

        return parser