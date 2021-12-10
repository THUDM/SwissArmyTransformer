import math
import torch
import torch.nn.functional as F
from .mixins import BaseMixin
from .encoder_decoder_model import EncoderDecoderModel
from SwissArmyTransformer.mpu import get_model_parallel_world_size
from SwissArmyTransformer.mpu.transformer import standard_attention, SelfAttention, CrossAttention, MLP
from SwissArmyTransformer.mpu.mappings import copy_to_model_parallel_region
from SwissArmyTransformer.mpu.utils import divide, split_tensor_along_last_dim


class T5PositionEmbeddingMixin(BaseMixin):
    def position_embedding_forward(self, position_ids, **kw_args):
        return None


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 or bfloat16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        elif self.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        return self.weight * hidden_states


class T5AttentionMixin(BaseMixin):
    def __init__(self, relative_attention_num_buckets, num_attention_heads, is_decoder=False):
        super().__init__()
        self.relative_attention_num_buckets = relative_attention_num_buckets
        world_size = get_model_parallel_world_size()
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.relative_attention_bias = torch.nn.Embedding(self.relative_attention_num_buckets,
                                                          self.num_attention_heads_per_partition)
        self.is_decoder = is_decoder

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, cross_attention=False):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        # shape (query_length, key_length, num_heads)
        if cross_attention:
            values = self.cross_relative_attention_bias(relative_position_bucket)
        else:
            values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def attention_forward(self, hidden_states, mask, position_bias=None, *args, layer_id=None, mems=None, **kw_args):
        attn_module = self.transformer.layers[layer_id].attention
        seq_length = hidden_states.size(1)
        memory_length = mems[layer_id].size(1) if mems else 0
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = attn_module.attention_dropout if attn_module.training else None

        query_layer = attn_module._transpose_for_scores(mixed_query_layer)
        key_layer = attn_module._transpose_for_scores(mixed_key_layer)
        value_layer = attn_module._transpose_for_scores(mixed_value_layer)

        if position_bias is None:
            position_bias = self.compute_bias(seq_length, memory_length + seq_length)
        context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn,
                                           log_attention_weights=position_bias, scaling_attention_score=False)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = attn_module.dense(context_layer)

        if attn_module.training:
            output = attn_module.output_dropout(output)

        return output, position_bias

    def cross_attention_forward(self, hidden_states, cross_attention_mask, encoder_outputs, layer_id=None, *args,
                                **kw_args):
        attn_module = self.transformer.layers[layer_id].cross_attention
        mixed_query_layer = attn_module.query(hidden_states)
        mixed_x_layer = attn_module.key_value(encoder_outputs)
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)

        dropout_fn = attn_module.attention_dropout if attn_module.training else None
        # Reshape and transpose [b, np, s, hn]
        query_layer = attn_module._transpose_for_scores(mixed_query_layer)
        key_layer = attn_module._transpose_for_scores(mixed_key_layer)
        value_layer = attn_module._transpose_for_scores(mixed_value_layer)

        context_layer = standard_attention(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn,
                                           scaling_attention_score=False)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = attn_module.dense(context_layer)
        if attn_module.training:
            output = attn_module.output_dropout(output)

        return output


class T5DecoderFinalMixin(BaseMixin):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def final_forward(self, logits, **kwargs):
        logits_parallel = copy_to_model_parallel_region(logits)
        logits_parallel = logits_parallel * (self.hidden_size ** -0.5)
        logits_parallel = F.linear(logits_parallel, self.transformer.word_embeddings.weight)
        return logits_parallel


class T5Model(EncoderDecoderModel):
    def __init__(self, args, **kwargs):
        self.init_method_std = args.init_method_std
        super().__init__(args, tie_word_embeddings=True, **kwargs, use_bias=False,
                         layernorm=T5LayerNorm, activation_func=torch.nn.functional.relu,
                         init_method=self._init_weights)
        self.encoder.add_mixin(
            "t5-attention", T5AttentionMixin(args.relative_attention_num_buckets, args.num_attention_heads)
        )
        self.encoder.add_mixin(
            "t5-position", T5PositionEmbeddingMixin()
        )
        del self.encoder.transformer.position_embeddings
        num_attention_heads = args.dec_num_attention_heads if args.dec_num_attention_heads is not None else args.num_attention_heads
        self.decoder.add_mixin(
            "t5-attention", T5AttentionMixin(args.relative_attention_num_buckets, num_attention_heads, is_decoder=True)
        )
        self.decoder.add_mixin(
            "t5-position", T5PositionEmbeddingMixin()
        )
        self.decoder.add_mixin(
            "t5-final", T5DecoderFinalMixin(args.hidden_size)
        )
        del self.decoder.transformer.position_embeddings

    def _init_weights(self, weight, module, name):
        init_method_std = self.init_method_std
        if isinstance(module, MLP):
            if name == "dense_h_to_4h":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size ** -0.5))
            elif name == "dense_4h_to_h":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.inner_hidden_size ** -0.5))
            else:
                raise NotImplementedError(name)
        elif isinstance(module, SelfAttention):
            if name == "query_key_value":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size ** -0.5))
                torch.nn.init.normal_(weight[:module.inner_hidden_size], mean=0, std=init_method_std * (
                        (module.hidden_size * module.hidden_size_per_attention_head) ** -0.5))
            elif name == "dense":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.inner_hidden_size ** -0.5))
            else:
                raise NotImplementedError(name)
        elif isinstance(module, CrossAttention):
            if name == "query":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (
                        (module.hidden_size * module.hidden_size_per_attention_head) ** -0.5))
            elif name == "key_value":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.hidden_size ** -0.5))
            elif name == "dense":
                torch.nn.init.normal_(weight, mean=0, std=init_method_std * (module.inner_hidden_size ** -0.5))
            else:
                raise NotImplementedError(name)
        else:
            raise NotImplementedError(module)

    @classmethod
    def add_model_specific_args(cls, parser):
        super().add_model_specific_args(parser)
        parser.add_argument("--relative-attention-num-buckets", type=int, default=None)
        parser.add_argument("--init-method-std", type=float, default=0.02)

    def encode(self, input_ids, attention_mask=None, **kw_args):
        return super().encode(input_ids, None, attention_mask, **kw_args)

    def decode(self, input_ids, attention_mask=None, encoder_outputs=None, cross_attention_mask=None, **kw_args):
        return super().decode(input_ids, None, attention_mask, encoder_outputs=encoder_outputs,
                              cross_attention_mask=cross_attention_mask, **kw_args)

    def forward(self, enc_input_ids, dec_input_ids, *, enc_attention_mask=None,
                dec_attention_mask=None, cross_attention_mask=None, **kw_args):
        batch_size, seq_length = enc_input_ids.size()[:2]
        if enc_attention_mask is None:
            enc_attention_mask = torch.ones(1, 1, 1, seq_length,
                                            dtype=self.encoder.transformer.word_embeddings.weight.dtype,
                                            device=enc_input_ids.device)
        if cross_attention_mask is None:
            cross_attention_mask = enc_attention_mask
        encoder_outputs = self.encode(enc_input_ids, enc_attention_mask, **kw_args)
        decoder_outputs, *mems = self.decode(dec_input_ids, dec_attention_mask,
                                             encoder_outputs=encoder_outputs, cross_attention_mask=cross_attention_mask,
                                             **kw_args)
        return encoder_outputs, decoder_outputs, *mems
