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

class PrefixTuningMixin(BaseMixin):
    def __init__(self, num_layers, hidden_size_per_attention_head, num_attention_heads, prefix_len):
        super().__init__()
        self.prefix = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(2, num_attention_heads, prefix_len, hidden_size_per_attention_head)*0.01)
            for layer_id in range(num_layers)
        ])
        self.prefix_len = prefix_len

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
        prefix_k, prefix_v = self.prefix[kw_args['layer_id']]

        b, nh, seq_len, hidden_size = k.shape
        prefix_k = prefix_k.unsqueeze(0).expand(b, nh, -1, hidden_size)
        prefix_v = prefix_v.unsqueeze(0).expand(b, nh, -1, hidden_size)

        k = torch.cat((k, prefix_k), dim=2)
        v = torch.cat((v, prefix_v), dim=2)
        if mask.numel() > 1:
            mask_prefixed = torch.ones(self.prefix_len, device=mask.device, dtype=mask.dtype)
            mask_prefixed = mask_prefixed.expand(*(mask.size()[:-1]), -1)
            mask = torch.cat((mask, mask_prefixed), dim=-1)
        return old_impl(q, k, v, mask, dropout_fn, **kw_args)

class CollectorMixin(BaseMixin):
    def __init__(self, number_layers, hidden_size_per_attention_head, num_attention_heads, collect_len):
        super().__init__()
        self.collectors = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(num_attention_heads, collect_len, hidden_size_per_attention_head)*0.002)
            for layer_id in range(number_layers)
        ])
        self.collect_len = collect_len

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, attention_output, log_attention_weights=None, scaling_attention_score=True, **kw_args):
        collector_q = self.collectors[kw_args['layer_id']]

        b, nh, seq_len, hidden_size = k.shape
        collector_q = collector_q.unsqueeze(0).expand(b, nh, -1, hidden_size)

        q = torch.cat((q, collector_q), dim=2)
        if scaling_attention_score:
            q = q / math.sqrt(q.shape[-1])
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        if log_attention_weights is not None:
            attention_scores += log_attention_weights

        if not (mask.shape[-2] == 1 and (mask > 0).all()):
            # if auto-regressive, skip
            attention_scores = torch.mul(attention_scores, mask) - \
                               10000.0 * (1.0 - mask)

        attention_probs = F.softmax(attention_scores, dim=-1)

        if dropout_fn is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = dropout_fn(attention_probs)
            else:
                attention_probs = dropout_fn(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        output_shape = context_layer.shape[:-3] + (self.collect_len, -1,)
        attention_output.append(context_layer[:,:,:self.collect_len].permute(0, 2, 1, 3).contiguous().view(*output_shape))
        return context_layer[:,:,self.collect_len:]


class CLSMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.cls_number = args.cls_number
        self.hidden_size = args.hidden_size
        self.cls_embeddings = torch.nn.Parameter(torch.zeros([self.cls_number, self.hidden_size]))
        torch.nn.init.normal_(self.cls_embeddings, mean=0.0, std=0.02)

    def word_embedding_forward(self, input_ids, **kw_tensors):
        origin_embeddings = self.transformer.word_embeddings(input_ids)
        CLS_embeddings = self.cls_embeddings.unsqueeze(0).repeat([origin_embeddings.shape[0], 1, 1])
        new_embeddings = torch.cat([CLS_embeddings, origin_embeddings[:, 1:]], dim=1)

        return new_embeddings

    def position_embedding_forward(self, position_ids, **kw_tensors):
        cls_id = position_ids[0,0].cpu().numpy().tolist()
        if self.cls_number > 1:
            position_ids = torch.cat([torch.ones(position_ids.shape[0], self.cls_number-1).long().cuda()*cls_id, position_ids], dim=1)
        position_embeddings = self.transformer.position_embeddings(position_ids)
        return  position_embeddings

    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
        if self.cls_number > 1 and mask.numel() > 1:
            mask_fixed = torch.ones(self.cls_number - 1, device=mask.device, dtype=mask.dtype)
            mask_fixed = mask_fixed.expand(*(mask.size()[:-1]), -1)
            mask = torch.cat((mask, mask_fixed), dim=-1)
        return old_impl(q, k, v, mask, dropout_fn, **kw_args)

    def reinit(self, *pre_mixins):
        old_weights = self.transformer.word_embeddings.weight.data[0]
        self.cls_embeddings.data[0].copy_(old_weights)

class LoRAMixin(BaseMixin):
    def __init__(
            self,
            hidden_size: int,
            layer_num: int = 24,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
    ):
        super().__init__()
        # Actual trainable parameters
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout and lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_linear = nn.ModuleList([
            nn.ParameterDict()
            for layer_id in range(layer_num)
        ])
        matrices = ["Q", "K", "V", "O"]

        for i in range(layer_num):
            for matrix in matrices:
                self.lora_linear[i][matrix+"_A"] = nn.Parameter(torch.zeros((r, hidden_size)))
                self.lora_linear[i][matrix+"_B"] = nn.Parameter(torch.zeros((hidden_size, r)))
                nn.init.kaiming_uniform_(self.lora_linear[i][matrix+"_A"], a=math.sqrt(5))
                nn.init.zeros_(self.lora_linear[i][matrix+"_B"])


        self.scaling = self.lora_alpha / self.r


    def attention_forward(self, hidden_states, mask, layer_id, **kw_args):
        attention_fn = standard_attention
        if 'attention_fn' in self.transformer.hooks:
            attention_fn = self.transformer.hooks['attention_fn']
        layer = self.transformer.layers[layer_id].attention
        lora_layer = self.lora_linear[layer_id]

        mixed_raw_layer = layer.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
        mixed_query_layer = mixed_query_layer + (self.lora_dropout(hidden_states) @ lora_layer["Q_A"].T @ lora_layer["Q_B"].T) * self.scaling
        mixed_key_layer = mixed_key_layer + (self.lora_dropout(hidden_states) @ lora_layer["K_A"].T @ lora_layer["K_B"].T) * self.scaling
        mixed_value_layer = mixed_value_layer + (self.lora_dropout(hidden_states) @ lora_layer["V_A"].T @ lora_layer["V_B"].T) * self.scaling


        dropout_fn = layer.attention_dropout if self.training else None

        query_layer = layer._transpose_for_scores(mixed_query_layer)
        key_layer = layer._transpose_for_scores(mixed_key_layer)
        value_layer = layer._transpose_for_scores(mixed_value_layer)

        context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (layer.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = layer.dense(context_layer)
        output = output + (self.lora_dropout(context_layer) @ lora_layer["O_A"].T @ lora_layer["O_B"].T ) * self.scaling

        if self.training:
            output = layer.output_dropout(output)
        return output

class LoRAM2Mixin(BaseMixin):
    def __init__(
            self,
            hidden_size: int,
            layer_num: int = 24,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
    ):
        super().__init__()
        # Actual trainable parameters
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout and lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_linear = nn.ModuleList([
            nn.ParameterDict()
            for layer_id in range(layer_num)
        ])
        matrices = ["M2"]

        for i in range(layer_num):
            for matrix in matrices:
                input_dim = 4 * hidden_size if matrix == "M2" else hidden_size
                output_dim = 4 * hidden_size if matrix == "M1" else hidden_size
                self.lora_linear[i][matrix+"_A"] = nn.Parameter(torch.zeros((r, input_dim)))
                self.lora_linear[i][matrix+"_B"] = nn.Parameter(torch.zeros((output_dim, r)))
                nn.init.kaiming_uniform_(self.lora_linear[i][matrix+"_A"], a=math.sqrt(5))
                nn.init.zeros_(self.lora_linear[i][matrix+"_B"])


        self.scaling = self.lora_alpha / self.r


    def mlp_forward(self, hidden_states, layer_id, **kw_args):
        layer = self.transformer.layers[layer_id].mlp
        lora_layer = self.lora_linear[layer_id]

        intermediate_parallel = layer.dense_h_to_4h(hidden_states)
        intermediate_parallel = layer.activation_func(intermediate_parallel)
        output = layer.dense_4h_to_h(intermediate_parallel)

        output = output + (self.lora_dropout(intermediate_parallel) @ lora_layer["M2_A"].T @ lora_layer["M2_B"].T) * self.scaling

        if self.training:
            output = layer.dropout(output)

        return output

class FFADDMixin(BaseMixin):
    def __init__(
            self,
            hidden_size: int,
            layer_num: int = 24,
            r: int = 0,
    ):
        super().__init__()
        # Actual trainable parameters
        self.r = r

        self.ffadd_linear = nn.ModuleList([
            nn.ModuleList()
            for layer_id in range(layer_num)
        ])

        for i in range(layer_num):
            self.ffadd_linear[i].append(torch.nn.Linear(hidden_size, r, bias=True))
            self.ffadd_linear[i].append(torch.nn.Linear(r, hidden_size, bias=True))
            nn.init.zeros_(self.ffadd_linear[i][1].weight)
            nn.init.zeros_(self.ffadd_linear[i][1].bias)


    def mlp_forward(self, hidden_states, layer_id,  attention_output = None, **kw_args):
        layer = self.transformer.layers[layer_id].mlp
        ffadd_layer = self.ffadd_linear[layer_id]

        intermediate_parallel = layer.dense_h_to_4h(hidden_states)
        intermediate_parallel = layer.activation_func(intermediate_parallel)
        output = layer.dense_4h_to_h(intermediate_parallel)

        intermediate_add = ffadd_layer[0](hidden_states)
        intermediate_add = layer.activation_func(intermediate_add)
        if attention_output is not None:
            attention_output.append(intermediate_add)
        output2 = ffadd_layer[1](intermediate_add)
        output = output + output2


        if self.training:
            output = layer.dropout(output)

        return output

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