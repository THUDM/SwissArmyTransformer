from lib2to3.pytree import Base
import torch
import torch.nn as nn
import math
from SwissArmyTransformer.mpu.transformer import LayerNorm, standard_attention
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim
roberta_gelu = nn.functional.gelu

# class CoMixin(BaseMixin):
#     def __init__(self):
#         super().__init__()
#     def 

class CLSMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.cls_embeddings = torch.nn.Parameter(torch.zeros([1, args.hidden_size]))
        torch.nn.init.normal_(self.cls_embeddings, mean=0.0, std=0.02)

    def word_embedding_forward(self, input_ids, **kw_tensors):
        origin_embeddings = self.transformer.word_embeddings(input_ids)
        CLS_embeddings = self.cls_embeddings.view([1,1,-1]).repeat([origin_embeddings.shape[0], 1, 1])
        new_embeddings = torch.cat([CLS_embeddings, origin_embeddings[:, 1:]], dim=1)
        return new_embeddings

    def reinit(self, *pre_mixins):
        old_weights = self.transformer.word_embeddings.weight.data[0]
        self.cls_embeddings.data.copy_(old_weights)
        
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