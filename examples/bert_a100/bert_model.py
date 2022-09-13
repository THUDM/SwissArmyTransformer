from lib2to3.pytree import Base
import torch
import torch.nn as nn
import math
from SwissArmyTransformer.mpu.transformer import LayerNorm, standard_attention
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel, non_conflict
from SwissArmyTransformer.mpu.utils import split_tensor_along_last_dim
import torch.nn.functional as F
from SwissArmyTransformer import mpu
bert_gelu = nn.functional.gelu

class bert_lm_head(torch.nn.Module):
    def __init__(self,  hidden_size, output_size, layernorm_epsilon=1.0e-5):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dense2 = nn.Linear(hidden_size, output_size, bias=True)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        x = self.dense1(x)
        x = nn.Tanh()(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class BertFinalMixin(BaseMixin):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lm_head = bert_lm_head(hidden_size, output_size)

    def final_forward(self, hidden_states, **kwargs):
        return self.lm_head(hidden_states[:, 0])



class BertPositionEmbeddingMixin(BaseMixin):
    def __init__(self, hidden_size):
        super().__init__()
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

    def position_embedding_forward(self, position_ids, **kw_args):
        position_embeddings = self.transformer.position_embeddings(position_ids)

        if 'token_type_ids' not in kw_args:
            #device = position_ids.device
            token_type_ids = torch.zeros(position_ids.size(), dtype=torch.long).cuda()
        else:
            token_type_ids =kw_args['token_type_ids']
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = position_embeddings + token_type_embeddings
        return position_embeddings

class MLP_bert_Mixin(BaseMixin):
    def __init__(self, hidden_size, *output_sizes, bias=True, activation_func=torch.nn.functional.relu, init_mean=0, init_std=0.005):
        super().__init__()
        self.activation_func = activation_func
        last_size = hidden_size
        self.layers = torch.nn.ModuleList()
        pooler_layer = torch.load('/thudm/workspace/guoyanhui/SwissArmyTransformer-main/models/bert/bert-large-uncased/pooler_parameter.pt')['pooler']
        print("pooler:")
        print(pooler_layer.weight.shape[0])
        print("hidden:")
        print(hidden_size)
        assert pooler_layer.weight.shape[0] == hidden_size, "dimension is different "
        self.layers.append(pooler_layer)
        for sz in output_sizes:
            this_layer = torch.nn.Linear(last_size, sz, bias=bias)
            last_size = sz
            torch.nn.init.normal_(this_layer.weight, mean=init_mean, std=init_std)
            self.layers.append(this_layer)

    def final_forward(self, logits, **kw_args):
        for i, layer in enumerate(self.layers):
            if i ==0:
                logits = layer(logits)
                logits =torch.nn.Tanh()(logits)
                logits = torch.nn.Dropout(p=0.1)(logits)
            if i > 0:
                logits = layer(logits)
        return logits


class BertModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(BertModel, self).__init__(args, transformer=transformer,
                                        activation_func=bert_gelu, **kwargs)
        self.add_mixin("position_embedding_forward", BertPositionEmbeddingMixin(args.hidden_size))
        #self.add_mixin("final_forward", BertFinalMixin(args.hidden_size, args.output_size))
        self.add_mixin("final_forward", BertFinalMixin(args.hidden_size, 2))

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