import torch
import torch.nn as nn
from SwissArmyTransformer.model.transformer import LayerNorm
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel

gelu = nn.functional.gelu

class lm_head(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, layernorm_epsilon=1.0e-5):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.decoder = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class BertFinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lm_head = lm_head(vocab_size, hidden_size)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)

class BertTypeMixin(BaseMixin):
    def __init__(self, num_types, hidden_size):
        super().__init__()
        self.type_embeddings = nn.Embedding(num_types, hidden_size)
    def word_embedding_forward(self, input_ids, **kwargs):
        return self.transformer.word_embeddings(input_ids) + self.type_embeddings(kwargs["token_type_ids"])

class ForwardMixin(BaseMixin):
    def __init__(self):
        super().__init__()
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

        return output#, kw_args['output_this_layer'], kw_args['output_cross_layer']


class BertModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(BertModel, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        self.add_mixin("bert-final", BertFinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("bert-type", BertTypeMixin(args.num_types, args.hidden_size))
        self.add_mixin("bert-forward", ForwardMixin())
        

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('BERT', 'BERT Configurations')
        group.add_argument('--num-types', type=int)
        return parser