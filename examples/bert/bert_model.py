import torch
import torch.nn as nn

from SwissArmyTransformer.mpu.transformer import LayerNorm
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel

bert_gelu = nn.functional.gelu

class bert_lm_head(torch.nn.Module):
    def __init__(self, hidden_size, output_size, layernorm_epsilon=1.0e-5):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size, bias = True)
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
        return self.lm_head(hidden_states[:,0])

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


class BertModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(BertModel, self).__init__(args, transformer=transformer,
                                        activation_func=bert_gelu, **kwargs)
        self.add_mixin("position_embedding_forward", BertPositionEmbeddingMixin(args.hidden_size))
        self.add_mixin("final_forward", BertFinalMixin(args.hidden_size, args.output_size))
