import torch
import torch.nn as nn

from .base_model import BaseModel
from .cached_autoregressive_model import CachedAutoregressiveModel


class GLMModel(CachedAutoregressiveModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.transformer.block_position_embeddings = torch.nn.Embedding(args.max_sequence_length, args.hidden_size)
        torch.nn.init.normal_(self.transformer.block_position_embeddings.weight, mean=0.0, std=0.02)

    def position_embedding_forward(self, position_ids, *other_tensors):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.transformer.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings
