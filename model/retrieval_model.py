# -*- encoding: utf-8 -*-
# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F

from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, AttentionMixin, ParallelLinearMixin

from mpu.transformer import split_tensor_along_last_dim
from mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker


class RetrievalModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.layout = args.layout
        self.txt_img_split = args.txt_img_split
        
        self.mixins.append(PositionEmbeddingMixin(
            2, args.hidden_size
        ))
        self.mixins.extend([
            ParallelLinearMixin(
                args.hidden_size, args.retrieval_size),
            ParallelLinearMixin(
                args.hidden_size, args.retrieval_size)
            ])
    
    def reinit(self):
        pass
    
    def position_embedding_forward(self, position_ids, *other_tensors):
        position_embeddings = torch.cat(
                (
                    self.transformer.position_embeddings(position_ids[:, :-2]),
                    self.mixins[0].position_embeddings(position_ids[:, -2:])
                ),
                dim=-2
            )
        return position_embeddings
    
    def final_forward(self, logits, *other_tensors):
        txt_logits = logits[:, -1, :]
        img_logits = logits[:, -2, :]
        return (self.mixins[1](txt_logits), self.mixins[2](img_logits))
    
    def disable_untrainable_params(self):
        pass
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('RetrievalModel', 'retrieval model configurations')
        group.add_argument('--txt-img-split', action='store_true')
        group.add_argument('--retrieval-temp', type=int, default=0)
        group.add_argument('--retrieval-mode', type=str, default='txt2img',
                            choices=['txt2img', 'img2txt', 'symmetric'])
        group.add_argument('--retrieval-hidden-size', type=int, default=2048)
        group.add_argument('--retrieval-size', type=int, default=1024)
        group.add_argument("--layout", type=str, default='64,1088')
        return parser