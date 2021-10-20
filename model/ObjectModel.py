import torch
import torch.nn.functional as F


from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, AttentionMixin

from mpu.transformer import split_tensor_along_last_dim
from mpu.local_attention_function import f_similar, f_weighting
from mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker

class ObjectModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        additional_seqlen = args.new_sequence_length - args.max_sequence_length
        self.mixins.append(PositionEmbeddingMixin(
            additional_seqlen, args.hidden_size
        ))
        self.layout = args.layout

    def position_embedding_forward(self, position_ids, *other_tensors):
        position = position_ids[..., :self.layout[1]]
        position_plus = position_ids[..., self.layout[1]:]
        position_embeddings = torch.cat(
                (
                    self.transformer.position_embeddings(position),
                    self.mixins[0].position_embeddings(position_plus)
                ),
                dim=-2
            )
        return position_embeddings

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ObjectModel', 'Object model configurations')
        group.add_argument("--layout", type=str, default='64,1088')
        group.add_argument("--new-sequence-length", type=int, default=5185)
        return parser