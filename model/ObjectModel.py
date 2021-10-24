import torch
import torch.nn.functional as F


from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, AttentionMixin, WordEmebeddingMixin

from mpu.transformer import split_tensor_along_last_dim
from mpu.local_attention_function import f_similar, f_weighting
from mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker
from mpu.mappings import gather_from_model_parallel_region, copy_to_model_parallel_region

class ObjectModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        additional_seqlen = args.new_sequence_length - args.max_sequence_length
        self.mixins.append(PositionEmbeddingMixin(
            additional_seqlen, args.hidden_size,
            reinit_slice=slice(-180, None)
        ))
        self.mixins.append(WordEmebeddingMixin(
            args.old_token_num, args.additional_token_num, args.hidden_size
        ))
        self.layout = args.layout

    def position_embedding_forward(self, position_ids, *other_tensors):
        # breakpoint()
        position_text = position_ids[..., :self.layout[0]]
        position_object = position_ids[..., self.layout[0]:self.layout[1]]
        position_image = position_ids[..., self.layout[1]:]
        position_embeddings = torch.cat(
                (
                    self.transformer.position_embeddings(position_text),
                    self.mixins[0].position_embeddings(position_object),
                    self.transformer.position_embeddings(position_image)
                ),
                dim=-2
            )
        return position_embeddings

    def word_embedding_forward(self, input_ids, *other_tensors):
        return self.mixins[1].word_embeddings(input_ids)

    def final_forward(self, logits, *other_tensors):
        logits = copy_to_model_parallel_region(logits)
        logits = F.linear(logits, self.mixins[1].word_embeddings.weight)
        return logits

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ObjectModel', 'Object model configurations')
        group.add_argument("--layout", type=str, default='64,246,1270')
        group.add_argument("--old-token-num", type=int, default=58219)
        group.add_argument("--additional-token-num", type=int, default=257)
        group.add_argument("--new-sequence-length", type=int, default=1271) #1089 + 180 + 2
        return parser