import math
import torch
import torch.nn.functional as F
from SwissArmyTransformer.model.base_model import BaseMixin, non_conflict
from SwissArmyTransformer.model import ViTModel
from SwissArmyTransformer.model.mixins import PrefixTuningMixin

class NewClassHeadMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.classifier = torch.nn.Linear(args.hidden_size, args.num_finetune_classes)

class InterpolatedPositionEmbeddingMixin(BaseMixin):
    def __init__(self, new_sequence_length, hidden_size, pre_interpolate, init_method_std=0.02):
        super(InterpolatedPositionEmbeddingMixin, self).__init__()
        self.pre_interpolate = pre_interpolate
        if self.pre_interpolate:
            self.new_sequence_length = new_sequence_length
            self.position_embeddings = torch.nn.Embedding(new_sequence_length, hidden_size)
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/models/vit/modeling_vit.py#L79

        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.
        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        N = embeddings.shape[0] - 1
        class_pos_embed = embeddings[:1]
        patch_pos_embed = embeddings[1:]
        dim = embeddings.shape[-1]
        h0 = height
        w0 = width
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def position_embedding_forward(self, position_ids, **kw_args):
        if self.pre_interpolate:
            return self.position_embeddings(position_ids)
        else:
            ini_pos_embed = self.transformer.position_embeddings.weight
            if position_ids.shape[-1] == ini_pos_embed.shape[-2]:
                return ini_pos_embed.unsqueeze(0).expand((position_ids.shape[0], -1, -1))
            else:
                new_height = int(math.sqrt(position_ids.shape[-1]-1))
                new_width = int(math.sqrt(position_ids.shape[-1]-1))
            return self.interpolate_pos_encoding(ini_pos_embed, new_height, new_width)

    def reinit(self, *pre_mixins):
        if self.pre_interpolate:
            old_weight = self.transformer.position_embeddings.weight.data
            old_len, hidden_size = old_weight.shape
            image_len_old = int(math.sqrt(old_len))
            image_len_new = int(math.sqrt(self.new_sequence_length-1))
            cls_weight = old_weight[:1]
            image_weight = old_weight[1:].reshape(1, image_len_old, image_len_old, hidden_size).permute(0, 3, 1, 2)
            image_weight = F.interpolate(image_weight, size=image_len_new, mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(image_len_new * image_len_new, hidden_size)
            new_weight = torch.cat([cls_weight, image_weight], dim=0)
            self.position_embeddings.weight.data.copy_(new_weight)

class ViTFinetuneModel(ViTModel):
    def __init__(self, args, transformer=None, parallel_output=True, layernorm_epsilon=1e-6):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, layernorm_epsilon=layernorm_epsilon)
        self.add_mixin('finetune_head', NewClassHeadMixin(args))
        self.add_mixin('interpolated_pos', InterpolatedPositionEmbeddingMixin(args.new_sequence_length, args.hidden_size, args.pre_interpolate))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    
    def final_forward(self, logits, **kw_args):
        logits = self.mixins["finetune_head"].classifier(logits[:, 0])
        return logits
