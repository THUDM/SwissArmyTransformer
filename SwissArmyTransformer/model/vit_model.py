# -*- encoding: utf-8 -*-
# @File    :   vit_model.py
# @Time    :   2021/12/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import argparse

import torch

from SwissArmyTransformer.model.base_model import BaseModel
from SwissArmyTransformer.model.mixins import BaseMixin
import torch.nn as nn

gelu = nn.functional.gelu

class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(self, in_channels, image_size, patch_size, hidden_size, flatten=True):
        super(ImagePatchEmbeddingMixin, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.flatten = flatten
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    def word_embedding_forward(self, input_ids, **kwargs):
        images = kwargs["image"]
        B, C, H, W = images.shape
        assert H == self.image_size
        assert W == self.image_size
        embeddings = self.proj(images)
        if self.flatten:
            embeddings = embeddings.flatten(2).transpose(1, 2)
        word_embeddings = self.transformer.word_embeddings(input_ids[:,:1])
        embeddings = torch.cat([word_embeddings, embeddings], dim=1)
        return embeddings


class ViTModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, activation_func=gelu, **kwargs)
        self.add_mixin("patch_embedding", ImagePatchEmbeddingMixin(args.in_channels, args.image_size, args.patch_size, args.hidden_size))
        self.classifier = nn.Linear(args.hidden_size, args.num_classes)

    def final_forward(self, logits, **kw_args):
        logits = self.classifier(logits[:, 0])
        return logits

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ViT', 'ViT Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        group.add_argument('--new-sequence-length', type=int, default=None)
        group.add_argument('--image-size', type=int, default=224)
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--num-classes', type=int, default=21843)
        group.add_argument('--patch-size', type=int, default=16)
        group.add_argument('--pre-interpolate', action='store_true')
        return parser


