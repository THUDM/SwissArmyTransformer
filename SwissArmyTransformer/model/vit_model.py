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
from collections.abc import Iterable
import math
import torch.nn.functional as F


gelu = nn.functional.gelu


class ViTProperty:
    """
    Store some hyper-parameters such as image size and patch size.
    We only support static image size for now.
    seq_len = pre_len + image_len + post_len
    """
    def __init__(self, image_size, patch_size, pre_len, post_len, **kwargs):
        assert isinstance(image_size, Iterable) and len(image_size) == 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.pre_len = pre_len
        self.post_len = post_len


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(self, in_channels, hidden_size, property, flatten=True):
        super(ImagePatchEmbeddingMixin, self).__init__()
        self.flatten = flatten
        self.property = property
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=property.patch_size, stride=property.patch_size)

    def word_embedding_forward(self, input_ids, **kwargs):
        images = kwargs["image"]
        embeddings = self.proj(images)
        if self.flatten:
            embeddings = embeddings.flatten(2).transpose(1, 2)
        pre_word_embeddings = self.transformer.word_embeddings(input_ids[:,:self.property.pre_len])
        post_word_embeddings = self.transformer.word_embeddings(input_ids[:,self.property.pre_len:self.property.pre_len+self.property.num_patches])
        embeddings = torch.cat([pre_word_embeddings, embeddings, post_word_embeddings], dim=1)
        return embeddings


class InterpolatedPositionEmbeddingMixin(BaseMixin):
    def __init__(self, hidden_size, old_property, property, init_method_std=0.02):
        super(InterpolatedPositionEmbeddingMixin, self).__init__()
        self.old_property = old_property
        self.property = property
        self.position_embeddings = torch.nn.Embedding(property.pre_len + property.num_patches + property.post_len, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)
    
    def position_embedding_forward(self, position_ids, **kwargs):
        """
        Although this is same as the default position_embedding_forward, we can use this to bypass an assert for now.
        (assert position_ids.shape[-1] == query_length in mpu/transformer.py)
        """
        return self.transformer.position_embeddings(position_ids)

    def reinit(self, *pre_mixins):
        """
        new pre_len, new num_patches and new post_len should all be larger or equal than the old ones.
        """
        old_weight = self.transformer.position_embeddings.weight.data
        pre_weight = old_weight[:self.old_property.pre_len]
        post_weight = old_weight[self.old_property.pre_len+self.old_property.num_patches:]
        image_weight = old_weight[self.old_property.pre_len:self.old_property.pre_len+self.old_property.num_patches].reshape(1, self.old_property.grid_size[0], self.old_property.grid_size[1], -1).permute(0, 3, 1, 2)
        image_weight = F.interpolate(image_weight, size=self.property.grid_size, mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(self.property.num_patches, -1)
        self.position_embeddings.weight.data[:self.old_property.pre_len] = pre_weight
        self.position_embeddings.weight.data[self.property.pre_len:self.property.pre_len+self.property.num_patches] = image_weight
        self.position_embeddings.weight.data[self.property.pre_len+self.property.num_patches:self.property.pre_len+self.property.num_patches+self.old_property.post_len] = post_weight
        self.transformer.position_embeddings = self.position_embeddings
        del self.position_embeddings


class ViTModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, activation_func=gelu, **kwargs)
        self.property = ViTProperty(args.image_size, args.patch_size, args.pre_len, args.post_len)
        if args.load:
            assert args.old_image_size is not None and args.old_pre_len is not None and args.old_post_len is not None
            self.old_property = ViTProperty(args.old_image_size, args.patch_size, args.old_pre_len, args.old_post_len)
        else:
            self.old_property = self.property
        self.add_mixin("patch_embedding", ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, self.property))
        self.add_mixin("pos_embedding", InterpolatedPositionEmbeddingMixin(args.hidden_size, self.old_property, self.property))
        self.classifier = nn.Linear(args.hidden_size, args.num_classes)

    def final_forward(self, logits, **kw_args):
        logits = self.classifier(logits[:, 0])
        return logits

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('ViT', 'ViT Configurations')
        group.add_argument('--num-finetune-classes', type=int, default=None)
        group.add_argument('--new-sequence-length', type=int, default=None)
        group.add_argument('--image-size', nargs='+', type=int, default=[224, 224])
        group.add_argument('--pre-len', type=int, default=1) # [cls] by default
        group.add_argument('--post-len', type=int, default=0) # empty by default, but sometimes with special tokens, such as [det] in yolos.
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--num-classes', type=int, default=21843)
        group.add_argument('--patch-size', type=int, default=16)
        group.add_argument('--old-image-size', nargs='+', type=int, default=None)
        group.add_argument('--old-pre-len', type=int, default=None)
        group.add_argument('--old-post-len', type=int, default=None)
        return parser


