# -*- encoding: utf-8 -*-
# @File    :   finetune_vit_cifar10.py
# @Time    :   2021/12/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

# here put the import lib
import os
import sys
import math
import random

from SwissArmyTransformer.data_utils.datasets import TSVDataset
import torch
import argparse
import numpy as np

from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.model import ViTModel
import torchvision
import math
import torchvision.transforms as transforms
import torch.nn.functional as F

class NewClassHeadMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.classifier = torch.nn.Linear(args.hidden_size, args.num_finetune_classes)

class InterpolatedPositionEmbeddingMixin(BaseMixin):
    def __init__(self, new_sequence_length, hidden_size, init_method_std=0.02):
        super(InterpolatedPositionEmbeddingMixin, self).__init__()
        self.pre_interpolate = args.pre_interpolate
        if self.pre_interpolate:
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
        h0 = height // self.mixins.patch_embedding.patch_size
        w0 = width // self.mixins.patch_embedding.patch_size
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
            cls_weight = old_weight[0].unsqueeze(0)
            image_weight = old_weight[1:].reshape(1, image_len_old, image_len_old, hidden_size).permute(0, 3, 1, 2)
            image_weight = F.interpolate(image_weight, size=image_len_new, mode='bicubic', align_corners=False).permute(0, 2, 3, 1).reshape(1, image_len_new * image_len_new, hidden_size)
            new_weight = torch.cat([cls_weight, image_weight], dim=1)
            self.position_embeddings.weight.data.copy_(new_weight)

class ViTFinetuneModel(ViTModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('finetune_head', NewClassHeadMixin(args))
        self.add_mixin('interpolated_pos', InterpolatedPositionEmbeddingMixin(args.new_sequence_length, args.hidden_size))
    
    def final_forward(self, logits, **kw_args):
        logits = self.mixins["finetune_head"].classifier(logits[:, 0])
        return logits

        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)

def get_batch(data_iterator, args, timers):
    # Items and their type.

    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    image_data = {"image":data[0]}
    label_data = {"label":data[1]}
    timers('data loader').stop()
    image_data = mpu.broadcast_data(["image"], image_data, torch.float32)
    label_data = mpu.broadcast_data(["label"], label_data, torch.int64)

    # Unpack.
    label_data = label_data['label'].long()
    image_data = image_data['image']
    batch_size = label_data.size()[0]
    seq_length = args.new_sequence_length
    position_ids = torch.zeros(seq_length, device=image_data.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[:seq_length])
    position_ids = position_ids.unsqueeze(0).expand([batch_size, -1])
    attention_mask = torch.ones((batch_size, 1, seq_length, seq_length), device=image_data.device)

    tokens = torch.zeros((batch_size, 1), device=image_data.device, dtype=torch.long)
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
        image_data = image_data.half()
    return tokens, image_data, label_data, attention_mask, position_ids


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, images, labels, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)

    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask, image=images)
    # logits = torch.softmax(logits, dim=-1)
    loss = F.cross_entropy(logits, labels)
    acc = (torch.argmax(logits, dim=-1) == labels).sum() / labels.numel()
    # pred = ((logits.contiguous().float().squeeze(-1))).sum(dim=-1) / loss_mask.sum(dim=-1)
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(
    #     pred,
    #     labels.float()
    # )
    # acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}

#/dataset/fd5061f6/SwissArmyTransformerDatasets/
def create_dataset_function(path, args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(224),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    return trainset

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser = ViTModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    training_main(args, model_cls=ViTFinetuneModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)
