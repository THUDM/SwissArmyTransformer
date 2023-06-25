# -*- encoding: utf-8 -*-
# @File    :   finetune_vit_cifar10.py
# @Time    :   2021/12/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

# here put the import lib
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from sat import mpu, get_args
from vit_ft_model import ViTFinetuneModel
from sat.training.deepspeed_training import training_main

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
    seq_length = args.pre_len + (args.image_size[0]//args.patch_size)*(args.image_size[1]//args.patch_size) + args.post_len
    position_ids = torch.zeros(seq_length, device=image_data.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[:seq_length])
    position_ids = position_ids.unsqueeze(0).expand([batch_size, -1])
    attention_mask = torch.ones((1, 1), device=image_data.device)

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
    loss = F.cross_entropy(logits, labels)
    acc = (torch.argmax(logits, dim=-1) == labels).sum() / labels.numel()
    return loss, {'acc': acc}

#/dataset/fd5061f6/satDatasets/
def create_dataset_function(path, args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(args.finetune_resolution),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/'.join(path.split('/')[:-1]), train=(path.split('/')[-1]=='train'),
                                            download=True, transform=transform)
    return trainset

def init_function(args, model):
    from sat.model.official.vit_model import ViTProperty
    old_prop = model.transformer.property
    new_prop = ViTProperty(args.finetune_resolution, old_prop.patch_size, old_prop.pre_len, old_prop.post_len)
    model.get_mixin("pos_embedding").reinit(property=new_prop)

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--from_pretrained', type=str)
    # py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser = ViTFinetuneModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    model, args = ViTFinetuneModel.from_pretrained(args.from_pretrained, args)
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, init_function=init_function)
