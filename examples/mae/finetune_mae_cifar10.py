# -*- encoding: utf-8 -*-

# here put the import lib
import argparse
from ssl import OPENSSL_VERSION
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from SwissArmyTransformer import mpu, get_args
from mae_model import MAE
from mae_finetune_model import MAE_finetune
from SwissArmyTransformer.training.deepspeed_training import training_main

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

    logits = model(tokens, position_ids, attention_mask, image=images, offline=True, mask_ratio=0.)
    loss = F.cross_entropy(logits, labels)
    acc = (torch.argmax(logits, dim=-1) == labels).sum() / labels.numel()
    return loss, {'acc': acc}

#/dataset/fd5061f6/SwissArmyTransformerDatasets/
def create_dataset_function(path, args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(224),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='/'.join(path.split('/')[:-1]), train=(path.split('/')[-1]=='train'),
                                            download=True, transform=transform)
    return trainset

def init_function(args, model):
    model.encoder.get_mixin("pos_embedding").reinit()

if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    # py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser = MAE.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed
    initialize_distributed(args)
    set_random_seed(args.seed)
    swiss_args = argparse.Namespace(
        num_layers=12,
        vocab_size=1,
        hidden_size=768,
        num_attention_heads=12,
        hidden_dropout=0.,
        attention_dropout=0.,
        in_channels=3,
        image_size=[224, 224],
        patch_size=16,
        pre_len=1,
        post_len=0,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        checkpoint_activations=True,
        checkpoint_num_layers=1,
        sandwich_ln=False,
        post_ln=False,
        # model_parallel_size=1,
        # world_size=1,
        # rank=0,
        num_classes=1000,
        dec_num_layers=8,
        dec_hidden_size=512,
        dec_num_attention_heads=16,
        load='/data/qingsong/pretrain/swiss-mae',
        old_image_size=[224, 224],
        old_pre_len=1,
        old_post_len=0,
        mode='finetune'
        )
    from SwissArmyTransformer.training.deepspeed_training import load_checkpoint
    swiss_model = MAE(swiss_args)
    load_checkpoint(swiss_model, swiss_args)
    swiss_model = swiss_model.encoder
    swiss_model = MAE_finetune(swiss_model, swiss_args.hidden_size, 10)
    if args.fp16:
        swiss_model.half()
    elif args.bf16:
        swiss_model.bfloat16()
    model = swiss_model.cuda(torch.cuda.current_device())
    override_attrs = vars(swiss_args).keys()
    for name in override_attrs:
        dec_attr = getattr(swiss_args, name)
        setattr(args, name, dec_attr)
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, init_function=init_function)
