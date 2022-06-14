# -*- encoding: utf-8 -*-
# @File    :   train_yolos.py
# @Time    :   2022/02/17
# @Author  :   Qingsong Lv
# @Contact :   lqs19@mails.tsinghua.edu.cn

# here put the import lib
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from SwissArmyTransformer import mpu, get_args
from SwissArmyTransformer.model.official.yolos_model import YOLOS
from SwissArmyTransformer.training.deepspeed_training import training_main
from util.misc import nested_tensor_from_tensor_list
from torchvision.transforms import ToPILImage

def get_batch(data_iterator, args, timers, mode):
    from datasets_.coco import make_coco_transforms
    transform = make_coco_transforms(mode, args)

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    image_data = {"image":data[0]}
    label_data = {"label":data[1]}
    timers('data loader').stop()
    # image_data = mpu.broadcast_data(["image"], image_data, torch.float32)
    # label_data = mpu.broadcast_data(["label"], label_data, torch.float32)
    

    # Unpack.
    label_data = label_data['label']
    image_data = image_data['image']

    ori_img = []
    ori_label = []

    # unpad label
    slice_key = ['boxes', 'labels', 'area', 'iscrowd']
    keep_key = ['image_id', 'orig_size', 'size']
    for i in range(label_data['mask'].shape[0]):
        ori_img.append(ToPILImage()(image_data[i]))
        target = {}
        for k in slice_key:
            target[k] = label_data[k][i][label_data['mask'][i]]
        for k in keep_key:
            target[k] = label_data[k][i]
        ori_label.append(target)


    image_data = ori_img
    label_data = ori_label

    image_data, label_data = list(zip(*[transform(x,y) for x, y in zip(image_data, label_data)]))
    image_data = nested_tensor_from_tensor_list(image_data).tensors

    # Convert
    # if args.fp16: # fp16 not supported for matcher
    #     image_data = image_data.half()
        # label_data = [{k:v.half() for k,v in x.items()} for x in label_data]
    # print(image_data)
    # print(label_data)
    # input()
    return image_data.cuda(), label_data


from models.matcher import build_matcher
from models.detector import SetCriterion, PostProcess

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    images, targets = get_batch(
        data_iterator, args, timers, 'train' if model.training else 'val')
    timers('batch generator').stop()

    device = images.device

    batch_size, _, height, width = images.shape
    num_patches = (height//16) * (width//16)
    seq_len = 1 + num_patches + model.module.get_mixin('det_head').num_det_tokens
    position_ids = torch.cat([torch.arange(seq_len)[None,]]*batch_size)
    encoded_input = {'input_ids':torch.cat([torch.arange(1+model.module.get_mixin('det_head').num_det_tokens)[None,]]*batch_size).long(), 'image':images, 'position_ids':position_ids}
    encoded_input = {k:v.to(device) for k,v in encoded_input.items()}
    encoded_input['attention_mask'] = None
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    outputs = model(**encoded_input, offline=False, height=height//16, width=width//16)[0]
    
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(91, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    # results = postprocessors['bbox'](outputs, orig_target_sizes)
    # if 'segm' in postprocessors.keys():
    #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
    # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
    return losses, {'loss': losses}


def create_dataset_function(path, args):
    from datasets_.coco import build
    ds = build(image_set=path, args=args)
    return build(image_set=path, args=args)

def init_function(args, model):
    model.get_mixin("pos_embedding").reinit()
    model.get_mixin('patch_embedding').reinit()


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--old_checkpoint', action="store_true")
    # py_parser.add_argument('--prefix_len', type=int, default=16)
    py_parser.add_argument('--coco_path', default='/data/qingsong/dataset/coco', type=str)
    py_parser.add_argument('--eval_size', default=512, type=int)
    # * Matcher
    py_parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    py_parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    py_parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients

    py_parser.add_argument('--dice_loss_coef', default=1, type=float)
    py_parser.add_argument('--bbox_loss_coef', default=5, type=float)
    py_parser.add_argument('--giou_loss_coef', default=2, type=float)
    py_parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    py_parser.add_argument('--md_type', type=str)

    py_parser = YOLOS.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed
    # initialize_distributed(args)
    # set_random_seed(args.seed)
    model, args = YOLOS.from_pretrained(args, args.md_type)
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, init_function=init_function)
