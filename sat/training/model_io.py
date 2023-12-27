# -*- encoding: utf-8 -*-
'''
@File    :   model_io.py
@Time    :   2021/10/05 18:39:55
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import numpy as np
import json
import argparse
from sat import mpu
from sat.helpers import print_rank0, print_all

def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    return os.path.join(checkpoints_path, 'latest')

def extract_model_specific_args_from_model(args, model):
    parser = argparse.ArgumentParser()
    
    if hasattr(model, 'module'):
        model = model.module
    if isinstance(model, torch.nn.Module):
        for md in model.modules(): # search 
            if hasattr(md, 'add_model_specific_args'):
                try:
                    md.add_model_specific_args(parser)
                except argparse.ArgumentError as e:
                    print(e)
    ret = {}
    for k in vars(parser.parse_args([])).keys():
        if hasattr(args, k):
            ret[k] = getattr(args, k)
    return ret

def extract_model_specific_args_to_dump(args, model):
    module = model.module if hasattr(model, 'module') else model
    # model_class
    to_dump = {'model_class': type(module).__name__} 
    # tokenizer_type
    if hasattr(args, 'tokenizer_type') and args.tokenizer_type != 'fake':
        to_dump['tokenizer_type'] = args.tokenizer_type 
    # architecture related args
    arch_args_list = ['num_layers', 'hidden_size', 'num_attention_heads', 'vocab_size',
        'layernorm_order', 'model_parallel_size', 'max_sequence_length',
        ]
    for name in arch_args_list: 
        if hasattr(args, name) and getattr(args, name) is not None:
            to_dump[name] = getattr(args, name)

    # optional architecture related args, only save if not default
    # optional means might be changed when loading 
    optional_arch_args_list = [
        ('is_decoder', False), 
        ('cross_attn_hidden_size', None), 
        ('use_bias', True),
        ('use_qkv_bias', False),
        ('inner_hidden_size', None),
        ('hidden_size_per_attention_head', None),
        ('cross_hidden_size_per_attention_head', None),
        ('use_final_layernorm', True),
        ('layernorm_epsilon', 1e-5),
        ('num_multi_query_heads', 0),
        ('cross_num_multi_query_heads', 0),
        ('row_parallel_linear_final_bias', True),
        ('is_gated_mlp', False)
    ]
    if hasattr(module, 'transformer'):
        for name, default in optional_arch_args_list:
            if module.transformer.__dict__[name] != default:
                to_dump[name] = module.transformer.__dict__[name]

    # model specific args
    model_specific_args = extract_model_specific_args_from_model(args, module)
    to_dump.update(model_specific_args)
    return to_dump


def update_ema_parameters_to_model(optimizer):
    """update ema parameters"""
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.runtime.utils import all_gather_dp_groups
    for i, (bit16_partitions, fp32_partition) in enumerate(
                zip(optimizer.parallel_partitioned_bit16_groups, optimizer.single_partition_of_fp32_groups)):
            ema_optimizer= optimizer.optimizer
            state = ema_optimizer.state[fp32_partition]
            partition_id = dist.get_rank(group=optimizer.real_dp_process_group[i])
            bit16_partitions[partition_id].data.copy_(state['shadow'].data)
    
    all_gather_dp_groups(groups_flat=optimizer.bit16_groups_flat,
                         partitioned_param_groups=optimizer.parallel_partitioned_bit16_groups,
                         dp_process_group=optimizer.real_dp_process_group,
                         start_alignment_factor=optimizer.nccl_start_alignment_factor,
                         allgather_bucket_size=optimizer.allgather_bucket_size)

def save_checkpoint(iteration, model, optimizer,
                    lr_scheduler, args):
    """Save a model checkpoint."""
    if hasattr(args, 'deepspeed') and args.deepspeed:
        if mpu.get_data_parallel_rank() == 0:
            print_rank0('Saving Model...')
            save_ds_checkpoint(iteration, model, lr_scheduler, args)
        if optimizer is not None and optimizer.optimizer.__class__.__name__ ==  "FusedEmaAdam" :
            update_ema_parameters_to_model(optimizer)
            if mpu.get_data_parallel_rank() == 0:
                print_rank0('Saving Ema Model...')
                save_ds_checkpoint(iteration, model, lr_scheduler, args, True)
    elif args.mode == 'inference':
        os.makedirs(os.path.join(args.save, str(iteration)), exist_ok=True)
        if torch.distributed.get_rank() < args.model_parallel_size:
            torch.save({'module': model.state_dict()}, os.path.join(args.save, str(iteration), 'mp_rank_{:02d}_model_states.pt'.format(torch.distributed.get_rank())))
    else:
        raise ValueError("training without deepspeed is not supported.")
    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
        # save model_config.json for from_pretrained().
        with open(os.path.join(args.save, 'model_config.json'), 'w') as f:
            to_dump = extract_model_specific_args_to_dump(args, model)
            json.dump(to_dump, f, indent=4)

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, lr_scheduler, args, use_ema = False):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    if lr_scheduler is not None:
        sd['client_lr_scheduler'] = lr_scheduler.state_dict()
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
    if not use_ema:
        save_ds_checkpoint_no_optim(model, args.save, str(iteration), client_state=sd)
    else:
        save_ds_checkpoint_no_optim(model, args.save, str(iteration)+'-ema', client_state=sd)


def save_ds_checkpoint_no_optim(model, save_dir, tag=None, client_state={}, save_latest=True):
    os.makedirs(save_dir, exist_ok=True)
    # Ensure tag is a string
    tag = str(tag)
    # Real save via deepspeed
    model._create_checkpoint_file(save_dir, tag, False)
    model._save_checkpoint(save_dir, tag, client_state=client_state)
    # Save latest checkpoint tag
    if save_latest:
        with open(os.path.join(save_dir, 'latest'), 'w') as fd:
            fd.write(tag)

    return True


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank0('could not find the metadata file {} '.format(
            tracker_filename))
        raise ValueError('could not find the metadata file {}, please check --load'.format(
            tracker_filename))
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    return iteration, release, True


def load_checkpoint(model, args, load_path=None, prefix=''):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = args.load

    # If model-only mode, set necessary args.
    if not hasattr(args, 'mode'):
        from copy import deepcopy
        args = deepcopy(args)
        args.mode = 'inference'

    iteration, release, success = get_checkpoint_iteration(load_path)
    if not success:
        return 0
    
    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
    if mpu.get_data_parallel_rank() == 0:
            print_all('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))
    sd = torch.load(checkpoint_name, map_location='cpu')
    new_sd = {'module':{}}
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]
    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['module'][k[len(prefix):]] = sd['module'][k]
    sd = new_sd
    
    if hasattr(model, 'module'):
        module = model.module
    else: # inference without deepspeed
        module = model

    # only load module, other hyperparameters are just for recording.
    missing_keys, unexpected_keys = module.load_state_dict(sd['module'], strict=False)
    if len(unexpected_keys) > 0:
        print_rank0(
            f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    if len(missing_keys) > 0:
        if args.mode == 'inference':
            if 'force_inference' in args and args.force_inference:
                print_rank0(f'Warning: Missing keys for inference: {missing_keys}.')
            else:
                raise ValueError(f'Missing keys for inference: {missing_keys}.\nIf you still want to inference anyway, pass --force_inference to args.')
        else: # new params
            assert all(name.find('mixins')>=0 or name.find('cross_attention')>=0 for name in missing_keys), missing_keys
            assert args.mode == 'finetune'
            # list all mixin names
            mixin_names = []
            for key_name in missing_keys:
                if key_name.find('mixins') < 0:
                    continue
                parts = key_name.split('.')
                mixin_name = parts[parts.index('mixins')+1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names) # initialize mixins

    # Do not need this any more, because we create optimizer after load now.
    # if args.mode != 'inference' and args.deepspeed and args.fp16:
    #     model.optimizer.refresh_fp32_params() # restore fp32 weights from module

    # Iterations.
    if args.mode == 'finetune':
        iteration = 0
    elif args.mode == 'pretrain' and not args.no_load_rng: # rng states.
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the random '
                         'state.'.format(checkpoint_name))
            exit()
    elif args.mode == 'inference':
        module.eval()

    if mpu.get_data_parallel_rank() == 0:
        print_all('> successfully loaded {}'.format(checkpoint_name))
    del sd
    return iteration
