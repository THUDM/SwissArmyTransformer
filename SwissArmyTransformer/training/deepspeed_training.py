# coding=utf-8
# Rewrite by Ming Ding, Tsinghua University
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import math
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from contextlib import ExitStack

import torch.distributed as dist
import deepspeed

from .learning_rates import AnnealingLR
from .model_io import load_checkpoint, save_checkpoint

from .utils import Timers
from .utils import report_memory
from .utils import print_args
from .utils import print_rank_0
from .utils import get_sample_writer

from SwissArmyTransformer import mpu
from SwissArmyTransformer.data_utils import make_loaders
from SwissArmyTransformer.ops import LayerNorm
from SwissArmyTransformer.arguments import set_random_seed, initialize_distributed

def training_main(args, model_cls, forward_step_function, create_dataset_function, handle_metrics=None, init_function=None):
    """Main training program."""
    hooks = {
        'forward_step': forward_step_function,
        'init_function': init_function,
        'create_dataset_function': create_dataset_function,
        'handle_metrics': handle_metrics,
    }

    timers = Timers()  # Timer.

    # Experiment Name
    if args.load and args.mode == 'pretrain':  # continue training
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")

    # Pytorch distributed. must before seed. ALREADY MOVED TO arguments.py!
    # if isinstance(model_cls, type):
    #     initialize_distributed(args)
    #     set_random_seed(args.seed)  # Random seeds for reproducability.

    # Data stuff.
    train_data, val_data, test_data = make_loaders(args, hooks['create_dataset_function'])
    if args.epochs:
        args.train_iters = len(train_data)
        if args.eval_interval is None:
            args.eval_interval = len(train_data)//args.epochs
        if args.save_interval is None:
            args.save_interval = len(train_data)//args.epochs

    # Build model
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    # Config model IO
    if args.load is not None:
        args.iteration = load_checkpoint(model, args)
        # if we don't load optim_states, filelock is no more needed.
        # with FileLock("/root/checkpoint_lock", timeout=-1):
        #     args.iteration = load_checkpoint(model, optimizer, args)
    else:
        args.iteration = 0
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    torch.distributed.barrier()

    # init hook before building deepspeed model and optimizer
    if hooks['init_function'] is not None:
        hooks['init_function'](args, model)

    # Optimization related things
    model, optimizer = setup_model_untrainable_params_and_optimizer(args, model)

    # initialize lr scheduler
    lr_scheduler = get_learning_rate_scheduler(optimizer, args.iteration, args)

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        if args.mode == 'pretrain':
            print('Pretraining or Continuing training the Model...')
        elif args.mode == 'finetune':
            print('Finetuning Model...')
        print_args(args)
        summary_writer = get_sample_writer(base=args.summary_dir, name=args.experiment_name, iteration=args.iteration)
        if hasattr(summary_writer, 'add_hparams'): # old version does not have this api
            summary_writer.add_hparams(vars(args), {'_dump':0}, name=args.experiment_name, global_step=0)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % len(train_data)
        if val_data is not None:
            start_iter_val = (args.train_iters // args.save_interval) * args.eval_interval
            val_data.batch_sampler.start_iter = start_iter_val % len(val_data)

    # training 
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)
                iteration, skipped = train(model, optimizer,
                    lr_scheduler,
                    train_data,
                    val_data,
                    timers, args, summary_writer=summary_writer,
                    hooks=hooks
                    )

    # final save
    if args.save and iteration != 0:  # TODO save
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    # final testing
    if args.do_test and test_data is not None:
        prefix = 'the end of training for test data'
        test_loss = evaluate_and_print_results(prefix, iter(test_data),
            model, len(test_data) if args.strict_eval else args.eval_iters, args, timers, True, split='test', hooks=hooks)

def get_model(args, model_cls):
    """Build the model."""

    print_rank_0(f'building {model_cls.__name__} model ...')
    model = model_cls(args)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    if args.fp16:
        model.half()
    elif args.bf16:
        model.bfloat16()
    model.cuda(torch.cuda.current_device())

    return model


def setup_model_untrainable_params_and_optimizer(args, model, config_params=None):
    """Setup model and optimizer."""

    if hasattr(model, 'disable_untrainable_params'):
        model.disable_untrainable_params() # mark trainable params

    param_groups = get_optimizer_param_groups(model)

    if args.train_data is not None:
        if args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")
            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                args=args,
                mpu=mpu,
                dist_init_required=False,
                config_params=args.deepspeed_config
            )
        else:
            raise ValueError('Currently, we only support training with deepspeed.')
    else:
        optimizer = None

    return model, optimizer


def get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias' and p.requires_grad])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias' and p.requires_grad])

    if len(weight_decay_params['params']) == 0:
        return (no_weight_decay_params,)
    elif len(no_weight_decay_params['params']) == 0:
        return (weight_decay_params,)

    return weight_decay_params, no_weight_decay_params


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    if hasattr(model, 'module'):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)  # TODO move to here
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False
    return param_groups


def get_learning_rate_scheduler(optimizer, iteration, args,
                                auto_warmup_steps=100, auto_warmup_rate=0.05):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = max(iteration - auto_warmup_steps, 0)
    if args.mode == 'pretrain' and iteration == 0:
        auto_warmup_steps = 0
    # If init_step <= current_steps <= init_step + auto_warmup_steps,
    # lr = auto_warmup_rate * args.lr.
    # This overrides other rules.
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio,
                               auto_warmup_steps=auto_warmup_steps,
                               auto_warmup_rate=auto_warmup_rate
                               )

    return lr_scheduler


def train(model, optimizer, lr_scheduler,
        train_data, val_data, timers, args, 
        summary_writer=None, hooks={}):
    """Train the model."""
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None
        
    # Turn on training mode which enables dropout.
    model.train()
    
    # Tracking loss.
    total_lm_loss = 0.0
    total_metrics = defaultdict(float)

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while args.iteration < args.train_iters:

        lm_loss, skipped_iter, metrics = train_step(train_data_iterator,
                                                    model,
                                                    optimizer,
                                                    lr_scheduler,
                                                    args, timers, hooks=hooks)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        for name in metrics:
            if not 'eval' in name:
                assert len(metrics[name].shape)==0, 'metrics without eval must be scalar'
                total_metrics[name] += metrics[name].data.detach().float().item()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            # average img & txt loss
            avg_metrics = {}
            for key in total_metrics:
                avg_metrics[key] = total_metrics[key] / args.log_interval

            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args,
                                     avg_metrics)
            total_lm_loss = 0.0
            total_metrics = defaultdict(float)
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False

            timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                        'batch generator', 'data loader'],
                       normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            if args.strict_eval:
                val_data_iterator = iter(val_data)
                eval_iters = len(val_data)
            else:
                eval_iters = args.eval_iters
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, eval_iters, args, timers, False, step=args.iteration, split='val', summary_writer=summary_writer, hooks=hooks)

        if args.exit_interval and args.iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, args.iteration), flush=True)
            exit()

    return args.iteration, skipped_iters


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers, hooks=None, single_step=False, **kwargs):
    """Single training step."""
    if hooks is None:
        hooks = {}
    lm_loss_total, metrics_total, count = 0.0, {}, 0
    forward_step = hooks['forward_step']

    while True:
        # Forward model for one step.
        timers('forward').start()
        forward_ret = forward_step(data_iterator, model, args, timers, **kwargs)
        if isinstance(forward_ret, tuple):
            lm_loss, metrics = forward_ret
        else:
            lm_loss, metrics = forward_ret, {}
        timers('forward').stop()

        # Check nan or inf in forward, preventing it from interfering loss scaler,
        # and all reduce metrics by the way
        lm_loss_reduced = lm_loss.detach().clone()
        torch.distributed.all_reduce(lm_loss_reduced.data)
        lm_loss_reduced.data = lm_loss_reduced.data / args.world_size

        loss_checker = lm_loss_reduced
        for name in metrics:
            if not 'eval' in name:
                metrics[name] = metrics[name].detach().clone()
                torch.distributed.all_reduce(metrics[name].data)
                metrics[name].data /= args.world_size
                loss_checker = loss_checker + metrics[name]
        if loss_checker.isnan().any() or loss_checker.isinf().any():
            print('Skipping backward and optimizer step for nan or inf in forwarding metrics/loss!')
            return lm_loss.detach(), 1, metrics

        # Accumulate the statistics
        lm_loss_total += lm_loss_reduced
        for name in metrics:
            if name not in metrics_total:
                metrics_total[name] = 0.0
            metrics_total[name] += metrics[name]
        count += 1
        # Calculate gradients, reduce across processes, and clip.
        timers('backward').start()
        backward_step(optimizer, model, lm_loss, args, timers)
        timers('backward').stop()
        # Update parameters.
        skipped_iter, complete = 0, False
        timers('optimizer').start()
        if args.deepspeed:
            if model.is_gradient_accumulation_boundary():
                model.step()
                complete = True
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
            else:
                model.step()
        else:
            raise ValueError('Currently, we only support training with deepspeed.')
        timers('optimizer').stop()
        if complete or single_step:
            break
    lm_loss_total /= count
    metrics_total = {key: value / count for key, value in metrics_total.items()}
    return lm_loss_total, skipped_iter, metrics_total


def backward_step(optimizer, model, loss, args, timers):
    """Backward step."""

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        raise ValueError('Currently, we only support training with deepspeed.')

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()

    return

def evaluate(data_iterator, model, eval_iters, args, timers, split, verbose=False, has_last=True, hooks={}):
    """Evaluation."""
    forward_step = hooks['forward_step']
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    total_lm_loss, metrics_total = 0, {}
    if split=='val':
        last_shape = args.val_last_shape
        drop_number = args.val_drop_number
    else:
        assert split=='test'
        last_shape = args.test_last_shape
        drop_number = args.test_drop_number
    is_scalar = {}
    with torch.no_grad():
        iteration = 0
        while iteration < eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, eval_iters))
            # Forward evaluation.
            # try:
            lm_loss, metrics = forward_step(data_iterator, model, args, timers)
            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()
            total_lm_loss += lm_loss.data.detach().float().item()
            is_last = True if iteration == eval_iters and args.strict_eval and len(last_shape)>0 else False
            for name in metrics:
                if name not in metrics_total:
                    metrics_total[name] = []
                is_scalar[name] = True if len(metrics[name].shape)==0 else False
                shape = list(metrics[name].shape)
                if not is_scalar[name] and is_last and metrics[name].shape[0] != last_shape[0]:
                    # pad tensor's first dim to args.batch_size
                    metrics[name] = torch.concat([metrics[name], torch.zeros([last_shape[0]-metrics[name].shape[0]] + shape[1:], dtype=metrics[name].dtype, device=metrics[name].device)])
                if rank==0:
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                else:
                    # metrics_gathered = None
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                # torch.distributed.gather(metrics[name], metrics_gathered, 0)
                torch.distributed.all_gather(metrics_gathered, metrics[name])

                if rank==0:
                    gathered_len = len(metrics_gathered) if not is_last else len(metrics_gathered) - drop_number
                    for i in range(gathered_len):
                        if is_scalar[name] or not is_last:
                            metrics_total[name].append(metrics_gathered[i].data.cpu())
                        else:
                            metrics_total[name].append(metrics_gathered[i][:last_shape[i]].data.cpu())
    # Move model back to the train mode.
    model.train()

    total_lm_loss /= eval_iters
    # metrics_avg = {key: value / eval_iters for key, value in metrics_total.items()}
    if rank==0:
        for name in metrics_total:
            if is_scalar[name]:
                metrics_total[name] = torch.stack(metrics_total[name], dim=0)
            else:
                metrics_total[name] = torch.concat(metrics_total[name], dim=0)
        if hooks['handle_metrics'] is not None:
            metrics = hooks['handle_metrics'](metrics_total)
        else:
            for name in metrics_total:
                assert is_scalar[name], 'you must return scalar metrics or implement handle_metrics hooks'
            metrics = {key: sum(value.split(1,0))/len(value) for key, value in metrics_total.items()}
    else:
        metrics = None
    return total_lm_loss, metrics

def evaluate_and_print_results(prefix, data_iterator, model, eval_iters,
                            args, timers, has_last, split, verbose=False, step=None, summary_writer=None, hooks={}):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, metrics = evaluate(data_iterator, model, eval_iters, args, timers, split, verbose, has_last, hooks=hooks)
    lm_ppl = math.exp(min(20, lm_loss))
    if torch.distributed.get_rank(group=mpu.get_data_parallel_group())==0:
        report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, step, metrics)
    return lm_loss


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args, avg_metrics):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' total loss {:.6E} |'.format(loss)
    for key in avg_metrics:
        log_string += ' {} {:.6E} |'.format(key, avg_metrics[key])
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/'+key, avg_metrics[key], step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, step, avg_metrics):
    string = ' validation loss at {} | '.format(prefix)
    string += 'loss: {:.6E} | '.format(loss)
    string += 'PPL: {:.6E}'.format(ppl)
    for key in avg_metrics:
        string += ' {} {:.6E} |'.format(key, avg_metrics[key].item())
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/valid_'+key, avg_metrics[key], step)
        