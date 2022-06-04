# coding=utf-8
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

"""argparser configuration"""

import argparse
import os
import torch
import deepspeed
import json


def add_model_config_args(parser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--num-layers', type=int, default=24,
                       help='num decoder layers')
    group.add_argument('--hidden-size', type=int, default=1024,
                       help='transformer hidden size')
    group.add_argument('--num-attention-heads', type=int, default=16,
                       help='num of transformer attention heads')
    group.add_argument('--vocab-size', type=int, default=0,
                       help='vocab size for tokenization. the size of word_embeddings.')
    group.add_argument('--max-sequence-length', type=int, default=512,
                       help='maximum number of position embeddings to use')
    
    # ---------------  Optional hyper-parameters --------------- 

    # The inner-hidden-size in MLP, default "None" means 4*hidden-size
    group.add_argument('--inner-hidden-size', type=int, default=None)
    # The hidden-size-per-attention-head in Self and Cross Attention, 
    # default "None" means hidden-size/num-attention-heads.
    group.add_argument('--hidden-size-per-attention-head', type=int, default=None)
    # TODO: upgrade it to pre-ln/post-ln/sandwich-ln, remember final layernorm.
    group.add_argument('--sandwich-ln', action='store_true',
                       help='add sandwich ln in cogview.')
    # TODO: fully test it, support the generation.
    group.add_argument('--model-parallel-size', type=int, default=1,
                       help='size of the model parallel.')
    
    # ---------------  Inessential hyper-parameters --------------- 

    # Dropout and eps hyper-parameters
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='layer norm epsilon')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='dropout probability for hidden state transformer')
    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='dropout probability for attention weights')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    
    return parser



def add_training_args(parser):
    """Training arguments."""

    group = parser.add_argument_group('train', 'training configurations')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--experiment-name', type=str, default="MyModel",
                       help="The experiment name for summary and checkpoint."
                       "Will load the previous name if mode==pretrain and with --load ")
    group.add_argument('--train-iters', type=int, default=1000000,
                       help='total number of iterations to train over all training runs')
    group.add_argument('--batch-size', type=int, default=4,
                       help='batch size on a single GPU. batch-size * world_size = total batch_size.')
    group.add_argument('--lr', type=float, default=1.0e-4,
                       help='initial learning rate')
    group.add_argument('--mode', type=str,
                       default='pretrain',
                       choices=['pretrain', # from_scratch / load ckpt for continue pretraining.
                                'finetune', # finetuning, auto-warmup 100 iters, new exp name.
                                'inference' # don't train.
                                ],
                       help='what type of task to use, will influence auto-warmup, exp name, iteration')
    group.add_argument('--seed', type=int, default=1234, help='random seed')
    group.add_argument('--zero-stage', type=int, default=0, choices=[0, 1, 2], 
                        help='deepspeed ZeRO stage. 0 means no ZeRO.')

    # ---------------  Optional hyper-parameters --------------- 

    # Efficiency.
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='checkpoint activation to allow for training '
                            'with larger models and sequences. become slow (< 1.5x), save CUDA memory.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1, # Inessential
                       help='chunk size (number of layers) for checkpointing. ')
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode')
    group.add_argument('--bf16', action='store_true', # only A100 supports it. Not fully tested.
                       help='Run model in bf16 mode')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='run optimizer after every gradient-accumulation-steps backwards.')

    # Log.
    group.add_argument('--log-interval', type=int, default=50,
                       help='report interval')
    group.add_argument('--summary-dir', type=str, default="", help="The directory to store the summary")
    group.add_argument('--save-args', action='store_true',
                       help='save args corresponding to the experiment-name')

    # Learning rate & weight decay.
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay LR over,'
                            ' If None defaults to `--train-iters`*`--epochs`')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    
    group.add_argument('--warmup', type=float, default=0.01,
                       help='percentage of data to warmup on (.01 = 1% of all '
                            'training iters). Default 0.01')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='weight decay coefficient for L2 regularization')
    
    # model checkpointing
    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--load', type=str, default=None,
                       help='Path to a directory containing a model checkpoint.')
    group.add_argument('--save-interval', type=int, default=5000,
                       help='number of iterations between saves')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--resume-dataloader', action='store_true',
                       help='Resume the dataloader when resuming training. ') 

    # distributed training related, don't use them.
    group.add_argument('--distributed-backend', default='nccl',
                       help='which backend to use for distributed '
                            'training. One of [gloo, nccl]')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher')

    # exit, for testing the first period of a long training
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after this many new iterations.')
    # TODO
    group.add_argument('--do-train', action='store_true')
    
    return parser


def add_evaluation_args(parser):
    """Evaluation arguments."""

    group = parser.add_argument_group('validation', 'validation configurations')

    group.add_argument('--eval-batch-size', type=int, default=None,
                       help='Data Loader batch size for evaluation datasets.'
                            'Defaults to `--batch-size`')
    group.add_argument('--eval-iters', type=int, default=100,
                       help='number of iterations to run for evaluation'
                            'validation/test for')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='interval between running evaluation on validation set')
    group.add_argument('--strict-eval', action='store_true',
                       help='won\'t enlarge or randomly map eval-data, and eval full eval-data.')
    return parser


def add_data_args(parser):
    """Train/valid/test data arguments."""

    group = parser.add_argument_group('data', 'data configurations')

    # Training datasets.
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated filenames or corpora names for training.'
                       'Use hf://path/to/dataset to load huggingface datasets.')
    group.add_argument('--train-data-weights', nargs='+', default=None, type=int,
                        help='scaling factors for different train-data, must the same number of'
                        '--train-data or None(==1).')

    # Validation and Test dataset.
    group.add_argument('--valid-data', nargs='*', default=None,
                       help="""Filename for validation data.""")
    group.add_argument('--test-data', nargs='*', default=None,
                       help="""Filename for testing""")
    group.add_argument('--split', default='1000,1,1',
                       help='comma-separated list of proportions for training,'
                            ' validation, and test split. '
                            'Only take effects when no valid/test specified.')

    # Efficiency.
    group.add_argument('--num-workers', type=int, default=1,
                       help="""Number of workers to use for dataloading""")
    # Sometimes, num-workders > 1 and cpu_offload (zero-stage 2) will make the validation dataloader hang.
    # I have no idea on the reason, and just set the default num-workers to 1.
    group.add_argument('--block-size', type=int, default=10000,
                       help="""Size of block to reduce memory in dataset, ignore it for most users.""")

    return parser


def add_text_generate_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('Text generation', 'configurations')
    group.add_argument("--temperature", type=float, default=1.0)
    group.add_argument("--top_p", type=float, default=0.0)
    group.add_argument("--top_k", type=int, default=0)
    group.add_argument("--num-beams", type=int, default=1)
    group.add_argument("--length-penalty", type=float, default=0.0)
    group.add_argument("--no-repeat-ngram-size", type=int, default=0)
    group.add_argument("--min-tgt-length", type=int, default=0)
    group.add_argument("--out-seq-length", type=int, default=256)
    group.add_argument('--input-source', type=str, default='interactive',
                       help='what input mode to use, interactive or path')
    group.add_argument('--output-path', type=str, default='./samples',
                       help='path to place the generated samples')
    group.add_argument('--with-id', action='store_true',
                       help='If each line is prepended with an id.')
    group.add_argument('--max-inference-batch-size', type=int, default=12)
    group.add_argument('--device', type=int, default=-1)
    return parser


    
def add_tokenization_args(parser):
    """tokenization arguments."""

    group = parser.add_argument_group('Tokenization', 'tokenization configurations')
    group.add_argument('--tokenizer-type', type=str, default='fake', help='type name of tokenizer')
    group.add_argument('--tokenizer-model-type', type=str,
                       default=None,
                       help="Model type to use for sentencepiece tokenization \
                           (one of ['bpe', 'char', 'unigram', 'word']) or \
                           bert vocab to use for BertWordPieceTokenizer (one of \
                           ['bert-large-uncased', 'bert-large-cased', etc.])")
    group.add_argument('--img-tokenizer-path', type=str, default=None,
                       help='The checkpoint file path of image tokenizer.')
    return parser


def get_args(args_list=None):
    """Parse all the args."""

    parser = argparse.ArgumentParser(description='SwissArmyTransformer')
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_data_args(parser)
    parser = add_tokenization_args(parser)
    parser = add_text_generate_args(parser)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args(args_list)

    if not args.train_data:
        print('WARNING: No training data specified')

    args.cuda = torch.cuda.is_available()

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    
    if args.device == -1: # not set manually
        args.device = args.rank % torch.cuda.device_count()
        if args.local_rank is not None:
            args.device = args.local_rank

    # args.model_parallel_size = min(args.model_parallel_size, args.world_size)
    if args.rank == 0:
        print('using world size: {} and model-parallel size: {} '.format(
            args.world_size, args.model_parallel_size))
    
    if args.train_data_weights is not None:
        assert len(args.train_data_weights) == len(args.train_data)
    
    if args.mode != 'inference': # training with deepspeed
        args.deepspeed = True
        if args.deepspeed_config is None: # not specified
            args.deepspeed_config = os.path.join(os.path.dirname(__file__), 'training', f'deepspeed_zero{args.zero_stage}.json')
            override_deepspeed_config = True
        else:
            override_deepspeed_config = False
    if args.zero_stage > 0:
        if args.rank == 0 and not args.fp16:
            print('Automatically set fp16=True to use ZeRO.')     
        args.fp16 = True
        args.bf16 = False

    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            with open(args.deepspeed_config) as file:
                deepspeed_config = json.load(file)
            
        if override_deepspeed_config: # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else: # override args with values in deepspeed_config
            if args.rank == 0:
                print('Will override arguments with manually specified deepspeed_config!')
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
            else:
                args.gradient_accumulation_steps = None
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get("params", {})
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get("weight_decay", args.weight_decay)
        args.deepspeed_config = deepspeed_config
    return args


def update_args_with_file(args, path='model_config.json'):
    with open(os.path.join(args.load, path), 'r', encoding='utf-8') as f:
        config = json.load(f)
    args = vars(args)
    for k in list(args.keys()):
        if k in config:
            del args[k]
    args = argparse.Namespace(**config, **args)
    return args