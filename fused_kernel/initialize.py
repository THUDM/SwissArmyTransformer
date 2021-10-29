# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron initialization."""

import random
import os
import time

import numpy as np
import torch
from datetime import timedelta

from . import fused_kernels


def initialize_fused_kernel(args=None):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only 
    data processing. In general this arg should not be set unless you know 
    what you are doing.
    Returns a function to finalize distributed env initialization 
    (optionally, only when args.lazy_mpu_init == True)
    """

    # Set pytorch JIT layer fusion options.
    _set_jit_fusion_options()
    
    # Compile dependencies.
    _compile_dependencies(args)

    # No continuation function
    return None


def _compile_dependencies(args):
    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    # seq_len = args.seq_length
    # attn_batch_size = \
    #     (args.num_attention_heads / args.tensor_model_parallel_size) * \
    #     args.micro_batch_size
    # # Constraints on sequence length and attn_batch_size to enable warp based
    # # optimization and upper triangular optimization (for causal mask)
    # custom_kernel_constraint = seq_len > 16 and seq_len <=2048 and \
    #     seq_len % 4 == 0 and attn_batch_size % 4 == 0
    # # Print a warning.
    # if not ((args.fp16 or args.bf16) and
    #         custom_kernel_constraint and
    #         args.masked_softmax_fusion):
    #     if args.rank == 0:
    #         print('WARNING: constraints for invoking optimized'
    #               ' fused softmax kernel are not met. We default'
    #               ' back to unfused kernel invocations.', flush=True)
    
    # Always build on rank zero first.
    if args.rank == 0:
        start_time = time.time()
        print('> compiling and loading fused kernels ...', flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if args.rank == 0:
        print('>>> done with compiling and loading fused kernels. '
              'Compilation time: {:.3f} seconds'.format(
                  time.time() - start_time), flush=True)


def _set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

