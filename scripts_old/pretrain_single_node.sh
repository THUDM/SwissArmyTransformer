#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"
OPTIONS_NCCL="NCCL_DEBUG=info"
HOST_FILE_PATH="hostfile_single"


config_json="$script_dir/ds_config_zero.json"
gpt_options=" \
       --experiment-name cogview-testlocal \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type BinaryDataset \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --save $main_dir/data/checkpoints \
       --train-iters 100000 \
       --resume-dataloader \
       --train-data /dataset/fd5061f6/cogview/cogdata_new/cogdata_task_3leveltokens/merge.bin \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 5184 \
       --max-memory-length 0 \
       --fp16 \
       --txt-loss-scale 2 \
       --sandwich-ln \
       --sparse-type cuda_2d \
       --save-interval 2500 
"

gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
