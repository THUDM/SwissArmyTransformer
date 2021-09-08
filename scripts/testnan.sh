#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 NCCL_IB_GID_INDEX=3 NCCL_NET_GDR_LEVEL=0"
OPTIONS_NCCL="NCCL_DEBUG=info"
HOST_FILE_PATH="hostfile_single"

small_data="/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_3leveltokens/zijian/zijian.bin.part_0.cogdata"
full_data="/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_3leveltokens/merge.bin"

config_json="$script_dir/ds_config.json"
gpt_options=" \
       --experiment-name cogview-testlocal \
       --img-tokenizer-num-tokens 8192 \
       --dataset-type CompactBinaryDataset \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --save $main_dir/data/checkpoints \
       --train-iters 100000 \
       --resume-dataloader \
       --test-data ${full_data} \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --max-position-embeddings 1089 \
       --max-memory-length 0 \
       --txt-loss-scale 1 \
       --sandwich-ln \
       --sparse-type standard \
       --save-interval 2500 \
       --fp16 \
       --eval-iters 1000 \
       --load pretrained/cogview/cogview-base
"
       # 
              # --load data/checkpoints/cogview-fixgrad-small08-25-09-38


gpt_options="${gpt_options}

"
            #    --deepspeed \
            #    --deepspeed_config ${config_json} \

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
