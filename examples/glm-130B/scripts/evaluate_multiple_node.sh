#!/bin/bash

NUM_WORKERS=15
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="/thudm/LargeScale/wudao-1"
OPTIONS_NCCL="NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 CUDA_LAUNCH_BLOCKING=0"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/config/model_glm_130B.sh"

ARGS="./evaluate.py \
       --mode inference \
       --data-path /thudm/LargeScale/data/zeroshot/ \
       --task $* \
       $MODEL_ARGS"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${MODEL_TYPE}-${TIMESTAMP}
mkdir -p logs/${EXP_NAME}

run_cmd="PYTHONPATH=/thudm/LargeScale/SwissArmyTransformer ${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} ${ARGS}"
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}.log
