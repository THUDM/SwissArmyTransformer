#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=$1
if [[ "$1" == "" ]] || [[ "$2" == "" ]];
then
    echo "Please pass in two root folders to save model and data!"
    exit
fi

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="swiss-vit-base-patch16-224-in21k"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/1381d0995d5e4634856e/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip"
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE \
            --num-finetune-classes 10 \
            --image-size 384 384"


OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="$2/train"
eval_data="$2/test"


config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-vit-cifar10 \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 1000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 6000 \
       --eval-interval 100 \
       --save /data/qingsong/checkpoints \
       --split 1 \
       --strict-eval \
       --eval-batch-size 8 \
       --lr 0.01 \
       --do-train
"



gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              

run_cmd="${OPTIONS_NCCL} deepspeed --master_port 16666 --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_vit_cifar10.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
