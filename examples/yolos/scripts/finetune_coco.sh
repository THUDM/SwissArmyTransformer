#! /bin/bash

CHECKPOINT_PATH=$1
if [[ "$1" == "" ]];
then
    echo "Please pass in root folder to save model!"
    exit
fi

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=2
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="yolos-tiny"
MODEL_ARGS="--image-size 800 1333 \
            --pre-len 1 \
            --post-len 100 \
            --num-det-tokens 100 \
            --num-det-classes 92
"

OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="train"
eval_data="val"


gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-coco \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 10000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 1000 \
       --eval-interval 100 \
       --save "$CHECKPOINT_PATH/checkpoints" \
       --split 1 \
       --strict-eval \
       --eval-batch-size 32 \
       --zero-stage 0 \
       --lr 0.00002 \
       --batch-size 4 \
       --md_type deit-tiny
"
              

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} train_yolos_coco.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
