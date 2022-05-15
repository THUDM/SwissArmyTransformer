#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=$1
if [[ "$1" == "" ]] || [[ "$2" == "" ]];
then
    echo "Please pass in two root folders to save model and data!"
    exit
fi
MODEL_TYPE="swiss-bert-base-uncased"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/9b4ab7c17ce842ea9c9d/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip"
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE"

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
echo $MODEL_TYPE

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

en_data="hf://super_glue/boolq/train"
eval_data="hf://super_glue/boolq/validation"
test_data="hf://super_glue/boolq/test"

config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-boolq \
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
       --checkpoint-num-layers 1 \
       --fp16 \
       --save-interval 1000 \
       --eval-interval 100 \
       --save checkpoints/ \
       --split 1 \
       --strict-eval \
       --eval-batch-size 8 \
       --do-train \
       --data_root $2
"



gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"


run_cmd="${OPTIONS_NCCL} deepspeed --include localhost:6,7,8,9 --hostfile ${HOST_FILE_PATH} finetune_bert_boolq.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x