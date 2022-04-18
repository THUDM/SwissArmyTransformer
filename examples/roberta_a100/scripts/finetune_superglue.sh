#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/roberta

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
source $main_dir/config/model_roberta_large.sh
echo $MODEL_TYPE

task_name=$1
seed=$2
gpu=$3
lr=$4


OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

dataset_name="$task_name"
if [[ "$task_name" == "wsc" ]]; then
  dataset_name="wsc.fixed"
fi

en_data="hf://super_glue/${dataset_name}/train"
eval_data="hf://super_glue/${dataset_name}/validation"

config_json="$script_dir/ds_config_${seed}.json"

finetune_type="froze"

gpt_options=" \
       --finetune-type ${finetune_type} \
       --experiment-name finetune-$MODEL_TYPE-${dataset_name}-${finetune_type}-lr${lr}-seed${seed}- \
       --summary-dir runs/finetune-$MODEL_TYPE-${dataset_name}-${finetune_type} \
       --cls-number 1 \
       --collect-len 2 \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 14000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --fp16 \
       --eval-interval 200 \
       --save checkpoints/ \
       --split 1 \
       --eval-batch-size 2 \
       --warmup 0.1 \
       --valid-data ${eval_data} \
       --strict-eval \
       --dataset-name ${dataset_name} \
       --warmup 0.1 \
       --save-interval 4000 \
       --seed ${seed} \
       --save-args \
"

#2step
gpt_options="${gpt_options}
        --step1-lr 1e-3 \
        --step1-iters 4000 \
"


#       --child-load /workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/finetune-roberta-large-boolq-pt-7e-3-seed408805958-03-25-13-25 \
#child part
gpt_options="${gpt_options}
       --child-type ChildTuning-D \
       --reserve-p 0.3 \
       --max-grad-norm 1.0 \
"

#load head part

# --head-load \
gpt_options="${gpt_options}
       --head-path /dataset/fd5061f6/yzy/roberta_v100/checkpoints/finetune-roberta-large-boolq-bitfit-1e-3-seed7549048-03-25-13-22 \
"
#       --body-path /dataset/fd5061f6/yzy/roberta_v100/checkpoints/finetune-roberta-large-boolq-all-1e-5-seed465049921-loadbithead-03-27-01-18 \

gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"

((port=$RANDOM+10000))

#if [ "$FINETUNE_GPU" ]; then
#  echo "use gpu $FINETUNE_GPU"
#else
#  export FINETUNE_GPU=0
#  echo "use gpu $FINETUNE_GPU"
#fi

run_cmd="${OPTIONS_NCCL} deepspeed --include=localhost:$gpu --master_port ${port} --hostfile ${HOST_FILE_PATH} finetune_roberta.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
