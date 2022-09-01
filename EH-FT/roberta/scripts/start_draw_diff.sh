#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
source $main_dir/config/model_roberta_large.sh
echo $MODEL_TYPE

task_name=$1

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

dataset_name="$task_name"
if [[ "$task_name" == "wsc" ]]; then
  dataset_name="wsc.fixed"
fi

en_data="hf://super_glue/${dataset_name}/train"
eval_data="hf://super_glue/${dataset_name}/validation"

config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-${dataset_name}-\
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 16000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${en_data} \
       --distributed-backend nccl \
       --lr-decay-style linear \
       --checkpoint-activations \
       --fp16 \
       --eval-interval 100 \
       --save checkpoints/ \
       --split 1 \
       --eval-batch-size 2 \
       --warmup 0.1 \
       --valid-data ${eval_data} \
       --strict-eval \
       --save-interval 4000\
"


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"

((port=$RANDOM+10000))

if [ "$FINETUNE_GPU" ]; then
  echo "use gpu $FINETUNE_GPU"
else
  export FINETUNE_GPU=0
  echo "use gpu $FINETUNE_GPU"
fi

run_cmd="${OPTIONS_NCCL} deepspeed --include=localhost:$FINETUNE_GPU --master_port ${port} --hostfile ${HOST_FILE_PATH} child_ex/draw_para_diff.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
