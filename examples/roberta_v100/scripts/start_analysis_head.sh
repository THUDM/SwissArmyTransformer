#! /bin/bash
# /sharefs/cognitive/workspace/cogview/43c333e5af5edd3666bcd54caecee8fe/yzy/
# Change for multinode config
CHECKPOINT_PATH=/sharefs/cogview-new/yzy/roberta

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

dataset_name="$task_name"
if [[ "$task_name" == "wsc" ]]; then
  dataset_name="wsc.fixed"
fi

hf_path="super_glue"
if [[ "$task_name" == "cola" || "$task_name" == "sst2" || "$task_name" == "qqp" || "$task_name" == "mrpc" || "$task_name" == "stsb" || "$task_name" == "mnli" || "$task_name" == "qnli" || "$task_name" == "wnli" ]]; then
    hf_path="glue"
fi
if [[ "$task_name" == "squad" ]]; then
  hf_path="squad"
  dataset_name="plain_text"
fi
if [[ "$task_name" == "squad_v2" ]]; then
  hf_path="squad_v2"
fi
if [[ "$task_name" == "conll2003" ]]; then
  hf_path="conll2003"
fi
if [[ "$task_name" == "emotion" ]]; then
  dataset_name="default"
  hf_path="emotion"
fi
if [[ "$task_name" == "wsc" ]]; then
  dataset_name="wsc.fixed"
fi

en_data="hf://${hf_path}/${dataset_name}/train"
eval_data="hf://${hf_path}/${dataset_name}/validation"

config_json="$script_dir/ds_config_ft.json"
gpt_options=" \
       --experiment-name ana_head-$MODEL_TYPE-${dataset_name}-\
       --dataset-name ${dataset_name} \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --epochs 1 \
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
       --eval-batch-size 32 \
       --warmup 0.1 \
       --valid-data ${eval_data} \
       --strict-eval \
       --save-interval 4000 \
       --ssl_load2 /workspace/yzy/ST_develop/SwissArmyTransformer/examples/roberta_test/checkpoints/finetune-roberta-large-boolq-baseline-1e-5-03-09-10-25 \
       --seed 1 \
"

run_cmd="python analysis_ffadd.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
