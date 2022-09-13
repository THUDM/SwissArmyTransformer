#! /bin/bash

# Change for multinode config

if [[ "$PLATFORM" ==  "wudao" ]]; then
  CHECKPOINT_PATH=/sharefs/cogview-new/yzy/bert
else
  #CHECKPOINT_PATH=/thudm/workspace/yzy/roberta
  CHECKPOINT_PATH=/thudm/workspace/guoyanhui/SwissArmyTransformer-main/models/bert

fi


NUM_WORKERS=1  # 机器个数
NUM_GPUS_PER_WORKER=1  #一个机器有多少张卡
MP_SIZE=1    # 机器个数*卡数=并行数
# $0 代表脚本本身，realpath表示获取当前工作目录的绝对路径的命令 $()表示命令替换,将命令的结果放到该位置
script_path=$(realpath $0)
# dirname 获取指定路径的上一级目录的命令
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
# source 在当前bash环境下读取并执行FileName中的命令
source $main_dir/config/model_bert_large.sh
# MODEL_TYPE为上个开启的进程里的变量，表示采用的是哪个模型
echo $MODEL_TYPE
# $1 代表传给shell的第一个参数, $0代表脚本本身
task_name=$1
seed=$2
gpu=$3
lr=$4
epochs=$5
step1_epochs=$6
type=$7
result_dir=$9

# 设置变量
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

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

# 数据集的位置
en_data="hf://${hf_path}/${dataset_name}/train"
eval_data="hf://${hf_path}/${dataset_name}/validation"
# 配置文件的路径 加速器的配置文件
config_json="$script_dir/ds_config_${seed}.json"

finetune_type="$type"

gpt_options=" \
      --finetune-type ${finetune_type} \
      --name-model $MODEL_TYPE \
      --experiment-name finetune-$MODEL_TYPE-${task_name}-${finetune_type}-lr${lr}-seed${seed}- \
      --summary-dir runs/finetune-$MODEL_TYPE-${task_name}-${finetune_type} \
      --cls-number 1 \
      --collect-len 2 \
      --model-parallel-size ${MP_SIZE} \
      --mode finetune \
      --epochs ${epochs} \
      --resume-dataloader \
      $MODEL_ARGS \
      --train-data ${en_data} \
      --distributed-backend nccl \
      --lr-decay-style linear \
      --save checkpoints/ \
      --split 1 \
      --save-interval 4000 \
      --eval-batch-size 32 \
      --num-workers 0 \
      --warmup 0.1 \
      --valid-data ${eval_data} \
      --strict-eval \
      --dataset-name ${dataset_name} \
      --seed ${seed} \
      --save-args \
      --lora-r 128 \
      --prefix_len 64 \
      --fp16 \
"
#fp16

#ffadd part
gpt_options="${gpt_options}
        --ffadd-r 32 \
"

#2step
STEP1LR="5e-4"
if [[ "$finetune_type" == "2step+lora" || "$finetune_type" == "2step+ffadd" ]]; then
  STEP1LR="5e-4"
fi

if [[ "$finetune_type" == "2step+pt" ]]; then
  STEP1LR="5e-3"
fi

gpt_options="${gpt_options}
        --step1-lr ${STEP1LR} \
        --step1-epochs ${step1_epochs} \
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
       --head-path  /workspace/yzy/ST_deve/SwissArmyTransformer/examples/roberta_v100/checkpoints/finetune-roberta-large-copa-pt-lr0.005-seed55883230-04-19-03-42 \
"
#       --body-path /dataset/fd5061f6/yzy/  roberta_v100/checkpoints/finetune-roberta-large-boolq-all-1e-5-seed465049921-loadbithead-03-27-01-18 \

gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"


# RANDOM 产生随机数
((port=$RANDOM+10000))

#if [ "$FINETUNE_GPU" ]; then
#  echo "use gpu $FINETUNE_GPU"
#else
#  export FINETUNE_GPU=0
#  echo "use gpu $FINETUNE_GPU"
#fi

#run_cmd="${OPTIONS_NCCL} deepspeed --include=localhost:$FINETUNE_GPU --master_port ${port} --hostfile ${HOST_FILE_PATH} finetune_bert_${task_name}.py ${gpt_options}"
#
run_cmd="${OPTIONS_NCCL} deepspeed --include=localhost:${gpu} --master_port ${port} --hostfile ${HOST_FILE_PATH} finetune_bert.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}
# set +x 关闭脚本调试，set -x开启脚本调试， set -o 查看脚本调试
set +x
