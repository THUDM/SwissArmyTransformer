#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/data/qingsong/pretrain

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
source $main_dir/config/model_bert_base.sh

run_cmd="python inference_bert.py --pretrain_path $CHECKPOINT_PATH --mode inference $MODEL_ARGS"
echo ${run_cmd}
eval ${run_cmd}

set +x
