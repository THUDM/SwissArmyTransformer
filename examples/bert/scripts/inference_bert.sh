#! /bin/bash

# Change for multinode config
CHECKPOINT_PATH=/data/qingsong/pretrain
MODEL_TYPE="swiss-bert-base-uncased"
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE"

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

run_cmd="python inference_bert.py --pretrain_path $CHECKPOINT_PATH --mode inference $MODEL_ARGS"
echo ${run_cmd}
eval ${run_cmd}

set +x
