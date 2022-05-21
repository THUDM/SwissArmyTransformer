#! /bin/bash

CHECKPOINT_PATH=$1
if [[ "$1" == "" ]];
then
    echo "Please pass in root folder to save model!"
    exit
fi
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
source $main_dir/config/model_clip.sh


run_cmd="python inference_clip.py --pretrain_path $PRETRAIN_PATH --mode inference $MODEL_ARGS"
echo ${run_cmd}
eval ${run_cmd}

set +x
