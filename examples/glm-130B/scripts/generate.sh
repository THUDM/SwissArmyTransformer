#!/bin/bash

NUM_GPUS_PER_WORKER=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/config/model_glm_130B.sh"

#SAMPLING ARGS
TEMP=0.9
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_WORKER --master_port $MASTER_PORT \
    ${main_dir}/generate.py \
       --mode inference \
       --sampling-strategy BeamSearchStrategy \
       --num-beams 4 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.7 \
       --out-seq-length 256 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --output-path samples_glm \
       --input-source ./input.txt \
       $MODEL_ARGS
