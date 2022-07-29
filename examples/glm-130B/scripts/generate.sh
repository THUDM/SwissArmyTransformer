#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/config/model_glm_130B.sh"

MAX_OUTPUT_LENGTH=256
# BeamSearch args
NUM_BEAMS=4
LENGTH_PENALTY=1.0
NO_REPEAT_NGRAM=3
# Greedy args
TEMP=0.9
TOPK=40
TOPP=0

ARGS="${main_dir}/generate.py \
       --mode inference \
       --sampling-strategy BeamSearchStrategy \
       --num-beams $NUM_BEAMS \
       --no-repeat-ngram-size $NO_REPEAT_NGRAM \
       --length-penalty $LENGTH_PENALTY \
       --out-seq-length $MAX_OUTPUT_LENGTH \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --output-path samples_glm \
       --input-source ./input.txt \
       --print-all-beams \
       $MODEL_ARGS"

run_cmd="PYTHONPATH=/thudm/LargeScale/SwissArmyTransformer torchrun --nproc_per_node $MP_SIZE ${ARGS}"
eval ${run_cmd}
