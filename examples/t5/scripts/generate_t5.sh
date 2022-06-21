#!/bin/bash
CHECKPOINT_PATH=/sharefs/english/yanan/huggingface_models

source $1
MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_t5_large.json"

python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT inference_t5.py \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --mode inference \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --num-beams 4 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.7 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm \
       --batch-size 2 \
       --out-seq-length 200 \
       --mode inference \
       --input-source ./input.txt \
       --checkpoint-activations \
       --sampling-strategy BeamSearchStrategy
