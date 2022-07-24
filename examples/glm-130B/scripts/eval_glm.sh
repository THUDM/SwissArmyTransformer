#!/bin/bash
CHECKPOINT_PATH=/thudm/workspace/hanyu/SwissArmyTransformer-1/data/ckpt

source $1
MPSIZE=8
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_config.json"

python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT evaluation/main.py \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.5 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm \
       --batch-size 4 \
       --out-seq-length 200 \
       --sampling-strategy BeamSearchStrategy \
       --micro-batch-size 1 \
       --num-workers 1 \
       --mode inference \
       --task task-test \
       --eval-data-path /thudm/workspace/hanyu/SwissArmyTransformer/examples/glm-130B
