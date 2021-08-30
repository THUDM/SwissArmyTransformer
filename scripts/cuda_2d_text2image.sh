#!/bin/bash

CHECKPOINT_PATH=data/checkpoints/cogview-fixgrad-small08-25-09-38
# CHECKPOINT_PATH=data/checkpoints/cogview-compare
NLAYERS=16
NHIDDEN=1024
NATT=16
MAXSEQLEN=5184
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.05
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=100
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

MASTER_PORT=${MASTER_PORT} python generate_samples.py \
       --deepspeed \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 5184 \
       --fp16 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --sandwich-ln \
       --img-tokenizer-path pretrained/vqvae/vqvae_hard_biggerset_011.pt \
       --sparse-type standard \
       --max-position-embeddings-finetune $MAXSEQLEN \
       --generation-task "cuda-2d generation" \
       --input-source ./input.txt \
       --output-path samples_text2image \
       --batch-size 2 \
       --max-inference-batch-size 4 \
       --device 0 \
       --sparse-type standard \
       $@


