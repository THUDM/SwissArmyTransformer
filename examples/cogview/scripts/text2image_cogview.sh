#!/bin/bash

CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/cogview/cogview-base
NLAYERS=48
NHIDDEN=2560
NATT=40
MAXSEQLEN=1089
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MPSIZE=1

#SAMPLING ARGS
TEMP=1.03
TOPK=200

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

MASTER_PORT=${MASTER_PORT} python inference_cogview.py \
       --tokenizer-type cogview \
       --img-tokenizer-path /dataset/fd5061f6/sat_pretrained/vqvae/l1+ms-ssim+revd_percep.pt \
       --mode inference \
       --distributed-backend nccl \
       --max-sequence-length 1089 \
       --sandwich-ln \
       --fp16 \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --temperature $TEMP \
       --top_k $TOPK \
       --sandwich-ln \
       --input-source ./input.txt \
       --output-path samples_text2image \
       --batch-size 4 \
       --max-inference-batch-size 8 \
       $@


