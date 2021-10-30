#!/bin/bash
CHECKPOINT_PATH=/dataset/fd5061f6/sat_pretrained/glm

# MODEL_ARGS="--block-lm \
#             --cloze-eval \
#             --num-layers 24 \
#             --hidden-size 1024 \
#             --num-attention-heads 16 \
#             --max-sequence-length 513 \
#             --tokenizer-model-type roberta \
#             --tokenizer-type glm_GPT2BPETokenizer \
#             --load ${CHECKPOINT_PATH}/glm-roberta-large-blank"

#MODEL_TYPE="blocklm-10B"
#MODEL_ARGS="--block-lm \
#            --cloze-eval \
#            --task-mask \
#            --num-layers 48 \
#            --hidden-size 4096 \
#            --num-attention-heads 64 \
#            --max-sequence-length 1025 \
#            --tokenizer-model-type gpt2 \
#            --tokenizer-type glm_GPT2BPETokenizer \
#            --old-checkpoint \
#            --load ${CHECKPOINT_PATH}/glm-en-10b"

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

config_json="$script_dir/ds_config.json"

python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT inference_glm.py \
       --mode inference \
       --model-parallel-size $MPSIZE \
       $MODEL_ARGS \
       --num-beams 4 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.7 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm \
       --batch-size 1 \
       --out-seq-length 200 \
       --mode inference
