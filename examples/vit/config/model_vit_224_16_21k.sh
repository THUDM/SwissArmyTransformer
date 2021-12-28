MODEL_TYPE="vit-base-224-16-21k"
MODEL_ARGS="
            --image-size 224 \
            --patch-size 16 \
            --vocab-size 1 \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --in-channels 3 \
            --num-classes 21843 \
            --num-finetune-classes 10 \
            --max-sequence-length 197 \
            --tokenizer-model-type roberta \
            --tokenizer-type glm_GPT2BPETokenizer \
            --new-sequence-length 197 \
            --load ${CHECKPOINT_PATH}/swiss-vit-base-patch16-224-in21k"