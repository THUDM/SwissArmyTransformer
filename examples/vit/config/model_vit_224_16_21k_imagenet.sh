MODEL_TYPE="vit-base-224-16-21k"
MODEL_ARGS="--image-size 384 384 \
            --patch-size 16 \
            --vocab-size 1 \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --in-channels 3 \
            --num-classes 21843 \
            --num-finetune-classes 1000 \
            --tokenizer-model-type roberta \
            --tokenizer-type Fake \
            --attention-dropout 0. \
            --hidden-dropout 0. \
            --load ${CHECKPOINT_PATH}/swiss-vit-base-patch16-224-in21k \
            --old-image-size 224 224 \
            --old-pre-len 1 \
            --old-post-len 0"
