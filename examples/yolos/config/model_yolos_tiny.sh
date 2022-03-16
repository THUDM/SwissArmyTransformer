MODEL_TYPE="yolos-tiny"
MODEL_ARGS="--image-size 800 1333 \
            --pre-len 1 \
            --post-len 100 \
            --patch-size 16 \
            --vocab-size 1 \
            --num-layers 12 \
            --hidden-size 192 \
            --num-attention-heads 3 \
            --in-channels 3 \
            --num-det-tokens 100 \
            --num-det-classes 92 \
            --tokenizer-model-type roberta \
            --tokenizer-type Fake \
            --attention-dropout 0. \
            --hidden-dropout 0. \
            --load ${CHECKPOINT_PATH}/swiss-deit-tiny \
            --old-image-size 224 224 \
            --old-pre-len 1 \
            --old-post-len 0"
