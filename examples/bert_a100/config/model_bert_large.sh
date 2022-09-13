MODEL_TYPE="bert-large-uncased"
MODEL_ARGS="--num-layers 24 \
            --vocab-size 30522 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 512 \
            --hidden-dropout 0.1 \
            --attention-dropout 0.1 \
            --load ${CHECKPOINT_PATH}/$MODEL_TYPE"