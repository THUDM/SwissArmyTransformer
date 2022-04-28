MODEL_TYPE="swiss-bert-base-uncased"
MODEL_ARGS="--num-layers 12 \
            --vocab-size 30522 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-sequence-length 512 \
            --num-types 2 \
            --hidden-dropout 0.1 \
            --attention-dropout 0.1 \
            --checkpoint-activations \
            --checkpoint-num-layers 1 \
            --load ${CHECKPOINT_PATH}/$MODEL_TYPE"