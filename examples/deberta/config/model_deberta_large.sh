MODEL_TYPE="deberta-large"
MODEL_ARGS="--num-layers 24 \
            --vocab-size 50265 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 512 \
            --hidden-dropout 0.1 \
            --attention-dropout 0.1 \
            --layernorm-epsilon 1e-7 \
            --max-relative-positions -1 \
            --checkpoint-activations \
            --checkpoint-num-layers 12 \
            --load ${CHECKPOINT_PATH}/$MODEL_TYPE"