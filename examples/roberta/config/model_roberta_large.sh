MODEL_TYPE="swiss-roberta-large"
MODEL_ARGS="--num-layers 24 \
            --vocab-size 50265 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 514 \
            --hidden-dropout 0.1 \
            --attention-dropout 0.1 \
            --checkpoint-activations \
            --checkpoint-num-layers 1 \
	    --post-ln \
            --load ${CHECKPOINT_PATH}/$MODEL_TYPE"
