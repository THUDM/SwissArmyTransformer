MODEL_TYPE="glm-130B"
CHECKPOINT_PATH=/thudm/workspace/hanyu/SwissArmyTransformer/data/ckpt
MODEL_ARGS="--model-parallel-size 8 \
            --num-layers 70 \
            --hidden-size 12288 \
            --inner-hidden-size 32768 \
            --vocab-size 150528 \
            --num-attention-heads 96 \
            --max-sequence-length 2048 \
            --tokenizer-type icetk-glm-130B \
            --layernorm-order post \
            --load ${CHECKPOINT_PATH}/iter_0049300 \
            --use-gpu-initialization \
            --skip-init \
            --fp16"
