MODEL_TYPE="blocklm-130B"
MODEL_ARGS="--block-lm \
            --cloze-eval \
            --task-mask \
            --num-layers 69 \
            --hidden-size 12288 \
            --inner-hidden-size 32768 \
            --vocab-size 150528 \
            --num-attention-heads 96 \
            --max-sequence-length 1025 \
            --tokenizer-model-type gpt2 \
            --tokenizer-type icetk \
            --load ${CHECKPOINT_PATH}/iter_0010400"
