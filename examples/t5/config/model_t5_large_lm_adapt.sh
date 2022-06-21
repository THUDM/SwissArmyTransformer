MODEL_TYPE="t5-large-lm-adapt"
MODEL_ARGS="--vocab-size 32128 \
            --num-layers 24 \
            --hidden-size 1024 \
            --inner-hidden-size 2816 \
            --num-attention-heads 16 \
            --hidden-size-per-attention-head 64 \
            --relative-attention-num-buckets 32 \
            --no-share-embeddings \
            --gated-gelu-mlp \
            --layernorm-epsilon 1e-6 \
            --tokenizer-type hf_T5Tokenizer \
            --tokenizer-model-type t5-large \
            --load ${CHECKPOINT_PATH}/t5-large-lm-adapt"