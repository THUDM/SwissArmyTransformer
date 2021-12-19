MODEL_TYPE="t5-xxl"
MODEL_ARGS="--t5-model \
            --vocab-size 32128 \
            --num-layers 24 \
            --hidden-size 4096 \
            --inner-hidden-size 10240 \
            --num-attention-heads 64 \
            --hidden-size-per-attention-head 64 \
            --relative-attention-num-buckets 32 \
            --no-share-embeddings \
            --gated-gelu-mlp \
            --layernorm-epsilon 1e-6 \
            --tokenizer-type hf_T5Tokenizer \
            --tokenizer-model-type t5-large \
            --load /dataset/fd5061f6/yanan/huggingface_models/t5-xxl-lm-adapt"