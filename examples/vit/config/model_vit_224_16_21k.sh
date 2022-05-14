MODEL_TYPE="vit-base-224-16-21k"
MODEL_ARGS="--load ${CHECKPOINT_PATH}/swiss-vit-base-patch16-224-in21k \
            --num-finetune-classes 10 \
            --image-size 384 384"
