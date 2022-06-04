MODEL_TYPE="swiss-deit-base"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/eac58c838baa4a61902c/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip"
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE \
            --num-finetune-classes 10 \
            --image-size 384 384"
