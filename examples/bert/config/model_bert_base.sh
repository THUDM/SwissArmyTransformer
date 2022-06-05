MODEL_TYPE="bert-base-uncased"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/f45fff1a308846cfa63a/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip"
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE"