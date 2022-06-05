MODEL_TYPE="swiss-clip"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/8f9336ec0fa84a81bc14/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip"
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE"

if [ -d "${CHECKPOINT_PATH}/clip-vit-base-patch32" ]
then
    PRETRAIN_PATH="${CHECKPOINT_PATH}"
else
    PRETRAIN_PATH=""
fi