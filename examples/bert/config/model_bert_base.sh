MODEL_TYPE="swiss-bert-base-uncased"
if [ ! -d "${CHECKPOINT_PATH}/$MODEL_TYPE" ]
then
    if [ ! -f "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" ]
    then
        wget "https://cloud.tsinghua.edu.cn/f/9b4ab7c17ce842ea9c9d/?dl=1" -O "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" --no-check-certificate
    fi
    unzip "${CHECKPOINT_PATH}/$MODEL_TYPE.zip" -d "${CHECKPOINT_PATH}"
fi
MODEL_ARGS="--load ${CHECKPOINT_PATH}/$MODEL_TYPE"

if [ -d "${CHECKPOINT_PATH}/bert-base-uncased" ]
then
    PRETRAIN_PATH="${CHECKPOINT_PATH}"
else
    PRETRAIN_PATH=""
fi