#! /bin/bash
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1
SP_SIZE=1

# export CUDA_VISIBLE_DEVICES=2,3
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="chatglm3-6b-base"
MODEL_ARGS="--prompt_column content \
    --response_column summary \
    --max_source_length 64 \
    --max_target_length 128 \
    --pre_seq_len 8"

# OPTIONS_SAT="SAT_HOME=$1" #"SAT_HOME=/raid/dm/sat_models"
# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
OPTIONS_NCCL=""
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

train_data="/inference/liji/sat/SwissArmyTransformer/examples/chatglm3/data/AdvertiseGen/train.json"
eval_data="/inference/liji/sat/SwissArmyTransformer/examples/chatglm3/data/AdvertiseGen/dev.json"


gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE-adgen-ptuning \
       --model-parallel-size ${MP_SIZE} \
       --sequence-parallel-size ${SP_SIZE} \
       --mode finetune \
       --train-iters 1000 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --valid-data ${eval_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 1000 \
       --eval-interval 10000 \
       --save "./checkpoints" \
       --split 1 \
       --eval-iters 10 \
       --eval-batch-size 8 \
       --zero-stage 1 \
       --lr 0.01 \
       --batch-size 8 \
       --skip-init \
       --fp16 \
       --num-multi-query-heads 2
       --use_ptuning
"

              
export ENV_OUTPUT_PATH=./output130 #${ENV_OUTPUT_PATH:-/data/output}
export ENV_LOGGING_DIR=${ENV_LOGGING_DIR:-${ENV_OUTPUT_PATH}/logs}
mkdir -p ${ENV_LOGGING_DIR}
# run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} /root/.local/bin/deepspeed --num_gpus ${NUM_GPUS_PER_WORKER} --master_port 16666 --include localhost:0 --hostfile ${HOST_FILE_PATH} finetune_chatglm2.py ${gpt_options}"
# run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} /root/.local/bin/deepspeed --num_gpus ${NUM_GPUS_PER_WORKER} --include localhost:2,3 finetune_chatglm2.py ${gpt_options} 2>&1 | tee ${ENV_LOGGING_DIR}/train-${ENV_BASE_MODEL_TYPE}-$(date +"%m-%d-%H-%M-%S").log"
run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} /root/.local/bin/deepspeed --hostfile ${HOST_FILE_PATH} --master_port 16666 --include localhost:0 finetune_chatglm2.py ${gpt_options} 2>&1 | tee ${ENV_LOGGING_DIR}/train-${ENV_BASE_MODEL_TYPE}-$(date +"%m-%d-%H-%M-%S").log"
echo ${run_cmd}
eval ${run_cmd}

set +x
