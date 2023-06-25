NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

MODEL_TYPE="gpt-2"

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

gpt_options=" \
       --batch-size 4 \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-data /zhangpai21/sxx/data/rm-static/data/train-00000-of-00001-2a1df75c6bce91ab.parquet\
       --train-iters 2000 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save "./checkpoints" \
       --save-interval 2000 \
       --zero-stage 1 \
       --lr 0.0001 \
       --skip-init \
       --para_lambda 1.0 \
       --para_delta 1.0 \
       --fp16
"

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} slic_hf.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
