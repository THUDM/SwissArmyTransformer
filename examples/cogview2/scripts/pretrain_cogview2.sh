#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

full_data="/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_4leveltokens/merge.bin"
small_data="/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_4leveltokens/zijian/zijian.bin.part_0.cogdata"

config_json="$script_dir/ds_config_zero.json"
gpt_options=" \
       --experiment-name pretrain-cogview2-test \
       --tokenizer-type cogview \
       --img-tokenizer-path /dataset/fd5061f6/sat_pretrained/vqvae/vqvae_hard_biggerset_011.pt \
       --model-parallel-size ${MP_SIZE} \
       --mode pretrain \
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --train-iters 200000 \
       --resume-dataloader \
       --train-data ${full_data} \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --checkpoint-activations \
       --max-sequence-length 1089 \
       --sandwich-ln \
       --fp16 \
       --save-interval 2000 \
       --eval-interval 1000 \
       --save $main_dir/checkpoints \
       --load /dataset/fd5061f6/sat_pretrained/cogview/cogview2-base
"
       # --load pretrained/cogview/cogview-base


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_cogview2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
