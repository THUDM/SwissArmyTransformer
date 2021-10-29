#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="hostfile"
HOST_FILE_PATH="hostfile_single"

full_data="/dataset/fd5061f6/HWY/cogdata_video/cogdata_task_video16frame_zh_animal/merge.bin"
small_data="/dataset/fd5061f6/HWY/cogdata_video/cogdata_task_video16frame_zh_animal/howto100m_animal/howto100m_animal.bin.part_0.cogdata"

config_json="$script_dir/ds_config_zero.json"
gpt_options=" \
       --experiment-name pretrain-cogvideo \
       --tokenizer-type cogview \
       --img-tokenizer-path /dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt \
       --model-parallel-size ${MP_SIZE} \
       --mode pretrain \
       --num-layers 48 \
       --hidden-size 2560 \
       --num-attention-heads 40 \
       --video-n-head 12 \
       --video-hidden-size 480 \
       --masked-softmax-fusion \
       --layout 64,1088,16448 \
       --new-sequence-length 16464 \
       --train-iters 200000 \
       --resume-dataloader \
       --train-data ${small_data} \
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
"
       # --load pretrained/cogview/cogview-base


gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              

run_cmd="${OPTIONS_NCCL} deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} pretrain_video.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
