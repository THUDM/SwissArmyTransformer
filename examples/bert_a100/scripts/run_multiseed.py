# -*- encoding: utf-8 -*-
# @File    :   run_multiseed.py
# @Time    :   2022/3/24
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import json
from multiprocessing import Process
import random
import argparse
import os

def change_ds_config(lr, seed, batch_size):
    ds_config = {
            "train_micro_batch_size_per_gpu":batch_size,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 10,
            "gradient_clipping": 0.1,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 400,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": lr,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-6,
                    "weight_decay": 0.01
                }
            },
            "activation_checkpointing": {
                "partition_activations": False,
                "contiguous_memory_optimization": False
            },
            "wall_clock_breakdown": False
        }
    with open(f"scripts/ds_config_{seed}.json", "w") as f:
        json.dump(ds_config, f)

def run(gpu, seed_per_gpu, dataset, log_dir, lr, batch_size, epochs, step1_epochs, finetune_type):
    os.makedirs(log_dir, exist_ok=True)
    f = open(log_dir + str(gpu) + ".txt", "w")
    for i in range(seed_per_gpu):
        seed = random.randint(1, 1000000000)
        change_ds_config(lr, seed, batch_size)
        f.write(f"{i} run begin")
        new_finetune_type = finetune_type
        os.system(f"bash scripts/finetune_superglue.sh {dataset} {seed} {gpu} {lr} {epochs} {step1_epochs} {new_finetune_type}")
        f.write(f"{i} run end")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-s', type=int, default=0)
    parser.add_argument('--number-gpu', type=int, default=4)
    parser.add_argument('--seed-per-gpu', type=int, default=1)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--node', type=int, default=3)       # node 代表什么？
    parser.add_argument('--lr-search', action='store_true')   # 是否启动学习率搜索
    args = parser.parse_args()
    gpu_start = args.gpu_s
    Plist = []

    log_dir = f"multiseed_node{args.node}_/"

    if args.lr_search:     # lr_search参数表明4个gpu是否采用不同的学习率
        lr_search = [5e-5, 1e-4, 5e-4, 1e-3]
        assert len(lr_search) == args.number_gpu
    else:
        lr_search = [3e-5] * args.number_gpu
    #finetune_type = 'ffadd'   pt， all， lora， bitfit
    finetune_type = 'all'
    #finetune_type = 'all'
    batch_size = 16
    epochs=4
    step1_epochs = 1
    # if args.dataset in ["rte", "mrpc"]:
    #     epochs=60      # 60
    #     step1_epochs = 6   # 6
    # elif args.dataset in ["boolq"]:
    #     epochs=40   #40
    #     step1_epochs = 4     #4
    # elif args.dataset in ["wnli", "cb", "copa"]:
    #     epochs=400      # 400
    #     step1_epochs = 40  #80
    # elif args.dataset in ["qqp", "qnli"]:
    #     epochs = 10
    #     step1_epochs = 3
    for i in range(args.number_gpu):
        p = Process(target=run, args=(gpu_start+i,args.seed_per_gpu,args.dataset,log_dir,lr_search[i], batch_size, epochs,step1_epochs,finetune_type  ))
        p.start()
        Plist.append(p)
    for i in range(args.number_gpu):
        Plist[i].join()
    print("*****************************all seed finished!!*****************************")


'''
finetune 1e-5
bitfit 1e-3
ptv2 5e-3
lora 5e-4

python scripts/run_multiseed.py --gpu-s 0 --number-gpu 4 --dataset rte
'''