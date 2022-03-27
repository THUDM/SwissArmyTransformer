# -*- encoding: utf-8 -*-
# @File    :   run_multiseed.py
# @Time    :   2022/3/24
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
from multiprocessing import Process
import random
import argparse
import os

def run(gpu, seed_per_gpu, dataset, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    f = open(log_dir + str(gpu) + ".txt", "w")
    for i in range(seed_per_gpu):
        seed = random.randint(1, 1000000000)
        f.write(f"{i} run begin")
        os.system(f"bash scripts/finetune_superglue.sh {dataset} {seed} {gpu}")
        f.write(f"{i} run end")
    f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-s', type=int, default=0)
    parser.add_argument('--number-gpu', type=int, default=4)
    parser.add_argument('--seed-per-gpu', type=int, default=1)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--node', type=int, default=3)
    args = parser.parse_args()

    gpu_start = args.gpu_s
    Plist = []

    log_dir = f"multiseed_node{args.node}_/"

    for i in range(args.number_gpu):
        p = Process(target=run, args=(gpu_start+i,args.seed_per_gpu,args.dataset,log_dir, ))
        p.start()
        Plist.append(p)
    for i in range(args.number_gpu):
        Plist[i].join()
    print("*****************************all seed finished!!*****************************")