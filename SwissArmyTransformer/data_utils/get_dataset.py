# -*- encoding: utf-8 -*-
# @File    :   get_dataset.py
# @Time    :   2021/12/14
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

import datasets
from datasets import load_dataset

def parse_huggingface_path(path):
    names = path.split('/')
    first_name = names[0]
    second_name = names[1] if len(names) >= 2 else None
    split = names[2] if len(names) >= 3 else None
    return first_name, second_name, split

def get_dataset(dataset_name, sub_name, process_fn, columns, split='train'):
    datasets.config.HF_DATASETS_OFFLINE = 1
    dataset = load_dataset(dataset_name, sub_name,  cache_dir='/dataset/fd5061f6/SwissArmyTransformerDatasets', split=split)
    dataset = dataset.map(process_fn, batched=False)
    dataset.set_format(type='torch', columns=columns)
    return dataset
