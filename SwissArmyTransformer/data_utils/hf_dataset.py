# -*- encoding: utf-8 -*-
# @File    :   get_dataset.py
# @Time    :   2021/12/14
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn

import datasets
from datasets import load_dataset

def parse_huggingface_path(path):
    if path.startswith('hf://'):
        path = path[5:]
    names = path.split('/')
    first_name = names[0]
    second_name = names[1] if len(names) >= 2 and names[1] != '*' else None
    split = names[2] if len(names) >= 3 else 'train'
    return first_name, second_name, split

def load_hf_dataset(path, process_fn, columns=None, cache_dir='~/.cache/huggingface/datasets', offline=False):
    dataset_name, sub_name, split = parse_huggingface_path(path)
    datasets.config.HF_DATASETS_OFFLINE = int(offline)
    dataset = load_dataset(dataset_name, sub_name, cache_dir=cache_dir, split=split, 
        download_config=datasets.utils.DownloadConfig(max_retries=20)) # TODO
    dataset = dataset.map(process_fn, batched=False)
    dataset.set_format(type='torch', columns=columns)
    return dataset
