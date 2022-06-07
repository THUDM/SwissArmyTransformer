# -*- encoding: utf-8 -*-
'''
@File    :   download.py
@Time    :   2022/06/05 17:08:29
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import requests
from tqdm import tqdm
from filelock import FileLock

def download_with_progress_bar(save_path, url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']), unit_scale=True)
            for chunk in r.iter_content(chunk_size=32 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def auto_create(name, *, path=None, url=None):
    if path is None:
        path = os.getenv('SAT_HOME', '~/.sat_models') # TODO Rename
    path = os.path.expanduser(path)
    file_path = os.path.join(path, name + '.zip')
    model_path = os.path.join(path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock = FileLock(file_path + '.lock')
    with lock:
        if os.path.exists(file_path) or os.path.isdir(model_path):
            pass
        else:
            if url is None:
                url = MODEL_ULRS[name]
            print(f'Downloading models {url} into {file_path} ...')
            download_with_progress_bar(file_path, url)
        # unzip
        if not os.path.isdir(model_path):
            import zipfile
            f = zipfile.ZipFile(file_path, 'r')
            f.extractall(path=path) # TODO check hierarcy of folders and name consistency
            assert os.path.isdir(model_path), f'Unzip failed, or the first-level folder in zip is not {name}.'
    return model_path # must return outside the `with lock` block

MODEL_ULRS = {
    'bert-base-uncased': 'https://cloud.tsinghua.edu.cn/f/f45fff1a308846cfa63a/?dl=1',
    'vit-base-patch16-224-in21k': 'https://cloud.tsinghua.edu.cn/f/fdf40233d9034b6a8bdc/?dl=1',
    'deit-tiny': 'https://cloud.tsinghua.edu.cn/f/b759657cb80e4bc69303/?dl=1',
    'deit-small': 'https://cloud.tsinghua.edu.cn/f/51498210e2c943dbbef1/?dl=1',
    'deit-base': 'https://cloud.tsinghua.edu.cn/f/9a26fd1aee7146e1a848/?dl=1',
    'cait-s24-224': 'https://cloud.tsinghua.edu.cn/f/bdfb12396000468b8bb9/?dl=1',
    'clip': 'https://cloud.tsinghua.edu.cn/f/bd29f0537f9949e6a4fb/?dl=1',
    'yolos-tiny': 'https://cloud.tsinghua.edu.cn/f/8ee048b6a1f1403d9253/?dl=1'
}