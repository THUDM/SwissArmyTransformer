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
from .urls import MODEL_ULRS

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

