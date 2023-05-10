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
from .urls import MODEL_URLS

def download_with_progress_bar(save_path, url, chunk_size=2048):
    resume_header = None
    file_size_downloaded = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        file_size_downloaded = os.path.getsize(save_path)
        resume_header = {'Range': f'bytes={file_size_downloaded}-'}

    response = requests.get(url, stream=True, headers=resume_header)
    total_size = int(response.headers.get('content-length', 0)) + file_size_downloaded
    if total_size == file_size_downloaded:
        return
    
    with open(save_path, 'ab') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path, initial=file_size_downloaded) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
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
        if os.path.isdir(model_path):
            pass
        else:
            if url is None:
                url = MODEL_URLS[name]
            print(f'Downloading models {url} into {file_path} ...')
            try:
                download_with_progress_bar(file_path, url)
            except Exception as e:
                print(f'Failed to download or check, if you already had the zip file, please unzip it manually as {model_path}!')
                raise e
        # unzip
        if not os.path.isdir(model_path):
            import zipfile
            print(f'Unzipping {file_path}...')
            f = zipfile.ZipFile(file_path, 'r')
            f.extractall(path=path) # TODO check hierarcy of folders and name consistency
            assert os.path.isdir(model_path), f'Unzip failed, or the first-level folder in zip is not {name}.'
    return model_path # must return outside the `with lock` block

