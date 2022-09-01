
import os
import datasets
from datasets import load_dataset

def parse_huggingface_path(path):
    if path.startswith('hf://'):
        path = path[5:]
    names = path.split('/')
    first_name = names[0]
    first_name = first_name.replace('!', '/')
    second_name = names[1] if len(names) >= 2 and names[1] != '*' else None
    split = names[2] if len(names) >= 3 else 'train'
    return first_name, second_name, split

def load_hf_dataset(path, process_fn, filter_fn = None, columns=None, cache_dir='~/.cache/huggingface/datasets', offline=False, transformer_name = None, low_resource=None):
    dataset_name, sub_name, split = parse_huggingface_path(path)
    datasets.config.HF_DATASETS_OFFLINE = int(offline)
    if transformer_name:
        dataset_path = cache_dir + '/' + dataset_name + "_" + sub_name + "_" + split + "_" + transformer_name + ".data"
    else:
        dataset_path = None

    keep_in_memory = True
    if dataset_path and os.path.exists(dataset_path):
        dataset = datasets.load_from_disk(dataset_path, keep_in_memory=keep_in_memory)
    else:
        if "SemEval2014Task4Raw" in dataset_name:
            cache_dir = None
        dataset = load_dataset(dataset_name, sub_name, cache_dir=cache_dir, split=split,
        download_config=datasets.utils.DownloadConfig(max_retries=20), keep_in_memory=keep_in_memory) # TODO
        # dataset = dataset.filter(lambda example, indice: indice % 100 == 0, with_indices=True)
        if filter_fn is not None:
            dataset = dataset.filter(filter_fn, batched=False)
        if low_resource and dataset.__len__() > 250:
            dataset = dataset.select(range(250))
        if "SemEval2014Task4Raw" in dataset_name:
            batched = True
        else:
            batched = False
        dataset = dataset.map(process_fn, batched=batched, batch_size=1, load_from_cache_file=True, keep_in_memory=keep_in_memory, remove_columns=dataset.column_names)
        if dataset_path:
            dataset.save_to_disk(dataset_path)
    dataset.set_format(type='torch', columns=columns)
    return dataset
