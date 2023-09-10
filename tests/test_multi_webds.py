from sat.data_utils import make_loaders
from sat import get_args
from sat.data_utils import MetaDistributedWebDataset

args = get_args(['--batch-size', '1', '--train-data', '/nxchinamobile2/shared/img_datasets/laion115m_grounding_small_objects_optimized/part-00032/320000{0..9}.tar', '/nxchinamobile2/shared/img_datasets/laion_aes_3m/part-00001/10000{0..4}.tar', '--train-data-weights', '2', '1', '--iterable-dataset', '--split','1', '--num-workers', '0'])

def process_fn(src):
    for x in src:
        if x["task_data"] is not None:
            yield 1
        else:
            yield 0

def create_func(path, args):
    return MetaDistributedWebDataset(path, process_fn, seed=0, meta_names=['task_data'])

train_loader, val_loader, test_loader = make_loaders(args,create_dataset_function=create_func)

a, b = 0, 0
for i, x in enumerate(train_loader):
    print(x)
    if x[0] == 0:
        a += 1
    else: 
        b += 1
    if i > 500:
        break

print(a, b)



