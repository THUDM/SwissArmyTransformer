import sys
import os
import torch
import copy

checkpoint = sys.argv[1]
target_path = sys.argv[2]

assert os.path.isdir(checkpoint)
iteration_file = os.path.join(checkpoint, 'latest_checkpointed_iteration.txt')
if os.path.exists(iteration_file):
    with open(iteration_file) as fin:
        iteration = int(fin.read().strip())
    checkpoint = os.path.join(checkpoint, str(iteration))
else:
    iteration = None

os.makedirs(target_path, exist_ok=True)
if iteration is not None:
    with open(os.path.join(target_path, "latest"), "w") as output:
        output.write(str(iteration))
    target_path = os.path.join(target_path, str(iteration))
    os.makedirs(target_path, exist_ok=True)


filenames = os.listdir(checkpoint)
filenames = [filename for filename in filenames if filename.startswith("mp_rank_")]
filenames = sorted(filenames,
                   key=lambda x: int(x.split('_')[2]))
filenames = [os.path.join(checkpoint, x) for x in filenames]

for filename in filenames:
    data = torch.load(filename)
    state_dict = data['module']
    state_dict['transformer.word_embeddings.weight'] = state_dict['word_embeddings.weight']
    del state_dict['word_embeddings.weight']
    # print(f"Target path: {os.path.join(target_path, os.path.basename(filename))}")
    torch.save(data, os.path.join(target_path, os.path.basename(filename)))
