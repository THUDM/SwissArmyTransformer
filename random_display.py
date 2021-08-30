from data_utils.datasets import BinaryDataset
from data_utils import get_tokenizer
import argparse
import os
import torch
import random
test_dir = 'tmp'
# bin_dir = '/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_3leveltokens/merge.bin'
bin_dir = '/dataset/fd5061f6/cogview/cogdata_new/cogdata_task_3leveltokens/quanjing005/quanjing005.bin.part_0.cogdata'
bin_ds = BinaryDataset(os.path.join(bin_dir), process_fn=lambda x:x, length_per_sample=64*64+32*32+64, dtype='int32', preload=False)
args = argparse.Namespace(img_tokenizer_path='pretrained/vqvae/vqvae_hard_biggerset_011.pt', img_tokenizer_num_tokens=None)
tokenizer = get_tokenizer(args)

bin_ds = [bin_ds[random.randint(0, len(bin_ds)-1)] for i in range(16)]
for x in bin_ds:
    end = x.tolist().index(-1)
    print(tokenizer.DecodeIds(x[:end])[0])

from torchvision.utils import save_image
imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(torch.tensor(x[64:64+64**2], dtype=torch.long, device='cuda')) for x in bin_ds], dim=0)
save_image(imgs, os.path.join(test_dir, 'testcase512.jpg'), normalize=True)
imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(torch.tensor(x[64+64**2:64+64**2+32**2], dtype=torch.long,device='cuda')) for x in bin_ds], dim=0)
save_image(imgs, os.path.join(test_dir, 'testcase256.jpg'), normalize=True)
# imgs = torch.cat([tokenizer.img_tokenizer.DecodeIds(torch.tensor(x[64+64**2+32**2:], dtype=torch.long,device='cuda')) for x in bin_ds], dim=0)
# save_image(imgs, os.path.join(test_dir, 'testcase128.jpg'), normalize=True)