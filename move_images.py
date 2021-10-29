# %%
coco_30k = '/workspace/dm/SwissArmyTransformer/coco30k.txt'
with open(coco_30k, 'r') as fin:
    lines = fin.readlines()
    
import os
from posixpath import join
import shutil
prefix0 = '/workspace/dm/SwissArmyTransformer/coco_samples'
prefix1 = '/dataset/fd5061f6/mingding/SwissArmyTransformer/coco_samples'
cnt = 0 
with open('coco_select.txt', 'w') as fout:
    for i, line in enumerate(lines):
        _id, text = line.strip().split('\t')
        if i % 200 == 0:
            print(i, cnt)
        src = os.path.join(prefix1, _id)
        if not os.path.exists(src):
            src = os.path.join(prefix0, _id)
        assert os.path.exists(src), _id
        fout.write(
            '\t'.join([text] + [
                os.path.join(src, f'{i}.jpg')
                for i in range(60)
                ]) + '\n'
        )
                
    