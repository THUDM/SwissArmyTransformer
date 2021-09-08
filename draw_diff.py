import numpy as np
import torch
def loadbao(name):
    ret = []
    with open(name, 'r') as fin:
        for line in fin:
            a, b = line.split()
            ret.append(abs(float(b)))
    return ret
import torchvision
import torchvision.transforms as transforms

def sq(img, x, y, lx, ly):
    assert len(img.shape) == 3
    img[:,x:x+lx,y] = torch.tensor([0,1,0]).unsqueeze(-1)
    img[:,x:x+lx,y+ly] = torch.tensor([0,1,0]).unsqueeze(-1)
    img[:,x,y:y+ly] = torch.tensor([0,1,0]).unsqueeze(-1)
    img[:,x+lx,y:y+ly] = torch.tensor([0,1,0]).unsqueeze(-1)

transform = transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
            ])
img = torchvision.io.read_image('bao.jpeg')
img = transform(img) / 255.
a = np.array(loadbao('bao2.txt'))
b = np.array(loadbao('bao3.txt'))
for t in (b-a>1).nonzero()[0]:
    x,y = t // 32, t % 32
    sq(img, x*16, y*16, 15, 15)
print(a.mean(), b.mean())
torchvision.utils.save_image(img, 'example_bao.jpg')
