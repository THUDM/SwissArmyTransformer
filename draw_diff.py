import numpy as np
import torch
def entropy(x):
    a = np.array(x)
    a = a / a.sum()
    return - np.sum(a * np.log(a))
print(entropy([0.9999,0.001]))
def loadbao(name):
    ret1, ret2 = [], []
    with open(name, 'r') as fin:
        for line in fin:
            a = line.split()
            aa = [float(x) for x in a[1:5]]
            b = entropy(aa)
            c = sum(aa)
            ret1.append(b)
            ret2.append(c)
    return np.array(ret1), np.array(ret2)

def loadlion(name):
    ret1, ret2 = [], []
    with open(name, 'r') as fin:
        for line in fin:
            a, b = line.split()
            ret1.append(abs(float(a)))
            ret2.append(abs(float(b)))
    return ret1, ret2
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
img = torchvision.io.read_image('cat2.jpeg')
img = transform(img) / 255.
# a,b = np.array(loadlion('bed6.txt'))
b, c = np.array(loadbao('bed1.txt'))
for t in (b>1.35).nonzero()[0]:
    x,y = t // 64, t % 64
    sq(img, x*8, y*8, 7, 7)
print(b.sum())
torchvision.utils.save_image(img, 'example_cat.jpg')
