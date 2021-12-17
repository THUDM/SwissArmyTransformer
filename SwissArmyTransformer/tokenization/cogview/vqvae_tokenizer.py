import math
import torch
import torch.nn.functional as F

from vqvae import load_default_HVQVAE, load_ckpt

def is_exp2(x):
    t = math.log2(x)
    return abs(t - int(t)) < 1e-4


def sqrt_int(x):
    r = int(math.sqrt(x) + 1e-4)
    assert r * r == x
    return r

class VQVAETokenizer(object):
    def __init__(self,
                    model_path=None,
                    device='cuda'):
        model = load_default_HVQVAE()
        model = load_ckpt(model, model_path)
        model = model.to(device)
        model.eval()

        self.model = model
        self.device = device
        self.image_tokens = model.quantize.n_embed
        self.num_tokens = model.quantize.n_embed

    def __len__(self):
        return self.num_tokens

    def img2code(self, img, l=None):
        '''Convert a batch of img to code
        Args:
            model: The tokenizer model.
            img: [b, c, h, w]
        '''
        if l is None:
            with torch.no_grad():
                quants, diffs, ids = self.model.encode(img)
            return [id.view(img.shape[0], -1) for id in ids]
        else:
            with torch.no_grad():
                quant, diff, id = self.model.single_encode(img, l)
            return id.view(img.shape[0], -1)

    def code2img(self, codes, l=None):
        '''Convert a batch of code to imgs
        Args:
            model: ...
            codes w/o l: [l, b, h, w] or [l, b, h*w] List of LongTensor
            codes w/ l: [b, h, w] or [b, h*w] LongTensor
        '''
        if l is None:
            if len(codes[0].shape) == 2:
                s = int(math.sqrt(len(codes[0].view(-1))) + 1e-5)
                b = codes[0].shape[0]
                codes = [code.view(b, s, s) for code in codes]
            with torch.no_grad():
                out = model.decode_code(codes)
                out = [item * torch.tensor([0.30379, 0.32279, 0.32800], device=out.device).view(1, -1, 1, 1) + torch.tensor([0.79093, 0.76271, 0.75340], device=out.device).view(1, -1, 1, 1) for item in out]
        else:
            if len(codes.shape) == 2:
                s = int(math.sqrt(len(codes.view(-1))) + 1e-5)
                codes = codes.view(codes.shape[0], s, s)
            with torch.no_grad():
                out = model.single_decode_code(codes)
                out = out * torch.tensor([0.30379, 0.32279, 0.32800], device=out.device).view(1, -1, 1, 1) + torch.tensor([0.79093, 0.76271, 0.75340], device=out.device).view(1, -1, 1, 1)
        return out

    def EncodeAsIds(self, img, l=None):
        assert len(img.shape) == 4 # [b, c, h, w]
        return self.img2code(img, l)
    
    def DecodeIds(self, codes, l=None):
        return self.code2img(codes, l)

    def read_img(self, path, img_size=256):
        tr = transform.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        img = tr(Image.open(path))
        if img.shape[0] == 4:
            img = img[:-1]
        tr_normalize = transforms.Normalize([0.79093, 0.76271, 0.75340], [
                                            0.30379, 0.32279, 0.32800])
        img = tr_normalize(img)
        img = img.unsqueeze(0).float().to(self.device) # size [1, 3, h, w]
        return img