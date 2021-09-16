# %%
p = 'people.jpeg'
from data_utils.vqvae_tokenizer import VQVAETokenizer
model = VQVAETokenizer(
    'pretrained/vqvae/vqvae_hard_biggerset_011.pt'
)
img = model.read_img(p, img_size=512)
# %%
test_dir = 'tmp'
import os
import torch
from torchvision.utils import save_image
img = model.EncodeAsIds(img)
imgs = model.DecodeIds(torch.tensor(img))
save_image(imgs, os.path.join(test_dir, 'show512_people.jpg'), normalize=True)
# %%
