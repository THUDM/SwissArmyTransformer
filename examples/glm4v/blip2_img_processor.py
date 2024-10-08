import torch
from functools import partial
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomCrop
import torchvision.transforms as T
import albumentations as A
import numpy as np
from copy import deepcopy
from PIL import Image
import re
import torch

class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

def blip2_image_processor_func_with_inputs(image_processor, image):
    return {'image': image_processor(image).unsqueeze(0), 'input_ids': torch.zeros(1, 1, dtype=torch.long), 'attention_mask': torch.ones(1, 1, dtype=torch.long)}

blip2_image_processor_sat_1120 = partial(blip2_image_processor_func_with_inputs, BlipImageEvalProcessor(1120))
