import torch
from ..mixout import Mixout
from typing import Optional
from collections import OrderedDict


def mixout(input:torch.Tensor, 
           target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
           p:float=0.0, 
           training:bool=False, 
           inplace:bool=False) -> torch.Tensor:

    return Mixout.apply(input, target, p, training, inplace)