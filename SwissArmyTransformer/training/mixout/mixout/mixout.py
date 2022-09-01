##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Cheolhyoung Lee
## Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
## Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
from torch.autograd.function import InplaceFunction
from typing import Optional
from collections import OrderedDict


class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns 
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p) 
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None. 
    # I modified the code of dropout in PyTorch. 
    @staticmethod
    def _make_noise(input:torch.Tensor) -> torch.Tensor:
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, 
                ctx, 
                input:torch.Tensor, 
                target:Optional["OrderedDict[str, torch.Tensor]"]=None, 
                p:float=0.0, 
                training:bool=False, 
                inplace:bool=False) -> torch.Tensor:

        if p < 0 or p > 1:
            raise ValueError(f"A mix probability of mixout has to be between 0 and 1,  but got {p}")

        if target is not None and input.size() != target.size():
            raise ValueError(f"A target tensor size must match with a input tensor size {input.size()}, but got {target.size()}")
        
        ctx.p = p    
        ctx.training = training
        
        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        
        if ctx.p == 0 or not ctx.training:
            return output
        
        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], *([1] * (len(input.size())-1)))
        ctx.noise.expand_as(input)
        
        if ctx.p == 1:
            output = target.clone()
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        
        return output


    @staticmethod
    def backward(ctx, grad_output:torch.Tensor) -> Optional[torch.Tensor]:
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None
