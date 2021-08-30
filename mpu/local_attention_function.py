import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)

__all__ = ['f_similar', 'f_weighting', 'LocalAttention', 'TorchLocalAttention']


class similarFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW, casual_mask=False):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        ctx.casual_mask = casual_mask
        output = similar_forward(x_ori, x_loc, kH, kW, casual_mask)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        casual_mask = ctx.casual_mask
        grad_outputs = grad_outputs.contiguous()
        grad_ori = similar_backward(x_ori, x_loc, grad_outputs, kH, kW, True, casual_mask)
        grad_loc = similar_backward(x_ori, x_loc, grad_outputs, kH, kW, False, casual_mask)

        return grad_ori, grad_loc, None, None, None


class weightingFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW, casual_mask=False):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        ctx.casual_mask = casual_mask
        output = weighting_forward(x_ori, x_weight, kH, kW, casual_mask)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        casual_mask = ctx.casual_mask
        grad_outputs = grad_outputs.contiguous()
        grad_ori = weighting_backward_ori(x_ori, x_weight, grad_outputs, kH, kW, casual_mask)
        grad_weight = weighting_backward_weight(x_ori, x_weight, grad_outputs, kH, kW, casual_mask)

        return grad_ori, grad_weight, None, None, None


f_similar = similarFunction.apply
f_weighting = weightingFunction.apply


class LocalAttention(nn.Module):
    def __init__(self, inp_channels, out_channels, kH, kW):
        super(LocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        weight = f_similar(x1, x2, self.kH, self.kW)
        weight = F.softmax(weight, -1)
        out = f_weighting(x3, weight, self.kH, self.kW)

        return out


class TorchLocalAttention(nn.Module):
    def __init__(self, inp_channels, out_channels, kH, kW):
        super(TorchLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(inp_channels, out_channels, kernel_size=1, bias=False)
        self.kH = kH
        self.kW = kW

    @staticmethod
    def f_similar(x_theta, x_phi, kh, kw, casual_mask=False):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2, kw // 2)
        x_theta = x_theta.permute(0, 2, 3, 1).contiguous()
        x_theta = x_theta.view(n * h * w, 1, c)

        x_phi = F.unfold(x_phi, kernel_size=(kh, kw), stride=1, padding=pad)
        x_phi = x_phi.contiguous().view(n, c, kh * kw, h * w)
        x_phi = x_phi.permute(0, 3, 1, 2).contiguous()
        x_phi = x_phi.view(n * h * w, c, kh * kw)
        out = x_theta @ x_phi
        out = out.view(n, h, w, kh * kw)
        if casual_mask:
            out = out[..., :kh * kw // 2 + 1]
        return out

    @staticmethod
    def f_weighting(x_theta, x_phi, kh, kw, casual_mask=False):
        n, c, h, w = x_theta.size()  # (N, inter_channels, H, W)
        pad = (kh // 2, kw // 2)
        x_theta = F.unfold(x_theta, kernel_size=(kh, kw), stride=1, padding=pad)
        x_theta = x_theta.permute(0, 2, 1).contiguous()
        x_theta = x_theta.view(n * h * w, c, kh * kw)

        if casual_mask:
            x_theta = x_theta[..., :kh * kw // 2 + 1]
            x_phi = x_phi.view(n * h * w, kh * kw // 2 + 1, 1)
        else:   
            x_phi = x_phi.view(n * h * w, kh * kw, 1)

        out = torch.matmul(x_theta, x_phi)
        out = out.squeeze(-1)
        out = out.view(n, h, w, c)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        weight = self.f_similar(x1, x2, self.kH, self.kW)
        weight = F.softmax(weight, -1)
        out = self.f_weighting(x3, weight, self.kH, self.kW)

        return out
    
