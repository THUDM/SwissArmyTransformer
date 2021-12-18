import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def nonlinearity(x):
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, 
                    in_channels,
                    with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2., mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class DownSample(nn.Module):
    def __init__(self,
                    in_channels,
                    with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    stride=2,
                                    padding=0)
    
    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResidualDownSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.pooling_down_sampler = DownSample(in_channels, with_conv=False)
        self.conv_down_sampler = DownSample(in_channels, with_conv=True)

    def forward(self, x):
        return self.pooling_down_sampler(x) + self.conv_down_sampler(x)

class ResnetBlock(nn.Module):
    def __init__(self,
                    in_channels,
                    dropout,
                    out_channels=None,
                    conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)
                                            
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.k = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.v = nn.Conv2d(in_channels,
                            in_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=1, 
                                    stride=1,
                                    padding=0)
    
    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        B, C, H, W = q.shape
        q = q.reshape(B, C, -1)
        q = q.permute(0, 2, 1) # (B, H*W, C)
        k = k.reshape(B, C, -1) # (B, C, H*W)
        w_ = torch.bmm(q, k) # (B, H*W, H*W)
        w_ = w_ * C**(-0.5)
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(B, C, -1) # (B, C, H*W)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(B, C, H, W)

        h_ = self.proj_out(h_)

        return x + h_

class Encoder(nn.Module):
    def __init__(self,
                    in_channels=3,
                    out_channels=3,
                    z_channels=256,
                    channels=128,
                    num_res_blocks=0,
                    resolution=256,
                    attn_resolutions=[16],
                    resample_with_conv=True,
                    channels_mult=(1,2,4,8),
                    dropout=0.
                    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.channels = channels
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.conv_in = nn.Conv2d(in_channels,
                                    channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        current_resolution = resolution
        in_channels_mult = (1,) + tuple(channels_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_channels_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                            out_channels=block_out,
                                            dropout=dropout))
                block_in = block_out
                if current_resolution in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = DownSample(block_in,
                                                resample_with_conv)
                current_resolution = current_resolution // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                    z_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def test_forward(self, x):
        # downsample
        import pdb
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            
        return hs

    def forward(self, x):
        # downsample
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        return h

class Decoder(nn.Module):
    def __init__(self,
                    in_channels=3,
                    out_channels=3,
                    z_channels=256,
                    channels=128,
                    num_res_blocks=0,
                    resolution=256,
                    attn_resolutions=[16],
                    channels_mult=(1,2,4,8),
                    resample_with_conv=True,
                    dropout=0.
                    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.channels = channels
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        
        in_channels_mult = (1,) + tuple(channels_mult)
        block_in = channels * channels_mult[self.num_resolutions - 1]
        current_resolution = resolution // 2**(self.num_resolutions - 1)
        self.z_shape = (1, z_channels, current_resolution, current_resolution)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels,
                                    block_in,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                        out_channels=block_in,
                                        dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                            out_channels=block_out,
                                            dropout=dropout))
                block_in = block_out
                if current_resolution in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in,
                                        resample_with_conv)
                current_resolution = current_resolution * 2
            self.up.insert(0, up)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                    out_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight
