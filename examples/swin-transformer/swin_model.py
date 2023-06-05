from sat.model import BaseModel, BaseMixin
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sat import mpu
import collections
from itertools import repeat
from sat.mpu.utils import split_tensor_along_last_dim
from torch import _assert

def to_pair(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return x
    return tuple(repeat(x, 2))

# ********************* Adapted from timm/models/swin_transformer.py *********************

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

# ********************* ******************************************** *********************

class WindowAttnMixin(BaseMixin):
    def __init__(self, num_heads, window_size):
        super().__init__()
        self.window_size = to_pair(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)
    
    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
        # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

        B_, N = query_layer.shape[0], query_layer.shape[2]

        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if log_attention_weights is not None:
            attention_scores += log_attention_weights

        attention_scores = attention_scores + self._get_rel_pos_bias()

        if attention_mask is not None: #.numel() != 1:
            num_win = attention_mask.shape[0]
            attention_scores = attention_scores.view(B_ // num_win, num_win, self.num_heads, N, N) + attention_mask.unsqueeze(1).unsqueeze(0) # - \
                        #    100.0 * (1.0 - attention_mask.unsqueeze(1).unsqueeze(0))
            attention_scores = attention_scores.view(-1, self.num_heads, N, N)

        attention_probs = F.softmax(attention_scores, dim=-1)

        if attention_dropout is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = attention_dropout(attention_probs)
            else:
                attention_probs = attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2).reshape(B_, N, -1)
        return context_layer

class SwinAttnMixin(BaseMixin):
    def __init__(self, num_layers, input_resolution, window_size=7, shift_sizes=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        if shift_sizes is None:
            self.shift_sizes = [0 if i % 2 == 0 else window_size // 2 for i in range(num_layers)]
        else:
            self.shift_size = shift_sizes
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_sizes = [0] * num_layers
            self.window_size = min(self.input_resolution)
        for shift_size in self.shift_sizes:
            assert shift_size == self.window_size // 2 or shift_size == 0, "shift_size must be 0 or window_size/2"

        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        cnt = 0
        for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.window_size//2),
                slice(-self.window_size//2, None)):
            for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size//2),
                    slice(-self.window_size//2, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # num_win, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.register_buffer("attn_mask", attn_mask)
    
    def attention_forward(self, hidden_states, mask, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = self.hooks['attention_fn']

        shift_size = origin.shift_sizes[kw_args['layer_id']]
        window_size = origin.window_size
        x = hidden_states
        H, W = origin.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        x = x.view(B, H, W, C)

        # cyclic shift
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # num_win*B, window_size, window_size, C
        x_windows = x_windows.view(-1, window_size * window_size, C)  # num_win*B, window_size*window_size, C

        mixed_raw_layer = self.query_key_value(x_windows)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        mask = None if shift_size == 0 else origin.attn_mask

        attn_windows = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        output = self.dense(x)

        if self.training:
            output = self.output_dropout(output)
        return output

class SwinModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(SwinModel, self).__init__(args, transformer=transformer, **kwargs)
        self.transformer.word_embeddings = None
        self.transformer.position_embeddings = None
        self.add_mixin("window", WindowAttnMixin(args.num_attention_heads, args.window_size))
        self.add_mixin("swin", SwinAttnMixin(args.num_layers, args.input_resolution, args.window_size, args.shift_sizes))
    
    def word_embedding_forward(self, input_ids, **kw_args):
        return input_ids
    
    def position_embedding_forward(self, position_ids, **kw_args):
        return None
    
    def final_forward(self, logits, **kwargs):
        return logits
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('SwinModel', 'SwinModel Configurations')
        group.add_argument('--window-size', type=int)
        group.add_argument('--input-resolution', nargs='+', type=int, default=[320, 240])
        group.add_argument('--shift-sizes', nargs='+', type=int, default=None)
        return super().add_model_specific_args(parser)