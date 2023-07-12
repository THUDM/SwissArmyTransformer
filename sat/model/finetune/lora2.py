"""
In this mixin, I use a different implementation than sat/model/finetune/lora.py
I just use a fake linear layer to replace any model with lora mixin.
"""

import torch
import torch.nn as nn
from sat.model.base_model import BaseMixin
import math
from sat.helpers import print_all
from sat.model.transformer import RowParallelLinear, ColumnParallelLinear

class HackLinear(nn.Linear):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix+'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix+'bias'])

class HackRowParallelLinear(RowParallelLinear):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix+'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix+'bias'])

class HackColumnParallelLinear(ColumnParallelLinear):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'weight' in state_dict:
            self.weight.data.copy_(state_dict[prefix+'weight'])
        if prefix + 'bias' in state_dict:
            self.bias.data.copy_(state_dict[prefix+'bias'])

try:
    from bitsandbytes.nn import LinearNF4
    def copy_nested_list(src, dst):
        for i in range(len(dst)):
            if type(dst[i]) is torch.Tensor:
                dst[i].copy_(src[i])
            elif type(dst[i]) is list:
                copy_nested_list(src[i], dst[i])
            else:
                dst[i] = src[i]
    class HackLinearNF4(LinearNF4):
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            if prefix + 'weight' in state_dict:
                self.weight.data.copy_(state_dict[prefix+'weight'])
                if self.weight.data.dtype == torch.uint8:
                    copy_nested_list(state_dict[prefix+'quant_state'], self.weight.quant_state)
            if prefix + 'bias' in state_dict:
                self.bias.data.copy_(state_dict[prefix+'bias'])
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            super()._save_to_state_dict(destination, prefix, keep_vars)
            destination[prefix+'quant_state'] = self.weight.quant_state
except Exception as exception:
    print_all("Failed to load bitsandbytes:" + str(exception), level='WARNING')


class HackParameterList(nn.ParameterList):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for i in range(len(self)):
            if prefix + str(i) in state_dict:
                self[i].data.copy_(state_dict[prefix+str(i)])

map_cls = {
    nn.Linear: (HackLinear, {}),
    ColumnParallelLinear: (HackColumnParallelLinear, {'gather_output': False}),
    RowParallelLinear: (HackRowParallelLinear, {'input_is_parallel': True})
}

class LoraLinear(nn.Module):
    def __init__(self, original_cls, partition, in_dim, out_dim, r, lora_alpha=1., lora_dropout=0., qlora=False, original_obj=None):
        super().__init__()
        if lora_dropout and lora_dropout > 0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        bias = True
        if original_obj:
            bias = original_obj.bias is not None
        if qlora:
            try:
                self.original = HackLinearNF4(in_dim, out_dim, bias=bias)
            except:
                raise Exception('Build 4bit layer failed. You need to install the latest bitsandbytes. Try `pip install bitsandbytes`. If you still meet error after installation, try running `from bitsandbytes.nn import LinearNF4` with python and fix the error.')
        else:
            base_cls, kwargs = map_cls[original_cls]
            self.original = base_cls(in_dim, out_dim, **kwargs, bias=bias)
        if original_obj:
            self.original.weight.data.copy_(original_obj.weight.data)
            if bias:
                self.original.bias.data.copy_(original_obj.bias.data)
        self.matrix_A = HackParameterList([nn.Parameter(torch.empty((r, in_dim))) for _ in range(partition)])
        self.matrix_B = HackParameterList([nn.Parameter(torch.empty((out_dim // partition, r))) for _ in range(partition)])
        for i in range(partition):
            nn.init.kaiming_uniform_(self.matrix_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.matrix_B[i])
        self.partition = partition

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # This is not a perfect version, becuase it doesn't handle errors and unexpected keys.
        if prefix + 'weight' in state_dict:
            # load from normal Linear
            self.original._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        else:
            # load from LoraLinear
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
            
    def forward(self, x):
        mixed_raw_layer = self.original(x)
        lora_outputs = []
        for i in range(self.partition):
            lora_outputs.append((self.lora_dropout(x) @ self.matrix_A[i].T @ self.matrix_B[i].T) * self.scaling)
        mixed_raw_layer = mixed_raw_layer + torch.cat(lora_outputs, -1)

        return mixed_raw_layer


def replace_linear_with_lora(lin, partition, r, *args, **kw_args):
    out_dim, in_dim = lin.weight.shape
    original_cls = type(lin)
    new_layer = LoraLinear(original_cls, partition, in_dim, out_dim, r, *args, **kw_args, original_obj=lin)
    del lin
    return new_layer

def merge_linear_lora(lin):
    if lin.original.weight.data.dtype is not torch.uint8:
        weight = lin.original.weight
        out_dim, in_dim = weight.shape
        new_lin = nn.Linear(in_dim, out_dim)
    else:
        import bitsandbytes.functional as F
        weight = F.dequantize_fp4(lin.original.weight.data, lin.original.weight.quant_state).to(lin.original.bias.data.dtype)
        out_dim, in_dim = weight.shape
        new_lin = HackLinearNF4(in_dim, out_dim)
    if lin.original.bias:
        new_lin.bias.data = lin.original.bias.data
    new_qkv = []
    for i in range(lin.partition):
        new_qkv.append(lin.matrix_A[i].data.T.float() @ lin.matrix_B[i].data.T.float() * lin.scaling)
    new_qkv = torch.cat(new_qkv, -1)
    new_lin.weight.data = weight + new_qkv.T.to(lin.original.bias.data.dtype)
    return new_lin.cuda() if torch.cuda.is_available() else new_lin

class LoraMixin(BaseMixin):
    def __init__(self, 
                layer_num,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.,
                layer_range = None,
                qlora = False,
                cross_attention = True):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        if layer_range is None:
            layer_range = [i for i in range(layer_num)]
        self.layer_range = layer_range

        self.scaling = self.lora_alpha / self.r
        self.qlora = qlora
        self.cross_attention = cross_attention

    def reinit(self, parent_model):
        for i in self.layer_range:
            print(f'replacing layer {i} attention with lora')
            parent_model.transformer.layers[i].attention.dense = replace_linear_with_lora(parent_model.transformer.layers[i].attention.dense, 1, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora)
            parent_model.transformer.layers[i].attention.query_key_value = replace_linear_with_lora(parent_model.transformer.layers[i].attention.query_key_value, 3, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora)
            if self.cross_attention and parent_model.transformer.layers[i].is_decoder:
                print(f'replacing layer {i} cross attention with lora')
                parent_model.transformer.layers[i].cross_attention.dense = replace_linear_with_lora(parent_model.transformer.layers[i].cross_attention.dense, 1, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora)
                parent_model.transformer.layers[i].cross_attention.query = replace_linear_with_lora(parent_model.transformer.layers[i].cross_attention.query, 1, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora)
                parent_model.transformer.layers[i].cross_attention.key_value = replace_linear_with_lora(parent_model.transformer.layers[i].cross_attention.key_value, 2, self.r, self.lora_alpha, self.lora_dropout, qlora=self.qlora)
        if self.qlora:
            print('replacing chatglm linear layer with 4bit')
            def replace_linear_with_nf4(model, name=None, cache={}):
                if type(model) in (nn.Linear, RowParallelLinear, ColumnParallelLinear):
                    out_dim, in_dim = model.weight.shape
                    return HackLinearNF4(in_dim, out_dim)
                names = set()
                for name, child in model.named_children():
                    if name not in names:
                        if child in cache:
                            new_child = cache[child]
                        else:
                            new_child = replace_linear_with_nf4(child, name=name, cache=cache)
                            cache[child] = new_child
                        setattr(model, name, new_child)
                        names.add(name)
                flag = True
                while flag:
                    flag = False
                    for name, child in model.named_children():
                        if name not in names:
                            setattr(model, name, cache[child])
                            names.add(name)
                            flag = True
                return model
            replace_linear_with_nf4(parent_model.transformer, None, {})

    def merge_lora(self):
        for i in self.layer_range:
            print(f'merge layer {i} lora attention back to linear')
            self.transformer.layers[i].attention.dense = merge_linear_lora(self.transformer.layers[i].attention.dense)
            self.transformer.layers[i].attention.query_key_value = merge_linear_lora(self.transformer.layers[i].attention.query_key_value)
            if self.transformer.layers[i].is_decoder:
                print(f'merge layer {i} lora cross attention back to linear')
                self.transformer.layers[i].cross_attention.dense = merge_linear_lora(self.transformer.layers[i].cross_attention.dense)
                self.transformer.layers[i].cross_attention.query = merge_linear_lora(self.transformer.layers[i].cross_attention.query)
                self.transformer.layers[i].cross_attention.key_value = merge_linear_lora(self.transformer.layers[i].cross_attention.key_value)

if __name__ == '__main__':
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = nn.Linear(100, 200)
        
        def forward(self, x):
            return self.child(x)

    model = Model()
    torch.save(model.state_dict(), "linear.pt")
    x = torch.randn(2, 100)
    out1 = model(x)
    model.child = LoraLinear(100, 200, 10)
    model.load_state_dict(torch.load("linear.pt"), strict=False)
    out2 = model(x)
    torch.save(model.state_dict(), "lora.pt")
    ckpt = torch.load("lora.pt")
    breakpoint()
    model.load_state_dict(ckpt, strict=False)
    out3 = model(x)
    breakpoint()