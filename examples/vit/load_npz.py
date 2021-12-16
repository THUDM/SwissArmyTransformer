# -*- encoding: utf-8 -*-
# @File    :   load_npz.py
# @Time    :   2021/12/16
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import math
import torch
import argparse
from SwissArmyTransformer.model import VitModel
def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


# m = timm.create_model('mobilenetv3_large_100',checkpoint_path='/workspace/yzy/mobilenetv3_large_100_ra-f55367f5.pth')
# m.eval()
# m = timm.create_model('mobilenetv3_large_100',checkpoint_path='/workspace/yzy/mobilenetv3_large_100_ra-f55367f5.pth')
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

if __name__ == "__main__":
    # def _load_old_weight():
    import os
    args = argparse.Namespace(
        num_layers = 12,
        hidden_size = 768,
        image_size = 224,
        patch_size = 16,
        vocab_size=1,
        max_sequence_length=197,
        num_attention_heads=12,
        sandwich_ln=False,
        checkpoint_activations=False,
        model_parallel_size=1,
        world_size=1,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        inner_hidden_size=None,
        hidden_size_per_attention_head = None,
        rank=0,
        checkpoint_num_layers=1,
        in_channels=3,
        num_classes=21843,
    )
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=args.world_size, rank=args.rank,init_method=init_method)
    import SwissArmyTransformer.mpu as mpu
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)
    print('bg')
    model = VitModel(args)
    old = {}
    old['module'] = {}

    #load npz
    import numpy as np
    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)
    with torch.no_grad():
        checkpoint_path='/workspace/yzy/ST_dev/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz'
        w = np.load(checkpoint_path)
        prefix=''
        embed_conv_w = adapt_input_conv(
            model.mixins["patch_embedding"].proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
        model.mixins["patch_embedding"].proj.weight.copy_(embed_conv_w)
        model.mixins["patch_embedding"].proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
        model.transformer.word_embeddings.weight.copy_(_n2p(w[f'{prefix}cls'], t=False).squeeze(0))
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
        model.transformer.position_embeddings.weight.copy_(pos_embed_w.squeeze(0))
        model.classifier.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.classifier.bias.copy_(_n2p(w[f'{prefix}head/bias']))
        for i in range(len(model.transformer.layers)):
            block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
            mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
            model.transformer.layers[i].attention.query_key_value.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
            model.transformer.layers[i].attention.query_key_value.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))

            model.transformer.layers[i].attention.dense.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            model.transformer.layers[i].attention.dense.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            #mlp
            model.transformer.layers[i].mlp.dense_h_to_4h.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{0}/kernel']))
            model.transformer.layers[i].mlp.dense_h_to_4h.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{0}/bias']))
            model.transformer.layers[i].mlp.dense_4h_to_h.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{1}/kernel']))
            model.transformer.layers[i].mlp.dense_4h_to_h.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{1}/bias']))

    state = {'module':model.state_dict()}
    torch.save(state, 'vit-Base-224-16-21k.ckpt')
    missing_keys, unexpected_keys = model.load_state_dict(state['module'], strict=False)
    print(missing_keys)
    print(unexpected_keys)
        # breakpoint()
        # print(pos_embed_w)
    #看看pos的长度