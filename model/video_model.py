# -*- encoding: utf-8 -*-
'''
@File    :   video_model.py
@Time    :   2021/10/09 20:27:32
@Author  :   Wenyi Hong 
@Contact :   hongwy18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F

from mpu import transformer


from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, VideoAttentionMixin, VideoMLPMixin, VideoLayerMixin

from mpu.transformer import split_tensor_along_last_dim
from mpu.local_attention_function import f_similar, f_weighting
from mpu.utils import sqrt, gelu
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker
from mpu.transformer import unscaled_init_method



class VideoModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        additional_seqlen = args.new_sequence_length - args.max_sequence_length
        self.mixins.append(PositionEmbeddingMixin(
            additional_seqlen, args.hidden_size, 
            reinit_slice=slice(-1025, None)
        ))
        self.mixins.append(VideoAttentionMixin(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            video_hidden_size=args.video_hidden_size,
            video_n_head=args.video_n_head,
            attention_dropout_prob=args.attention_dropout,
            masked_softmax_fusion=args.masked_softmax_fusion,
            init_method=self.transformer.init_method,
            output_layer_init_method=self.transformer.output_layer_init_method,
            args=args
        ))
        self.mixins.append(VideoMLPMixin(
            num_layers=args.num_layers,
            video_hidden_size=args.video_hidden_size,
            hidden_size=args.hidden_size,
            output_dropout_prob=args.hidden_dropout,
            init_method=self.transformer.init_method,
            output_layer_init_method=self.transformer.output_layer_init_method
        ))
        self.mixins.append(VideoLayerMixin(
            num_layers=args.num_layers,
            video_hidden_size=args.video_hidden_size,
            hidden_size=args.hidden_size,
            sandwich_ln=args.sandwich_ln,
            layernorm_epsilon=args.layernorm_epsilon,
            init_method=self.transformer.init_method
        ))
        
        self.layout = args.layout
        self.sandwich_ln = args.sandwich_ln
        self.hidden_size = args.hidden_size
        self.video_hidden_size = args.video_hidden_size
        self.masked_softmax_fusion = args.masked_softmax_fusion
        # [PAD]... [ROI1] text ...  {layout[0]} [BOI1] 1024 {layout[1]} [BOI2] 1024 [BOI3] 1024 [EOI2] 1024 ... (16 frames)[POS8]1024 {layout[2]}
        # output ends with [EOI1]
        # frame beginer: [BOI1][BOI2][BOI3][EOI2][EOI3][ROI2][ROI3][POS0][POS1][POS2][POS3][POS4][POS5][POS6][POS7][POS8]
        # self.frame_nums = args.frame_nums
        # attend to 1 frame before
        
    def position_embedding_forward(self, position_ids, **kw_tensors):
        position = position_ids[..., :self.layout[1]]
        position_plus = position_ids[..., self.layout[1]:]
        position_embeddings = torch.cat(
                (
                    self.transformer.position_embeddings(position),
                    self.mixins[0].position_embeddings(position_plus)
                ),
                dim=-2
            )
        return position_embeddings
    
    def my_attention_forward(self, hidden_states, mask, layer_id=None, **kw_tensors):
        per_frame_len = self.layout[1]-self.layout[0]
        text_image_len = self.layout[1]
        
        attn_module = self.transformer.layers[layer_id].attention
        attn_mixin = self.mixins[1]
        
        attention_mask_i2i = mask[..., :self.layout[1], :]
        attention_mask_f2i = mask[..., self.layout[1]:, :]
        attention_mask_f2f = torch.ones((attention_mask_f2i.shape[0], 1, per_frame_len, per_frame_len), 
                                        device=attention_mask_i2i.device).type(torch.float16)
        attention_mask_f2f.tril_()
        attention_mask_f2f.unsqueeze(0).unsqueeze(0)
        
        hidden_states_image, hidden_states_frame = hidden_states
        # base model qkv
        mixed_raw_layer = attn_module.query_key_value(hidden_states_image)
        qi, ki, vi = split_tensor_along_last_dim(mixed_raw_layer, 3)
        # video frames qkv
        mixed_raw_layer = attn_mixin.query_key_value[layer_id](hidden_states_frame)
        qf, kf, vf = split_tensor_along_last_dim(mixed_raw_layer, 3)
        dropout_fn = attn_module.attention_dropout if self.training else None
        dropout_fn_frame = attn_mixin.attention_dropout[layer_id] if self.training else None
              
        # video attention: attend to text+lastframe+self
        if self.masked_softmax_fusion:
            context_image, context_frame = video_attention_attend_fused(qi=qi, ki=ki, vi=vi, 
                                                         qf=qf, kf=kf, vf=vf, 
                                                         n_head=attn_module.num_attention_heads_per_partition,
                                                         video_n_head=self.mixins[1].video_n_head,
                                                         text_image_len=text_image_len,
                                                         per_frame_len=per_frame_len, 
                                                         attention_mask_i2i=attention_mask_i2i, 
                                                         attention_mask_f2i=attention_mask_f2i, 
                                                         attention_mask_f2f=attention_mask_f2f,
                                                         keymap_i2v=attn_mixin.keymap_i2v[layer_id],
                                                         valmap_i2v=attn_mixin.valmap_i2v[layer_id],
                                                         mask_softmax=attn_mixin.mask_softmax,
                                                         causal_mask_softmax=attn_mixin.causal_mask_softmax,
                                                         attention_dropout_i=dropout_fn,
                                                         attention_dropout_f=dropout_fn_frame,
                                                         layer_id=layer_id)
        else:
            context_image, context_frame = video_attention_attend1(qi=qi, ki=ki, vi=vi, 
                                                            qf=qf, kf=kf, vf=vf, 
                                                            n_head=attn_module.num_attention_heads_per_partition,
                                                            video_n_head=self.mixins[1].video_n_head,
                                                            text_image_len=text_image_len,
                                                            per_frame_len=per_frame_len, 
                                                            attention_mask_i2i=attention_mask_i2i, 
                                                            attention_mask_f2i=attention_mask_f2i, 
                                                            attention_mask_f2f=attention_mask_f2f,
                                                            keymap_i2v=attn_mixin.keymap_i2v[layer_id],
                                                            valmap_i2v=attn_mixin.valmap_i2v[layer_id],
                                                            attention_dropout_i=dropout_fn,
                                                            attention_dropout_f=dropout_fn_frame,
                                                            layer_id=layer_id)
        
        output_image = attn_module.dense(context_image)
        output_frame = attn_mixin.dense[layer_id](context_frame)
                
        return (output_image, output_frame), None
    
    def my_mlp_forward(self, hidden_states, layer_id=None, **kw_tensors):
        image_hidden_states, frame_hidden_states = hidden_states
        intermediate_parallel_image = self.transformer.layers[layer_id].mlp.dense_h_to_4h(image_hidden_states)
        intermediate_parallel_image = gelu(intermediate_parallel_image)
        output_image = self.transformer.layers[layer_id].mlp.dense_4h_to_h(intermediate_parallel_image)
        intermediate_parallel_frame = self.mixins[2].dense_h_to_4h[layer_id](frame_hidden_states)
        intermediate_parallel_frame = gelu(intermediate_parallel_frame)
        output_frame = self.mixins[2].dense_4h_to_h[layer_id](intermediate_parallel_frame)
        
        return output_image, output_frame
    
    def layer_forward(self, hidden_states, mask, layer_id=None, **kw_tensors):
        if layer_id == 0:
            hidden_state_image = hidden_states[..., :self.layout[1], :]
            hidden_state_frame = hidden_states[..., self.layout[1]:, :]
            hidden_state_frame = self.mixins[3].startmap_i2v(hidden_state_frame)
            
        else:
            hidden_state_image = hidden_states[..., :self.layout[1], :]
            hidden_state_frame = hidden_states[..., self.layout[1]:, :].reshape(hidden_states.shape[0], -1, self.video_hidden_size)
            # hidden_state_image, hidden_state_frame = hidden_states
            
        # Layer norm at the begining of the transformer layer.
        layernorm_output1_image = self.transformer.layers[layer_id].input_layernorm(hidden_state_image)
        layernorm_output1_frame = self.mixins[3].frame_input_layernorms[layer_id](hidden_state_frame)
        # self attention
        attention_output, output_this_layer = self.my_attention_forward((layernorm_output1_image, layernorm_output1_frame), 
                                                                    mask, **kw_tensors, layer_id=layer_id)
        attention_output_image, attention_output_frame = attention_output
        # third layernorm
        if self.sandwich_ln:
            attention_output_image = self.transformer.layers[layer_id].third_layernorm(attention_output_image)
            attention_output_frame = self.mixins[3].frame_third_layernorms[layer_id](attention_output_frame)
        # Residual connection.
        layernorm_input_image = hidden_state_image + attention_output_image
        layernorm_input_frame = hidden_state_frame + attention_output_frame
        # Layer norm post the self attention.
        layernorm_output_image = self.transformer.layers[layer_id].post_attention_layernorm(layernorm_input_image)
        layernorm_output_frame = self.mixins[3].frame_post_attention_layernorms[layer_id](layernorm_input_frame)
        # MLP.
        mlp_output_image, mlp_output_frame = self.my_mlp_forward((layernorm_output_image, layernorm_output_frame), layer_id=layer_id)
        # Fourth LayerNorm
        if self.sandwich_ln:
            mlp_output_image = self.transformer.layers[layer_id].fourth_layernorm(mlp_output_image)
            mlp_output_frame = self.mixins[3].frame_fourth_layernorms[layer_id](mlp_output_frame)
        # Second residual connection.
        output_image = layernorm_input_image + mlp_output_image
        output_frame = layernorm_input_frame + mlp_output_frame

        if layer_id != len(self.transformer.layers)-1:
            output_frame = output_frame.reshape(output_frame.shape[0], -1, output_image.shape[-1])
            output = torch.cat((output_image, output_frame), dim=-2)
            return output, output_this_layer
            # return (output_image, output_frame), output_this_layer
        else:
            # the last layer
            output_frame = self.mixins[3].endmap_v2i(output_frame) / math.sqrt(self.video_hidden_size)
            output = torch.cat((output_image, output_frame), dim=-2)
            
            return output, output_this_layer
    
    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VideoModel', 'video model configurations')
        group.add_argument("--layout", type=str, default='64,1089,16464')
        group.add_argument("--new-sequence-length", type=int, default=16464)
        group.add_argument("--video-hidden-size", type=int, default=320)
        group.add_argument("--video-n-head", type=int, default=10)
        group.add_argument("--masked-softmax-fusion", action='store_true', help="enable mask&softmax fusion")
        group.add_argument("--second-frame-scale", type=float, default=1, help="scale second frame loss for larger impact")
        return parser

def video_attention_attend_fused(qi, ki, vi, qf, kf, vf, n_head, video_n_head, text_image_len, per_frame_len, 
                      attention_mask_i2i, attention_mask_f2i, attention_mask_f2f, keymap_i2v, valmap_i2v, 
                      mask_softmax, causal_mask_softmax,
                      attention_dropout_i=None, attention_dropout_f=None, layer_id=None, **kwargs):
    b, si0, hi0 = qi.shape
    b, sf0, hf0 = qf.shape
    hf = hf0 // video_n_head
    frame_num = int(sf0 / per_frame_len)
    assert frame_num * per_frame_len == sf0
    
    # map to video hidden size
    ki_small = keymap_i2v(ki)
    vi_small = valmap_i2v(vi)
        
    # standard attention for image(frame 0) to image
    hi = hi0 // n_head
    qi = qi.reshape(b, si0, n_head, hi).permute(0, 2, 1, 3)
    vi = vi.reshape(b, si0, n_head, hi).permute(0, 2, 1, 3)
    kiT = ki.reshape(b, si0, n_head, hi).permute(0, 2, 3, 1)
    attention_scores = torch.matmul(qi / math.sqrt(qi.shape[-1]), kiT) # 0.21ms

    # attention_scores = torch.mul(attention_scores, attention_mask_i2i) - \
    #                 10000.0 * (1.0 - attention_mask_i2i)
    # attention_probs_image = F.softmax(attention_scores, dim=-1) 
    attention_probs_image = mask_softmax(attention_scores, torch.logical_not(attention_mask_i2i).type(torch.float16))
    
    # can broadcast automatically
    # attend to self
    q11 = qf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 2, 4) # (b, n_head, frame_num, per_frame_len, h)
    k11T = kf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 4, 2)
    v11 = vf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 2, 4)
    attention_scores11 = torch.matmul(q11 / math.sqrt(q11.shape[-1]), k11T)
    attention_scores11 = attention_scores11.reshape(b, video_n_head*frame_num, per_frame_len, per_frame_len)
                    
    # attend to image(frame 0)
    ki_smallT = ki_small.reshape(b, si0, video_n_head, hf).permute(0, 2, 3, 1)
    vi_small = vi_small.reshape(b, si0, video_n_head, hf).transpose(1, 2)
    qf2i = qf.reshape(b, sf0, video_n_head, hf).transpose(1, 2)
    attention_scoresf2i = torch.matmul(qf2i / math.sqrt(qf2i.shape[-1]), ki_smallT) # 中间相乘的那一维变短（总video hidden从320变成40），时间反而增加了
    attention_scoresf2i = attention_scoresf2i.reshape(b, video_n_head*frame_num, per_frame_len, -1)
    attention_probs_f2i = mask_softmax(attention_scoresf2i, torch.logical_not(attention_mask_f2i).type(torch.float16))
    attention_probs_f2i = attention_probs_f2i.reshape(b, video_n_head, sf0, text_image_len)
    # attention_scoresf2i = attention_scoresf2i.reshape(b, video_n_head, frame_num, per_frame_len, si0)
    # attention_scoresf2i = torch.mul(attention_scoresf2i, attention_mask_f2i) - \
    #                 10000.0 * (1.0 - attention_mask_f2i)
    
    # attend to frame before
    attention_scores10 = torch.matmul(q11[:, :, 1:] / math.sqrt(q11.shape[-1]), k11T[:, :, :-1])
    attention_scores10 = torch.cat((torch.full(attention_scores10[:, :, :1].shape, -10000.0, device=attention_scores10.device).type(torch.float16),
                                    attention_scores10), dim=2)
    attention_scores10 = attention_scores10.reshape(b, video_n_head*frame_num, per_frame_len, per_frame_len)
    # attention probs
    all_mask = torch.cat((torch.zeros(attention_mask_f2i.shape[0], attention_mask_f2i.shape[1], per_frame_len, per_frame_len, device=attention_mask_f2i.device), 
                          torch.ones(attention_mask_f2i.shape[0], attention_mask_f2i.shape[1], per_frame_len, per_frame_len, device=attention_mask_f2i.device).triu(diagonal=1)
                          ), dim=-1)
    attention_score_f2f = torch.cat((attention_scores10, 
                                      attention_scores11), 
                                      dim=-1) # 这个cat和下面的softmax很费时间，softmax要3.5ms
    
    attention_probs_f2f = mask_softmax(attention_score_f2f, all_mask)
    attention_probs_f2f = attention_probs_f2f.reshape(b, video_n_head, frame_num, per_frame_len, 2*per_frame_len)

    if attention_dropout_i is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_image = attention_dropout_i(attention_probs_image)
    if attention_dropout_f is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_f2i = attention_dropout_f(attention_probs_f2i)
            attention_probs_f2f = attention_dropout_f(attention_probs_f2f) #5526MB

    
    # images' context
    context_image = torch.matmul(attention_probs_image, vi) # [b, n_head, si0, hi] 
    # frame's context
    contextf2i = torch.matmul(attention_probs_f2i, vi_small)
    context11 = torch.matmul(attention_probs_f2f[..., per_frame_len:], v11)
    v10 = torch.cat((v11[:, :, -1:, :, :], v11[:, :, :-1, :, :]), dim=2)
    context10 = torch.matmul(attention_probs_f2f[..., :per_frame_len], v10)
    context_frame = (context11+context10).reshape(b, video_n_head, sf0, hf)+contextf2i

    return context_image.transpose(1, 2).reshape(b, si0, hi0), context_frame.transpose(1, 2).reshape(b, sf0, hf0)
    
    
def video_attention_attend1(qi, ki, vi, qf, kf, vf, n_head, video_n_head, text_image_len, per_frame_len, 
                      attention_mask_i2i, attention_mask_f2i, attention_mask_f2f, keymap_i2v, valmap_i2v, 
                      attention_dropout_i=None, attention_dropout_f=None, layer_id=None, **kwargs):
    b, si0, hi0 = qi.shape
    b, sf0, hf0 = qf.shape
    hf = hf0 // video_n_head
    frame_num = int(sf0 / per_frame_len)
    assert frame_num * per_frame_len == sf0
    
    # map to video hidden size
    ki_small = keymap_i2v(ki)
    vi_small = valmap_i2v(vi)
        
    # standard attention for image(frame 0) to image
    hi = hi0 // n_head
    qi = qi.reshape(b, si0, n_head, hi).permute(0, 2, 1, 3)
    vi = vi.reshape(b, si0, n_head, hi).permute(0, 2, 1, 3)
    kiT = ki.reshape(b, si0, n_head, hi).permute(0, 2, 3, 1)
    attention_scores = torch.matmul(qi / math.sqrt(qi.shape[-1]), kiT) # 0.21ms

    attention_scores = torch.mul(attention_scores, attention_mask_i2i) - \
                    10000.0 * (1.0 - attention_mask_i2i)
    attention_probs_image = F.softmax(attention_scores, dim=-1) 
    
    # special attention for other frames
    attention_mask_f2i = attention_mask_f2i.unsqueeze(2)
    attention_mask_f2f = attention_mask_f2f.unsqueeze(2)
    # can broadcast automatically
    # attend to self
    q11 = qf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 2, 4) # (b, n_head, frame_num, per_frame_len, h)
    k11T = kf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 4, 2)
    v11 = vf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 2, 4)
    attention_scores11 = torch.matmul(q11 / math.sqrt(q11.shape[-1]), k11T) #0.4ms(video hidden=320)
    attention_scores11 = torch.mul(attention_scores11, attention_mask_f2f) - \
                    10000.0 * (1.0 - attention_mask_f2f)
    # attention_scores11 = torch.mul(attention_scores11, attention_mask_f2f)
    # attention_scores11 -= 10000.0 * (1.0 - attention_mask_f2f)
                    
    # attend to image(frame 0)
    ki_smallT = ki_small.reshape(b, si0, video_n_head, hf).permute(0, 2, 3, 1)
    vi_small = vi_small.reshape(b, si0, video_n_head, hf).transpose(1, 2)
    qf2i = qf.reshape(b, sf0, video_n_head, hf).transpose(1, 2)
    attention_scoresf2i = torch.matmul(qf2i / math.sqrt(qf2i.shape[-1]), ki_smallT) # 中间相乘的那一维变短（总video hidden从320变成40），时间反而增加了
    attention_scoresf2i = attention_scoresf2i.reshape(b, video_n_head, frame_num, per_frame_len, si0)
    attention_scoresf2i = torch.mul(attention_scoresf2i, attention_mask_f2i) - \
                    10000.0 * (1.0 - attention_mask_f2i)
    # attend to frame before
    attention_scores10 = torch.matmul(q11[:, :, 1:] / math.sqrt(q11.shape[-1]), k11T[:, :, :-1])
    # frame_before_midtime = time.time()
    attention_scores10 = torch.cat((torch.full(attention_scores10[:, :, :1].shape, -10000.0, device=attention_scores10.device).type(torch.float16),
                                    attention_scores10), dim=2)
    # attention probs
    attention_score_frame = torch.cat((attention_scoresf2i, 
                                      attention_scores10, 
                                      attention_scores11), 
                                      dim=-1) # 这个cat和下面的softmax很费时间，softmax要3.5ms
    attention_probs_frame = F.softmax(attention_score_frame, dim=-1)

    if attention_dropout_i is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_image = attention_dropout_i(attention_probs_image)
    if attention_dropout_f is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_frame = attention_dropout_f(attention_probs_frame) #5526MB

    
    # images' context
    context_image = torch.matmul(attention_probs_image, vi) # [b, n_head, si0, hi] # L1 1.3ms
    # frame's context
    contextf2i = torch.matmul(attention_probs_frame[..., :text_image_len].reshape(b, video_n_head, sf0, text_image_len), vi_small) # L1 0.91ms
    context11 = torch.matmul(attention_probs_frame[..., text_image_len+per_frame_len:], v11) # L1 0.91ms
    v10 = torch.cat((v11[:, :, -1:, :, :], v11[:, :, :-1, :, :]), dim=2)
    context10 = torch.matmul(attention_probs_frame[..., text_image_len:text_image_len+per_frame_len], v10) # L1 0.91ms
    context_frame = (context11+context10).reshape(b, video_n_head, sf0, hf)+contextf2i

    return context_image.transpose(1, 2).reshape(b, si0, hi0), context_frame.transpose(1, 2).reshape(b, sf0, hf0)
    
    