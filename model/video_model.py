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


from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, VideoAttentionMixin

from mpu.transformer import split_tensor_along_last_dim
from mpu.local_attention_function import f_similar, f_weighting
from mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker



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
        ))
        self.layout = args.layout
        # [PAD]... [ROI1] text ...  {layout[0]} [BOI1] 1024 {layout[1]} [BOI2] 1024 [BOI3] 1024 [EOI2] 1024 ... (16 frames)[POS8]1024 {layout[2]}
        # output ends with [EOI1]
        # frame beginer: [BOI1][BOI2][BOI3][EOI2][EOI3][ROI2][ROI3][POS0][POS1][POS2][POS3][POS4][POS5][POS6][POS7][POS8]
        # self.frame_nums = args.frame_nums
        # attend to 1 frame before
        
    def position_embedding_forward(self, position_ids, *other_tensors):
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
    
    def attention_forward(self, hidden_states, mask, *other_tensors, layer_id=None):
        per_frame_len = self.layout[1]-self.layout[0]
        text_image_len = self.layout[1]
        attn_module = self.transformer.layers[layer_id].attention
        query_key_value_plus = self.mixins[1].query_key_value[layer_id] 
        densemap_i2v =self.mixins[1].densemap_i2v
        densemap_v2i = self.mixins[1].densemap_v2i
        attention_mask_i2i = mask[..., :self.layout[1], :]
        attention_mask_f2i = mask[..., self.layout[1]:, :]
        attention_mask_f2f = torch.ones((attention_mask_f2i.shape[0], 1, per_frame_len, per_frame_len), 
                                        device=attention_mask_i2i.device).type(torch.float16)
        attention_mask_f2f.tril_()
        attention_mask_f2f.unsqueeze(0).unsqueeze(0)
        
        hidden_states_plus = hidden_states[:, self.layout[1]:]
        hidden_states = hidden_states[:, :self.layout[1]]
        # base model qkv
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        # breakpoint()
        qi, ki, vi = split_tensor_along_last_dim(mixed_raw_layer, 3)
        # video frames qkv
        hidden_states_plus = densemap_i2v(hidden_states_plus)/math.sqrt(hidden_states_plus.shape[-1])
        # hidden_states_plus = densemap_i2v(hidden_states_plus/50.6)
        # hidden_states_plus = densemap_i2v(hidden_states_plus)
        mixed_raw_layer = query_key_value_plus(hidden_states_plus)
        qf, kf, vf = split_tensor_along_last_dim(mixed_raw_layer, 3)
        # 初始化到和原来一样!!!

        dropout_fn = attn_module.attention_dropout if self.training else None
              
        # video attention: attend to text+lastframe+self
        context_image, context_frame = video_attention_attend1(qi=qi, ki=ki, vi=vi, 
                                                         qf=qf, kf=kf, vf=vf, 
                                                         n_head=attn_module.num_attention_heads_per_partition,
                                                         video_n_head=self.mixins[1].video_n_head,
                                                         text_image_len=text_image_len,
                                                         per_frame_len=per_frame_len, 
                                                         attention_mask_i2i=attention_mask_i2i, 
                                                         attention_mask_f2i=attention_mask_f2i, 
                                                         attention_mask_f2f=attention_mask_f2f,
                                                         densemap_i2v=densemap_i2v,
                                                         attention_dropout=dropout_fn)
        
        output_image = attn_module.dense(context_image)
        output_frame = attn_module.dense(densemap_v2i(context_frame)/math.sqrt(context_frame.shape[-1]))
        # output_frame = attn_module.dense(densemap_v2i(context_frame))/17.9
        output_frame = attn_module.dense(densemap_v2i(context_frame))
        output = torch.cat((output_image, output_frame), dim=1)
        
        return output, None
    
    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VideoModel', 'video model configurations')
        group.add_argument("--layout", type=str, default='64,1089,16464')
        group.add_argument("--new-sequence-length", type=int, default=16464)
        group.add_argument("--video-hidden-size", type=int, default=320)
        group.add_argument("--video-n-head", type=int, default=10)
        return parser

def video_attention_attend1(qi, ki, vi, qf, kf, vf, n_head, video_n_head, text_image_len, per_frame_len, 
                      attention_mask_i2i, attention_mask_f2i, attention_mask_f2f, densemap_i2v, attention_dropout=None, **kwargs):
    b, si0, hi0 = qi.shape
    b, sf0, hf0 = qf.shape
    hf = hf0 // video_n_head
    frame_num = int(sf0 / per_frame_len)
    assert frame_num * per_frame_len == sf0
    
    # map to video hidden size
    ki_small = densemap_i2v(ki)/math.sqrt(ki.shape[-1])
    # ki_small = densemap_i2v(ki)/50.6
    # ki_small = densemap_i2v(ki)
    vi_small = densemap_i2v(vi)/math.sqrt(vi.shape[-1])
    # vi_small = densemap_i2v(vi)/50.6
    # vi_small = densemap_i2v(vi)
    
    # standard attention for image(frame 0) to image
    hi = hi0 // n_head
    qi = qi.reshape(b, si0, n_head, hi).permute(0, 2, 1, 3)
    vi = vi.reshape(b, si0, n_head, hi).permute(0, 2, 1, 3)
    kiT = ki.reshape(b, si0, n_head, hi).permute(0, 2, 3, 1)
    attention_scores = torch.matmul(qi / math.sqrt(qi.shape[-1]), kiT) # 97MB

    attention_scores = torch.mul(attention_scores, attention_mask_i2i) - \
                    10000.0 * (1.0 - attention_mask_i2i) # 92MB
    attention_probs_image = F.softmax(attention_scores, dim=-1) # 92MB
    
    # special attention for other frames
    attention_mask_f2i = attention_mask_f2i.unsqueeze(2)
    attention_mask_f2f = attention_mask_f2f.unsqueeze(2)
    # can broadcast automatically
    # attend to self
    q11 = qf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 2, 4) # (b, n_head, frame_num, per_frame_len, h)
    k11T = kf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 4, 2)
    v11 = vf.reshape(b, frame_num, per_frame_len, video_n_head, hf).permute(0, 3, 1, 2, 4)
    attention_scores11 = torch.matmul(q11 / math.sqrt(q11.shape[-1]), k11T) #1204MB
    attention_scores11 = torch.mul(attention_scores11, attention_mask_f2f) - \
                    10000.0 * (1.0 - attention_mask_f2f)    # 1204MB+1204
    # attend to image(frame 0)
    ki_smallT = ki_small.reshape(b, si0, video_n_head, hf).permute(0, 2, 3, 1)
    vi_small = vi_small.reshape(b, si0, video_n_head, hf).transpose(1, 2)
    qf2i = qf.reshape(b, sf0, video_n_head, hf).transpose(1, 2)
    attention_scoresf2i = torch.matmul(qf2i / math.sqrt(qf2i.shape[-1]), ki_smallT) #1278MB
    attention_scoresf2i = attention_scoresf2i.reshape(b, video_n_head, frame_num, per_frame_len, si0)
    attention_scoresf2i = torch.mul(attention_scoresf2i, attention_mask_f2i) - \
                    10000.0 * (1.0 - attention_mask_f2i) #1278MB+1278MB
    # attend to frame before
    # in place!!!
    k10T = torch.cat((k11T[:, :, -1:, :, :], k11T[:, :, :-1, :, :]), dim=2)
    attention_scores10 = torch.matmul(q11 / math.sqrt(q11.shape[-1]), k10T)
    attention_scores10[:, :, 0] = -10000.0 # specially set frame 1
    # attention probs
    attention_score_frame = torch.cat((attention_scoresf2i, 
                                      attention_scores10, 
                                      attention_scores11), 
                                      dim=-1) #3684MB
    attention_probs_frame = F.softmax(attention_score_frame, dim=-1) #3684MB

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs_frame = attention_dropout(attention_probs_frame) #5526MB
            attention_probs_image = attention_dropout(attention_probs_image)

    
    # images' context
    context_image = torch.matmul(attention_probs_image, vi) # [b, n_head, si0, hi]
    # frame's context
    contextf2i = torch.matmul(attention_probs_frame[..., :text_image_len].reshape(b, video_n_head, sf0, text_image_len), vi_small)
    context11 = torch.matmul(attention_probs_frame[..., text_image_len+per_frame_len:], v11)
    v10 = torch.cat((v11[:, :, -1:, :, :], v11[:, :, :-1, :, :]), dim=2)
    context10 = torch.matmul(attention_probs_frame[..., text_image_len:text_image_len+per_frame_len], v10)
    context_frame = (context11+context10).reshape(b, video_n_head, sf0, hf)+contextf2i
    # if context_frame.isnan().any():
    #     breakpoint()
    return context_image.transpose(1, 2).reshape(b, si0, hi0), context_frame.transpose(1, 2).reshape(b, sf0, hf0)
    
    