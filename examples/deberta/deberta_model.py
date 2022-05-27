# -*- encoding: utf-8 -*-
# @File    :   deberta_model.py
# @Time    :   2022/5/7
# @Author  :   Zhuoyi Yang
# @Contact :   yangzhuo18@mails.tsinghua.edu.cn
import torch
from SwissArmyTransformer.model.base_model import BaseMixin, BaseModel
from SwissArmyTransformer.mpu.transformer import LayerNorm
import torch.nn as nn
from SwissArmyTransformer.mpu.transformer import standard_attention
from SwissArmyTransformer.mpu.utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from torch import _softmax_backward_data, nn
from packaging import version
import math

is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")

deberta_gelu = nn.functional.gelu

def softmax_backward_data(parent, grad_output, output, dim, self):
    """
    A function that calls the internal `_softmax_backward_data` PyTorch method and that adjusts the arguments according
    to the torch version detected.
    """

    if is_torch_less_than_1_11:
        return _softmax_backward_data(grad_output, output, parent.dim, self)
    else:
        return _softmax_backward_data(grad_output, output, parent.dim, self.dtype)

def build_relative_position(query_size, key_size, device):
    """
    Build relative position according to the query and key

    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """

    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta.modeling_deberta import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Byte"],
        )
        output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf"))))
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))

@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))

class DebertaAttentionMixin(BaseMixin):
    def __init__(self, args, transformer=None, **kwargs):
        super(DebertaAttentionMixin, self).__init__()
        self.max_relative_positions = args.max_relative_positions
        if self.max_relative_positions < 1:
            self.max_relative_positions = args.max_sequence_length

        self.pos_att_type=["c2p", "p2c"]
        layer_num = args.num_layers

        self.module = nn.ModuleList([
            nn.ModuleDict()
            for layer_id in range(layer_num)
        ])
        self.para = nn.ModuleList([
            nn.ParameterDict()
            for layer_id in range(layer_num)
        ])

        self.num_attention_heads = args.num_attention_heads
        self.all_head_size = args.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        for i in range(layer_num):
            # self.para[i]["rel_embeddings"] = nn.Embedding(args.max_relative_positions * 2, args.hidden_size)
            if "c2p" in self.pos_att_type:
                self.module[i]["pos_proj"] = nn.Linear(args.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type:
                self.module[i]["pos_q_proj"] = nn.Linear(args.hidden_size, self.all_head_size)
        for i in range(layer_num):
            self.para[i]["q_bias"] = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
            self.para[i]["v_bias"] = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor, module_dict):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
                         self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
                         ].unsqueeze(0)

        score = 0

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer = module_dict["pos_proj"](rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer = module_dict["pos_q_proj"](rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)

            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            score += p2c_att
            return score

    def attention_forward(self, hidden_states, mask, layer_id, relative_pos, rel_embeddings, **kw_args):
        module_dict = self.module[layer_id]
        para_dict = self.para[layer_id]
        # relative_pos, rel_embeddings = kw_args['rel']
        origin_attention = self.transformer.layers[layer_id].attention

        qp = origin_attention.query_key_value(hidden_states)  # .split(self.all_head_size, dim=-1)
        query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)

        query_layer = query_layer + self.transpose_for_scores(para_dict["q_bias"][None, None, :])
        value_layer = value_layer + self.transpose_for_scores(para_dict["v_bias"][None, None, :])
        # breakpoint()
        # query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        # value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor, module_dict)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        attention_probs = XSoftmax.apply(attention_scores, mask, -1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = origin_attention.dense(context_layer)
        context_layer = origin_attention.output_dropout(context_layer)

        return context_layer


class deberta_lm_head(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, layernorm_epsilon=1e-7):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.dense(x)
        x = deberta_gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class DebertaFinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = deberta_lm_head(vocab_size, hidden_size)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)

class DebertaModel(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(DebertaModel, self).__init__(args, transformer=transformer, activation_func=deberta_gelu, **kwargs)
        self.add_mixin("deberta-attention", DebertaAttentionMixin(args))
        self.add_mixin("deberta-final", DebertaFinalMixin(args.vocab_size, args.hidden_size))
        self.max_relative_positions = args.max_relative_positions
        if self.max_relative_positions < 1:
            self.max_relative_positions = args.max_sequence_length
        self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, args.hidden_size)

    def get_rel_pos(self, hidden_states):
        q = hidden_states.size(-2)
        relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    def layer_forward(self, hidden_states, mask, *args, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        if kw_args['layer_id'] == 0:
            relative_pos = self.get_rel_pos(hidden_states)
            rel_embeddings = self.rel_embeddings.weight
            kw_args['relative_pos'] = relative_pos
            kw_args['rel_embeddings'] = rel_embeddings
            output_cross_layer = {'relative_pos': relative_pos, 'rel_embeddings':rel_embeddings}
        else:
            output_cross_layer = {'relative_pos': kw_args['relative_pos'], 'rel_embeddings':kw_args['rel_embeddings']}
        layer = self.transformer.layers[kw_args['layer_id']]
        # Layer norm at the begining of the transformer layer.
        # breakpoint()
        hidden_states = layer.input_layernorm(hidden_states)
        # Self attention.
        attention_output = layer.attention(hidden_states, mask, **kw_args)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = layer.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = layer.mlp(layernorm_output, **kw_args)

        # Second residual connection.
        output = layernorm_output + mlp_output

        return output, kw_args['output_this_layer'], output_cross_layer

    @classmethod
    def add_model_specific_args(cls, parser):
        """Arguments for Deberta"""
        group = parser.add_argument_group('deberta', 'deberta Configurations')
        group.add_argument('--max-relative-positions', type=int, default=-1)
        return parser