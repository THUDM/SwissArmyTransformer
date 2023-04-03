import math
import copy
import os
import warnings
import re
import sys
from transformers.utils import logging
logger = logging.get_logger(__name__)

import torch
import torch.nn as nn
from transformers import GenerationMixin
from SwissArmyTransformer import AutoModel
from typing import Optional, Tuple, Union, List, Callable
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.generation.logits_process import LogitsProcessor
class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 20005] = 5e4
        return scores

from transformers import AutoConfig

class ChatModel(nn.Module, GenerationMixin):
    def __init__(self, args, model=None):
        super().__init__()
        self.position_encoding_2d = True
        self.config = AutoConfig.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        if model is None:
            self.model, args = AutoModel.from_pretrained(args, "chatglm-6b")
        else:
            self.model = model
        self.device = self.model.parameters().__next__().device
        self.main_input_name = 'input_ids'
    
    @classmethod
    def from_pretrained(cls, args, name, base_cls=None, *, home_path=None, url=None, prefix='', **kwargs):
        if base_cls is None:
            model, args = AutoModel.from_pretrained(args, name, home_path=home_path, url=url, prefix=prefix, **kwargs)
        else:
            model, args = base_cls.from_pretrained(args, name, home_path=home_path, url=url, prefix=prefix, **kwargs)
        return cls(args, model), args
    
    def can_generate(self):
        return True

    def get_masks_and_position_ids(self, input_ids, mask_positions, device, gmask=False):
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
        attention_mask.tril_()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask.unsqueeze_(1)
        # attention_mask = (attention_mask < 0.5).bool()

        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).expand(batch_size, seq_length)
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [torch.cat((
                torch.zeros(context_length, dtype=torch.long, device=device),
                torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
            )) for context_length in context_lengths]
            block_position_ids = torch.stack(block_position_ids, dim=0)
            position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).expand(batch_size, seq_length)
            if not gmask:
                for i, context_length in enumerate(context_lengths):
                    position_ids[context_length:] = mask_positions[i]

        return attention_mask, position_ids

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs
    ) -> dict:
        batch_size, seq_length = input_ids.shape
        MASK, gMASK = 150000, 150001
        mask_token = MASK if MASK in input_ids else gMASK
        use_gmask = False if MASK in input_ids else gMASK
        seqs = input_ids.tolist()
        mask_positions = [seq.index(mask_token) for seq in seqs]

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
            last_token = input_ids[:, -1].unsqueeze(-1)
            if self.position_encoding_2d:
                position_ids = torch.tensor(
                    [[mask_position, seq_length - context_length] for mask_position, context_length in
                     zip(mask_positions, context_lengths)], dtype=torch.long, device=input_ids.device).unsqueeze(-1)
            else:
                position_ids = torch.tensor([mask_position for mask_position in mask_positions], dtype=torch.long,
                                            device=input_ids.device).unsqueeze(-1)

            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
            }
        else:
            attention_mask, position_ids = self.get_masks_and_position_ids(
                input_ids,
                mask_positions=mask_positions,
                device=input_ids.device,
                gmask=use_gmask
            )

            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            **kw_args
    ):
        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_layers)

            if attention_mask is None:
                attention_mask = self.model.get_masks(
                    input_ids=input_ids,
                    device=input_ids.device
                )

            if position_ids is None:
                MASK, gMASK = 150000, 150001
                mask_token = MASK if MASK in input_ids else gMASK
                use_gmask = False if MASK in input_ids else gMASK

                mask_positions = [seq.tolist().index(mask_token) for seq in input_ids]
                position_ids = self.model.get_position_ids(
                    input_ids=input_ids,
                    mask_positions=mask_positions,
                    device=input_ids.device,
                    gmask=use_gmask
                )
        if attention_mask is None:
            attention_mask = torch.ones(1, 1, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(input_ids.device)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

        lm_logits = outputs[0]
        past_key_values = [x['past_key_values'] for x in outputs[1:]]

        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=past_key_values
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    @torch.no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        input_ids = tokenizer([prompt], return_tensors="pt", padding=True)
        input_ids = input_ids.to(self.device)
        outputs = self.generate(**input_ids, **gen_kwargs)
        outputs = outputs.tolist()[0][len(input_ids["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history
