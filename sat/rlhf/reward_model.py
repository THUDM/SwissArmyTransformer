import torch
import torch.nn as nn

from sat import get_tokenizer
from sat.model.transformer import LayerNorm
from sat.model.base_model import BaseModel, BaseMixin


class rm_head(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size, layernorm_epsilon=1.0e-5):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return x


class RewardModelFinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rm_head = rm_head(vocab_size, hidden_size)

    def final_forward(self, logits, **kwargs):
        return self.rm_head(logits)


class RewardModel(BaseModel):

    def __init__(self, base_model, args, num_padding_at_beginning=0, **kwargs):
        super(RewardModel, self).__init__(args, transformer=base_model, **kwargs)

        self.add_mixin("rm-final", RewardModelFinalMixin(args.vocab_size, args.hidden_size))

        self.args = args
        self.num_padding_at_beginning = num_padding_at_beginning
        
        tokenizer = get_tokenizer(args)
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        **kwargs
    ):
        # modified from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
        loss = None

        rewards = super(RewardModel, self).forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs
        )[0].squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        return_value_only=False,
        prompt_length=0,
        **kwargs
    ):
        # modified from https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
        values = super(RewardModel, self).forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs
        )[0].squeeze(-1)

        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
