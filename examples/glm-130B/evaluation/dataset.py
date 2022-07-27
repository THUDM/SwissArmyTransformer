import os
import json

import numpy as np
import torch

from SwissArmyTransformer import get_tokenizer
from scipy.linalg import block_diag


def build_dataset(path, unified_multitask_encoding=False):
    """Helper function to select and build dataset."""
    return ZeroShotDataset(path, unified_multitask_encoding=unified_multitask_encoding)


def pad_batch(tokens, targets, position_ids, attention_mask, max_seq_length=None):
    assert len(tokens) <= max_seq_length
    attention_mask.append(np.zeros((max_seq_length - len(tokens), max_seq_length - len(tokens)), dtype=np.long))
    tokens = np.concatenate((tokens, np.zeros(max_seq_length - len(tokens), dtype=np.long)))
    targets = np.concatenate((targets, np.zeros(max_seq_length - len(targets), dtype=np.long)))
    position_ids = np.concatenate((position_ids, np.zeros(max_seq_length - len(position_ids), dtype=np.long)))
    return tokens, targets, position_ids, attention_mask


def build_multiple_choice_sample(text, choices, is_single_token, unified_multitask_encoding=False):
    tokenizer = get_tokenizer()

    dtype = np.int64
    sop_id = tokenizer.get_command("sop")
    mask_id = tokenizer.get_command("[MASK]")

    token = np.array(text, dtype=dtype)
    target = np.array(text, dtype=dtype)
    position_id = np.arange(len(text), dtype=dtype)
    choice_target_id = []

    blank_filling = mask_id in text
    if not blank_filling:
        mask_position = len(token)
        token = np.concatenate((token, [mask_id]))
        target = np.concatenate((target, [mask_id]))
        position_id = np.concatenate((position_id, [mask_position]))
    else:
        mask_position = text.index(mask_id)

    division = len(token)
    attention_mask = [np.ones((len(token), len(token)), dtype=dtype)]

    for choice in choices:
        position_id = np.concatenate(
            (
                position_id,
                [mask_position] * len(choice)
                if blank_filling or not unified_multitask_encoding
                else np.arange(mask_position, mask_position + len(choice), dtype=dtype),
            )
        )
        choice_target_id.append(np.arange(len(token), len(token) + len(choice), dtype=dtype))
        attention_mask.append(np.tril(np.ones((len(choice), len(choice)), dtype=np.long)))
        token = np.concatenate((token, [sop_id], choice[:-1]))
        target = np.concatenate((target, choice))

        if is_single_token:
            break

    # pad batch
    seq_length = len(token)
    TILE = 32
    token, target, position_id, attention_mask = pad_batch(
        token, target, position_id, attention_mask, ((seq_length + TILE - 1) // TILE) * TILE
    )

    attention_mask = block_diag(*attention_mask)
    attention_mask[:seq_length, :division] = 1

    item = {
        "tokens": token,
        "targets": target,
        "position_ids": position_id,
        "attention_mask": attention_mask < 0.5,
        "choice_target_ids": choice_target_id,
        "is_single_token": is_single_token,
    }
    if is_single_token:
        item["choices"] = np.array(choices, dtype=dtype).squeeze()
    return item


def build_generation_sample(text, max_seq_length, use_task_mask, unidirectional=True):
    tokenizer = get_tokenizer()

    dtype = np.int64
    sop_id = tokenizer.get_command("sop")
    mask_id = tokenizer.get_command("[gMASK]") if use_task_mask else tokenizer.get_command("[MASK]")

    token = np.array(text, dtype=dtype)

    blank_filling = mask_id in text
    if blank_filling:
        assert not unidirectional, "Unidirectional attention doesn't support blank filling"
        assert not use_task_mask, "Unidirectional attention doesn't support task mask"
        mask_position = text.index(mask_id)
        token = np.concatenate((token, [sop_id]))
    else:
        mask_position = len(token)
        if unidirectional:
            token = np.concatenate(([mask_id, sop_id], token))
        else:
            token = np.concatenate((token, [mask_id, sop_id]))
    context_length = len(token)

    position_id = np.arange(0, max_seq_length, dtype=dtype)
    if not use_task_mask:
        position_id[context_length - 1 :] = mask_position

    attention_mask = np.tril(np.ones((max_seq_length, max_seq_length), dtype=np.long))
    if not unidirectional:
        attention_mask[: context_length - 1, : context_length - 1] = 1

    item = {
        "tokens": np.concatenate((token, np.zeros(max_seq_length - len(token), dtype=np.long))),
        "position_ids": position_id,
        "attention_mask": attention_mask < 0.5,
        "context_length": context_length,
    }
    return item


class ZeroShotDataset(torch.utils.data.Dataset):
    """
    Jsonlines of {
        "text": context
        "choices": [choice_id1,...], if not None, len(target) == 1
        "label": If generation task -1, else [0, len(choices))
        "type": 'mul' | 'gen'
    }
    If [MASK] not in context, will append [MASK] after text
    """

    def __init__(
        self, path, max_seq_length=2048, use_task_mask=False, unidirectional=False, unified_multitask_encoding=False
    ):
        self.path = path
        self.max_seq_length = max_seq_length
        self.data = []

        self.dtype = np.long

        tokenizer = get_tokenizer(tokenizer_type="icetk-glm-130B")
        self.mask_id = tokenizer.get_command("[MASK]")
        self.gmask_id = tokenizer.get_command("[gMASK]")
        self.use_task_mask = use_task_mask
        self.unidirectional = unidirectional

        self.unified_multitask_encoding = unified_multitask_encoding

        self.task_type = None

        with open(os.path.join(path), "r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if "task_type" not in item:
                    item["task_type"] = "mul"
                if self.task_type == None:
                    self.task_type = item["task_type"]
                else:
                    assert self.task_type == item["task_type"]
                if self.task_type == "mul":
                    text, choices, label = item["inputs"], item["choices"], item["label"]

                    tgt_seq_length = sum([len(choice) for choice in choices])
                    if tgt_seq_length == len(choices):
                        # For single token, we only insert one [sop]
                        tgt_seq_length = 1

                    assert tgt_seq_length < max_seq_length
                    if len(text) + tgt_seq_length + 2 > max_seq_length:
                        text_length = max_seq_length - tgt_seq_length - 2
                        text = text[len(text) - text_length : len(text)]

                    assert not (
                        self.mask_id in text and self.unified_multitask_encoding
                    ), "Unified multitask encoding don't support blank filling"

                    self.data.append(
                        {
                            "text": text,
                            "choices": choices,
                            "label": label,
                            "is_single_token": tgt_seq_length == 1,
                        }
                    )
                elif self.task_type == "gen":
                    text, targets = item["inputs"], item["targets"]
                    max_tgt_seq_length = max([len(target) for target in targets])
                    if len(text) + max_tgt_seq_length + 2 > max_seq_length:
                        text_length = max_seq_length - max_tgt_seq_length - 2
                        text = text[len(text) - text_length : len(text)]
                    self.data.append({"text": text, "targets": targets, "task_type": "gen"})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.task_type == "mul":  # Multi-choice data
            sample = build_multiple_choice_sample(
                item["text"],
                item["choices"],
                item["is_single_token"],
                unified_multitask_encoding=self.unified_multitask_encoding,
            )
            sample["label"] = item["label"]
            return sample
        elif self.task_type == "gen":  # generative data
            sample = build_generation_sample(
                item["text"],
                max_seq_length=self.max_seq_length,
                use_task_mask=self.use_task_mask,
                unidirectional=self.unidirectional,
            )
            sample["targets"] = [np.array(target, dtype=self.dtype) for target in item["targets"]]
            return sample
        else:
            assert False
