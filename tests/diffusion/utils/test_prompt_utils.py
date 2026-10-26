# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.utils.prompt_utils import (
    pre_tokenized_negative_prompt_ids,
    pre_tokenized_prompt_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reads_prompt_ids():
    assert pre_tokenized_prompt_ids({"prompt_ids": [1, 2, 3]}) == [1, 2, 3]


def test_reads_native_prompt_token_ids():
    assert pre_tokenized_prompt_ids({"prompt_token_ids": [4, 5]}) == [4, 5]
    assert pre_tokenized_prompt_ids({"prompt_ids": [1], "prompt_token_ids": [2]}) == [1]


def test_accepts_a_tuple_of_token_ids():
    assert pre_tokenized_prompt_ids({"prompt_ids": (6, 7)}) == [6, 7]


def test_unwraps_a_single_batched_prompt():
    assert pre_tokenized_prompt_ids({"prompt_ids": [[1, 2, 3]]}) == [1, 2, 3]


@pytest.mark.parametrize("ids", [[[]], [()], ([],), ((),)])
@pytest.mark.parametrize(
    ("key", "read_ids"),
    [("prompt_ids", pre_tokenized_prompt_ids), ("negative_prompt_ids", pre_tokenized_negative_prompt_ids)],
)
def test_rejects_empty_batched_prompt(ids, key, read_ids):
    with pytest.raises(ValueError, match=f"`{key}` must be a non-empty list or tuple of token ids"):
        read_ids({key: ids})


def test_rejects_more_than_one_prompt():
    with pytest.raises(ValueError, match="carries a single prompt"):
        pre_tokenized_prompt_ids({"prompt_ids": [[1], [2]]})


def test_reads_negative_prompt_ids():
    assert pre_tokenized_negative_prompt_ids({"negative_prompt_ids": [7]}) == [7]


def test_reads_native_negative_prompt_token_ids():
    assert pre_tokenized_negative_prompt_ids({"negative_prompt_token_ids": [8]}) == [8]
    assert pre_tokenized_negative_prompt_ids({"negative_prompt_ids": [1], "negative_prompt_token_ids": [2]}) == [1]


def test_positive_ids_are_not_read_as_negative_ids():
    assert pre_tokenized_negative_prompt_ids({"prompt_ids": [1]}) is None


def test_negative_ids_are_not_read_as_positive_ids():
    assert pre_tokenized_prompt_ids({"negative_prompt_ids": [1]}) is None


def test_empty_ids_are_absent():
    assert pre_tokenized_prompt_ids({"prompt_ids": []}) is None


def test_text_only_prompt_has_no_ids():
    assert pre_tokenized_prompt_ids({"prompt": "a cat"}) is None


@pytest.mark.parametrize("prompt", [None, "a cat", ["a", "cat"], 7])
def test_non_dict_prompt_has_no_ids(prompt: object):
    assert pre_tokenized_prompt_ids(prompt) is None
    assert pre_tokenized_negative_prompt_ids(prompt) is None


@pytest.mark.parametrize(
    ("key", "mask_key", "read_ids"),
    [
        ("prompt_ids", "prompt_mask", pre_tokenized_prompt_ids),
        ("prompt_token_ids", "prompt_mask", pre_tokenized_prompt_ids),
        ("negative_prompt_ids", "negative_prompt_mask", pre_tokenized_negative_prompt_ids),
        ("negative_prompt_token_ids", "negative_prompt_mask", pre_tokenized_negative_prompt_ids),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "mask_values,expected", [([1, 1, 0], [7, 8]), ([0, 1, 1], [8, 9]), ([1, 0, 1], [7, 9]), ([1, 1, 1], [7, 8, 9])]
)
def test_applies_attention_mask(key, mask_key, read_ids, batched, mask_values, expected):
    ids = [[7, 8, 9]] if batched else [7, 8, 9]
    mask = torch.tensor([mask_values] if batched else mask_values)
    assert read_ids({key: ids, mask_key: mask}) == expected


@pytest.mark.parametrize(
    ("key", "mask_key", "read_ids"),
    [
        ("prompt_ids", "prompt_mask", pre_tokenized_prompt_ids),
        ("prompt_token_ids", "prompt_mask", pre_tokenized_prompt_ids),
        ("negative_prompt_ids", "negative_prompt_mask", pre_tokenized_negative_prompt_ids),
        ("negative_prompt_token_ids", "negative_prompt_mask", pre_tokenized_negative_prompt_ids),
    ],
)
@pytest.mark.parametrize(
    "mask_values", [[], [1], [[1, 1, 1], [1, 1, 1]], [[[1, 1, 1]]], [1, 0.5, 1], [1, float("nan"), 1], [0, 0, 0]]
)
def test_rejects_invalid_attention_mask(key, mask_key, read_ids, mask_values):
    with pytest.raises(ValueError, match=mask_key):
        read_ids({key: [7, 8, 9], mask_key: torch.tensor(mask_values)})


def test_positive_and_negative_masks_are_independent():
    prompt = {
        "prompt_ids": [7, 8, 9],
        "negative_prompt_ids": [4, 5, 6],
        "prompt_mask": torch.tensor([1, 1, 0]),
        "negative_prompt_mask": torch.tensor([0, 1, 1]),
    }
    assert pre_tokenized_prompt_ids(prompt) == [7, 8]
    assert pre_tokenized_negative_prompt_ids(prompt) == [5, 6]
