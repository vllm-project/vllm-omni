# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.utils.prompt_utils import (
    pre_tokenized_negative_prompt_ids,
    pre_tokenized_prompt_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reads_prompt_ids():
    assert pre_tokenized_prompt_ids({"prompt_ids": [1, 2, 3]}) == [1, 2, 3]


def test_native_prompt_token_ids_are_not_read():
    # ``prompt_token_ids`` belongs to native vLLM / AR-stage hand-off, not to
    # ``OmniCustomPrompt``.
    assert pre_tokenized_prompt_ids({"prompt_token_ids": [4, 5]}) is None
    assert pre_tokenized_prompt_ids({"prompt_ids": [1], "prompt_token_ids": [2]}) == [1]


def test_accepts_a_tuple_of_token_ids():
    assert pre_tokenized_prompt_ids({"prompt_ids": (6, 7)}) == [6, 7]


def test_unwraps_a_single_batched_prompt():
    assert pre_tokenized_prompt_ids({"prompt_ids": [[1, 2, 3]]}) == [1, 2, 3]


def test_rejects_more_than_one_prompt():
    with pytest.raises(ValueError, match="carries a single prompt"):
        pre_tokenized_prompt_ids({"prompt_ids": [[1], [2]]})


def test_reads_negative_prompt_ids():
    assert pre_tokenized_negative_prompt_ids({"negative_prompt_ids": [7]}) == [7]


def test_native_negative_prompt_token_ids_are_not_read():
    assert pre_tokenized_negative_prompt_ids({"negative_prompt_token_ids": [8]}) is None


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
