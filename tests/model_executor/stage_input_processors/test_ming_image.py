# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Ming-Image stage input processor."""

import pytest

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.stage_input_processors.ming_image import (
    _IMAGE_END_TOKEN_ID,
    _IMAGE_PATCH_TOKEN_ID,
    _NUM_QUERY_TOKENS,
    _validate_query_suffix,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TEXT_PROMPT_IDS = [7, 8, 9]
# The thinker appends ``<image><imagePatch>*256</image>``, so the query block is
# ``_NUM_QUERY_TOKENS + 2`` tokens wide. ``_validate_query_suffix`` deliberately
# does not inspect the opening ``<image>`` id, so any id distinct from the patch
# and end tokens stands in for it here.
_STAND_IN_OPEN_TOKEN_ID = 157158


def _with_query_block(prompt_ids: list[int]) -> list[int]:
    return [*prompt_ids, _STAND_IN_OPEN_TOKEN_ID, *([_IMAGE_PATCH_TOKEN_ID] * _NUM_QUERY_TOKENS), _IMAGE_END_TOKEN_ID]


def test_prompt_without_query_suffix_is_a_client_error():
    # A prompt that cannot even hold the appended query block is malformed
    # input, not broken server state: it must fail as HTTP 400 on its own
    # instead of escaping the orchestrator dispatch guard (#11198).
    with pytest.raises(OmniClientError) as exc_info:
        _validate_query_suffix(_TEXT_PROMPT_IDS)

    assert exc_info.value.status_code == 400


def test_prompt_one_token_short_of_the_query_block_is_a_client_error():
    prompt_ids = _with_query_block([])[:-1]

    with pytest.raises(OmniClientError) as exc_info:
        _validate_query_suffix(prompt_ids)

    assert exc_info.value.status_code == 400


def test_wrong_query_end_token_is_a_client_error():
    prompt_ids = _with_query_block(_TEXT_PROMPT_IDS)
    prompt_ids[-1] = _IMAGE_PATCH_TOKEN_ID

    with pytest.raises(OmniClientError) as exc_info:
        _validate_query_suffix(prompt_ids)

    assert exc_info.value.status_code == 400


def test_corrupted_patch_in_query_block_is_a_client_error():
    prompt_ids = _with_query_block(_TEXT_PROMPT_IDS)
    prompt_ids[len(_TEXT_PROMPT_IDS) + 1 + _NUM_QUERY_TOKENS // 2] = _IMAGE_END_TOKEN_ID

    with pytest.raises(OmniClientError) as exc_info:
        _validate_query_suffix(prompt_ids)

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("text_prompt_ids", [[], _TEXT_PROMPT_IDS], ids=["block-only", "with-text"])
def test_valid_query_suffix_is_located(text_prompt_ids):
    prompt_ids = _with_query_block(text_prompt_ids)

    start, end = _validate_query_suffix(prompt_ids)

    assert (start, end) == (len(text_prompt_ids) + 1, len(prompt_ids) - 1)
    assert prompt_ids[start] == _IMAGE_PATCH_TOKEN_ID
    assert prompt_ids[end] == _IMAGE_END_TOKEN_ID
