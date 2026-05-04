# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.mimo_audio import _flush_remaining_codes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sentinel():
    return {"codes": {"audio": []}, "meta": {"finished": torch.tensor(True, dtype=torch.bool)}}


def test_flush_remaining_codes_when_no_codes_accumulated_missing_request_id():
    """No entry for request_id: treat as empty, return finished sentinel with empty codes."""
    tm = SimpleNamespace(code_prompt_token_ids={})
    out = _flush_remaining_codes(tm, "missing", chunk_size=3, left_context_size=3)
    assert out["codes"]["audio"] == _sentinel()["codes"]["audio"]
    assert out["meta"]["finished"].item() is True


def test_flush_remaining_codes_when_no_codes_accumulated_empty_list():
    """Explicit empty accumulation list returns the same sentinel."""
    tm = SimpleNamespace(code_prompt_token_ids={"r": []})
    out = _flush_remaining_codes(tm, "r", chunk_size=3, left_context_size=3)
    assert out["codes"]["audio"] == []
    assert out["meta"]["finished"].item() is True


def test_flush_remaining_codes_partial_chunk_remaining():
    """
    length=7, chunk_size=3 -> remainder 1 frame in last chunk; context window uses
    left_context + that partial (3+1), capped by length -> last 4 scalar entries.
    """
    tm = SimpleNamespace(
        code_prompt_token_ids={"r": [[1], [2], [3], [4], [5], [6], [7]]},
    )
    out = _flush_remaining_codes(tm, "r", chunk_size=3, left_context_size=3)
    assert out["meta"]["finished"].item() is True
    assert out["codes"]["audio"] == [4, 5, 6, 7]


def test_flush_remaining_codes_when_length_is_exact_multiple_of_chunk_size():
    """length % chunk_size == 0 uses full last chunk as context_length (not remainder)."""
    tm = SimpleNamespace(
        code_prompt_token_ids={"r": [[1], [2], [3], [4], [5], [6]]},
    )
    out = _flush_remaining_codes(tm, "r", chunk_size=3, left_context_size=3)
    # context_length = chunk_size = 3, end_index = min(6, 6) -> all 6
    assert out["codes"]["audio"] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize(
    "length,chunk_size,left_context,expected_end_index",
    [
        (2, 3, 3, 2),  # chunk_length=2 -> context 2, min(2,5)=2
        (5, 3, 3, 5),  # chunk_length=2 -> min(5,5)=5
        (7, 3, 3, 4),  # chunk_length=1 -> min(7,4)=4
        (6, 3, 3, 6),  # chunk_length=0 -> context 3, min(6,6)=6
        (10, 3, 3, 4),  # chunk_length=1 -> min(10,4)=4
        (1, 5, 10, 1),  # chunk_length=1 -> min(1,11)=1
    ],
)
def test_flush_remaining_codes_context_window_end_index(
    length: int, chunk_size: int, left_context: int, expected_end_index: int
) -> None:
    """Mirror _flush_remaining_codes context_length and end_index rules."""
    accumulated = [[i] for i in range(length)]
    tm = SimpleNamespace(code_prompt_token_ids={"r": accumulated})
    out = _flush_remaining_codes(tm, "r", chunk_size=chunk_size, left_context_size=left_context)
    expected_flat = list(range(length - expected_end_index, length))
    assert out["codes"]["audio"] == expected_flat
    assert out["meta"]["finished"].item() is True
