# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the CosyVoice3 talker speech-embedding OOB guard.

See https://github.com/vllm-project/vllm-omni/issues/4721 — a non-multimodal request
(e.g. text-only ``/v1/completions`` without an audio prompt) reaches the talker's ``else``
branch in ``embed_input_ids`` and indexes ``speech_embedding`` with out-of-range text
token ids, triggering a CUDA device-side assert that kills the EngineCore. The guard
turns that into a clear ``ValueError`` before the gather.
"""

from __future__ import annotations

import re

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _guard():
    # Defer the heavy cosyvoice3 import until a test actually runs.
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _validate_speech_token_ids

    return _validate_speech_token_ids


def test_inclusive_boundaries_pass() -> None:
    # 0 and num_speech_tokens - 1 are valid and must not raise.
    _guard()(torch.tensor([0, 1, 5, 9], dtype=torch.long), num_speech_tokens=10)


def test_empty_ids_pass() -> None:
    _guard()(torch.empty(0, dtype=torch.long), num_speech_tokens=10)


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([10], id="equal_to_size"),
        pytest.param([0, 1, 6562], id="text_token_id_above_codec_range"),
        pytest.param([-1], id="negative"),
    ],
)
def test_out_of_range_ids_raise(ids: list[int]) -> None:
    with pytest.raises(ValueError, match="out of range"):
        _guard()(torch.tensor(ids, dtype=torch.long), num_speech_tokens=10)


def test_out_of_range_error_message_guides_user() -> None:
    with pytest.raises(ValueError, match="provide the required audio prompt"):
        _guard()(torch.tensor([6562], dtype=torch.long), num_speech_tokens=10)


def test_error_message_caps_listed_ids() -> None:
    # Many bad ids must not make the error message blow up; at most 8 are listed.
    ids = torch.arange(100, 150, dtype=torch.long)  # 50 out-of-range ids
    with pytest.raises(ValueError) as exc:
        _guard()(ids, num_speech_tokens=10)
    match = re.search(r"ids \[(.*?)\]", str(exc.value))
    assert match is not None, f"Expected 'ids [...]' in error message, got: {exc.value}"
    listed = match.group(1)
    assert 0 < len([x for x in listed.split(",") if x.strip()]) <= 8
