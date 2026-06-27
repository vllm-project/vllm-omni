# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the CosyVoice3 talker speech-embedding OOB guard.

See https://github.com/vllm-project/vllm-omni/issues/4721 — a non-multimodal request
(e.g. text-only ``/v1/completions`` without an audio prompt) reaches the talker's ``else``
branch in ``embed_input_ids`` and indexes ``speech_embedding`` with out-of-range text
token ids, triggering a CUDA device-side assert that kills the EngineCore. The serving
layer rejects unsupported generate routes before dispatch; this guard remains
defense-in-depth for any path that reaches the talker prefill, turning OOB ids into a
clear ``ValueError`` before the gather.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _guard():
    # Defer the heavy cosyvoice3 import until a test actually runs.
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _validate_speech_token_ids

    return _validate_speech_token_ids


def _prefill_detector():
    # Defer the heavy cosyvoice3 import until a test actually runs.
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import _is_prefill_step

    return _is_prefill_step


def test_is_prefill_step_returns_false_for_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    # Decode must return False so embed_input_ids skips the OOB guard and avoids per-step sync.
    ctx = SimpleNamespace(attn_metadata={"backend": SimpleNamespace(max_query_len=1)})
    monkeypatch.setattr(cosyvoice3, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(cosyvoice3, "get_forward_context", lambda: ctx)

    assert _prefill_detector()() is False


def test_is_prefill_step_returns_true_for_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    ctx = SimpleNamespace(attn_metadata={"backend": SimpleNamespace(max_query_len=8)})
    monkeypatch.setattr(cosyvoice3, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(cosyvoice3, "get_forward_context", lambda: ctx)

    assert _prefill_detector()() is True


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(AttributeError("attn metadata unavailable"), id="attribute_error"),
        pytest.param(KeyError("metadata"), id="key_error"),
        pytest.param(RuntimeError("backend metadata unavailable"), id="runtime_error"),
    ],
)
def test_is_prefill_step_fails_safe_when_metadata_unavailable(
    monkeypatch: pytest.MonkeyPatch, exception: Exception
) -> None:
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as cosyvoice3

    class BrokenForwardContext:
        @property
        def attn_metadata(self):
            raise exception

    monkeypatch.setattr(cosyvoice3, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(cosyvoice3, "get_forward_context", lambda: BrokenForwardContext())

    assert _prefill_detector()() is True


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
