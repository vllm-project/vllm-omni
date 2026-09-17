# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Commit-only PCM buffer for the Qwen3-Omni duplex plugin."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.model_executor.models.qwen3_omni.duplex.input import Qwen3OmniPcmAppendBuffer
from vllm_omni.model_executor.models.qwen3_omni.duplex.turn_commit import TurnCommitPcmAppendBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: int = 1600, *, speech: bool = True) -> dict[str, object]:
    audio = np.ones(samples, dtype="<f4").tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(audio).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "is_speech": speech,
    }


def test_commit_only_buffer_emits_on_commit() -> None:
    buf = Qwen3OmniPcmAppendBuffer()
    reservation = buf.prepare_append(
        _payload(),
        operation_id="op1",
        chunk_period_ms=1000,
        allow_emit=True,
    )
    assert reservation is None
    assert buf.has_pending()
    commit = buf.prepare_commit(operation_id="c1", chunk_period_ms=1000)
    assert commit.payload is not None
    assert commit.payload.get("final") is True
    assert commit.payload.get("turn_commit") is True
    assert commit.payload.get("is_speech") is True
    commit.commit()
    assert not buf.has_pending()


def test_commit_only_buffer_rejects_sample_rate_change() -> None:
    buf = Qwen3OmniPcmAppendBuffer()
    buf.prepare_append(_payload(), operation_id="op1", chunk_period_ms=1000, allow_emit=True)
    changed = _payload()
    changed["sample_rate_hz"] = 24000
    with pytest.raises(ValueError, match="sample_rate_hz changed"):
        buf.prepare_append(changed, operation_id="op2", chunk_period_ms=1000, allow_emit=True)


def test_empty_prepare_commit_returns_no_payload() -> None:
    buf = Qwen3OmniPcmAppendBuffer()
    commit = buf.prepare_commit(operation_id="c0", chunk_period_ms=1000)
    assert commit.payload is None
    commit.commit()
    assert not buf.has_pending()


def test_generic_turn_commit_buffer_has_no_qwen3_imports() -> None:
    import inspect

    source = inspect.getsource(TurnCommitPcmAppendBuffer)
    assert "qwen3" not in source.lower()
    buf = TurnCommitPcmAppendBuffer()
    buf.prepare_append(_payload(), operation_id="op", chunk_period_ms=1000, allow_emit=True)
    commit = buf.prepare_commit(operation_id="c", chunk_period_ms=1000)
    assert commit.payload is not None
    commit.commit()
