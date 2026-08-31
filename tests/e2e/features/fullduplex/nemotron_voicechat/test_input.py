from __future__ import annotations

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.nemotron_voicechat.input import (
    NemotronVoiceChatPcmAppendBuffer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _payload(samples: np.ndarray, *, sample_rate_hz: int = 16000, fmt: str = "pcm_f32le") -> dict[str, object]:
    return {
        "type": "audio",
        "audio": base64.b64encode(np.asarray(samples, dtype=np.float32).tobytes()).decode("ascii"),
        "format": fmt,
        "sample_rate_hz": sample_rate_hz,
    }


def _append(buffer, samples, operation_id):
    return buffer.prepare_append(
        _payload(np.asarray(samples, dtype=np.float32)),
        operation_id=operation_id,
        chunk_period_ms=80,
        allow_emit=True,
    )


def test_irregular_browser_packets_emit_exact_ordered_80ms_frame() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    assert _append(buffer, np.arange(1000), "packet-1") is None

    reservation = _append(buffer, np.arange(1000, 1280), "packet-2")

    assert reservation is not None
    assert reservation.operation_id == "packet-2"
    decoded = np.frombuffer(base64.b64decode(reservation.payload["audio"]), dtype=np.float32)
    np.testing.assert_array_equal(decoded, np.arange(1280, dtype=np.float32))
    assert buffer.pending_byte_count == 0


def test_append_rejects_invalid_frame_contract() -> None:
    cases = (
        (1281, 16000, "pcm_f32le", 0.0, "at most 1280"),
        (1, 8000, "pcm_f32le", 0.0, "sample_rate_hz"),
        (1, 16000, "audio/pcm", 0.0, "format"),
        (1, 16000, "pcm_f32le", np.nan, "finite"),
    )
    for samples, sample_rate_hz, fmt, value, message in cases:
        with pytest.raises(ValueError, match=message):
            NemotronVoiceChatPcmAppendBuffer().prepare_append(
                _payload(np.full(samples, value, dtype=np.float32), sample_rate_hz=sample_rate_hz, fmt=fmt),
                operation_id="invalid",
                chunk_period_ms=80,
                allow_emit=True,
            )


def test_terminal_commit_rejects_more_than_one_frame_tail() -> None:
    buffer = NemotronVoiceChatPcmAppendBuffer()
    for index, samples in enumerate((1000, 281)):
        buffer.prepare_append(
            _payload(np.zeros(samples, dtype=np.float32)),
            operation_id=f"tail-{index}",
            chunk_period_ms=80,
            allow_emit=False,
        )
    with pytest.raises(ValueError, match="more than one 80 ms frame"):
        buffer.prepare_commit(operation_id="commit", chunk_period_ms=80)
