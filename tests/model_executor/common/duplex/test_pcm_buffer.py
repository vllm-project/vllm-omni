# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64

import numpy as np
import pytest

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation
from vllm_omni.model_executor.common.duplex.pcm_buffer import (
    FixedFramePcmAppendBuffer,
    FixedFramePcmAppendReservation,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAME = 1920
RATE = 24000
PERIOD = 80


def _pcm_payload(samples: np.ndarray, *, sample_rate_hz: int = RATE, fmt: str = "pcm_f32le") -> dict[str, object]:
    samples = np.ascontiguousarray(samples, dtype="<f4")
    return {
        "type": "audio",
        "format": fmt,
        "sample_rate_hz": sample_rate_hz,
        "audio": base64.b64encode(samples.tobytes()).decode("ascii"),
    }


def _buffer() -> FixedFramePcmAppendBuffer:
    return FixedFramePcmAppendBuffer(sample_rate_hz=RATE, frame_samples=FRAME, chunk_period_ms=PERIOD, model="Test")


def _decode(payload: dict[str, object]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(str(payload["audio"])), dtype="<f4")


def test_buffer_implements_the_framework_contracts() -> None:
    buffer = _buffer()

    assert isinstance(buffer, PcmAppendBuffer)
    assert buffer.frame_bytes == FRAME * 4
    reservation = buffer.prepare_append(
        _pcm_payload(np.arange(FRAME)), operation_id="op", chunk_period_ms=PERIOD, allow_emit=True
    )
    assert isinstance(reservation, FixedFramePcmAppendReservation)
    assert isinstance(reservation, PcmAppendReservation)


def test_one_whole_frame_is_reserved_transactionally() -> None:
    buffer = _buffer()

    reservation = buffer.prepare_append(
        _pcm_payload(np.arange(FRAME)), operation_id="op-1", chunk_period_ms=PERIOD, allow_emit=True
    )

    assert reservation is not None
    assert reservation.active
    assert reservation.byte_count == FRAME * 4
    assert reservation.payload is not None and reservation.payload["sample_rate_hz"] == RATE
    assert _decode(reservation.payload).tolist() == list(range(FRAME))
    assert buffer.pending_byte_count == 0
    assert buffer.has_reserved() and not buffer.has_pending()

    reservation.rollback()

    assert not reservation.active
    assert buffer.pending_byte_count == FRAME * 4
    assert buffer.has_pending() and not buffer.has_reserved()


def test_partial_chunks_accumulate_until_a_whole_frame_exists() -> None:
    buffer = _buffer()
    half = _pcm_payload(np.zeros(FRAME // 2))

    assert buffer.prepare_append(half, operation_id="op-1", chunk_period_ms=PERIOD, allow_emit=True) is None
    assert buffer.pending_byte_count == FRAME * 2
    reservation = buffer.prepare_append(half, operation_id="op-2", chunk_period_ms=PERIOD, allow_emit=True)

    assert reservation is not None
    assert reservation.byte_count == FRAME * 4
    assert buffer.pending_byte_count == 0


def test_appends_are_not_emitted_while_emission_is_disallowed() -> None:
    buffer = _buffer()

    assert (
        buffer.prepare_append(
            _pcm_payload(np.zeros(FRAME)), operation_id="op", chunk_period_ms=PERIOD, allow_emit=False
        )
        is None
    )
    assert buffer.pending_byte_count == FRAME * 4


def test_commit_flushes_a_padded_final_frame_and_an_empty_buffer_reserves_nothing() -> None:
    buffer = _buffer()
    buffer.prepare_append(_pcm_payload(np.ones(10)), operation_id="op-1", chunk_period_ms=PERIOD, allow_emit=False)

    reservation = buffer.prepare_commit(operation_id="commit", chunk_period_ms=PERIOD)

    assert reservation.payload is not None
    assert reservation.payload["final"] is True
    decoded = _decode(reservation.payload)
    assert decoded.size == FRAME
    assert decoded[:10].tolist() == [1.0] * 10 and not decoded[10:].any()
    assert reservation.byte_count == 40
    reservation.commit()
    assert not buffer.has_reserved()

    empty = buffer.prepare_commit(operation_id="commit-2", chunk_period_ms=PERIOD)
    assert empty.payload is None and empty.byte_count == 0
    empty.commit()
    assert buffer.flush(chunk_period_ms=PERIOD) is None


def test_rolling_back_an_earlier_reservation_restores_later_ones_in_order() -> None:
    buffer = _buffer()
    first = buffer.prepare_append(
        _pcm_payload(np.full(FRAME, 1.0)), operation_id="op-1", chunk_period_ms=PERIOD, allow_emit=True
    )
    second = buffer.prepare_append(
        _pcm_payload(np.full(FRAME, 2.0)), operation_id="op-2", chunk_period_ms=PERIOD, allow_emit=True
    )
    assert first is not None and second is not None

    first.rollback()

    assert not first.active and not second.active
    assert buffer.pending_byte_count == 2 * FRAME * 4
    replay = buffer.prepare_append(
        _pcm_payload(np.zeros(0)), operation_id="op-3", chunk_period_ms=PERIOD, allow_emit=True
    )
    assert replay is not None and _decode(replay.payload)[0] == 1.0


def test_clear_drops_pending_audio_and_deactivates_reservations() -> None:
    buffer = _buffer()
    reservation = buffer.prepare_append(
        _pcm_payload(np.zeros(FRAME + 8)), operation_id="op-1", chunk_period_ms=PERIOD, allow_emit=True
    )
    assert reservation is not None and buffer.pending_byte_count == 32

    buffer.clear()
    buffer.clear_force_listen()

    assert not reservation.active
    assert buffer.pending_byte_count == 0
    reservation.rollback()  # no-op after clear
    assert buffer.pending_byte_count == 0


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_pcm_payload(np.zeros(8), sample_rate_hz=16000), "sample_rate_hz must be 24000"),
        (_pcm_payload(np.zeros(8), fmt="pcm16"), "format must be pcm_f32le"),
        (_pcm_payload(np.array([0.0, np.nan])), "finite"),
    ],
)
def test_malformed_audio_is_rejected_and_nothing_is_buffered(payload: dict[str, object], match: str) -> None:
    buffer = _buffer()

    with pytest.raises(ValueError, match=match):
        buffer.prepare_append(payload, operation_id="op", chunk_period_ms=PERIOD, allow_emit=True)

    assert buffer.pending_byte_count == 0


def test_chunk_period_and_duplicate_operations_are_refused() -> None:
    buffer = _buffer()

    with pytest.raises(ValueError, match="chunk_period_ms must be 80"):
        buffer.prepare_append(_pcm_payload(np.zeros(8)), operation_id="op", chunk_period_ms=1000, allow_emit=True)
    buffer.prepare_append(_pcm_payload(np.zeros(FRAME)), operation_id="op", chunk_period_ms=PERIOD, allow_emit=True)
    with pytest.raises(ValueError, match="duplicate active operation_id"):
        buffer.prepare_append(_pcm_payload(np.zeros(8)), operation_id="op", chunk_period_ms=PERIOD, allow_emit=True)


def test_constructor_validates_its_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        FixedFramePcmAppendBuffer(sample_rate_hz=0, frame_samples=FRAME, chunk_period_ms=PERIOD)
