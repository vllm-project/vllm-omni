# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.minicpmo45.input import MiniCPMO45PcmAppendBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def pcm_payload(samples: int, *, speech: bool = True) -> dict[str, object]:
    audio = np.ones(samples, dtype=np.float32).tobytes()
    return {
        "type": "audio",
        "audio": base64.b64encode(audio).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
        "is_speech": speech,
    }


def test_commit_does_not_add_silence_after_incremental_audio_was_drained():
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    committed = buffer.commit(chunk_period_ms=1_000)

    assert emitted is not None
    assert not buffer.has_pending()
    assert committed is None


def test_append_emits_one_model_unit_when_multiple_units_are_buffered():
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(pcm_payload(32_000), chunk_period_ms=1_000)

    assert emitted is not None
    assert len(base64.b64decode(emitted["audio"])) == 16_000 * 4
    assert buffer.pending_byte_count == 16_000 * 4


def test_commit_without_speech_does_not_synthesize_terminal_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    buffer.append(pcm_payload(8_000, speech=False), chunk_period_ms=1_000)

    committed = buffer.commit(chunk_period_ms=1_000)

    assert committed is None


def test_commit_resets_cumulative_turn_accounting():
    buffer = MiniCPMO45PcmAppendBuffer()
    buffer.append(pcm_payload(16_000), chunk_period_ms=1_000)
    buffer.commit(chunk_period_ms=1_000)

    empty = buffer.commit(chunk_period_ms=1_000)

    assert empty is None


def test_pcm_append_reservation_rollback_restores_emitted_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    original = pcm_payload(16_000)

    reservation = buffer.prepare_append(
        original,
        operation_id="append-1",
        chunk_period_ms=1_000,
    )

    assert reservation is not None
    assert not buffer.has_pending()
    reservation.rollback()
    assert buffer.has_pending()

    retried = buffer.flush(chunk_period_ms=1_000)
    assert retried is not None
    assert base64.b64decode(retried["audio"]) == base64.b64decode(original["audio"])


def test_pcm_commit_keeps_prior_append_reservation_active():
    buffer = MiniCPMO45PcmAppendBuffer()
    append_reservation = buffer.prepare_append(
        pcm_payload(16_000),
        operation_id="append-before-commit",
        chunk_period_ms=1_000,
    )

    commit_reservation = buffer.prepare_commit(
        operation_id="commit-after-append",
        chunk_period_ms=1_000,
    )

    assert append_reservation is not None
    assert append_reservation.active
    assert commit_reservation.payload is None
    append_reservation.commit()
    commit_reservation.commit()


def test_pcm_commit_reservation_rollback_restores_residual_audio():
    buffer = MiniCPMO45PcmAppendBuffer()
    original = pcm_payload(8_000)
    assert (
        buffer.prepare_append(
            original,
            operation_id="buffer-half-chunk",
            chunk_period_ms=1_000,
        )
        is None
    )

    reservation = buffer.prepare_commit(
        operation_id="final-half-chunk",
        chunk_period_ms=1_000,
    )

    assert reservation.payload is not None
    assert reservation.payload["final"] is True
    reservation.rollback()

    retried = buffer.prepare_commit(
        operation_id="retry-final-half-chunk",
        chunk_period_ms=1_000,
    )
    assert retried.payload is not None
    assert base64.b64decode(retried.payload["audio"]) == (base64.b64decode(original["audio"]) + b"\x00" * (8_000 * 4))
