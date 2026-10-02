# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64

import numpy as np
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input import MiniCPMO45PcmAppendBuffer

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


def test_speech_marker_does_not_leak_across_irregular_chunk_boundaries():
    buffer = MiniCPMO45PcmAppendBuffer()

    assert buffer.append(pcm_payload(15_000), chunk_period_ms=1_000) is None
    speech_unit = buffer.append(pcm_payload(2_000, speech=False), chunk_period_ms=1_000)
    silence_unit = buffer.append(pcm_payload(15_000, speech=False), chunk_period_ms=1_000)

    assert speech_unit is not None
    assert speech_unit["is_speech"] is True
    assert silence_unit is not None
    assert silence_unit["is_speech"] is False


def test_force_listen_marker_does_not_leak_across_irregular_chunk_boundaries():
    buffer = MiniCPMO45PcmAppendBuffer()
    forced = pcm_payload(15_000, speech=False)
    forced["force_listen"] = True

    assert buffer.append(forced, chunk_period_ms=1_000) is None
    forced_unit = buffer.append(pcm_payload(2_000, speech=False), chunk_period_ms=1_000)
    unforced_unit = buffer.append(pcm_payload(15_000, speech=False), chunk_period_ms=1_000)

    assert forced_unit is not None
    assert forced_unit["force_listen"] is True
    assert unforced_unit is not None
    assert unforced_unit["force_listen"] is False


def test_pcm_append_rollback_restores_per_span_speech_markers():
    buffer = MiniCPMO45PcmAppendBuffer()
    assert buffer.append(pcm_payload(15_000), chunk_period_ms=1_000) is None
    reservation = buffer.prepare_append(
        pcm_payload(2_000, speech=False),
        operation_id="mixed-unit",
        chunk_period_ms=1_000,
    )

    assert reservation is not None
    reservation.rollback()
    first = buffer.append(pcm_payload(15_000, speech=False), chunk_period_ms=1_000)
    second = buffer.flush(chunk_period_ms=1_000)

    assert first is not None
    assert first["is_speech"] is True
    assert second is not None
    assert second["is_speech"] is False


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


def _frame_payload(samples: int, frames: list[str]) -> dict[str, object]:
    payload = pcm_payload(samples)
    payload["video_frames"] = frames
    return payload


def test_append_attaches_every_frame_of_the_unit_closing_append():
    """A base frame and its stacked composite arrive on one append and must
    enter the same model unit (official ``frame_list``), not one per unit."""
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted = buffer.append(_frame_payload(16_000, ["base-0", "stack-0"]), chunk_period_ms=1_000)

    assert emitted is not None
    assert emitted["video_frames"] == ["base-0", "stack-0"]
    assert buffer._frame_queue == []


def test_stacked_frames_do_not_accumulate_across_units():
    buffer = MiniCPMO45PcmAppendBuffer()
    attached: list[list[str]] = []

    for unit in range(64):
        emitted = buffer.append(
            _frame_payload(16_000, [f"base-{unit}", f"stack-{unit}"]),
            chunk_period_ms=1_000,
        )
        assert emitted is not None
        attached.append(list(emitted["video_frames"]))

    assert attached[0] == ["base-0", "stack-0"]
    assert attached[-1] == ["base-63", "stack-63"]
    assert buffer._frame_queue == []


def test_frames_from_partial_appends_ride_the_unit_they_close():
    """200 ms client chunks: the frames arrive with the 5th chunk of each
    second and attach to the unit that chunk completes, one group per unit."""
    buffer = MiniCPMO45PcmAppendBuffer()

    emitted_units: list[dict[str, object]] = []
    for unit in range(3):
        for chunk in range(5):
            frames = [f"base-{unit}", f"stack-{unit}"] if chunk == 4 else []
            emitted = buffer.append(_frame_payload(3_200, frames), chunk_period_ms=1_000)
            if emitted is not None:
                emitted_units.append(emitted)

    assert [unit["video_frames"] for unit in emitted_units] == [
        ["base-0", "stack-0"],
        ["base-1", "stack-1"],
        ["base-2", "stack-2"],
    ]
    assert buffer._frame_queue == []


def test_single_frame_per_unit_still_attaches_one_frame_per_unit():
    buffer = MiniCPMO45PcmAppendBuffer()

    first = buffer.append(_frame_payload(16_000, ["base-0"]), chunk_period_ms=1_000)
    second = buffer.append(_frame_payload(16_000, ["base-1"]), chunk_period_ms=1_000)

    assert first is not None and first["video_frames"] == ["base-0"]
    assert second is not None and second["video_frames"] == ["base-1"]


def test_frame_groups_stay_queued_when_audio_outruns_units():
    """Two appends land before one unit closes: the later group waits for the
    next unit instead of merging into the first."""
    buffer = MiniCPMO45PcmAppendBuffer()

    assert buffer.append(_frame_payload(8_000, ["base-0", "stack-0"]), chunk_period_ms=1_000) is None
    first = buffer.append(_frame_payload(16_000, ["base-1", "stack-1"]), chunk_period_ms=1_000)
    second = buffer.append(pcm_payload(8_000), chunk_period_ms=1_000)

    assert first is not None and first["video_frames"] == ["base-0", "stack-0"]
    assert second is not None and second["video_frames"] == ["base-1", "stack-1"]
    assert buffer._frame_queue == []


def test_rollback_restores_frame_groups_in_wire_order():
    buffer = MiniCPMO45PcmAppendBuffer()

    first = buffer.prepare_append(
        _frame_payload(16_000, ["base-0", "stack-0"]),
        operation_id="append-0",
        chunk_period_ms=1_000,
    )
    second = buffer.prepare_append(
        _frame_payload(16_000, ["base-1", "stack-1"]),
        operation_id="append-1",
        chunk_period_ms=1_000,
    )
    assert first is not None and second is not None
    assert first.payload is not None and first.payload["video_frames"] == ["base-0", "stack-0"]

    first.rollback()

    assert buffer._frame_queue == [["base-0", "stack-0"], ["base-1", "stack-1"]]
    retried = buffer.flush(chunk_period_ms=1_000)
    assert retried is not None
    assert retried["video_frames"] == ["base-0", "stack-0"]
    assert buffer._frame_queue == [["base-1", "stack-1"]]
