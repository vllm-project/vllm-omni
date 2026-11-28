# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The cumulative Stage1 audio cursor.

Stage1 hands the data plane a snapshot of every sample it has generated for the
request so far, and only the new tail may reach the client. The cursor that
makes that possible is the one piece of per-request state whose loss is silent
and expensive: rewinding it re-sends the whole session as a single delta.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind

from tests.engine.test_output_processor import _make_state
from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.data_plane import (
    MiniCPMO45DataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REQUEST_ID = "duplex-s.c2Vzc2lvbg==.e.0.r.stage0"


def _session() -> MiniCPMO45DataPlaneSession:
    return MiniCPMO45DataPlaneSession(lambda *args, **kwargs: "")


def _snapshot(num_samples: int) -> np.ndarray:
    return np.arange(num_samples, dtype=np.float32)


def _sliced_len(session: MiniCPMO45DataPlaneSession, num_samples: int) -> int:
    sliced = session.slice_cumulative_audio(REQUEST_ID, _snapshot(num_samples))
    if sliced is None:
        return 0
    return int(np.asarray(sliced).reshape(-1).size)


def test_a_growing_snapshot_yields_only_the_new_tail() -> None:
    session = _session()
    assert _sliced_len(session, 24000) == 24000
    assert _sliced_len(session, 48000) == 24000
    assert _sliced_len(session, 48000) == 0  # unchanged snapshot emits nothing


def test_a_placeholder_unit_does_not_rewind_the_cursor() -> None:
    """The failure behind a 300s benchmark case producing 1373s of audio.

    A continuation unit arrives as a one-sample snapshot. Taking that as "the
    producer restarted" drops the cursor to 1, so the next real snapshot is
    sliced from the start and the client is handed the entire session again.
    """
    session = _session()
    assert _sliced_len(session, 637440) == 637440
    assert _sliced_len(session, 1) == 1
    assert _sliced_len(session, 1) == 1
    assert _sliced_len(session, 1) == 1

    # 649920 - 637440, not 649920 - 1.
    assert _sliced_len(session, 649920) == 12480


def test_a_genuinely_restarted_producer_is_still_picked_up() -> None:
    """Two consecutive below-cursor snapshots that grow are a new buffer."""
    session = _session()
    assert _sliced_len(session, 48000) == 48000
    assert _sliced_len(session, 1000) == 1000  # head of the new buffer
    assert _sliced_len(session, 3000) == 2000  # its tail, not 3000 - 48000
    assert _sliced_len(session, 5000) == 2000


def test_a_restarted_waveform_can_grow_across_turns() -> None:
    session = _session()
    session.slice_cumulative_audio(REQUEST_ID, _snapshot(8), turn_id=0)
    session.slice_cumulative_audio(REQUEST_ID, _snapshot(1), turn_id=0)
    waveform = _snapshot(12)
    first = session.slice_cumulative_audio(REQUEST_ID, waveform[:4], turn_id=1)
    assert session.slice_cumulative_audio(REQUEST_ID, waveform[:4], turn_id=1) is None
    placeholder = session.slice_cumulative_audio(REQUEST_ID, _snapshot(1), turn_id=2)
    np.testing.assert_array_equal(placeholder, _snapshot(1))
    tail = session.slice_cumulative_audio(REQUEST_ID, waveform, turn_id=2)
    np.testing.assert_array_equal(np.concatenate([first, tail]), waveform)


def test_close_stream_releases_the_cursor_for_the_next_stream() -> None:
    session = _session()
    assert _sliced_len(session, 48000) == 48000
    session.close_stream(REQUEST_ID)
    assert _sliced_len(session, 24000) == 24000


@pytest.mark.parametrize("kind", [RequestOutputKind.CUMULATIVE, RequestOutputKind.DELTA])
@pytest.mark.parametrize("first_size", [4, 6])
def test_resident_request_receives_exact_pcm_across_accumulators(kind, first_size):
    pcm = []

    def encode(audio, *_):
        if audio is None:
            return None
        pcm.extend(audio.tolist())
        return "pcm"

    plane = MiniCPMO45DataPlaneSession(encode)
    old, delayed, new = _make_state(kind), _make_state(kind), _make_state(kind)

    def emit(state, turn, seq, samples, *, active_turn):
        state.add_multimodal_tensor(
            {
                "model_outputs": torch.tensor(samples),
                "meta.duplex_turn_id": torch.tensor(turn),
                "meta.duplex_audio_cache_epoch": torch.tensor(turn),
                "meta.duplex_audio_chunk_seq": torch.tensor(seq),
            },
            "audio",
        )
        output = state.make_request_output([1], None, None, None)
        context = DuplexDataPlaneContext(auto_responds=True, turn_id=active_turn)
        list(plane.project_output(output, context=context))
        return output

    late = emit(old, 0, 0, [10] * 4, active_turn=0)
    emit(new, 1, 0, [20] * first_size, active_turn=1)
    assert pcm == [10] * 4 + [20] * first_size
    context = DuplexDataPlaneContext(auto_responds=True, turn_id=1)
    list(plane.project_output(late, context=context))
    # A retired accumulator can deliver its very first snapshot late.
    emit(delayed, 0, 0, [99] * 12, active_turn=1)
    new.add_multimodal_tensor({"model_outputs": torch.tensor([21])}, "audio")
    packet = emit(new, 1, 1, [21] * 2, active_turn=1)
    list(plane.project_output(packet, context=context))
    # Cache epochs can change while the same accumulator keeps growing.
    emit(new, 2, 0, [22] * 2, active_turn=2)
    assert pcm == [10] * 4 + [20] * first_size + [21] * 3 + [22] * 2
