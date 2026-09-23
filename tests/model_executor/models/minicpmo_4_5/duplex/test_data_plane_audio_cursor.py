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


def test_close_stream_releases_the_cursor_for_the_next_stream() -> None:
    session = _session()
    assert _sliced_len(session, 48000) == 48000
    session.close_stream(REQUEST_ID)
    assert _sliced_len(session, 24000) == 24000
