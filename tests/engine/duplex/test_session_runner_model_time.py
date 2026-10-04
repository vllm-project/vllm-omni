# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session options that control how the runner advances model time between client inputs."""

from __future__ import annotations

import pytest

from tests.engine.duplex.test_session_runner import (
    append_audio,
    close_harness,
    open_harness,
    tts_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def test_silence_continuation_can_be_turned_off_per_session() -> None:
    """``extra_body.silence_continuation: false``: the model only hears audio the client sent."""
    h = await open_harness(extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True)
        )

        assert len(h.port.submissions) == 1, "no silence unit may be invented"
        assert h.runner.model_state.continuation_units == 0
        assert h.session.active_response_id is not None
    finally:
        await close_harness(h)
