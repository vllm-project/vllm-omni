# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression for #7729: auto-response duplex silence continuation must not be
lost when the answer opens during the continuation's chunk-period wait."""

import asyncio

import pytest

from tests.engine.duplex.test_session_runner import (
    append_audio,
    close_harness,
    open_harness,
    tts_output,
)
from vllm_omni.engine.duplex import commands

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_answer_continues_after_the_last_real_unit() -> None:
    """The reporter's timeline (issue #7729):

    A TTS segment ends with no text or audio before any response exists. The
    session schedules a model-turn continuation which waits up to one chunk
    period. During the wait, the last real audio units arrive and the answer
    opens. Before the fix the continuation was dropped after the wait and
    nothing scheduled another one, leaving ``response.done`` un-emitted.
    """
    h = await open_harness()  # auto_response=True
    try:
        for _ in range(4):
            await h.run(append_audio())
        request_id = h.stage0_request_id()

        # A TTS segment ends with no text/audio before any response exists.
        # This schedules a continuation, which waits for up to one chunk period.
        h.deliver(tts_output(request_id, samples=0, text="", tts_is_last_chunk=True))
        await asyncio.sleep(0.1)

        # During that wait the last real units arrive and the answer starts.
        h.submit(append_audio())
        await asyncio.sleep(0.1)
        h.deliver(tts_output(request_id, samples=24000, text="no"))
        await asyncio.sleep(0.1)
        h.submit(append_audio())
        h.submit(commands.Commit())
        await h.settle(timeout_s=5.0)

        # More of the answer. Its segment only ends after another unit is processed.
        for samples, text in ((48000, "no fi"), (72000, "no fight, bu"), (96000, "no fight, but yes,")):
            await h.deliver_and_settle(tts_output(request_id, samples=samples, text=text))
        await h.settle(idle_s=2.5, timeout_s=8.0)

        assert h.session.active_response_id is not None, "the answer is still open"
        assert len(h.port.submissions) == 7, (
            f"expected 6 real units + 1 silence continuation, got {len(h.port.submissions)}"
        )
    finally:
        await close_harness(h)
