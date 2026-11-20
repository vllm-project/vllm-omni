# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real input must supersede a pending Nemotron silence continuation."""

import asyncio
from types import SimpleNamespace

import pytest

import vllm_omni.engine.duplex.session.model_channel as model_channel_module
import vllm_omni.engine.duplex.session.runner as runner_module
from tests.engine.duplex.test_deadline_pacing_scheduler import FakeClock, _continuation_kwargs
from tests.engine.duplex.test_session_runner import append_audio, tts_output, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import open_nemotron_harness

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("arrival_jitter_s", [0.005, 0.15])
async def test_real_nemotron_input_supersedes_silence_after_client_jitter(
    monkeypatch: pytest.MonkeyPatch, arrival_jitter_s: float
) -> None:
    clock = FakeClock(start=100.0)
    real_sleep = asyncio.sleep
    # Leave the event loop's clock real so the harness can drain its mailbox.
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(monotonic=clock))
    monkeypatch.setattr(model_channel_module, "time", SimpleNamespace(monotonic=clock))
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), samples=1764, finished=False),
            stage_id=2,
            segment_finished=False,
        )
        assert h.session.active_response_id is not None
        assert h.session.capabilities.chunk_period_ms == 80
        state = h.runner.model_state
        state.last_native_submit_monotonic = 100.0
        state.silence_deadline_monotonic = None
        before = len(h.port.submissions)
        clock.value = 100.04
        timer_entered = asyncio.Event()
        release_timer = asyncio.Event()
        wake_times: list[float] = []

        async def controlled_sleep(delay: float) -> None:
            if delay > 0 and asyncio.current_task() is scheduled:
                wake_times.append(clock.value + delay)
                timer_entered.set()
                await release_timer.wait()
            else:
                await real_sleep(delay)

        monkeypatch.setattr(asyncio, "sleep", controlled_sleep)
        scheduled = asyncio.create_task(
            h.runner._schedule_silence_continuation(h.runner.model.silence_unit_payload(), **_continuation_kwargs(h))
        )
        try:
            await asyncio.wait_for(timer_entered.wait(), timeout=2)
            real_arrival = 100.08 + arrival_jitter_s
            if wake_times[0] < real_arrival:
                # The old deadline fires first, advancing the model once for
                # silence and again for the real frame below.
                clock.value = wake_times[0]
                release_timer.set()
                await scheduled
                if h.runner.tasks.append_tail is not None:
                    await h.runner.tasks.append_tail
                clock.value = real_arrival
                events = await h.run(append_audio(samples=1280))
            else:
                # The jitter margin lets real input invalidate the old timer.
                clock.value = real_arrival
                events = await h.run(append_audio(samples=1280))
                clock.value = wake_times[0]
                release_timer.set()
                await scheduled
                if h.runner.tasks.append_tail is not None:
                    await h.runner.tasks.append_tail
            assert "error" not in types(events)
            assert len(h.port.submissions) - before == 1, "silence and real input both advanced the model"
            assert state.last_native_submit_monotonic == pytest.approx(real_arrival)
            assert state.silence_deadline_monotonic is None
        finally:
            release_timer.set()
            if not scheduled.done():
                scheduled.cancel()
            await asyncio.gather(scheduled, return_exceptions=True)
