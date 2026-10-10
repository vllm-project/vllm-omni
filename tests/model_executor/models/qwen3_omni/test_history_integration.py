# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import asyncio

import pytest

from tests.engine.duplex.test_session_runner import append_audio, tts_output
from tests.model_executor.models.qwen3_omni.test_duplex_plugin import open_qwen
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    CancelResponse,
    ClearOutputAudio,
    Commit,
    DeleteItem,
    TruncateItem,
    UpdateSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEXT = "Hello world. This part was not heard."


async def completed_response(h, *, played_ms=1000):
    await h.run(append_audio())
    await h.run(Commit(final=True, create_response=True))
    request = h.port.submissions[-1].context.request_id
    response = h.session.active_response_id
    await h.deliver_and_settle(tts_output(request, samples=0, text=TEXT, finished=True), stage_id=0)
    await h.deliver_and_settle(tts_output(request, samples=240000, finished=True), stage_id=2)
    await h.run(AckPlayback(response_id=response, played_ms=played_ms))
    return response


@pytest.mark.asyncio
async def test_text_only_estimator_shares_history_fences_without_caching_pcm():
    async def estimate(snapshot):
        assert snapshot.pcm_f32le == b""
        assert snapshot.sample_rate_hz == 0
        return 5 if snapshot.played_ms >= 1000 else 0

    h = await open_qwen(calibrate=estimate, requires_audio=False, history_max_bytes=1)
    try:
        response = await completed_response(h)
        controller = h.runner.ctx.history_calibration
        assert controller.retained_bytes == 0
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == "Hello"
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=500))
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == ""
        await h.run(AckPlayback(response_id=response, played_ms=10000))
        assert h.session.history[-1]["content"] == ""
        await h.run(DeleteItem(item_id=f"item_{response}"))
        assert not controller.audio and not controller.tasks
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("requires_audio", [True, False])
async def test_nonunit_playback_speed_refuses_both_policies(requires_audio):
    async def calibrate(snapshot):
        pytest.fail("Recorded audio time cannot attest non-unit playback time")

    h = await open_qwen(calibrate=calibrate, requires_audio=requires_audio)
    try:
        await h.run(UpdateSession(patch={"speed": 2.0}))
        await completed_response(h)
        await h.runner.ctx.history_calibration.before_prompt()
        assert h.session.history[-1]["content"] == ""
        assert h.runner.ctx.history_calibration.retained_bytes == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_next_submitted_prompt_waits_and_uses_heard_prefix_in_original_position():
    entered, release = asyncio.Event(), asyncio.Event()

    async def calibrate(snapshot):
        assert snapshot.text == TEXT
        assert snapshot.played_ms == 1000
        assert len(snapshot.pcm_f32le) == 240000 * 4
        entered.set()
        await release.wait()
        return len("Hello")

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=1000))
        await asyncio.wait_for(entered.wait(), 2)
        await h.run(append_audio())
        next_turn = asyncio.create_task(h.run(Commit(final=True, create_response=True)))
        await asyncio.sleep(0.02)
        assert len(h.port.submissions) == 1
        release.set()
        await asyncio.wait_for(next_turn, 2)
        for _ in range(100):
            if len(h.port.submissions) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(h.port.submissions) == 2
        messages = ast.literal_eval(h.port.submissions[-1].prompt["prompt"])
        assistants = [m for m in messages if m["role"] == "assistant"]
        assert assistants == [{"role": "assistant", "content": "Hello"}]
        assert messages.index(assistants[0]) < len(messages) - 1
    finally:
        release.set()
        await h.manager.shutdown()
    assert h.runner.ctx.history_calibration.retained_bytes == 0


@pytest.mark.asyncio
async def test_stricter_truncate_clears_calibrated_prefix_and_late_full_ack_cannot_restore_it():
    async def calibrate(snapshot):
        return 5 if snapshot.played_ms >= 1000 else None

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        controller = h.runner.ctx.history_calibration
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == "Hello"
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=500))
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == ""
        await h.run(AckPlayback(response_id=response, played_ms=10000))
        assert h.session.history[-1]["content"] == ""
        assert h.session.history_audio_cutoff(response) == 500
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "exception", "invalid_offset"])
async def test_failed_calibration_keeps_empty_history_and_does_not_retry_same_snapshot(failure):
    calls = 0

    async def calibrate(snapshot):
        nonlocal calls
        calls += 1
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "exception":
            raise ValueError("invalid evidence")
        return len(snapshot.text) + 1

    h = await open_qwen(calibrate=calibrate, calibration_timeout_ms=30)
    try:
        await completed_response(h)
        controller = h.runner.ctx.history_calibration
        await asyncio.wait_for(controller.before_prompt(), 2)
        await controller.before_prompt()
        assert calls == 1
        assert h.session.history[-1]["content"] == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("requires_audio", [True, False])
@pytest.mark.parametrize("start_next_before_truncate", [True, False])
async def test_late_full_ack_preserves_calibrated_reply_for_stricter_truncate(
    requires_audio, start_next_before_truncate
):
    cutoffs = []

    async def calibrate(snapshot):
        cutoffs.append(snapshot.played_ms)
        return 5 if snapshot.played_ms >= 1000 else 0

    h = await open_qwen(calibrate=calibrate, requires_audio=requires_audio)
    try:
        response = await completed_response(h)
        controller = h.runner.ctx.history_calibration
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=1000))
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == "Hello"
        await h.run(AckPlayback(response_id=response, played_ms=10000))
        if start_next_before_truncate:
            # A new response resets the active playback cursor. The old item
            # must still use its own retained evidence and permanent limit.
            await h.run(append_audio())
            await h.run(Commit(final=True, create_response=True))
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=500))
        await controller.before_prompt()
        assert [m for m in h.session.history if m["role"] == "assistant"] == [{"role": "assistant", "content": ""}]
        if not start_next_before_truncate:
            await h.run(append_audio())
            await h.run(Commit(final=True, create_response=True))
            messages = ast.literal_eval(h.port.submissions[-1].prompt["prompt"])
            assert [m for m in messages if m["role"] == "assistant"] == [{"role": "assistant", "content": ""}]
        assert cutoffs == [1000, 500]
        assert h.session.history_audio_cutoff(response) == 500
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("requires_audio", [True, False])
@pytest.mark.parametrize("start_next_before_truncate", [True, False])
async def test_duplicate_full_ack_preserves_cancelled_reply_for_stricter_truncate(
    requires_audio, start_next_before_truncate
):
    cutoffs = []

    async def calibrate(snapshot):
        cutoffs.append(snapshot.played_ms)
        return len("Hello") if snapshot.played_ms >= 10000 else len("He")

    h = await open_qwen(calibrate=calibrate, requires_audio=requires_audio)
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request = h.port.submissions[-1].context.request_id
        response = h.session.active_response_id
        await h.deliver_and_settle(tts_output(request, samples=0, text=TEXT, finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request, samples=240000, finished=False), stage_id=2)
        await h.run(AckPlayback(response_id=response, played_ms=10000))
        await h.run(CancelResponse())
        controller = h.runner.ctx.history_calibration
        await controller.before_prompt()
        assert h.session.history[-1]["content"] == "Hello"

        await h.run(AckPlayback(response_id=response, played_ms=10000))
        if start_next_before_truncate:
            await h.run(append_audio())
            await h.run(Commit(final=True, create_response=True))
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=5000))
        await controller.before_prompt()
        assert [m for m in h.session.history if m["role"] == "assistant"] == [{"role": "assistant", "content": "He"}]
        if not start_next_before_truncate:
            await h.run(append_audio())
            await h.run(Commit(final=True, create_response=True))
            messages = ast.literal_eval(h.port.submissions[-1].prompt["prompt"])
            assert [m for m in messages if m["role"] == "assistant"] == [{"role": "assistant", "content": "He"}]
        assert cutoffs == [10000, 5000]
        assert h.session.history_audio_cutoff(response) == 5000
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_deleted_reply_cannot_be_resurrected_by_async_result():
    entered, release = asyncio.Event(), asyncio.Event()

    async def calibrate(snapshot):
        entered.set()
        await release.wait()
        return 5

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=1000))
        await asyncio.wait_for(entered.wait(), 2)
        await h.run(DeleteItem(item_id=f"item_{response}"))
        release.set()
        await h.settle()
        assert all(m["role"] != "assistant" for m in h.session.history)
        assert h.runner.ctx.history_calibration.retained_bytes == 0
    finally:
        release.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_oversized_audio_falls_back_without_sending_a_suffix_to_asr():
    async def calibrate(snapshot):
        pytest.fail("An incomplete audio cache must not be passed to ASR")

    h = await open_qwen(calibrate=calibrate, history_max_bytes=1024)
    try:
        await completed_response(h)
        await h.runner.ctx.history_calibration.before_prompt()
        assert h.session.history[-1]["content"] == ""
        assert h.runner.ctx.history_calibration.retained_bytes == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_generation_calibrates_across_epoch_change():
    async def calibrate(snapshot):
        return 5

    h = await open_qwen(calibrate=calibrate)
    try:
        await h.run(append_audio())
        await h.run(Commit(final=True, create_response=True))
        request = h.port.submissions[-1].context.request_id
        response = h.session.active_response_id
        epoch = h.session.epoch
        await h.deliver_and_settle(tts_output(request, samples=0, text=TEXT, finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request, samples=240000, finished=False), stage_id=2)
        await h.run(AckPlayback(response_id=response, played_ms=1000))
        await h.run(CancelResponse())
        await h.runner.ctx.history_calibration.before_prompt()
        assert h.session.epoch > epoch
        assert h.session.history[-1]["content"] == "Hello"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_smaller_ordinary_ack_does_not_shrink_prefix_and_full_ack_releases_audio():
    async def calibrate(snapshot):
        return 5

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        await h.runner.ctx.history_calibration.before_prompt()
        await h.run(AckPlayback(response_id=response, played_ms=200))
        assert h.session.history[-1]["content"] == "Hello"
        await h.run(AckPlayback(response_id=response, played_ms=10000))
        assert h.session.history[-1]["content"] == TEXT
        assert h.runner.ctx.history_calibration.retained_bytes == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_late_result_cannot_override_newer_stricter_calibration():
    first_started = asyncio.Event()
    release_old = asyncio.Event()

    async def calibrate(snapshot):
        if snapshot.played_ms == 1000:
            first_started.set()
            try:
                await release_old.wait()
            except asyncio.CancelledError:
                # Simulate an executor whose in-flight work cannot be stopped.
                await release_old.wait()
            return 11
        return 5

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=1000))
        await asyncio.wait_for(first_started.wait(), 2)
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=500))
        await h.runner.ctx.history_calibration.before_prompt()
        assert h.session.history[-1]["content"] == "Hello"
        release_old.set()
        await asyncio.sleep(0.02)
        assert h.session.history[-1]["content"] == "Hello"
    finally:
        release_old.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_opted_in_calibration_does_not_override_commit_all_policy():
    async def calibrate(snapshot):
        pytest.fail("commit_all_on_done must keep its existing semantics")

    h = await open_qwen(calibrate=calibrate)
    try:
        await h.run(UpdateSession(patch={"playback_commit_policy": "commit_all_on_done"}))
        await completed_response(h)
        await h.runner.ctx.history_calibration.before_prompt()
        assert h.session.history[-1]["content"] == TEXT
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_next_prompt_waits_for_replacement_after_stricter_truncate():
    started, replacement_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def calibrate(snapshot):
        if snapshot.played_ms == 1000:
            started.set()
            await asyncio.Event().wait()
        replacement_started.set()
        await release.wait()
        return 5

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        await h.run(TruncateItem(item_id=f"item_{response}", audio_end_ms=1000))
        await asyncio.wait_for(started.wait(), 2)
        await h.run(append_audio())
        h.submit(Commit(final=True, create_response=True))
        await asyncio.sleep(0.02)
        assert len(h.port.submissions) == 1
        h.submit(TruncateItem(item_id=f"item_{response}", audio_end_ms=500))
        await asyncio.wait_for(replacement_started.wait(), 2)
        await asyncio.sleep(0.02)
        assert len(h.port.submissions) == 1
        release.set()
        await h.settle()
        assert len(h.port.submissions) == 2
        messages = ast.literal_eval(h.port.submissions[-1].prompt["prompt"])
        assert [m for m in messages if m["role"] == "assistant"] == [{"role": "assistant", "content": "Hello"}]
    finally:
        release.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_prompt_deadline_rejects_executor_result_after_cancellation():
    cancelled, release = asyncio.Event(), asyncio.Event()

    async def calibrate(snapshot):
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return 5

    h = await open_qwen(calibrate=calibrate, calibration_timeout_ms=30)
    try:
        await completed_response(h)
        controller = h.runner.ctx.history_calibration
        await asyncio.wait_for(controller.before_prompt(), 1)
        await asyncio.wait_for(cancelled.wait(), 1)
        assert h.session.history[-1]["content"] == ""
        release.set()
        await h.settle()
        assert h.session.history[-1]["content"] == ""
        await controller.before_prompt()
        assert not controller.tasks
    finally:
        release.set()
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_playback_ack_during_normal_playback_does_not_start_asr():
    async def calibrate(snapshot):
        pytest.fail("Periodic playback progress must not start ASR")

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h)
        for played_ms in (2000, 3000, 4000):
            await h.run(AckPlayback(response_id=response, played_ms=played_ms))
        assert not h.runner.ctx.history_calibration.tasks
        assert not h.runner.ctx.history_calibration.versions
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_ack_after_playback_clear_starts_calibration_before_next_input():
    entered = asyncio.Event()

    async def calibrate(snapshot):
        assert snapshot.played_ms == 1000
        entered.set()
        return len("Hello")

    h = await open_qwen(calibrate=calibrate)
    try:
        response = await completed_response(h, played_ms=0)
        await h.run(ClearOutputAudio())
        assert not entered.is_set()
        await h.run(AckPlayback(response_id=response, played_ms=1000))
        await asyncio.wait_for(entered.wait(), timeout=0.5)
        assert len(h.port.submissions) == 1, "Calibration must start before another turn is submitted"
        await h.runner.ctx.history_calibration.before_prompt()
        assert h.session.history[-1]["content"] == "Hello"
    finally:
        await h.manager.shutdown()
