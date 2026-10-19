# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Received audio must win over silence before PCM buffering, too."""

import asyncio
import base64
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import append_audio, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import open_nemotron_harness
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexOverlapPolicy
from vllm_omni.engine.duplex.session import model_channel as model_channel_module
from vllm_omni.engine.duplex.session import runner as runner_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def activate_response(h, monkeypatch):
    # Only replace these modules' clock reference; the event loop keeps its
    # real clock. The silence deadline is already due, so no timed sleep runs.
    clock = SimpleNamespace(monotonic=lambda: 100.081)
    monkeypatch.setattr(runner_module, "time", clock)
    monkeypatch.setattr(model_channel_module, "time", clock)
    h.runner.model_state.last_native_submit_monotonic = 100.0
    h.session.begin_response(turn_id=h.session.turn_id)


async def schedule_silence(h):
    response_id = h.session.active_response_id
    assert response_id is not None
    return await h.runner._schedule_silence_continuation(
        h.runner.model.silence_unit_payload(),
        request_id=h.session.active_request_id,
        owner_id=f"response:{response_id}",
        response_id=response_id,
        response_owned=True,
        expected_epoch=h.session.epoch,
        expected_model_turn_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting_at", ["mailbox", "decode"])
async def test_received_audio_prevents_silence_before_pcm_buffering(monkeypatch, waiting_at):
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        entered, release = asyncio.Event(), asyncio.Event()
        if waiting_at == "decode":
            original = h.runner.offload

            async def gated_decode(fn, *args, **kwargs):
                if fn is runner_module.convert_input_audio_with_rate:
                    entered.set()
                    await release.wait()
                return await original(fn, *args, **kwargs)

            monkeypatch.setattr(h.runner, "offload", gated_decode)
            h.submit(append_audio(samples=1280))
        else:
            original = h.runner._handle_item
            blocker = commands.UpdateSession(patch={})

            async def gated_mailbox(item):
                if item is blocker:
                    entered.set()
                    await release.wait()
                return await original(item)

            monkeypatch.setattr(h.runner, "_handle_item", gated_mailbox)
            h.submit(blocker)
            h.submit(append_audio(samples=1280))
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            # The old predicate sees both as empty and accepts a full synthetic
            # acoustic frame ahead of the client's already-received frame.
            assert not h.runner.model_state.audio_buffer.has_pending()
            assert not h.runner.model_state.audio_buffer.has_reserved()
            assert await schedule_silence(h) is False
            assert len(h.port.submissions) == 1
        finally:
            release.set()
            await h.settle()
        assert len(h.port.submissions) == 2
        assert [s.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] for s in h.port.submissions] == [
            1,
            2,
        ]
        # Successful dispatch releases input priority; reply continuation can
        # resume after the real frame has been submitted.
        h.runner.model_state.last_native_submit_monotonic = 100.0
        assert await schedule_silence(h) is True
        assert await h.runner.tasks.append_tail
        assert len(h.port.submissions) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [ValueError("invalid audio"), RuntimeError("decoder failed")])
async def test_rejected_audio_releases_silence_priority(monkeypatch, failure):
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        original = h.runner.offload

        async def reject_decode(fn, *args, **kwargs):
            if fn is runner_module.convert_input_audio_with_rate:
                raise failure
            return await original(fn, *args, **kwargs)

        monkeypatch.setattr(h.runner, "offload", reject_decode)
        events = await h.run(append_audio(samples=1280))
        assert "error" in types(events)
        assert len(h.port.submissions) == 1
        # Neither the validation-error return nor the worker's exception path
        # may leave a permanent real-input flag that suppresses continuation.
        assert await schedule_silence(h) is True
        assert await h.runner.tasks.append_tail
        assert len(h.port.submissions) == 2


def _pcm_bytes(payload: dict[str, object]) -> bytes:
    audio = payload["audio"]
    if isinstance(audio, str):
        return base64.b64decode(audio)
    if isinstance(audio, bytes):
        return audio
    raise AssertionError(type(audio))


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap_action", ["continue_output", "ack", "drop"])
async def test_ignored_overlap_frame_rearms_the_blocked_continuation(monkeypatch, overlap_action):
    """A valid frame that overlap discards must not swallow the only continuation.

    The continuation runs while the frame is still decoding, sees the pending
    counter, and returns. After the frame is ignored there is no second client
    input and no manual reschedule; the same epoch and response must still
    gain exactly one silence unit.
    """
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        # MiniCPM-o shape: auto-response is already on, and barge-in-on-speech
        # is what keeps overlap active for continue_output / ack / drop.
        h.session.capabilities = replace(h.session.capabilities, supports_barge_in=True)
        h.session.config.overlap_policy = DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value
        received = append_audio(samples=1280)
        ignored = commands.AppendAudio(
            audio=received.audio,
            format=received.format,
            sample_rate_hz=received.sample_rate_hz,
            is_speech=True,
            hints={"overlap_action": overlap_action},
        )
        entered, release = asyncio.Event(), asyncio.Event()
        original = h.runner.offload

        async def gated_decode(fn, *args, **kwargs):
            if fn is runner_module.convert_input_audio_with_rate:
                entered.set()
                await release.wait()
            return await original(fn, *args, **kwargs)

        monkeypatch.setattr(h.runner, "offload", gated_decode)
        response_id = h.session.active_response_id
        epoch = h.session.epoch
        h.submit(ignored)
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            await h.runner.model.maybe_continue_response(
                expected_epoch=epoch,
                expected_response_id=response_id,
            )
            assert len(h.port.submissions) == 1
            assert h.runner.model_state.continuation_units == 0
        finally:
            release.set()
        await h.settle()
        assert h.session.epoch == epoch
        assert h.session.active_response_id == response_id
        assert len(h.port.submissions) == 2
        assert h.runner.model_state.continuation_units == 1
        silence = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]
        assert silence["epoch"] == epoch
        assert set(_pcm_bytes(silence["payload"])) == {0}


def enable_overlap_discard(h) -> None:
    h.session.capabilities = replace(h.session.capabilities, supports_barge_in=True)
    h.session.config.overlap_policy = DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value


def ignored_append(overlap_action: str) -> commands.AppendAudio:
    received = append_audio(samples=1280)
    return commands.AppendAudio(
        audio=received.audio,
        format=received.format,
        sample_rate_hz=received.sample_rate_hz,
        is_speech=True,
        hints={"overlap_action": overlap_action},
    )


@pytest.mark.asyncio
async def test_stale_sleeper_keeps_another_owners_deferred_continuation(monkeypatch):
    """An older pacing sleep must not clear a slot another continuation just stored."""
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        enable_overlap_discard(h)
        # The deadline is not yet due, so the older continuation blocks in sleep.
        clock = SimpleNamespace(monotonic=lambda: 100.0)
        monkeypatch.setattr(runner_module, "time", clock)
        monkeypatch.setattr(model_channel_module, "time", clock)
        h.runner.model_state.last_native_submit_monotonic = 100.0
        h.runner.model_state.silence_deadline_monotonic = None
        entered_sleep, release_sleep = asyncio.Event(), asyncio.Event()
        entered_decode, release_decode = asyncio.Event(), asyncio.Event()
        slept = False
        real_sleep = runner_module.asyncio.sleep

        async def gated_sleep(delay, *args, **kwargs):
            nonlocal slept
            if delay > 0 and not slept:
                slept = True
                entered_sleep.set()
                await release_sleep.wait()
                return None
            return await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(runner_module.asyncio, "sleep", gated_sleep)
        original = h.runner.offload

        async def gated_decode(fn, *args, **kwargs):
            if fn is runner_module.convert_input_audio_with_rate:
                entered_decode.set()
                await release_decode.wait()
            return await original(fn, *args, **kwargs)

        monkeypatch.setattr(h.runner, "offload", gated_decode)
        response_id = h.session.active_response_id
        old_epoch = h.session.epoch
        older = asyncio.create_task(
            h.runner.model.maybe_continue_response(
                expected_epoch=old_epoch,
                expected_response_id=response_id,
            )
        )
        await asyncio.wait_for(entered_sleep.wait(), timeout=2)
        h.session.epoch = old_epoch + 1
        h.submit(ignored_append("drop"))
        await asyncio.wait_for(entered_decode.wait(), timeout=2)
        try:
            await h.runner.model.maybe_continue_response(
                expected_epoch=h.session.epoch,
                expected_response_id=response_id,
            )
            deferred = h.runner._deferred_silence
            assert deferred is not None
            assert deferred.expected_epoch == h.session.epoch
            assert deferred.expected_response_id == response_id
            release_sleep.set()
            await older
            assert h.runner._deferred_silence is deferred
            assert len(h.port.submissions) == 1
        finally:
            release_sleep.set()
            release_decode.set()
        await h.settle()
        assert h.session.active_response_id == response_id
        assert len(h.port.submissions) == 2
        assert h.runner.model_state.continuation_units == 1
        silence = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]
        assert silence["epoch"] == old_epoch + 1
        assert set(_pcm_bytes(silence["payload"])) == {0}


@pytest.mark.asyncio
async def test_model_turn_replay_skips_a_response_begun_from_queued_output(monkeypatch):
    """A deferred model-turn continuation must not feed a response begun later."""
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        h.session.mark_audio_sent(duration_ms=80)
        h.session.end_response(commit_text=False, preserve_request=True)
        assert h.session.active_response_id is None
        assert isinstance(h.session.active_request_id, str)
        # Playback keeps overlap active without an open response, so the frame
        # can be discarded while the continuation is still model-turn owned.
        assert h.session.playback.sent_ms > h.session.playback.committed_ms
        enable_overlap_discard(h)
        entered, release = asyncio.Event(), asyncio.Event()
        original = h.runner.offload

        async def gated_decode(fn, *args, **kwargs):
            if fn is runner_module.convert_input_audio_with_rate:
                entered.set()
                await release.wait()
            return await original(fn, *args, **kwargs)

        monkeypatch.setattr(h.runner, "offload", gated_decode)
        epoch = h.session.epoch
        turn_id = h.session.turn_id
        h.submit(ignored_append("continue_output"))
        await asyncio.wait_for(entered.wait(), timeout=2)
        try:
            await h.runner.model.maybe_continue_response(
                expected_epoch=epoch,
                expected_model_turn_id=turn_id,
            )
            deferred = h.runner._deferred_silence
            assert deferred is not None
            assert deferred.expected_response_id is None
            assert deferred.expected_model_turn_id == turn_id
            h.deliver(
                SimpleNamespace(
                    request_id=h.stage0_request_id(),
                    finished=False,
                    multimodal_output={},
                    outputs=[SimpleNamespace(text="", token_ids=[42], multimodal_output={})],
                ),
                stage_id=0,
                segment_finished=True,
                segment_token_ids=[42],
            )
            assert len(h.port.submissions) == 1
        finally:
            release.set()
        await h.settle()
        assert h.session.active_response_id is not None
        assert len(h.port.submissions) == 1
        assert h.runner.model_state.continuation_units == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("overlap_action", ["drop", "ack", "continue_output"])
async def test_old_append_tail_keeps_same_owners_new_deferred_continuation(monkeypatch, overlap_action):
    """Accepted input can move the append chain without changing response ownership."""
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        enable_overlap_discard(h)
        now = {"value": 100.0}
        clock = SimpleNamespace(monotonic=lambda: now["value"])
        monkeypatch.setattr(runner_module, "time", clock)
        monkeypatch.setattr(model_channel_module, "time", clock)
        h.runner.model_state.last_native_submit_monotonic = 100.0
        h.runner.model_state.silence_deadline_monotonic = None
        entered_sleep, release_sleep = asyncio.Event(), asyncio.Event()
        entered_decode, release_decode = asyncio.Event(), asyncio.Event()
        real_sleep = asyncio.sleep
        slept = False

        async def gated_sleep(delay, *args, **kwargs):
            nonlocal slept
            if delay > 0 and not slept:
                slept = True
                entered_sleep.set()
                await release_sleep.wait()
                return None
            return await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(runner_module.asyncio, "sleep", gated_sleep)
        epoch, response_id = h.session.epoch, h.session.active_response_id
        old_tail = h.runner.tasks.append_tail
        older = asyncio.create_task(
            h.runner.model.maybe_continue_response(expected_epoch=epoch, expected_response_id=response_id)
        )
        try:
            await asyncio.wait_for(entered_sleep.wait(), timeout=2)
            h.submit(append_audio(samples=1280, is_speech=False))

            async def wait_for_new_tail():
                while h.runner.tasks.append_tail is old_tail:
                    await real_sleep(0)

            await asyncio.wait_for(wait_for_new_tail(), timeout=2)
            assert await h.runner.tasks.append_tail
            assert len(h.port.submissions) == 2
            assert (h.session.epoch, h.session.active_response_id) == (epoch, response_id)
            now["value"] = 100.081
            original = h.runner.offload

            async def gated_decode(fn, *args, **kwargs):
                if fn is runner_module.convert_input_audio_with_rate:
                    entered_decode.set()
                    await release_decode.wait()
                return await original(fn, *args, **kwargs)

            monkeypatch.setattr(h.runner, "offload", gated_decode)
            # A real nonterminal listen output starts the newer continuation.
            # It runs after the following frame enters offloaded decoding.
            h.deliver(
                SimpleNamespace(
                    request_id=h.stage0_request_id(),
                    finished=False,
                    multimodal_output={},
                    outputs=[SimpleNamespace(text="", token_ids=[12], multimodal_output={})],
                ),
                stage_id=0,
                segment_finished=True,
                segment_token_ids=[12],
            )
            h.submit(ignored_append(overlap_action))
            await asyncio.wait_for(entered_decode.wait(), timeout=2)

            async def wait_for_deferred():
                while h.runner._deferred_silence is None:
                    await real_sleep(0)

            await asyncio.wait_for(wait_for_deferred(), timeout=2)
            deferred = h.runner._deferred_silence
            assert deferred.expected_epoch == epoch
            assert deferred.expected_response_id == response_id
            release_sleep.set()
            await older
            assert h.runner._deferred_silence is deferred
        finally:
            release_sleep.set()
            release_decode.set()
            await asyncio.gather(older, return_exceptions=True)
        await h.settle()
        assert (h.session.epoch, h.session.active_response_id) == (epoch, response_id)
        assert len(h.port.submissions) == 3
        assert h.runner.model_state.continuation_units == 1
        silence = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]
        assert silence["epoch"] == epoch
        assert set(_pcm_bytes(silence["payload"])) == {0}


def _due_silence_clock(h, monkeypatch):
    now = {"value": 100.0}
    clock = SimpleNamespace(monotonic=lambda: now["value"])
    monkeypatch.setattr(runner_module, "time", clock)
    monkeypatch.setattr(model_channel_module, "time", clock)
    h.runner.model_state.last_native_submit_monotonic = 100.0
    h.runner.model_state.silence_deadline_monotonic = None
    return now


@pytest.mark.asyncio
@pytest.mark.parametrize("move_tail", [False, True])
async def test_unset_append_tail_cannot_drop_a_newer_deferred_slot(monkeypatch, move_tail):
    """None is not a chain id. An older sleeper must not clear or replace the newer slot."""
    async with open_nemotron_harness(monkeypatch) as h:
        await h.run(append_audio(samples=1280))
        activate_response(h, monkeypatch)
        h.runner.tasks.append_tail = None
        _due_silence_clock(h, monkeypatch)
        entered_sleep, release_sleep = asyncio.Event(), asyncio.Event()
        real_sleep = asyncio.sleep
        slept = False

        async def gated_sleep(delay, *args, **kwargs):
            nonlocal slept
            if delay > 0 and not slept:
                slept = True
                entered_sleep.set()
                await release_sleep.wait()
                return None
            return await real_sleep(delay, *args, **kwargs)

        monkeypatch.setattr(runner_module.asyncio, "sleep", gated_sleep)
        older = asyncio.create_task(schedule_silence(h))
        try:
            await asyncio.wait_for(entered_sleep.wait(), timeout=2)
            h.runner._pending_input_appends = 1
            assert await schedule_silence(h) is False
            deferred = h.runner._deferred_silence
            assert deferred is not None
            assert deferred.append_tail is None
            if move_tail:
                h.runner.tasks.append_tail = asyncio.create_task(real_sleep(0))
            release_sleep.set()
            assert await older is False
            assert h.runner._deferred_silence is deferred
        finally:
            release_sleep.set()
            h.runner._pending_input_appends = 0
            await asyncio.gather(older, return_exceptions=True)
