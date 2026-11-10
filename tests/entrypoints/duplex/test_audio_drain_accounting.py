# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The original nine #7530 accounting cases, at the shared serving boundary.

Real manager/runner, plugin, StagePool, orchestrator flush, scheduler-owned
codec adapter, handle and attachment sends. The same-loop utility bridge,
model execution, codec decoding and socket are doubles: no GPU/model claim.
PCM group sizes are synthetic decoder emissions whose total is checked
against the actual de-delayed codec payloads, not invented EOF rows.
"""

import asyncio

import pytest

from tests.distributed.omni_connectors.test_chunk_transfer_adapter import build_adapter as _build_adapter_fixture
from tests.engine.duplex.test_audio_drain import _generated, _until
from tests.engine.duplex.test_session_runner import Harness, close_harness
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame, open_personaplex_harness
from tests.engine.test_codec_prefix_flush import _CodecScheduler, _request, _row, _send_queue
from tests.engine.test_orchestrator import FakeStageClient, _build_stage_pools
from tests.entrypoints.duplex.test_audio_send_completion import _next_audio
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta, ErrorEvent, TranscriptDelta
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexSessionCommandMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.model_executor.stage_input_processors.personaplex import talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
build_adapter = _build_adapter_fixture


@pytest.fixture(autouse=True)
def _prefill(monkeypatch):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)


class _Composition:
    def __init__(self, h, adapter, connector, mocker):
        self.harnesses = [h]
        self.adapter, self.connector = adapter, connector
        adapter.custom_process_next_stage_input_func = talker2code2wav_async_chunk
        self.core = StageEngineCoreProc.__new__(StageEngineCoreProc)
        self.core.scheduler = _CodecScheduler(requests={}, chunk_transfer_adapter=adapter)
        client = FakeStageClient(stage_type="llm")
        self.pool = _build_stage_pools([[client]])[0]
        self.orchestrator = DuplexOrchestrator.__new__(DuplexOrchestrator)
        self.orchestrator.stage_pools = [self.pool]
        self.orchestrator.request_states = {}
        self.flushes: list[tuple[str, int]] = []
        original_submit = h.port.submit

        async def submit(submission):
            result = await original_submit(submission)
            context = submission.context
            state = self.orchestrator.request_states.setdefault(
                context.request_id,
                DuplexOrchestratorRequestState(request_id=context.request_id, session_id=context.session_id),
            )
            state.stage_fences[0] = context.fence
            self.pool._request_bindings[context.request_id] = result.replica_id
            return result

        async def utility(request_id, sequence):
            future = self.core.omni_flush_codec_prefix(request_id, sequence)
            assert not future.done()
            _send_queue(adapter)
            result = await asyncio.wrap_future(future)
            self.flushes.append((request_id, sequence))
            return result

        mocker.patch.object(client, "flush_codec_prefix_async", new=utility, create=True)
        mocker.patch.object(h.port, "submit", new=submit)
        mocker.patch.object(h.port, "flush_audio_prefix", new=self.orchestrator.flush_audio_prefix)
        self.configure(h, mocker)

    def configure(self, h, mocker):
        mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))

        async def submit_command_async(session_id, command):
            h.manager.dispatch(DuplexSessionCommandMessage(session_id=session_id, command=command))

        # Preserve the typed frontend and this session's actual runtime limits.
        engine = DuplexOmniEngine.__new__(DuplexOmniEngine)
        engine.duplex_session_config = h.manager.runtime_config
        mocker.patch.object(engine, "submit_command_async", new=submit_command_async)
        omni = DuplexOmni.__new__(DuplexOmni)
        omni.engine = engine
        h.handle = DuplexSessionHandle(omni, h.session.session_id)
        h.handle._outbox = h.output_buffer
        h.handler = OmniDuplexSessionHandler(duplex_omni=omni)
        h.sent = []

    async def attach(self, h, mocker):
        async def send(payload):
            h.sent.append(payload)

        await h.handler._attachment_registry.create(h.session.session_id, send=send, close=mocker.AsyncMock())

    async def second(self, mocker, *, suffix="second"):
        first = self.harnesses[0]
        buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
        session_id = first.session.session_id + "-" + suffix
        await first.manager.handle(
            OpenDuplexSessionMessage(
                control_id=suffix + "-open",
                session_id=session_id,
                output_buffer=buffer,
                session_config=DuplexSessionConfig(
                    model="nvidia/personaplex-7b-v1", modalities=["audio", "text"], voice="NATF2.pt"
                ),
            )
        )
        result = await asyncio.wait_for(first.results.get(), 2)
        assert result.ok
        h = Harness(first.manager, first.port, first.output, buffer, first.results, first.manager.runners[session_id])
        self.harnesses.append(h)
        self.configure(h, mocker)
        await h.settle()
        return h

    async def generate(self, h):
        request = _request(h.stage0_request_id())
        self.core.scheduler.requests[request.request_id] = request
        for sequence in range(1, h.session.audio_delivery.accepted_seq + 1):
            self.adapter.save_async({"codes": {"audio": _row(sequence - 1)}}, request, is_segment_finished=True)
            _generated(h, sequence)
        _send_queue(self.adapter)
        await _until(lambda: h.session.audio_delivery.generated_seq == h.session.audio_delivery.accepted_seq)

    def submitted_samples(self, h):
        total = 0
        for call in self.connector.put.call_args_list:
            if not call.kwargs["put_key"].startswith(h.stage0_request_id() + "_0_"):
                continue
            payload = call.kwargs["data"]
            assert not payload.meta.finished.item()
            # The regular first one-row segment may send an existing empty
            # boundary marker. It contributes no fabricated acoustic frame.
            if payload.codes is not None and payload.codes.audio is not None:
                total += payload.codes.audio.numel() // 8 * 1920
        return total

    async def project(self, h, samples, text=""):
        before = h.session.audio_delivery.projected_samples
        h.deliver(code2wav_output(h.stage0_request_id(), samples=samples, text=text))
        if samples:
            await _until(lambda: h.session.audio_delivery.projected_samples == before + samples)
            return await _next_audio(h.output_buffer)
        await _until(lambda: h.runner._mailbox.empty())
        assert h.session.audio_delivery.projected_samples == before
        assert not [pending.event for pending in h.output_buffer._pending if isinstance(pending.event, AudioDelta)]
        return None

    async def send(self, h, event):
        before = h.session.audio_delivery.completed_samples
        await h.handler._send_event(h.session.session_id, event, handle=h.handle)
        await _until(lambda: h.session.audio_delivery.completed_samples > before)


async def _open_composition(build_adapter, mocker, *, attach=True):
    h = await open_personaplex_harness(runtime_config=DuplexSessionRuntimeConfig(max_sessions=2))
    adapter, connector = build_adapter(
        stage_id=0, connector_extra={"codec_chunk_frames": 5, "initial_codec_chunk_frames": 1}
    )
    result = _Composition(h, adapter, connector, mocker)
    if attach:
        await result.attach(h, mocker)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", [(1920, 1920), (3840, 1920), (0, 1920, 0, 1920)])
@pytest.mark.parametrize("sessions", [1, 2])
async def test_delta_totals_are_distinct_from_delivered_watermark(sizes, sessions, build_adapter, mocker):
    c = await _open_composition(build_adapter, mocker)
    try:
        if sessions == 2:
            await c.attach(await c.second(mocker), mocker)
        accepted = sum(sizes) // 1920 + 1
        for h in c.harnesses:
            for _ in range(accepted):
                # Accepted silence counts exactly like speech; no wall-clock
                # auto-continuation is allowed to alter this frozen target.
                await h.run(frame(value=0))
        drains = [h.runner.begin_audio_drain(timeout=5) for h in c.harnesses]
        for h in c.harnesses:
            await c.generate(h)
        await _until(lambda: len(c.flushes) == sessions)
        for h in c.harnesses:
            assert c.submitted_samples(h) == sum(sizes)
            assert len(c.adapter.request_payload[h.stage0_request_id()]["personaplex_frames"]) == 1
        total = 0
        for size in sizes:
            for h, drain in zip(c.harnesses, drains):
                event = await c.project(h, size)
                state = h.session.audio_delivery
                assert state.projected_samples == total + size
                assert state.completed_samples == total
                assert not drain.done()
                if event is not None:
                    await c.send(h, event)
                    assert state.completed_samples == total + size
            total += size
        for h, drain in zip(c.harnesses, drains):
            target = await drain
            assert (target.accepted_seq, target.expected_samples) == (accepted, total)
            assert h.session.playback.played_ms == 0
    finally:
        await close_harness(c.harnesses[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty", "exception"])
async def test_failed_encode_cannot_advance_delivery_or_transcript(failure, build_adapter, mocker):
    c = await _open_composition(build_adapter, mocker)
    h = c.harnesses[0]
    fail = False

    def encode(audio, _rate, _format, _speed):
        if fail:
            if failure == "exception":
                raise ValueError("encoder failure")
            return None
        return "encoded"

    mocker.patch.object(h.runner.plugin.data_plane, "_encode_audio", new=encode)
    try:
        for _ in range(3):
            await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=5)
        await c.generate(h)
        await _until(lambda: len(c.flushes) == 1)
        assert c.submitted_samples(h) == 3840
        await c.send(h, await c.project(h, 1920, "first"))
        fail = True
        pending = code2wav_output(h.stage0_request_id(), samples=1920, text="first second")
        h.deliver(pending)

        async def error():
            while True:
                event = await h.output_buffer.get()
                if isinstance(event, ErrorEvent):
                    return event

        assert "encod" in (await asyncio.wait_for(error(), 2)).message
        state = h.session.audio_delivery
        assert state.projected_samples == state.completed_samples == 1920
        assert not drain.done()
        fail = False
        h.deliver(pending)
        retried = await _next_audio(h.output_buffer)
        assert state.projected_samples == 3840 and state.completed_samples == 1920
        await c.send(h, retried)
        assert (await drain).expected_samples == 3840
        transcript_deltas = []
        while h.output_buffer.pending_events:
            event = await h.output_buffer.get()
            if isinstance(event, TranscriptDelta):
                transcript_deltas.append(event.delta)
        assert transcript_deltas == [" second"]
        assert len(h.sent) == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_close_resets_totals_without_touching_another_stream(build_adapter, mocker):
    c = await _open_composition(build_adapter, mocker)
    first = c.harnesses[0]
    old_receipt = None
    try:
        second = await c.second(mocker)
        await c.attach(second, mocker)
        for h in c.harnesses:
            for _ in range(2):
                await h.run(frame())
            drain = h.runner.begin_audio_drain(timeout=5)
            await c.generate(h)
            await _until(lambda: (h.stage0_request_id(), 2) in c.flushes)
            event = await c.project(h, 1920, "hello")
            if h is first:
                old_receipt = h.output_buffer.send_receipt(event)
                assert old_receipt is not None
            await c.send(h, event)
            assert (await drain).expected_samples == 1920
        assert old_receipt is not None
        sibling = second.session.audio_delivery
        old_request = first.stage0_request_id()
        first.submit(commands.CancelResponse())
        await _until(lambda: first.session.epoch == 1)
        c.adapter.cleanup_sender(old_request)
        await first.run(frame())
        fresh = first.session.audio_delivery
        assert fresh.request_id != old_request and fresh.epoch == 1
        assert fresh.accepted_seq == 1
        assert fresh.generated_seq == fresh.projected_samples == fresh.completed_samples == 0
        assert sibling is second.session.audio_delivery
        assert (sibling.accepted_seq, sibling.generated_seq, sibling.completed_samples) == (2, 2, 1920)
        # Close through the manager, not runner.close: cleanup must finish and
        # release one of the two admission slots before a replacement can open.
        closing_request = first.stage0_request_id()
        await first.manager.handle(
            CloseDuplexSessionMessage(
                control_id="accounting-close", session_id=first.session.session_id, reason="accounting-test"
            )
        )
        closed = await asyncio.wait_for(first.results.get(), 2)
        assert closed.ok and closed.operation == "close"
        assert first.session.session_id not in first.manager.runners
        assert first.session.session_id not in first.manager._closing
        assert any(event.type == "session.closed" for event in await first.settle())
        # The recording StagePort does not run a scheduler cleanup utility.
        c.adapter.cleanup_sender(closing_request)

        # Session ids are never reused. Reopen with a fresh id on the same
        # manager, while the second stream remains alive with its own counters.
        replacement = await c.second(mocker, suffix="replacement")
        await c.attach(replacement, mocker)
        assert set(first.manager.runners) == {second.session.session_id, replacement.session.session_id}
        assert replacement.session.audio_delivery is None
        assert replacement.session.epoch == 0
        for _ in range(2):
            await replacement.run(frame())
        fresh = replacement.session.audio_delivery
        assert fresh.request_id not in {old_request, closing_request}
        assert fresh.epoch == 0 and fresh.accepted_seq == 2
        assert fresh.generated_seq == fresh.projected_samples == fresh.completed_samples == 0
        replacement.submit(commands.AudioSendCompleted(receipt=old_receipt))
        await replacement.settle()
        assert fresh.generated_seq == fresh.projected_samples == fresh.completed_samples == 0
        drain = replacement.runner.begin_audio_drain(timeout=5)
        await c.generate(replacement)
        await _until(lambda: (replacement.stage0_request_id(), 2) in c.flushes)
        assert c.submitted_samples(replacement) == 1920
        event = await c.project(replacement, 1920, "fresh")
        assert fresh.projected_samples == 1920 and fresh.completed_samples == 0
        assert not drain.done()
        # Now the old receipt's epoch and sample interval would otherwise fit
        # this replacement. Only the request identity can reject its credit.
        assert old_receipt.watermark.epoch == fresh.epoch
        assert (old_receipt.watermark.start_samples, old_receipt.watermark.samples) == (0, 1920)
        assert old_receipt.watermark.request_id != fresh.request_id
        complete = mocker.spy(replacement.session, "complete_audio_send")
        replacement.submit(commands.AudioSendCompleted(receipt=old_receipt))
        await _until(lambda: complete.call_count == 1)
        assert fresh.completed_samples == 0 and not drain.done()
        await c.send(replacement, event)
        assert (await drain).expected_samples == 1920
        assert sibling is second.session.audio_delivery
        assert (sibling.accepted_seq, sibling.generated_seq, sibling.projected_samples, sibling.completed_samples) == (
            2,
            2,
            1920,
            1920,
        )
        assert fresh.completed_samples == 1920
        assert replacement.session.playback.played_ms == second.session.playback.played_ms == 0
    finally:
        await close_harness(first)
