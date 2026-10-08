# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Gander history through the engine session and ordinary stage port."""

import asyncio
import base64
from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason

from tests.engine.duplex.test_session_runner import append_audio, close_harness, open_harness, pcm_f32
from vllm_omni.engine.duplex.commands import SignalTurn
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.events import ErrorEvent
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.engine.duplex.session.context_history import DuplexContextHistory
from vllm_omni.engine.duplex_orchestrator import DuplexOrchestrator, DuplexOrchestratorRequestState
from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import GanderContextPolicy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def initialized():
    h = await open_harness()
    h.session.replace_runtime_config(
        {**h.session.runtime_config, "gander_enabled": True, "gander_context_version": 0, "duplex_context_version": 0}
    )
    h.runner.ctx.history = DuplexContextHistory(
        h.runner.ctx,
        GanderContextPolicy(),
        out=h.runner.out,
        model=h.runner.model,
        max_tokens=40960,
        wait_for_append_tail=h.runner._wait_for_append_tail,
        close_from_runtime=h.runner._close_from_runtime,
    )
    await h.run(append_audio())
    complete(h, seq=1)
    await h.settle()
    return h


def complete(h, *, seq):
    rid = h.stage0_request_id()
    output = SimpleNamespace(
        request_id=rid,
        finished=True,
        outputs=[SimpleNamespace(token_ids=[7], stop_reason=7)],
        multimodal_output={"special_token_ids": {"listen_token_id": 7, "gander_append_seq": seq}},
    )
    h.deliver(output, stage_id=0, segment_finished=True, segment_token_ids=(7,))


def edit(h, edits, event_id="edit"):
    return SignalTurn(
        event="input.context.replace",
        signal_payload={
            "kind": "history_edit",
            "event_id": event_id,
            "epoch": h.session.epoch,
            "base_version": h.session.runtime_config["duplex_context_version"],
            "edits": edits,
        },
    )


@pytest.mark.asyncio
async def test_invalid_edit_preserves_request_and_accepts_next_audio():
    h = await initialized()
    try:
        ids = h.session.resource_request_ids()
        before = h.runner.ctx.history.snapshot()
        await h.run(edit(h, [{"op": "delete", "unit_id": "missing"}]))
        assert h.runner.ctx.history.snapshot() == before
        assert h.session.resource_request_ids() == ids
        assert not h.port.cleanups
        await h.run(append_audio())
        assert len(h.port.submissions) == 2
        assert h.port.submissions[-1].already_submitted
        complete(h, seq=2)
        await h.runner.ctx.history.wait_applied()
        assert len(h.runner.ctx.history.prompts) == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_invalid_edit_waits_for_accepted_append_reply(monkeypatch):
    h = await initialized()
    accepted = asyncio.Event()
    reply = asyncio.Event()
    tasks = []
    original_submit = h.port.submit

    async def hold_reply(submission):
        result = await original_submit(submission)
        accepted.set()
        await reply.wait()
        return result

    try:
        ids = h.session.resource_request_ids()
        epoch = h.session.epoch
        monkeypatch.setattr(h.port, "submit", hold_reply)
        append = await h.runner._start_append({"audio": "", "sample_rate": 16000}, final=False)
        tasks.append(append)
        await asyncio.wait_for(accepted.wait(), 2)
        validation = asyncio.Event()
        original_prepare = h.runner.ctx.history.policy.prepare_replacement

        def prepare(*args, **kwargs):
            validation.set()
            return original_prepare(*args, **kwargs)

        monkeypatch.setattr(h.runner.ctx.history.policy, "prepare_replacement", prepare)
        rejected = asyncio.create_task(h.runner._on_command(edit(h, [{"op": "delete", "unit_id": "missing"}])))
        tasks.append(rejected)
        await asyncio.sleep(0)
        assert not rejected.done()
        assert not validation.is_set(), "validation must wait for the accepted append's receipt"
        assert not append.cancelled()
        complete(h, seq=2)
        reply.set()
        assert await asyncio.wait_for(append, 2)
        await asyncio.wait_for(rejected, 2)
        assert validation.is_set()
        assert h.session.epoch == epoch
        assert h.session.resource_request_ids() == ids
        assert not h.port.cleanups
        await h.run(append_audio())
        complete(h, seq=3)
        await h.runner.ctx.history.wait_applied()
        assert len(h.port.submissions) == 3
        assert len(h.runner.ctx.history.prompts) == 3
    finally:
        reply.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_call_registered_during_result_append_accepts_its_result(monkeypatch):
    from tests.model_executor.models.minicpmo_4_5.duplex.test_gander_tools import Tokenizer
    from vllm_omni.model_executor.models.minicpmo_4_5 import gander_tools

    h = await initialized()
    accepted = asyncio.Event()
    reply = asyncio.Event()
    task = None
    original_append = h.runner.model.append_runtime_input

    async def hold_reply(*args, **kwargs):
        result = await original_append(*args, **kwargs)
        accepted.set()
        await reply.wait()
        return result

    async def register(call_id):
        await h.runner.model._send_one_model_output_event(
            {"function_call": True, "name": "lookup", "arguments": "{}", "call_id": call_id},
            expected_epoch=h.session.epoch,
        )

    def result(call_id):
        return {"kind": "tool_result", "event_id": f"result-{call_id}", "epoch": 0, "call_id": call_id, "output": "ok"}

    try:
        monkeypatch.setattr(gander_tools, "tokenizer_for", lambda path: Tokenizer())
        h.session.replace_runtime_config(
            {
                **h.session.runtime_config,
                "gander_tokenizer_path": "fake",
                "gander_tools": [{"name": "lookup", "parameters": {"type": "object"}}],
            }
        )
        await register("c1")
        assert "c1" in h.session.runtime_config["gander_calls"]
        monkeypatch.setattr(h.runner.model, "append_runtime_input", hold_reply)
        task = asyncio.create_task(h.runner.ctx.history.handle("input.context.append", result("c1")))
        await asyncio.wait_for(accepted.wait(), 2)
        assert not task.done()
        await register("c2")
        complete(h, seq=2)
        reply.set()
        assert await asyncio.wait_for(task, 2)
        assert h.session.runtime_config["gander_calls"]["c2"]["result"] is None
        task = asyncio.create_task(h.runner.ctx.history.handle("input.context.append", result("c2")))
        for _ in range(100):
            if len(h.port.submissions) == 3:
                break
            await asyncio.sleep(0.005)
        assert len(h.port.submissions) == 3
        complete(h, seq=3)
        assert await asyncio.wait_for(task, 2)
        assert all(call["result"] is not None for call in h.session.runtime_config["gander_calls"].values())
    finally:
        reply.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


async def pending_tool_call(monkeypatch):
    from tests.model_executor.models.minicpmo_4_5.duplex.test_gander_tools import Tokenizer
    from vllm_omni.model_executor.models.minicpmo_4_5 import gander_tools

    h = await initialized()
    monkeypatch.setattr(gander_tools, "tokenizer_for", lambda path: Tokenizer())
    h.session.replace_runtime_config(
        {
            **h.session.runtime_config,
            "gander_tokenizer_path": "fake",
            "gander_tools": [{"name": "lookup", "parameters": {"type": "object"}}],
        }
    )
    await h.runner.model._send_one_model_output_event(
        {"function_call": True, "name": "lookup", "arguments": "{}", "call_id": "pending"},
        expected_epoch=h.session.epoch,
    )
    return h


@pytest.mark.asyncio
async def test_targeted_cancel_does_not_invalidate_pending_tool_or_input(monkeypatch):
    h = await pending_tool_call(monkeypatch)
    try:
        epoch, requests = h.session.epoch, h.session.resource_request_ids()
        # An inactive response ID is a no-op while the native session listens.
        await h.runner._on_cancel({"type": "response.cancel", "response_id": "unrelated"})
        response = h.session.begin_response()
        # A different active response must also survive the wrong target.
        await h.runner._on_cancel({"type": "response.cancel", "response_id": "unrelated"})
        assert h.session.active_response_id == response
        assert h.session.epoch == epoch
        assert h.session.resource_request_ids() == requests
        assert not h.port.aborts
        runtime, payload = h.runner.ctx.history.policy.prepare_input(
            {"kind": "tool_result", "event_id": "result", "epoch": epoch, "call_id": "pending", "output": "ok"},
            dict(h.session.runtime_config),
            epoch=epoch,
        )
        assert payload is not None and runtime["gander_calls"]["pending"]["result"] is not None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_effective_cancel_rejects_late_tool_result_and_allows_next_input(monkeypatch):
    h = await pending_tool_call(monkeypatch)
    try:
        epoch = h.session.epoch
        h.session.begin_response()
        await h.run(SignalTurn(event="input.cancel", signal_payload={}))
        assert h.session.epoch > epoch
        before = len(h.port.submissions)
        receipts = dict(h.session.runtime_config.get("gander_context_receipts", {}))
        for result_epoch in (epoch, h.session.epoch):
            await h.run(
                SignalTurn(
                    event="input.context.append",
                    signal_payload={
                        "kind": "tool_result",
                        "event_id": "late-result",
                        "epoch": result_epoch,
                        "call_id": "pending",
                        "output": "obsolete",
                    },
                )
            )
        assert len(h.port.submissions) == before
        assert h.session.runtime_config.get("gander_context_receipts", {}) == receipts
        assert h.session.runtime_config["gander_calls"]["pending"]["result"] is None
        await h.run(append_audio())
        assert len(h.port.submissions) == before + 1
        complete(h, seq=1)
        await h.runner.ctx.history.wait_applied()
        assert not h.runner.ctx.run.closing
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_cleanup_failure_retains_gander_requests_for_close_retry(monkeypatch):
    h = await initialized()
    cleanup = h.port.cleanup
    attempts = []

    async def fail_once(request_ids: list[str], *, abort: bool = False) -> None:
        attempts.append((list(request_ids), abort))
        if len(attempts) == 1:
            raise RuntimeError("injected cleanup failure")
        await cleanup(request_ids, abort=abort)

    try:
        epoch = h.session.epoch
        request_ids = h.session.resource_request_ids()
        assert request_ids
        monkeypatch.setattr(h.port, "cleanup", fail_once)
        h.session.begin_response()
        events = await h.run(SignalTurn(event="input.cancel", signal_payload={}))
        assert h.session.epoch > epoch
        assert any(isinstance(event, ErrorEvent) and event.code == "runtime_signal_failed" for event in events)
        assert h.session.resource_request_ids() == request_ids
        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="close-after-failure", session_id=h.session.session_id)
        )
        result = await asyncio.wait_for(h.results.get(), timeout=2)
        assert result.ok
        assert attempts == [(request_ids, True), (request_ids, True)]
        assert not h.session.resource_request_ids()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_native_speech_interrupt_preserves_pending_application_tool(monkeypatch):
    h = await pending_tool_call(monkeypatch)
    try:
        epoch = h.session.epoch
        response = h.session.begin_response()
        await h.runner.model._send_one_model_output_event(
            {"is_interrupt": True, "is_listen": True}, expected_epoch=epoch
        )
        events = await h.settle()
        assert any(event.type == "output_audio_buffer.cleared" and event.response_id == response for event in events)
        assert h.session.epoch == epoch
        assert h.session.active_response_id is None
        runtime, payload = h.runner.ctx.history.policy.prepare_input(
            {"kind": "tool_result", "event_id": "result", "epoch": epoch, "call_id": "pending", "output": "ok"},
            dict(h.session.runtime_config),
            epoch=epoch,
        )
        assert payload is not None and runtime["gander_calls"]["pending"]["result"] is not None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_replacement_waits_for_model_completion_and_suppresses_replay():
    h = await initialized()
    task = None
    try:
        key = h.runner.ctx.history.snapshot()["units"][0]["unit_id"]
        task = asyncio.create_task(h.runner._on_command(edit(h, [{"op": "pin", "unit_id": key}])))
        for _ in range(100):
            if len(h.port.submissions) == 2:
                break
            await asyncio.sleep(0.005)
        assert len(h.port.submissions) == 2
        assert not task.done(), "submission is not a reconstruction completion receipt"
        assert h.session.epoch == 1
        assert not h.port.submissions[-1].already_submitted
        assert h.port.cleanups and h.port.cleanups[-1][1]
        before = list(h.runner.ctx.history.prompts[0]["model_intermediate_buffer"]["duplex"]["gander_output_ids"])
        complete(h, seq=1)
        await asyncio.wait_for(task, 2)
        await h.settle()
        assert h.runner.ctx.history.snapshot()["units"][0]["pinned"]
        assert h.runner.ctx.history.prompts[0]["model_intermediate_buffer"]["duplex"]["gander_output_ids"] == before
        assert any(e.to_realtime().get("type") == "duplex.input.context.replaced" for e in h.events)
        await h.run(append_audio())
        assert h.port.submissions[-1].already_submitted
        complete(h, seq=2)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_replacement_releases_dropped_journal_budget():
    h = await initialized()
    task = None
    try:
        history = h.runner.ctx.history
        await h.run(append_audio())
        complete(h, seq=2)
        await history.wait_applied()
        removed = history.snapshot()["units"][0]["unit_id"]
        task = asyncio.create_task(h.runner._on_command(edit(h, [{"op": "delete", "unit_id": removed}])))
        for _ in range(100):
            if len(h.port.submissions) == 3:
                break
            await asyncio.sleep(0.005)
        assert len(h.port.submissions) == 3
        complete(h, seq=1)
        await asyncio.wait_for(task, 2)
        assert len(history.prompts) == 1
        next_input = {"prompt_token_ids": [1]}
        history.max_bytes = history._size([*history.prompts, next_input])
        history.max_tokens = sum(history.policy.token_count(p) for p in history.prompts) + 2
        history.record("budget-probe", next_input)
        assert len(history.prompts) == 2
        with pytest.raises(ValueError, match="budget"):
            history.record("over-budget", next_input)
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


def test_context_wire_command_uses_engine_command_parser():
    command = translate_realtime_command({"type": "input.context.replace", "context": {"event_id": "x"}})
    assert isinstance(command, SignalTurn)
    assert command.event == "input.context.replace"
    assert command.signal_payload["event_id"] == "x"


@pytest.mark.asyncio
async def test_automatic_rollover_submits_next_audio_in_new_epoch():
    h = await initialized()
    task = None
    try:
        h.session.replace_runtime_config(
            {**h.session.runtime_config, "gander_history": {"max_units": 2, "retain_units": 1}}
        )
        await h.run(append_audio())
        complete(h, seq=2)
        task = asyncio.create_task(
            h.runner.model.append_runtime_input(
                {"audio": "", "sample_rate": 16000},
                final=False,
                expected_epoch=h.session.epoch,
            )
        )
        for _ in range(100):
            if len(h.port.submissions) >= 3:
                break
            await asyncio.sleep(0.005)
        assert h.session.epoch == 1
        complete(h, seq=1)
        ok, _ = await asyncio.wait_for(task, 2)
        assert ok
        assert h.port.submissions[-1].context.fence.epoch == 1
        assert h.port.submissions[-1].already_submitted
        complete(h, seq=2)
        await h.runner.ctx.history.wait_applied()
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_function_result_acknowledged_only_after_model_application(monkeypatch):
    h = await initialized()
    task = None
    try:

        def prepare(item, runtime, *, epoch):
            return {**runtime, "duplex_context_version": 1}, {"gander_control": True, "token_ids": [42]}

        monkeypatch.setattr(h.runner.ctx.history.policy, "prepare_input", prepare)
        item = {"type": "function_call_output", "id": "result", "call_id": "call", "output": "ok"}
        task = asyncio.create_task(h.runner.control._on_conversation_item_create({"payload": {"item": item}}))
        for _ in range(100):
            if len(h.port.submissions) == 2:
                break
            await asyncio.sleep(0.005)
        assert not task.done()
        assert not any(e.to_realtime().get("item", {}).get("id") == "result" for e in h.events)
        complete(h, seq=2)
        await asyncio.wait_for(task, 2)
        await h.settle()
        assert any(
            e.to_realtime().get("type") == "conversation.item.created"
            and e.to_realtime().get("item", {}).get("id") == "result"
            for e in h.events
        )
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_queued", [False, True])
async def test_rollover_preserves_queued_inputs_but_cancel_invalidates_them(cancel_queued):
    h = await initialized()
    tasks = []
    try:
        history = h.runner.ctx.history
        h.session.replace_runtime_config(
            {**h.session.runtime_config, "gander_history": {"max_units": 2, "retain_units": 1}}
        )
        await h.run(append_audio())
        # Leave Stage0 pending so A, B and C capture the same admission epoch.
        for index, name in enumerate(("A", "B", "C"), 1):
            tasks.append(
                await h.runner._start_append(
                    {
                        "audio": base64.b64encode(pcm_f32(16000, value=index / 10)).decode(),
                        "format": "pcm_f32le",
                        "sample_rate_hz": 16000,
                        "probe": name,
                    },
                    final=False,
                )
            )
        complete(h, seq=2)

        async def finish_inputs():
            while len(h.port.submissions) < 3:
                await asyncio.sleep(0)
            assert h.session.epoch == 1
            complete(h, seq=1)  # rollover replay
            assert await tasks[0]
            assert history.prompts[-1]["model_intermediate_buffer"]["duplex"]["payload"]["probe"] == "A"
            if cancel_queued:
                h.runner._cancel_pending_input(reason="test")
            else:
                # Complete every real/replayed unit, including another rollover
                # before C. Queued inputs must survive more than one rollover.
                while not all(task.done() for task in tasks):
                    if history.pending is not None and not history.pending.done():
                        complete(h, seq=history.prompts[-1]["model_intermediate_buffer"]["duplex"]["seq"])
                    await asyncio.sleep(0)
            await asyncio.gather(*tasks)

        await asyncio.wait_for(finish_inputs(), timeout=2)
        submitted = [
            d["payload"]["probe"]
            for entry in h.port.submissions
            if (d := entry.prompt["model_intermediate_buffer"]["duplex"])["payload"].get("probe")
            and not d["payload"].get("gander_replay")
        ]
        assert submitted == (["A"] if cancel_queued else ["A", "B", "C"])
        assert not h.runner.ctx.run.closing
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["input.cancel", "barge_in"])
async def test_cancel_pending_model_unit_then_query_context_without_new_audio(event):
    h = await initialized()
    try:
        await h.run(append_audio())
        history = h.runner.ctx.history
        pending = history.pending
        assert pending is not None and not pending.done()
        await h.runner._on_command(SignalTurn(event=event, signal_payload={}))
        assert pending.done(), "epoch invalidation must wake existing waiters"
        assert history.pending is None
        assert not history.prompts
        await asyncio.wait_for(
            h.runner._on_command(SignalTurn(event="input.context.get", signal_payload={})), timeout=1
        )
        await h.settle()
        assert any(e.to_realtime().get("error", {}).get("code") == "context_not_initialized" for e in h.events)
        assert not h.runner.ctx.run.closing
        await h.run(append_audio())
        complete(h, seq=1)
        await history.wait_applied()
        assert len(history.prompts) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_during_automatic_rollover_keeps_session_open(monkeypatch):
    h = await initialized()
    rollover_started = asyncio.Event()

    async def block_cleanup(request_ids, *, abort=False):
        del request_ids, abort
        rollover_started.set()
        await asyncio.Event().wait()

    task = None
    try:
        history = h.runner.ctx.history
        h.session.replace_runtime_config(
            {**h.session.runtime_config, "gander_history": {"max_units": 2, "retain_units": 1}}
        )
        await h.run(append_audio())
        complete(h, seq=2)
        await history.wait_applied()
        monkeypatch.setattr(h.port, "cleanup", block_cleanup)

        task = asyncio.create_task(history.replace(None, automatic=True))
        await asyncio.wait_for(rollover_started.wait(), 1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Drive the real input.cancel command path: its epoch transition owns
        # the shared journal cleanup after the append task unwinds.
        await h.runner._on_command(SignalTurn(event="input.cancel", signal_payload={}))
        await h.settle()
        assert not h.runner.ctx.run.closing
        assert h.session.state.name == "OPEN"
        assert history.pending is None
        assert history.pending_request is None
        assert not history.prompts
        assert not history.replaying
        assert h.runner.ctx.run.stream_request_id is None
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancel_mid_replay_retires_replayed_requests(monkeypatch):
    h = await initialized()
    task = None
    try:
        history = h.runner.ctx.history
        h.session.replace_runtime_config(
            {**h.session.runtime_config, "gander_history": {"max_units": 3, "retain_units": 2}}
        )
        for seq, value in ((2, 0.2), (3, 0.3)):
            await h.run(append_audio(value=value))
            complete(h, seq=seq)
            await history.wait_applied()

        # The next append rolls max_units over inside its own append task and
        # parks on the first replayed unit's completion.
        task = await h.runner._start_append(
            {
                "audio": base64.b64encode(pcm_f32(16000, value=0.4)).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "probe": "C",
            },
            final=False,
        )
        for _ in range(400):
            if len(h.port.submissions) >= 4:  # initial, A, B, replay unit 1
                break
            await asyncio.sleep(0.005)
        assert len(h.port.submissions) >= 4
        replayed_request_id = h.port.submissions[-1].context.request_id
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        await h.runner._on_command(SignalTurn(event="input.cancel", signal_payload={}))
        await h.settle()
        assert not h.runner.ctx.run.closing
        assert not history.prompts
        assert history.pending is None
        # The replay request submitted before the cancel is retired through the
        # ordinary abort-cleanup path instead of idling until session close.
        retired = [request_id for ids, abort in h.port.cleanups if abort for request_id in ids]
        assert replayed_request_id in retired

        await h.run(append_audio())
        complete(h, seq=1)
        await history.wait_applied()
        assert len(history.prompts) == 1
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_stage_request_error_fails_pending_and_closes_session():
    h = await initialized()
    try:
        await h.run(append_audio())
        history = h.runner.ctx.history
        pending = history.pending
        assert pending is not None and not pending.done()

        # A session-owned request retired by the scheduler (e.g. an invalid
        # model input isolated to this request) must wake the journal now --
        # its processed terminal output will never arrive.
        h.runner.on_request_error(0, h.stage0_request_id(), "native_duplex_prefill_failed: bad input")
        with pytest.raises(DuplexRuntimeConfigError):
            await pending
        await h.settle()
        assert h.runner.ctx.run.closing
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_old_request_error_does_not_fail_new_epoch_journal():
    h = await initialized()
    try:
        old_request_id = h.stage0_request_id()
        old_fence = h.session.fence
        await h.runner._on_command(SignalTurn(event="input.cancel", signal_payload={}))
        await h.run(append_audio())
        history = h.runner.ctx.history
        pending = history.pending
        assert pending is not None and not pending.done()

        orchestrator = object.__new__(DuplexOrchestrator)
        orchestrator.session_manager = h.manager
        orchestrator.output_async_queue = asyncio.Queue()
        # Old resources remain addressable until their asynchronous abort finishes.
        state = DuplexOrchestratorRequestState(
            request_id=old_request_id,
            session_owned=True,
            fence=h.session.fence,
            stage_fences={0: old_fence},
        )
        await orchestrator._report_duplex_session_request_error(
            0, 0, SimpleNamespace(finish_reason=FinishReason.ERROR, stop_reason="old input failed"), state
        )
        await h.settle()
        assert not pending.done()
        assert not h.runner.ctx.run.closing
        assert orchestrator.output_async_queue.empty()
        complete(h, seq=1)
        await history.wait_applied()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_current_epoch_error_from_previous_turn_still_closes_session():
    h = await initialized()
    try:
        await h.run(append_audio())
        pending = h.runner.ctx.history.pending
        h.session.complete_model_turn(h.session.turn_id)
        orchestrator = object.__new__(DuplexOrchestrator)
        orchestrator.session_manager = h.manager
        orchestrator.output_async_queue = asyncio.Queue()
        state = DuplexOrchestratorRequestState(
            request_id=h.stage0_request_id(),
            session_owned=True,
            fence=DuplexFence(h.session.session_id, epoch=h.session.epoch, turn_id=h.session.turn_id - 1),
        )
        await orchestrator._report_duplex_session_request_error(
            0, 0, SimpleNamespace(finish_reason=FinishReason.ERROR, stop_reason="current input failed"), state
        )
        with pytest.raises(DuplexRuntimeConfigError):
            await pending
        await h.settle()
        assert h.runner.ctx.run.closing
        assert not orchestrator.output_async_queue.empty()
    finally:
        await close_harness(h)
