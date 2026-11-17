# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Essential Nemotron regressions through the real session runner and plugin.

The second half drives the tool-call lifecycle the way a client does: the
model's frame-locked function channel exposes a call, the client answers it with
a ``function_call_output`` conversation item, and the session's ledger decides
whether that answer is admissible. Scheduling is not waited for -- the harness
injects the ledger's own clock and its ``call_later`` seam, so every ordering
(result before the deadline, at it, after a cancellation, after the plugin ate
the budget while rendering the result) is enumerated instead of timed.

These are runner tests over the existing ``RecordingStagePort`` fake, not GPU or
endpoint end-to-end runs: the stage outputs are synthesized exactly as the
harness in ``test_session_runner.py`` does, and everything below the stage port
-- mailbox, control, ledger, plugin, worker-facing runtime config -- is real.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
import torch

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import (
    SESSION_ID,
    Harness,
    RecordingStagePort,
    _fake_encode_audio,
    append_audio,
    tts_output,
    types,
)
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig, DuplexSessionState
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import ErrorEvent
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage, OpenDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import load_duplex_plugin
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState
from vllm_omni.model_executor.models.nemotron_voicechat.duplex import plugin as plugin_module
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    NemotronVoiceChatThinkerForConditionalGeneration as Thinker,
)
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import NEMOTRON_VOICECHAT_PIPELINE
from vllm_omni.protocol.realtime.items import normalize_conversation_item
from vllm_omni.utils.mm_outputs import build_mm_cpu
from vllm_omni.worker.output.payload_build import build_omni_mm_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class NemotronHarness(Harness):
    """The shared harness plus the two seams the tool-result deadline runs on."""

    #: Injected session clock; ``None`` means the session used ``time.monotonic``.
    clock: FakeClock | None = None
    #: Installed in place of the loop timer when a test drives expiry by hand.
    scheduler: FakeTimerScheduler | None = None
    #: The tokenizer the plugin caches; tests mutate ``encode`` to inject work.
    tokenizer: SimpleNamespace | None = None


@asynccontextmanager
async def open_nemotron_harness(
    monkeypatch,
    *,
    clock: FakeClock | None = None,
    extra_body: dict[str, object] | None = None,
    scheduler: FakeTimerScheduler | None = None,
    runtime_config: DuplexSessionRuntimeConfig | None = None,
):
    """Open one Nemotron session over the recording stage port."""
    monkeypatch.setenv("NEMOTRON_VOICECHAT_LLM_PATH", "test-tokenizer")
    special_ids = dict(
        nvc_text_bos_id=0,
        nvc_text_eos_id=1,
        nvc_text_pad_id=12,
        nvc_function_sotc_id=20,
        nvc_function_eotc_id=21,
        nvc_function_eotr_id=22,
        nvc_tokenizer_ref="test-tokenizer",
    )
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [42],
        decode=lambda ids, **kwargs: '[{"name":"lookup","arguments":{"x":1}}]',
    )
    monkeypatch.setattr(plugin_module, "_load_tokenizer_runtime", lambda _: (special_ids, tokenizer))
    plugin = load_duplex_plugin(NEMOTRON_VOICECHAT_PIPELINE.duplex_plugin, _fake_encode_audio)
    port = RecordingStagePort(stage_count=3)
    output: asyncio.Queue = asyncio.Queue()
    results: asyncio.Queue = asyncio.Queue()
    limits = runtime_config or DuplexSessionRuntimeConfig()
    output_buffer = DuplexOutputBuffer(
        max_bytes=limits.max_pending_output_bytes_per_session,
        max_events=limits.max_pending_output_events_per_session,
    )
    manager = DuplexSessionManager(
        plugin=plugin,
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=limits,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(stt_cfg={}), max_model_len=8192),
        clock=clock,
    )
    try:
        config = DuplexSessionConfig(
            model="nemotron",
            instructions="hi",
            extra_body={"auto_response": True, **(extra_body or {})},
        )
        await manager.handle(
            OpenDuplexSessionMessage(
                control_id="open", session_id=SESSION_ID, session_config=config, output_buffer=output_buffer
            )
        )
        result = await asyncio.wait_for(results.get(), timeout=2)
        assert result.ok, result
        h = NemotronHarness(
            manager=manager,
            port=port,
            output=output,
            output_buffer=output_buffer,
            results=results,
            runner=manager.runners[SESSION_ID],
            clock=clock,
            scheduler=scheduler,
            tokenizer=tokenizer,
        )
        if scheduler is not None:
            # The session arms its real loop timers in ``start()``; a test that
            # drives the deadline by hand takes the seam over before any call
            # exists, so no timer is ever armed on the real loop.
            h.runner.tool_calls.set_scheduler(scheduler)
        await h.settle()
        yield h
    finally:
        await manager.shutdown()


@pytest_asyncio.fixture
async def harness(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        yield h


async def deliver(h, token_id=12, **metadata):
    output = SimpleNamespace(
        request_id=h.stage0_request_id(),
        finished=False,
        multimodal_output={},
        outputs=[SimpleNamespace(text="", token_ids=[token_id], multimodal_output={})],
    )
    return await h.deliver_and_settle(
        output,
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[token_id],
        segment_output_metadata=metadata,
    )


@pytest.mark.asyncio
async def test_frame_buffering_prefill_and_resumable_append(harness):
    h = harness
    await h.run(append_audio(samples=1000))
    assert h.port.submissions == []
    await h.run(append_audio(samples=280))
    await h.run(append_audio(samples=1280))
    first, second = h.port.submissions
    assert first.prompt["prompt_token_ids"] == [0, 42, 1, 12]
    assert second.prompt["prompt_token_ids"] == [12]
    assert first.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 1
    assert second.prompt["model_intermediate_buffer"]["duplex"]["source_input_seq"] == 2
    assert not first.already_submitted and second.already_submitted
    assert first.context.request_id == second.context.request_id == h.stage0_request_id()
    assert "error" not in types(h.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_ids", [("active",), ("other", "active"), ("active", "other")])
async def test_tool_batches_reach_worker_and_drain_before_ninth_result(harness, request_ids):
    h = harness
    await h.run(append_audio(samples=1280))
    worker_state = {"function_response_generation": 0, "forced_function_tokens": []}
    thinker = Thinker.__new__(Thinker)
    torch.nn.Module.__init__(thinker)
    thinker._sessions = {"active": worker_state, "other": {"function_response_generation": 2}}
    thinker._use_function_head, thinker._dtype = True, torch.float32
    thinker.function_head = torch.nn.Identity()
    key = "nvc_function_response_consumed_generation"
    assert key in thinker.omni_client_multimodal_output_keys
    infos = [{"request_id": rid} for rid in request_ids]
    hidden = torch.zeros((len(infos) + 1, 2))
    for generation in range(1, 10):
        for token_id in (20, 7, 21):
            events = await deliver(h, nvc_function_token=[token_id])
        done = next(event for event in events if event.type == "response.function_call_arguments.done")
        item = {"type": "function_call_output", "call_id": done.call_id, "output": "20"}
        events = await h.run(commands.CreateItem(item=normalize_conversation_item(item)))
        assert "error" not in types(events)
        runtime = h.session.runtime_config
        assert runtime["nvc_function_response_generation"] == generation
        assert runtime["nvc_function_response_batches"][-1] == {"generation": generation, "token_ids": [42]}
        if generation < 8:
            continue
        if generation == 8:
            with pytest.raises(plugin_module.NemotronVoiceChatClientRuntimeConfigError, match="faster than"):
                h.runner.plugin.runtime_config_for_function_output(h.session.config, runtime, item)
        # Copy into model state exactly once, even if the same runtime snapshot is replayed.
        Thinker._sync_forced_function_response(worker_state, runtime)
        Thinker._sync_forced_function_response(worker_state, runtime)
        assert worker_state["forced_function_tokens"] == [42] * generation
        assert worker_state["function_response_generation"] == generation
        output = thinker.make_omni_output(hidden, model_intermediate_buffer=infos)
        for idx, rid in enumerate(request_ids):
            payload = build_omni_mm_payload(
                combined_multimodal_outputs=None,
                mm_cpu=build_mm_cpu(output.multimodal_outputs),
                rid=rid,
                idx=idx,
                start=0 if idx == 0 else 2,
                end=2 if idx == 0 else 3,
                audio_sparse_output=False,
                sparse_mm_index={},
                hidden_seq_len=len(hidden),
                scheduled_seq_len=len(hidden),
            )
            assert int(payload[key]) == (generation if rid == "active" else 2)
            if rid == "active":
                await deliver(h, **payload)
        assert h.session.runtime_config["nvc_function_response_batches"] == []
        assert h.session.runtime_config["nvc_function_response_generation"] == generation
    assert "error" not in types(h.events)


# --------------------------------------------------------------------------- #
# Tool-call lifecycle: exposure, admission, cancellation, deadline             #
# --------------------------------------------------------------------------- #


def errors(events) -> list[ErrorEvent]:
    return [event for event in events if isinstance(event, ErrorEvent)]


def error_codes(events) -> list[str]:
    return [event.code for event in errors(events)]


def output_item(call_id: str, output: str = "20", **extra: object) -> dict[str, object]:
    item: dict[str, object] = {"type": "function_call_output", "call_id": call_id, "output": output}
    item.update(extra)
    return normalize_conversation_item(item)


async def expose_function_call(h: Harness, *, tokens: tuple[int, ...] = (20, 7, 21)):
    """Emit one frame-locked function call; return its call id and the events.

    The ids come from the projected ``response.function_call_arguments.done``,
    which is exactly what a client would answer -- so the assertions below are
    about the same call id the wire carries, not about the ledger's copy.
    """
    events: list = []
    call_id: str | None = None
    for token_id in tokens:
        events.extend(await deliver(h, nvc_function_token=[token_id]))
        done = next((event for event in events if event.type == "response.function_call_arguments.done"), None)
        if done is not None:
            call_id = done.call_id
    assert call_id, f"the model exposed no function call: {types(events)}"
    return call_id, events


async def answer(h: Harness, call_id: str, output: str = "20", **extra: object):
    return await h.run(commands.CreateItem(item=output_item(call_id, output, **extra)))


async def prepare(h: Harness) -> str:
    """Append a frame and expose one call: the starting point of every case."""
    await h.run(append_audio(samples=1280))
    call_id, _ = await expose_function_call(h)
    return call_id


def generation(h: Harness) -> int:
    return int(h.session.runtime_config.get("nvc_function_response_generation", 0))


def batches(h: Harness) -> list:
    return list(h.session.runtime_config.get("nvc_function_response_batches") or [])


@pytest.mark.asyncio
async def test_the_exposed_call_is_registered_before_the_client_can_answer_it(harness):
    h = harness
    call_id = await prepare(h)

    record = h.runner.tool_calls.record(call_id)
    assert record is not None
    assert record.state is ToolCallState.PENDING
    assert record.origin.epoch == h.session.epoch
    assert record.origin.turn_id == h.session.turn_id
    # The record holds the same ids the projection handed the client, so the
    # answer can be matched to the item the client saw.
    done = next(event for event in h.events if event.type == "response.function_call_arguments.done")
    assert record.origin.item_id == done.item_id
    assert record.origin.response_id == done.response_id
    assert error_codes(h.events) == []


@pytest.mark.asyncio
async def test_call_budget_refuses_new_calls_without_interrupting_pending_results_or_audio(monkeypatch):
    limits = DuplexSessionRuntimeConfig(max_tool_calls_per_session=1)
    async with open_nemotron_harness(monkeypatch, runtime_config=limits) as h:
        call_id = await prepare(h)
        pending = h.runner.tool_calls.record(call_id)
        events = []
        for token_id in (20, 7, 21):
            events.extend(await deliver(h, nvc_function_token=[token_id]))

        assert error_codes(events) == ["resource_exhausted"]
        assert errors(events)[0].to_realtime()["error"]["type"] == "rate_limit_error"
        assert "response.function_call_arguments.done" not in types(events)
        assert "session.closed" not in types(events)
        assert not h.runner.closing
        assert pending.state is ToolCallState.PENDING
        assert len(h.runner.tool_calls) == 1

        submissions_before = len(h.port.submissions)
        assert error_codes(await h.run(append_audio(samples=1280))) == []
        assert len(h.port.submissions) == submissions_before + 1
        assert error_codes(await answer(h, call_id)) == []
        assert pending.state is ToolCallState.ACCEPTED
        worker_state = _worker()
        Thinker._sync_forced_function_response(worker_state, h.session.runtime_config)
        assert worker_state["forced_function_tokens"] == [42]

        audio = tts_output(h.stage0_request_id(), samples=1764)
        events = await h.deliver_and_settle(audio, stage_id=2, segment_finished=True)
        assert error_codes(events) == []
        delta = next(event for event in events if event.type == "response.output_audio.delta")
        assert delta.delta == "wav-1764"
        assert not h.runner.closing


@pytest.mark.asyncio
async def test_duplicate_call_registration_still_closes_the_session(harness):
    h = harness
    call_id = await prepare(h)
    await h.runner.model._expose_function_call({"call_id": call_id, "name": "lookup", "arguments": "{}"})
    events = await h.settle()

    assert error_codes(events) == ["tool_call_registration_failed"]
    assert errors(events)[0].to_realtime()["error"]["type"] == "server_error"
    assert "response.function_call_arguments.done" not in types(events)
    assert types(events).count("session.closed") == 1
    assert h.session.state is DuplexSessionState.CLOSED
    assert error_codes(await answer(h, call_id)) == ["unknown_session"]


@pytest.mark.asyncio
async def test_a_result_is_admitted_once_and_a_replay_is_refused(harness):
    h = harness
    call_id = await prepare(h)

    events = await answer(h, call_id)
    assert error_codes(events) == []
    assert generation(h) == 1
    assert batches(h)[-1] == {"generation": 1, "token_ids": [42]}
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.ACCEPTED

    # The refusal is about the *call*, not the item: the client may re-send the
    # same call id under a fresh auto item id and it is still a duplicate.
    replay = await answer(h, call_id, "different", id=f"item_{uuid4().hex}")
    assert error_codes(replay) == ["invalid_function_call_output"]
    error = errors(replay)[0]
    assert error.param == "item.output"
    assert "already exists" in error.message
    assert generation(h) == 1 and len(batches(h)) == 1


@pytest.mark.asyncio
async def test_an_unknown_or_client_imported_call_is_never_answerable(harness):
    h = harness
    await h.run(append_audio(samples=1280))

    unknown = await answer(h, "call_never_exposed")
    assert error_codes(unknown) == ["invalid_function_call_output"]
    assert errors(unknown)[0].param == "item.call_id"
    assert generation(h) == 0

    # A client can inject a ``function_call`` item into history itself. History
    # is not admission authority: only a call this session exposed is
    # answerable, so the imported call id is refused exactly like an invented
    # one even though the projection holds a matching function_call item.
    imported = normalize_conversation_item(
        {"type": "function_call", "call_id": "call_imported", "name": "lookup", "arguments": "{}"}
    )
    await h.run(commands.CreateItem(item=imported))
    refused = await answer(h, "call_imported", "42")
    assert error_codes(refused) == ["invalid_function_call_output"]
    assert errors(refused)[0].param == "item.call_id"
    assert h.runner.tool_calls.record("call_imported") is None
    assert generation(h) == 0


@pytest.mark.asyncio
async def test_a_plugin_refusal_keeps_the_call_answerable_and_the_retry_is_admitted(harness):
    h = harness
    call_id = await prepare(h)
    plugin = h.runner.plugin
    real = plugin.runtime_config_for_function_output
    attempts = {"n": 0}

    def flaky(config, current, item, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise plugin_module.NemotronVoiceChatClientRuntimeConfigError(
                "the tool-result channel is busy", code="invalid_function_call_output"
            )
        return real(config, current, item, **kwargs)

    plugin.runtime_config_for_function_output = flaky
    first = await answer(h, call_id)
    assert error_codes(first) == ["invalid_function_call_output"]
    assert "busy" in errors(first)[0].message
    # Nothing was admitted, and the call is still pending rather than burned.
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
    assert generation(h) == 0

    retry = await answer(h, call_id)
    assert error_codes(retry) == []
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.ACCEPTED
    assert generation(h) == 1


@pytest.mark.asyncio
async def test_an_effective_cancellation_refuses_a_late_result(harness):
    h = harness
    call_id = await prepare(h)
    epoch_before = h.session.epoch

    cancelled = await h.run(commands.CancelInput())
    # Cancelling is the client's own action: the session reports no error for
    # the calls it retires, only refuses the results that follow.
    assert error_codes(cancelled) == []
    assert h.session.epoch == epoch_before + 1
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.CANCELLED

    late = await answer(h, call_id)
    assert error_codes(late) == ["invalid_function_call_output"]
    assert errors(late)[0].param == "item.output"
    assert "cancelled" in errors(late)[0].message
    assert generation(h) == 0 and batches(h) == []


@pytest.mark.asyncio
async def test_an_unsupported_barge_in_is_a_noop_and_the_result_stays_admissible(harness):
    h = harness
    call_id = await prepare(h)
    epoch_before = h.session.epoch

    barge_in = await h.run(commands.BargeIn())
    assert error_codes(barge_in) == ["barge_in_unsupported"]
    # Nemotron does not support barge-in, so nothing was cancelled: the epoch
    # did not move and the pending call is still answerable.
    assert h.session.epoch == epoch_before
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING

    admitted = await answer(h, call_id)
    assert error_codes(admitted) == []
    assert generation(h) == 1


@pytest.mark.asyncio
async def test_a_wrong_response_cancel_leaves_the_call_pending(harness):
    h = harness
    call_id = await prepare(h)
    epoch_before = h.session.epoch

    await h.run(commands.CancelResponse(response_id="resp_not_this_session"))
    assert h.session.epoch == epoch_before
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING

    admitted = await answer(h, call_id)
    assert error_codes(admitted) == []
    assert generation(h) == 1


@pytest.mark.asyncio
async def test_cancellation_retires_a_result_the_model_never_consumed(harness):
    h = harness
    call_id = await prepare(h)
    await answer(h, call_id)
    assert generation(h) == 1 and len(batches(h)) == 1

    await h.run(commands.CancelInput())

    # The accepted result was queued for a frame the cancelled epoch never ran,
    # so it must not be speakable now. The monotonic cursor is *not* rewound:
    # the new epoch's request starts a fresh worker session (the stage request
    # id is per fence, and a cancel is a new fence), and the worker passes the
    # retired generation once, queueing nothing, because there is no batch left
    # to copy.
    assert batches(h) == []
    assert generation(h) == 1
    # The dropped generation is named, so the worker can tell "retired" from
    # "the batch went missing" -- the two look identical otherwise.
    assert h.session.runtime_config["nvc_function_response_floor_generation"] == 1
    assert h.runner.tool_calls.record(call_id).state is ToolCallState.ACCEPTED
    worker_state = {"function_response_generation": 0, "forced_function_tokens": []}
    Thinker._sync_forced_function_response(worker_state, h.session.runtime_config)
    assert worker_state["forced_function_tokens"] == []
    assert worker_state["function_response_generation"] == 1

    # A late second copy is still refused; the tombstone survived the epoch.
    late = await answer(h, call_id)
    assert error_codes(late) == ["invalid_function_call_output"]
    assert generation(h) == 1 and batches(h) == []

    # The *next* epoch's result is still contiguous for that same worker
    # session: this is why the retirement keeps the generation, so the cursor
    # the worker passed cannot make a later result look like a missing one.
    next_call = await prepare(h)
    admitted = await answer(h, next_call)
    assert error_codes(admitted) == []
    assert generation(h) == 2
    assert h.session.runtime_config["nvc_function_response_floor_generation"] == 1
    Thinker._sync_forced_function_response(worker_state, h.session.runtime_config)
    assert worker_state["forced_function_tokens"] == [42]
    assert worker_state["function_response_generation"] == 2


@pytest.mark.asyncio
async def test_a_failed_retirement_ends_the_session_instead_of_leaking_the_batch(harness):
    h = harness
    call_id = await prepare(h)
    await answer(h, call_id)
    assert len(batches(h)) == 1

    def explode(config, current):
        del config, current
        raise RuntimeError("the plugin cannot prove its queue is clean")

    h.runner.plugin.retire_pending_function_output = explode
    events = await h.run(commands.CancelInput())
    await h.settle()

    assert "tool_result_retirement_failed" in error_codes(events) or "tool_result_retirement_failed" in error_codes(
        h.events
    )
    assert "audio.cancelled" not in types(events)
    assert "input.cancelled" not in types(events)
    # Fail closed: the session is terminating (synchronously, at the moment the
    # retirement failed) and its ledger is already gone.
    assert h.runner.closing is True
    assert h.session.state is not DuplexSessionState.OPEN
    assert len(h.runner.tool_calls) == 0
    # The batch was never retired -- nothing refreshed the runtime on the
    # plugin's word -- but it is now unreachable: the session is gone from the
    # manager, so a later result is refused before it can reach the ledger, and
    # the runtime config still holds the same generation and batch it held
    # before the cancellation.
    assert error_codes(await answer(h, call_id)) == ["unknown_session"]
    assert generation(h) == 1 and len(batches(h)) == 1


def _worker() -> dict[str, object]:
    return {"function_response_generation": 0, "forced_function_tokens": []}


def test_the_floor_steps_the_cursor_over_generations_the_runtime_no_longer_holds():
    """What the floor buys: a cursor that crosses it, and nothing else.

    The worker cannot tell a consumed or retired batch from a lost one by
    absence alone, so the runtime records which generations left its queue.
    These are the sides of that contract, driven through the real worker
    function.
    """
    floor_then_applied = _worker()
    Thinker._sync_forced_function_response(
        floor_then_applied,
        {
            "nvc_function_response_generation": 2,
            "nvc_function_response_batches": [{"generation": 2, "token_ids": [42]}],
            "nvc_function_response_floor_generation": 1,
        },
    )
    assert floor_then_applied["forced_function_tokens"] == [42]
    assert floor_then_applied["function_response_generation"] == 2

    # A gap the floor does not cover is a lost batch, not a consumption.
    with pytest.raises(ValueError, match="not contiguous"):
        Thinker._sync_forced_function_response(
            _worker(),
            {
                "nvc_function_response_generation": 3,
                "nvc_function_response_batches": [{"generation": 3, "token_ids": [42]}],
                "nvc_function_response_floor_generation": 1,
            },
        )

    # The newest generation is never excused by an older floor: the floor is a
    # checkpoint of what was taken *below* the cursor, not a licence to skip.
    with pytest.raises(ValueError, match="incomplete"):
        Thinker._sync_forced_function_response(
            _worker(),
            {
                "nvc_function_response_generation": 2,
                "nvc_function_response_batches": [],
                "nvc_function_response_floor_generation": 1,
            },
        )

    # A floor above the newest generation is nonsense, and is refused rather
    # than trusted to excuse arbitrary gaps.
    with pytest.raises(ValueError, match="out of range"):
        Thinker._sync_forced_function_response(
            _worker(),
            {"nvc_function_response_generation": 1, "nvc_function_response_floor_generation": 9},
        )


def test_a_malformed_sequence_leaves_the_worker_cursor_and_queue_untouched():
    """The whole snapshot is validated before the first token is queued.

    A batch that cannot be spoken poisons the run it belongs to, so the worker
    must not have applied the *earlier* generations of the same snapshot by the
    time it notices. The cursor and the queue are therefore exactly what they
    were before the call.
    """
    partial = {"function_response_generation": 1, "forced_function_tokens": [7]}
    with pytest.raises(ValueError, match="requires token ids"):
        Thinker._sync_forced_function_response(
            partial,
            {
                "nvc_function_response_generation": 3,
                # Generation 2 is complete and generation 3 is not: the snapshot
                # as a whole is malformed, so generation 2 is not applied either.
                "nvc_function_response_batches": [
                    {"generation": 2, "token_ids": [42]},
                    {"generation": 3, "token_ids": []},
                ],
            },
        )
    assert partial == {"function_response_generation": 1, "forced_function_tokens": [7]}

    # A duplicate generation would speak one result twice, not replay it.
    duplicate = {"function_response_generation": 1, "forced_function_tokens": [7]}
    with pytest.raises(ValueError, match="not contiguous"):
        Thinker._sync_forced_function_response(
            duplicate,
            {
                "nvc_function_response_generation": 3,
                "nvc_function_response_batches": [
                    {"generation": 2, "token_ids": [42]},
                    {"generation": 2, "token_ids": [43]},
                ],
            },
        )
    assert duplicate == {"function_response_generation": 1, "forced_function_tokens": [7]}

    # A batch object that is not an object at all is still a malformed snapshot.
    malformed = {"function_response_generation": 0, "forced_function_tokens": []}
    with pytest.raises(ValueError, match="must be an object"):
        Thinker._sync_forced_function_response(
            malformed,
            {"nvc_function_response_generation": 1, "nvc_function_response_batches": ["nope"]},
        )
    assert malformed == {"function_response_generation": 0, "forced_function_tokens": []}


def test_a_malformed_floor_is_refused_instead_of_read_as_no_floor():
    state = _worker()
    with pytest.raises(ValueError, match="floor must be an integer"):
        Thinker._sync_forced_function_response(
            state,
            {"nvc_function_response_generation": 2, "nvc_function_response_floor_generation": "one"},
        )
    assert state == _worker()


def test_the_runtime_engine_refuses_a_malformed_floor_instead_of_reopening_generations():
    """A floor the runtime itself cannot read must not degrade to zero.

    Reading a malformed floor as "no floor" would make a generation the runtime
    already dropped indistinguishable from a lost batch, which is exactly the
    ambiguity the floor exists to remove.
    """
    plugin = load_duplex_plugin(NEMOTRON_VOICECHAT_PIPELINE.duplex_plugin, _fake_encode_audio)
    with pytest.raises(plugin_module.NemotronVoiceChatClientRuntimeConfigError, match="nonnegative integer"):
        plugin.retire_pending_function_output(
            None,
            {
                "nvc_function_response_generation": 1,
                "nvc_function_response_batches": [],
                "nvc_function_response_floor_generation": "one",
            },
        )


def test_retirement_names_the_generation_even_when_no_batch_is_left_to_drop():
    """An acknowledged result must survive the epoch change by name.

    The worker session of the cancelled epoch is gone, so nothing else on the
    request can tell the next epoch's worker that generation 1 was already
    taken: the floor is the only carrier. Writing it is a real change -- the
    no-op return is reserved for the runtime that already holds that state.
    """
    plugin = load_duplex_plugin(NEMOTRON_VOICECHAT_PIPELINE.duplex_plugin, _fake_encode_audio)
    retired = plugin.retire_pending_function_output(
        None,
        {
            "nvc_function_response_generation": 1,
            "nvc_function_response_batches": [],
            "nvc_function_response_floor_generation": 0,
        },
    )
    assert retired is not None
    assert retired["nvc_function_response_floor_generation"] == 1
    assert retired["nvc_function_response_batches"] == []
    assert retired["nvc_function_response_generation"] == 1

    # Already represented: the runtime holds this state, so nothing is written.
    assert (
        plugin.retire_pending_function_output(
            None,
            {
                "nvc_function_response_generation": 1,
                "nvc_function_response_batches": [],
                "nvc_function_response_floor_generation": 1,
            },
        )
        is None
    )


def test_a_client_cannot_inject_the_runtime_owned_function_response_floor():
    """The floor is the server's checkpoint, not a client knob.

    A client that could raise its own floor could excuse a batch the runtime
    never took. The key is server-owned, so the update path refuses the body
    that names it before it can reach a live session.
    """
    plugin = load_duplex_plugin(NEMOTRON_VOICECHAT_PIPELINE.duplex_plugin, _fake_encode_audio)
    assert "nvc_function_response_floor_generation" in plugin.private_runtime_config_keys
    config = DuplexSessionConfig(
        model="nemotron",
        instructions="hi",
        extra_body={"auto_response": True, "nvc_function_response_floor_generation": 9},
    )
    with pytest.raises(
        plugin_module.NemotronVoiceChatClientRuntimeConfigError, match="nvc_function_response_floor_generation"
    ):
        plugin.runtime_config_for_update(config, {"nvc_function_response_generation": 1})


@pytest.mark.asyncio
async def test_a_cancel_does_not_poison_the_next_epoch(harness):
    h = harness
    await h.run(commands.CancelInput())
    epoch_after_cancel = h.session.epoch

    call_id = await prepare(h)
    assert h.runner.tool_calls.record(call_id).origin.epoch == epoch_after_cancel

    events = await answer(h, call_id)
    assert error_codes(events) == []
    assert generation(h) == 1
    assert batches(h)[-1] == {"generation": 1, "token_ids": [42]}


# --------------------------------------------------------------------------- #
# Deadline                                                                    #
# --------------------------------------------------------------------------- #


TIMEOUT_S = 5.0


@asynccontextmanager
async def open_timed_harness(monkeypatch, *, timeout_s: float = TIMEOUT_S):
    clock = FakeClock(100.0)
    scheduler = FakeTimerScheduler()
    async with open_nemotron_harness(
        monkeypatch,
        clock=clock,
        scheduler=scheduler,
        extra_body={"tool_result_timeout_s": timeout_s},
    ) as h:
        assert h.clock is clock and h.scheduler is scheduler
        yield h


@pytest.mark.asyncio
async def test_the_expiry_timer_settles_through_the_mailbox_and_refuses_a_late_result(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        assert h.scheduler.pending_delays() == [TIMEOUT_S]

        # Reaching the deadline does not settle anything by itself: the timer
        # only hands the call id to the mailbox, which decides.
        h.clock.advance(TIMEOUT_S)
        h.scheduler.fire_all()
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING

        events = await h.settle()
        assert error_codes(events) == ["tool_result_timeout"]
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT
        assert h.scheduler.pending_delays() == []

        late = await answer(h, call_id)
        assert error_codes(late) == ["invalid_function_call_output"]
        error = errors(late)[0]
        assert error.param == "item.call_id"
        assert "tool_result_timeout_s" in error.message
        assert generation(h) == 0 and batches(h) == []

        # The channel is not poisoned: the next call in the same epoch works.
        next_call_id, _ = await expose_function_call(h)
        assert next_call_id != call_id
        admitted = await answer(h, next_call_id)
        assert error_codes(admitted) == []
        assert generation(h) == 1
        assert batches(h)[-1] == {"generation": 1, "token_ids": [42]}


@pytest.mark.asyncio
async def test_a_result_that_arrives_at_the_deadline_is_refused_without_the_timer(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)

        # Strictly before the deadline is accepted, at the deadline is not.
        # The timer has not fired. Settling the late result still emits exactly
        # one tool_result_timeout, and the later timer callback does not repeat it.
        h.clock.advance(TIMEOUT_S)
        refused = await answer(h, call_id)
        assert error_codes(refused) == ["invalid_function_call_output", "tool_result_timeout"]
        assert errors(refused)[0].param == "item.call_id"
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT
        assert "tool_result_timeout_s" in errors(refused)[0].message
        assert h.scheduler.pending_delays() == []
        h.scheduler.fire_all()
        assert error_codes(await h.settle()) == []
        assert generation(h) == 0


@pytest.mark.asyncio
async def test_a_result_that_misses_its_deadline_while_the_append_tail_waits_is_refused(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)

        # The append tail is awaited between the eligibility check and the
        # commit. A tail that takes longer than the remaining budget is exactly
        # the ordering this test needs, and the fake clock makes it a fact
        # rather than a race.
        async def slow_tail() -> bool:
            h.clock.advance(TIMEOUT_S)
            return True

        h.runner.control._wait_for_append_tail = slow_tail  # type: ignore[method-assign]
        refused = await answer(h, call_id)

        assert error_codes(refused) == ["invalid_function_call_output", "tool_result_timeout"]
        assert errors(refused)[0].param == "item.call_id"
        assert "tool_result_timeout_s" in errors(refused)[0].message
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT
        assert generation(h) == 0 and batches(h) == []


@pytest.mark.asyncio
async def test_a_result_that_misses_its_deadline_while_the_plugin_renders_it_is_refused(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        tokenizer = h.tokenizer
        assert tokenizer is not None
        real_encode = tokenizer.encode

        # Building the candidate is synchronous but not free: tokenizing a tool
        # result is work, and the fake clock crosses the deadline inside it. The
        # candidate must not be installed and no continuation may be spawned.
        def encode_and_cross_the_deadline(text, **kwargs):
            tokens = real_encode(text, **kwargs)
            h.clock.advance(TIMEOUT_S)
            return tokens

        tokenizer.encode = encode_and_cross_the_deadline
        submissions_before = len(h.port.submissions)
        refused = await answer(h, call_id)

        assert error_codes(refused) == ["invalid_function_call_output", "tool_result_timeout"]
        assert errors(refused)[0].param == "item.call_id"
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT
        # Nothing was committed: no item for the client, no queued batch, and no
        # continuation into the next stage frame.
        assert "conversation.item.created" not in types(refused)
        assert generation(h) == 0 and batches(h) == []
        assert len(h.port.submissions) == submissions_before


@pytest.mark.asyncio
async def test_the_deadline_is_armed_on_the_sessions_real_loop_timer(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"tool_result_timeout_s": 0.02}) as h:
        call_id, events = await expose_function_call(h)
        events.extend(await h.settle())

        # No clock or scheduler seam was injected here: this is the session's own
        # ``loop.call_later`` firing while the loop is idle.
        assert error_codes(events) == ["tool_result_timeout"]
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT


@pytest.mark.asyncio
async def test_close_disarms_the_deadline_timer(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        assert h.scheduler.pending_delays() == [TIMEOUT_S]

        await h.manager.handle(CloseDuplexSessionMessage(control_id="close", session_id=SESSION_ID))
        await h.settle()

        assert h.scheduler.pending_delays() == []
        assert len(h.runner.tool_calls) == 0
        # A callback that still fired (the scheduler remembers the callback even
        # for a cancelled timer) settles nothing and emits nothing.
        h.scheduler.fire_all()
        events = await h.settle()
        assert error_codes(events) == []
        assert call_id not in h.runner.tool_calls


@pytest.mark.asyncio
async def test_an_accepted_result_reaches_the_next_model_turn_after_the_call_became_stale(harness):
    """Admission is about eligibility; the continuation follows the live turn.

    The call is emitted in turn T. The model then completes that turn while the
    client is answering -- ``turn_id`` moves only through the session's own
    ``complete_model_turn`` (which re-syncs the fence), and the Nemotron data
    plane never emits a ``model_turn_id``, so this harness cannot produce that
    completion from a stage output. The ledger must still admit the result (its
    eligibility is the origin epoch plus settlement, not the turn), and the
    accepted result must reach the *next* worker frame instead of being dropped
    because the origin turn is no longer current.
    """
    h = harness
    call_id = await prepare(h)
    origin_turn = h.runner.tool_calls.record(call_id).origin.turn_id

    h.session.complete_model_turn(origin_turn)
    assert h.session.turn_id == origin_turn + 1
    submissions_before = len(h.port.submissions)

    events = await answer(h, call_id)
    assert error_codes(events) == []

    record = h.runner.tool_calls.record(call_id)
    assert record.state is ToolCallState.ACCEPTED
    assert record.origin.turn_id == origin_turn  # identity is preserved
    assert generation(h) == 1 and len(batches(h)) == 1

    # The continuation ran against the current turn: the runner submitted the
    # next stage frame for it, on the new turn's request.
    assert len(h.port.submissions) == submissions_before + 1
    assert h.port.submissions[-1].context.fence.turn_id == origin_turn + 1

    # ...and the worker picks the queued result up on that frame.
    worker_state = {"function_response_generation": 0, "forced_function_tokens": []}
    Thinker._sync_forced_function_response(worker_state, h.session.runtime_config)
    assert worker_state["forced_function_tokens"] == [42]
    assert worker_state["function_response_generation"] == 1
