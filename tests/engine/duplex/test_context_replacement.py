# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import base64

import pytest

from tests.engine.duplex.test_duplex_control_plane import _native_session, _NativeStagePort
from vllm_omni.engine.duplex.control_plane import DuplexControlPlane
from vllm_omni.engine.duplex.messages import AppendDuplexInputMessage, DuplexFence, SignalDuplexTurnMessage
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.runtime import MiniCPMO45DuplexRuntimeExtension
from vllm_omni.model_executor.models.minicpmo_4_5.gander_context import metadata, unit_id

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
AUDIO = base64.b64encode(bytes(64000)).decode()


async def setup_history(*, maximum=128, retain=96):
    sink: asyncio.Queue = asyncio.Queue()
    port = _NativeStagePort()
    plane = DuplexControlPlane(
        extension=MiniCPMO45DuplexRuntimeExtension(), stage_port=port, result_sink=sink, rollover_trigger_fraction=0
    )
    old = DuplexFence("context-test")
    session = _native_session(plane, old)
    session.replace_runtime_config(
        {
            "gander_enabled": True,
            "duplex_context_version": 0,
            "gander_history": {"max_units": maximum, "retain_units": retain},
        }
    )
    for seq in range(1, 4):
        await plane.handle_append(
            AppendDuplexInputMessage(
                control_id=f"a{seq}",
                operation_id=f"a{seq}",
                session_id=old.session_id,
                fence=old,
                mode="append_audio_chunk",
                payload={"audio": AUDIO, "format": "pcm_f32le"},
            )
        )
        assert (await sink.get()).ok
    return plane, port, sink, session, old


def replacement(old, *, context=None, runtime=None, event="context.replace"):
    return SignalDuplexTurnMessage(
        control_id="replace",
        session_id=old.session_id,
        fence=old,
        next_fence=DuplexFence(old.session_id, epoch=old.epoch + 1),
        event=event,
        runtime_config=runtime
        or {
            "gander_enabled": True,
            "duplex_context_version": 1,
            "gander_context_version": 1,
            "duplex_first_append_context_tokens": 7,
        },
        context=context or {"base_version": 0, "edits": []},
    )


@pytest.mark.asyncio
async def test_replacement_retires_old_kv_and_replays_new_prefix_before_receipt():
    plane, port, sink, session, old = await setup_history()
    prior_id = plane.stage_request_id(old, stage_id=0)
    await plane.handle_signal(replacement(old))
    receipt = await sink.get()
    assert receipt.ok, receipt
    result = receipt.stage_results[0]["result"]
    assert session.epoch == 1 and session.resource_generation == 1
    assert port.cleanup_calls == [([prior_id], True)]
    replay = [s for s in port.submit_calls if s.recovery_replay]
    assert len(replay) == 3
    assert len(replay[0].prompt["prompt_token_ids"]) == 18  # prefix 7 + first audio unit 11
    assert all(metadata(s.prompt)["fence"].epoch == 1 for s in replay)
    assert port.wait_replay_safe_calls[-1] == result["request_id"]
    assert session.input_seq == 3
    assert result["retained_unit_ids"] == ["u0-1", "u0-2", "u0-3"]
    await plane.handle_append(
        AppendDuplexInputMessage(
            control_id="late",
            operation_id="late",
            fence=old,
            session_id=old.session_id,
            mode="append_audio_chunk",
            payload={"audio": AUDIO},
        )
    )
    assert not (await sink.get()).ok


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    [
        {"base_version": 1, "edits": []},
        {"base_version": 0, "edits": [{"op": "delete", "unit_id": "missing"}]},
        {"base_version": 0, "edits": [{"op": "delete", "unit_id": "prefix-seed"}]},
    ],
)
async def test_invalid_replacement_does_not_abort_or_mutate_old_context(context):
    plane, port, sink, session, old = await setup_history()
    journal = tuple(session.replay_appends)
    await plane.handle_signal(replacement(old, context=context))
    assert not (await sink.get()).ok
    assert not port.cleanup_calls and session.fence == old
    assert tuple(session.replay_appends) == journal


@pytest.mark.asyncio
async def test_failed_rebuild_never_leaves_partially_ready_session(monkeypatch):
    plane, port, sink, session, old = await setup_history()

    async def fail(submission):
        raise RuntimeError("injected replay failure")

    monkeypatch.setattr(port, "submit", fail)
    await plane.handle_signal(replacement(old))
    assert not (await sink.get()).ok
    assert session.lease.terminal_reason == "context_replacement_failed"
    assert not session.resource_request_ids(submitted=True)


@pytest.mark.asyncio
async def test_model_window_rollover_keeps_pinned_unit_and_recent_suffix():
    plane, port, sink, session, old = await setup_history(maximum=4, retain=2)
    # Pin the oldest unit using an actual context edit.
    await plane.handle_signal(
        replacement(
            old,
            context={"base_version": 0, "edits": [{"op": "pin", "unit_id": "u0-1"}]},
            runtime={
                "gander_enabled": True,
                "duplex_context_version": 1,
                "gander_context_version": 1,
                "gander_history": {"max_units": 4, "retain_units": 2},
            },
        )
    )
    assert (await sink.get()).ok
    for seq in range(4, 8):
        await plane.handle_append(
            AppendDuplexInputMessage(
                control_id=f"b{seq}",
                operation_id=f"b{seq}",
                session_id=old.session_id,
                fence=session.fence,
                mode="append_audio_chunk",
                payload={"audio": AUDIO},
            )
        )
        assert (await sink.get()).ok
    assert len(session.replay_appends) <= 4
    assert any(unit_id(p.prompt) == "u0-1" for p in session.replay_appends)
    assert session.resource_generation >= 2


@pytest.mark.asyncio
async def test_fast_unit_output_before_append_receipt_is_journaled(monkeypatch):
    from vllm_omni.engine.duplex.contracts import DuplexContextOutput

    plane, port, sink, session, old = await setup_history()
    submit = port.submit

    async def fast_output(submission):
        plane._record_context_output(session, DuplexContextOutput(unit_sequence=4, data={"output_ids": [12, 104, 13]}))
        return await submit(submission)

    monkeypatch.setattr(port, "submit", fast_output)
    await plane.handle_append(
        AppendDuplexInputMessage(
            control_id="a4",
            operation_id="a4",
            session_id=old.session_id,
            fence=old,
            mode="append_audio_chunk",
            payload={"audio": AUDIO},
        )
    )
    assert (await sink.get()).ok
    assert metadata(session.replay_appends[-1].prompt)["gander_output_ids"] == [12, 104, 13]
    assert not session.pending_context_outputs


@pytest.mark.asyncio
async def test_window_compaction_waits_for_reply_delivery_but_hard_limits_remain():
    plane, port, sink, session, old = await setup_history(maximum=4, retain=2)
    for seq in range(4, 7):
        await plane.handle_append(
            AppendDuplexInputMessage(
                control_id=f"defer-{seq}",
                operation_id=f"defer-{seq}",
                session_id=old.session_id,
                fence=old,
                mode="append_audio_chunk",
                payload={"audio": AUDIO, "gander_defer_rollover": True},
            )
        )
        assert (await sink.get()).ok
    assert session.resource_generation == 0 and len(session.replay_appends) == 6
    await plane.handle_append(
        AppendDuplexInputMessage(
            control_id="delivered",
            operation_id="delivered",
            session_id=old.session_id,
            fence=old,
            mode="append_audio_chunk",
            payload={"audio": AUDIO},
        )
    )
    assert (await sink.get()).ok
    assert session.resource_generation == 1 and len(session.replay_appends) <= 4
    session.recovery_max_replay_tokens = session.replay_token_count
    await plane.handle_append(
        AppendDuplexInputMessage(
            control_id="exhaust",
            operation_id="exhaust",
            session_id=old.session_id,
            fence=old,
            mode="append_audio_chunk",
            payload={"audio": AUDIO, "gander_defer_rollover": True},
        )
    )
    assert not (await sink.get()).ok
    assert session.resource_generation == 1
