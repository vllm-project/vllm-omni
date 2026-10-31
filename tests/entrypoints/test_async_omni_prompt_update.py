# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Entrypoint contract tests for ``AsyncOmni.submit_interaction_async``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints import async_omni_base
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.inputs.data import OmniInteractionPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_async_omni(*, num_stages: int = 1, stage_type: str = "diffusion") -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni.log_stats = False
    omni.request_states = {
        "external-abc-uuid-1": ClientRequestState(
            request_id="external-abc-uuid-1",
            external_request_id="external-abc",
            queue=AsyncMock(),
        ),
        "external-abc-uuid-2": ClientRequestState(
            request_id="external-abc-uuid-2",
            external_request_id="external-abc",
            queue=AsyncMock(),
        ),
        "other-req-uuid": ClientRequestState(
            request_id="other-req-uuid",
            external_request_id="other-req",
            queue=AsyncMock(),
        ),
    }
    omni.engine = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        num_stages=num_stages,
        get_stage_metadata=lambda stage_id: SimpleNamespace(stage_type=stage_type),
        submit_interaction_async=AsyncMock(),
    )
    return omni


@pytest.mark.asyncio
async def test_playback_feedback_maps_session_id_and_ignores_finished_session(mocker) -> None:
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    rpc = mocker.patch.object(omni, "_engine_core_rpc", new=mocker.AsyncMock())
    await omni.update_streaming_playback("external-abc", 0.5)
    rpc.assert_awaited_once_with("update_streaming_playback", stage_ids=[0], args=("external-abc-uuid-1", 0.5))
    await omni.update_streaming_playback("finished", 9.0)
    assert rpc.await_count == 1


def _prompt_interaction(prompt: str = "new prompt", transition_chunks: int | None = None) -> OmniInteractionPrompt:
    interaction: OmniInteractionPrompt = {"event_id": "ui-update-1", "event": {"prompt": prompt}}
    if transition_chunks is not None:
        interaction["transition_chunks"] = transition_chunks
    return interaction


@pytest.mark.asyncio
async def test_paced_interaction_uses_engine_stamp_and_releases_failed_reservation(mocker):
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    rpc = mocker.patch.object(omni, "_engine_core_rpc", new=mocker.AsyncMock(return_value=[12.0]))
    interaction = {**_prompt_interaction(), "received_at": 0.0}
    assert await omni.submit_interaction_async("external-abc", interaction=interaction, track_playback=True) == 12.0
    payload = omni.engine.submit_interaction_async.call_args.kwargs["interaction"]
    assert payload["received_at"] == 12.0
    assert interaction["received_at"] == 0.0
    omni.engine.submit_interaction_async.side_effect = ValueError("failed")
    with pytest.raises(ValueError, match="failed"):
        await omni.submit_interaction_async("external-abc", interaction=interaction, track_playback=True)
    assert rpc.call_args.args == ("track_streaming_interaction",)
    assert rpc.call_args.kwargs["args"] == ("external-abc-uuid-1", "ui-update-1", False)


@pytest.mark.asyncio
async def test_cancel_during_ingress_reservation_releases_it(mocker):
    import asyncio

    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    entered = asyncio.Event()

    async def rpc(method, *, stage_ids, args):
        if len(args) == 2:
            entered.set()
            await asyncio.Event().wait()

    reserve = mocker.patch.object(omni, "_engine_core_rpc", new=mocker.AsyncMock(side_effect=rpc))
    task = asyncio.create_task(
        omni.submit_interaction_async("external-abc", interaction=_prompt_interaction(), track_playback=True)
    )
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reserve.call_args.kwargs["args"] == ("external-abc-uuid-1", "ui-update-1", False)
    omni.engine.submit_interaction_async.assert_not_called()


@pytest.mark.asyncio
async def test_submit_interaction_async_maps_external_to_internal_id() -> None:
    omni = _make_async_omni()
    with pytest.raises(ValueError, match="exactly one active request"):
        await omni.submit_interaction_async("external-abc", interaction=_prompt_interaction(transition_chunks=2))

    omni.request_states.pop("external-abc-uuid-2")
    await omni.submit_interaction_async(
        "external-abc",
        interaction=_prompt_interaction(transition_chunks=2),
    )
    omni.engine.submit_interaction_async.assert_awaited_once_with(  # pyright: ignore[reportAttributeAccessIssue]
        "external-abc-uuid-1",
        interaction=_prompt_interaction(transition_chunks=2),
    )


@pytest.mark.asyncio
async def test_submit_interaction_async_passes_missing_transition_chunks() -> None:
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    await omni.submit_interaction_async("external-abc", interaction=_prompt_interaction())
    omni.engine.submit_interaction_async.assert_awaited_once_with(  # pyright: ignore[reportAttributeAccessIssue]
        "external-abc-uuid-1",
        interaction=_prompt_interaction(),
    )


@pytest.mark.asyncio
async def test_submit_interaction_async_ignores_direct_received_at() -> None:
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    interaction = {**_prompt_interaction(), "received_at": 0.0}

    await omni.submit_interaction_async("external-abc", interaction=interaction)

    forwarded = omni.engine.submit_interaction_async.call_args.kwargs["interaction"]
    assert "received_at" not in forwarded


@pytest.mark.asyncio
async def test_submit_interaction_async_rejects_empty_prompt() -> None:
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        await omni.submit_interaction_async("external-abc", interaction=_prompt_interaction(""))


@pytest.mark.asyncio
async def test_submit_interaction_async_rejects_negative_transition_chunks() -> None:
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    with pytest.raises(ValueError, match="transition_chunks must be >= 0"):
        await omni.submit_interaction_async(
            "external-abc",
            interaction=_prompt_interaction(transition_chunks=-1),
        )


@pytest.mark.asyncio
async def test_submit_interaction_async_rejects_inactive_request() -> None:
    omni = _make_async_omni()
    omni.request_states.clear()
    with pytest.raises(ValueError, match="No active request"):
        await omni.submit_interaction_async("missing", interaction=_prompt_interaction(transition_chunks=2))


@pytest.mark.asyncio
async def test_submit_interaction_async_rejects_non_diffusion() -> None:
    omni = _make_async_omni(stage_type="llm")
    omni.request_states = {
        "req-uuid": ClientRequestState(
            request_id="req-uuid",
            external_request_id="req",
            queue=AsyncMock(),
        ),
    }
    with pytest.raises(ValueError, match="requires a diffusion stage"):
        await omni.submit_interaction_async("req", interaction=_prompt_interaction(transition_chunks=2))


@pytest.mark.asyncio
async def test_submit_interaction_async_forwards_unsupported_dict_shape_to_engine() -> None:
    omni = _make_async_omni()
    omni.request_states.pop("external-abc-uuid-2")
    payload: OmniInteractionPrompt = {  # pyright: ignore[reportAssignmentType]  # deliberately wrong
        "event": {"multi_modal_data": {"unsupported_modality": {"type": "pose"}}},
        "transition_chunks": 2,
    }

    await omni.submit_interaction_async(
        "external-abc",
        interaction=payload,
    )

    omni.engine.submit_interaction_async.assert_awaited_once_with(  # pyright: ignore[reportAttributeAccessIssue]
        "external-abc-uuid-1",
        interaction=payload,
    )


def _make_rejection_omni(mocker, messages: list) -> tuple[AsyncOmni, ClientRequestState, list[str]]:
    """AsyncOmni whose output loop reads ``messages`` once and records RPC/callback order."""
    omni = object.__new__(AsyncOmni)
    omni.final_output_task = None
    req_state = ClientRequestState(request_id="req-uuid", external_request_id="req")
    omni.request_states = {"req-uuid": req_state}
    order: list[str] = []
    pending = [messages]

    async def get_outputs_async(timeout: float):
        if pending:
            return pending.pop()
        await asyncio.sleep(0.01)
        return []

    async def engine_core_rpc(method, *, stage_ids, args):
        order.append(f"rpc:{method}:{args}")
        return [None]

    omni.engine = SimpleNamespace(get_outputs_async=get_outputs_async)  # pyright: ignore[reportAttributeAccessIssue]
    mocker.patch.object(omni, "_engine_core_rpc", side_effect=engine_core_rpc)
    return omni, req_state, order


@pytest.mark.asyncio
async def test_event_scoped_rejection_keeps_stream_and_releases_before_notifying(mocker) -> None:
    rejected = ErrorMessage(error="Failed interaction: bad pose", request_id="req-uuid", event_id="ev-1", stage_id=0)
    overloaded = ErrorMessage(error="Too many pending interactions", request_id="req-uuid", stage_id=0)
    omni, req_state, order = _make_rejection_omni(mocker, [rejected, overloaded])
    notified = asyncio.Event()

    async def on_error(event_id: str, message: str) -> None:
        order.append(f"notify:{event_id}:{message}")
        notified.set()

    req_state.interaction_error_handler = on_error
    omni._final_output_handler()
    try:
        # The request-scoped overload error still reaches generate() and ends it.
        assert await asyncio.wait_for(req_state.queue.get(), 1) is overloaded
        await asyncio.wait_for(notified.wait(), 1)
        assert req_state.queue.empty()
        assert order == [
            "rpc:track_streaming_interaction:('req-uuid', 'ev-1', False)",
            "notify:ev-1:Failed interaction: bad pose",
        ]
    finally:
        assert omni.final_output_task is not None
        omni.final_output_task.cancel()
        await asyncio.gather(omni.final_output_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_submit_interaction_async_registers_rejection_callback() -> None:
    omni = _make_async_omni()

    async def on_error(event_id: str, message: str) -> None:
        del event_id, message

    await omni.submit_interaction_async("other-req", interaction=_prompt_interaction(), on_error=on_error)

    assert omni.request_states["other-req-uuid"].interaction_error_handler is on_error


@pytest.mark.asyncio
async def test_pending_rejection_callbacks_are_drained_then_cancelled(monkeypatch) -> None:
    monkeypatch.setattr(async_omni_base, "INTERACTION_ERROR_DRAIN_TIMEOUT_S", 0.05)
    omni = object.__new__(AsyncOmni)
    req_state = ClientRequestState(request_id="req-uuid")
    stuck_cancelled = asyncio.Event()

    async def quick() -> None:
        await asyncio.sleep(0)

    async def stuck() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stuck_cancelled.set()

    quick_task = asyncio.ensure_future(quick())
    stuck_task = asyncio.ensure_future(stuck())
    req_state.interaction_error_tasks.update({quick_task, stuck_task})

    await asyncio.wait_for(omni._drain_interaction_error_tasks(req_state), 1)

    assert quick_task.done() and not quick_task.cancelled()
    await asyncio.wait_for(stuck_cancelled.wait(), 1)
    assert stuck_task.cancelled()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_rejection_callbacks(mocker) -> None:
    omni = object.__new__(AsyncOmni)
    omni.final_output_task = None
    req_state = ClientRequestState(request_id="req-uuid")
    omni.request_states = {"req-uuid": req_state}
    task = asyncio.ensure_future(asyncio.Event().wait())
    req_state.interaction_error_tasks.add(task)
    mocker.patch("vllm_omni.entrypoints.async_omni_base.OmniBase.shutdown")

    omni.shutdown()
    await asyncio.gather(task, return_exceptions=True)

    assert task.cancelled()
