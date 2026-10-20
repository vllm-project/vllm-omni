# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Entrypoint contract tests for ``AsyncOmni.submit_interaction_async``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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
