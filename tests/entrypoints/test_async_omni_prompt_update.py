# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


def _prompt_interaction(prompt: str = "new prompt", transition_chunks: int | None = None) -> OmniInteractionPrompt:
    interaction: OmniInteractionPrompt = {"event_id": "ui-update-1", "event": {"prompt": prompt}}
    if transition_chunks is not None:
        interaction["transition_chunks"] = transition_chunks
    return interaction


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
