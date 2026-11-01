# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from vllm_omni.experimental.ar_diffusion.consumer import ARDiffusionOmniTickConsumer
from vllm_omni.experimental.ar_diffusion.session import (
    ARDiffusionSession,
    ARDiffusionSessionEvent,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniPromptType,
    OmniSamplingParams,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeStagePool:
    def __init__(self, num_replicas: int = 1) -> None:
        self.num_replicas = num_replicas


class FakeEngine:
    def __init__(self, num_replicas: int = 1) -> None:
        self.stage_pools = [FakeStagePool(num_replicas)]


class FakeGenerateClient:
    def __init__(
        self,
        *,
        include_metadata: bool = True,
        applied_event_ids: list[int] | None = None,
        output_error: str | None = None,
        num_replicas: int = 1,
    ) -> None:
        self.engine = FakeEngine(num_replicas)
        self.include_metadata = include_metadata
        self.applied_event_ids = applied_event_ids
        self.output_error = output_error
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        prompt: OmniPromptType,
        *,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams],
        output_modalities: list[str] | None = None,
    ) -> AsyncIterator[OmniRequestOutput]:
        self.calls.append(
            {
                "prompt": prompt,
                "request_id": request_id,
                "sampling_params_list": sampling_params_list,
                "output_modalities": output_modalities,
            }
        )
        params = sampling_params_list[0]
        assert isinstance(params, OmniDiffusionSamplingParams)
        tick = ARDiffusionTickRequest.from_extra_args(params.extra_args)
        assert tick is not None
        chunk_metadata = ARDiffusionChunkMetadata.from_tick(tick).to_dict()
        if self.applied_event_ids is not None:
            chunk_metadata["applied_event_ids"] = self.applied_event_ids
        metadata = {"metadata": {"ar_diffusion": chunk_metadata}} if self.include_metadata else {}
        yield OmniRequestOutput(
            request_id=request_id,
            stage_id=0,
            final_output_type="video",
            images=["chunk"],  # type: ignore[list-item]
            _multimodal_output=metadata,
            finished=True,
            error=self.output_error,
        )


class FakeLifecycle:
    async def reset_session(self, session_id: str) -> None:
        pass

    async def close_session(self, session_id: str) -> None:
        pass


def make_tick() -> ARDiffusionTickRequest:
    return ARDiffusionTickRequest(
        session_id="world-1",
        request_id="world-1-chunk-3",
        chunk_index=3,
        applied_event_ids=(7, 8),
        prompt="turn left",
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="example.camera.v1",
                data={"samples": [{"x": 1.0}]},
            ),
        ),
    )


def test_consumer_rejects_replicated_ar_diffusion_stage() -> None:
    client = FakeGenerateClient(num_replicas=2)

    with pytest.raises(
        ValueError,
        match=r"require exactly one replica.*num_replicas=2",
    ):
        ARDiffusionOmniTickConsumer(
            client,
            prompt_provider=lambda tick: tick.prompt or "",
            sampling_params_list=[OmniDiffusionSamplingParams()],
            diffusion_stage_id=0,
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_consumer_submits_typed_tick_and_returns_standard_output() -> None:
    client = FakeGenerateClient()
    template = OmniDiffusionSamplingParams(
        output_type="latent",
        extra_args={"deployment_option": "kept"},
    )
    consumer = ARDiffusionOmniTickConsumer(
        client,
        prompt_provider=lambda tick: {
            "prompt": tick.prompt,
            "multi_modal_data": {"image": "initial-image"},
        },
        sampling_params_list=[template],
        diffusion_stage_id=0,
        output_modalities=["video"],
    )
    tick = make_tick()

    output = await consumer.execute_tick(tick)

    assert output.images == ["chunk"]
    assert consumer.chunk_metadata(output) == ARDiffusionChunkMetadata.from_tick(tick)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["request_id"] == tick.request_id
    assert call["output_modalities"] == ["video"]
    assert call["prompt"] == {
        "prompt": "turn left",
        "multi_modal_data": {"image": "initial-image"},
    }
    submitted_params = call["sampling_params_list"][0]
    assert submitted_params.extra_args["deployment_option"] == "kept"
    assert ARDiffusionTickRequest.from_extra_args(submitted_params.extra_args) == tick
    assert "ar_diffusion_tick" not in template.extra_args


@pytest.mark.asyncio
async def test_consumer_rejects_missing_chunk_metadata() -> None:
    consumer = ARDiffusionOmniTickConsumer(
        FakeGenerateClient(include_metadata=False),
        prompt_provider=lambda tick: tick.prompt or "",
        sampling_params_list=[OmniDiffusionSamplingParams()],
        diffusion_stage_id=0,
    )

    with pytest.raises(ValueError, match="multimodal_output.metadata"):
        await consumer.execute_tick(make_tick())


@pytest.mark.asyncio
async def test_consumer_rejects_non_monotonic_applied_event_metadata() -> None:
    consumer = ARDiffusionOmniTickConsumer(
        FakeGenerateClient(applied_event_ids=[8, 7]),
        prompt_provider=lambda tick: tick.prompt or "",
        sampling_params_list=[OmniDiffusionSamplingParams()],
        diffusion_stage_id=0,
    )

    with pytest.raises(ValueError, match="unique and strictly increasing"):
        await consumer.execute_tick(make_tick())


@pytest.mark.asyncio
async def test_consumer_preserves_engine_failure_before_metadata_validation() -> None:
    consumer = ARDiffusionOmniTickConsumer(
        FakeGenerateClient(include_metadata=False, output_error="model failed"),
        prompt_provider=lambda tick: tick.prompt or "",
        sampling_params_list=[OmniDiffusionSamplingParams()],
        diffusion_stage_id=0,
    )

    with pytest.raises(RuntimeError, match="model failed"):
        await consumer.execute_tick(make_tick())


@pytest.mark.asyncio
async def test_session_commits_only_after_standard_engine_output_metadata() -> None:
    client = FakeGenerateClient()
    consumer = ARDiffusionOmniTickConsumer(
        client,
        prompt_provider=lambda tick: tick.prompt or "",
        sampling_params_list=[OmniDiffusionSamplingParams()],
        diffusion_stage_id=0,
    )
    session = ARDiffusionSession(
        "world-1",
        tick_consumer=consumer,
        lifecycle=FakeLifecycle(),
        max_pending_events=4,
        request_id_factory=lambda session_id, chunk_index: f"{session_id}-chunk-{chunk_index}",
    )
    await session.accept_event(ARDiffusionSessionEvent(event_id=7, prompt="turn left"))

    output = await session.next_chunk()

    assert output.request_id == "world-1-chunk-0"
    assert session.chunk_index == 1
    assert session.pending_event_count == 0
    assert client.calls[0]["request_id"] == "world-1-chunk-0"
