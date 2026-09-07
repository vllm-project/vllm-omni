# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for midway prompt update across runner, batch, pipeline, and entrypoint layers."""

from __future__ import annotations

import asyncio
import queue
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.engine.test_orchestrator import (
    OrchestratorFixture,
    _build_harness,
    _wait_for,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.diffusion.interaction.coordinator import InteractionCoordinator
from vllm_omni.diffusion.interaction.modality_handlers.prompt import (
    PromptInteractionHandler,
    PromptSession,
)
from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.engine.async_omni_engine import StageRuntimeInfo
from vllm_omni.engine.messages import (
    ErrorMessage,
    InteractionMessage,
    ShutdownRequestMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_client import StagePoolClient
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.entrypoints.async_omni import AsyncEventResolver, AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniInteractionPrompt
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def pipeline() -> HeliosPipeline:
    pipeline = object.__new__(HeliosPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)  # pyright: ignore[reportAttributeAccessIssue]
    pipeline.encode_prompt = MagicMock(
        return_value=(
            torch.full((1, 4, 2), 2.0),
            None,
        )
    )
    pipeline.prepare_next_chunk = MagicMock()
    od_config = SimpleNamespace(model_class_name="HeliosPipeline")
    pipeline._interaction_coordinator = InteractionCoordinator.build(pipeline, od_config)
    return pipeline


def _make_diffusion_request_state(*, request_id: str = "req-1", fps: int | None = None) -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(  # pyright: ignore[reportArgumentType]
            num_outputs_per_prompt=1,
            max_sequence_length=226,
            fps=fps,
        ),
        prompt="hello",
    )
    state.prompt_embeds = torch.zeros(1, 4, 2)
    state.extra = {"window_num_frames": 8}
    return state


def _make_diffusion_model_runner(
    *,
    pipeline,
    streaming_output: bool = True,
    model_class_name: str = "HeliosPipeline",
) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.state_cache = {}
    runner.od_config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        model_class_name=model_class_name,
        streaming_output=streaming_output,
        cache_backend=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.cache_backend = None
    runner.offload_backend = None
    runner.prompt_embed_cache = None
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda *a, **k: None,
    )
    # HeliosPipeline skeletons from object.__new__ are not full SupportsStepExecution
    # instances; unit tests of submit_interaction stub this. Stepwise path tests
    # rebind to the real method.
    runner._supports_step_mode = lambda: True
    runner._interaction_coordinator = None
    return runner


def _prompt_interaction(
    prompt: str = "updated",
    *,
    event_id: str = "ui-update-1",
    transition_chunks: int | None = None,
) -> OmniInteractionPrompt:
    interaction: dict[str, Any] = {
        "event_id": event_id,
        "event": {"prompt": prompt},
    }
    if transition_chunks is not None:
        interaction["transition_chunks"] = transition_chunks
    return cast(OmniInteractionPrompt, interaction)


class TestPromptUpdateExecution:
    """Runner, InputBatch cache, and pipeline chunk-boundary behavior."""

    def test_runner_interaction_delegates_to_helios_pipeline(self, pipeline: HeliosPipeline) -> None:
        """Runner encodes the new prompt and queues pending embeds on the request state."""
        runner = _make_diffusion_model_runner(pipeline=pipeline)
        state = _make_diffusion_request_state()
        runner.state_cache["req-1"] = state

        runner.submit_interaction("req-1", _prompt_interaction(transition_chunks=2))

        pipeline.encode_prompt.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
        session = state.interaction_sessions["prompt"]
        assert isinstance(session, PromptSession)
        pending = session.pending_event
        assert pending is not None
        assert pending.event_id == "ui-update-1"
        assert pending.transition_chunks == 2
        assert torch.equal(pending.target_prompt_embeds, torch.full((1, 4, 2), 2.0))

    def test_runner_prompt_update_rejects_unsupported_pipeline(self) -> None:
        """Runner rejects prompt updates when the pipeline lacks prompt-update support."""

        class _UnsupportedPipeline:
            supports_step_execution = True

        runner = _make_diffusion_model_runner(pipeline=_UnsupportedPipeline())
        with pytest.raises(ValueError, match="not supported"):
            runner.submit_interaction("req-1", _prompt_interaction())

    def test_runner_prompt_update_rejects_missing_request(self, pipeline) -> None:
        """Runner rejects prompt updates when no active request state exists."""
        runner = _make_diffusion_model_runner(pipeline=pipeline)
        with pytest.raises(ValueError, match="No active request state"):
            runner.submit_interaction("missing", _prompt_interaction())

    @pytest.mark.parametrize(
        "interaction",
        [
            {},
            {"multi_modal_data": {"camera": {"type": "pose"}}},
            {"event": {"prompt": "updated", "multi_modal_data": {"camera": {"type": "pose"}}}},
            {"event": {"prompt": "updated", "multi_modal_data": {}}},
            {"event": {"prompt": "updated", "multi_modal_data": None}},
            {"event": {"prompt": "updated", "multi_modal_data": "bad"}},
        ],
    )
    def test_runner_interaction_rejects_structural_payloads_until_implemented(
        self,
        pipeline: HeliosPipeline,
        interaction: dict[str, Any],
    ) -> None:
        """Unsupported interaction dict shapes are preserved to and rejected by the runner."""
        runner = _make_diffusion_model_runner(pipeline=pipeline)
        runner.state_cache["req-1"] = _make_diffusion_request_state()

        with pytest.raises(NotImplementedError, match="Only text-only prompt update interactions"):
            runner.submit_interaction("req-1", cast(Any, interaction))

    @pytest.mark.parametrize("model_class_name", ["HeliosPipeline", "HeliosPyramidPipeline"])
    def test_coordinator_registers_prompt_for_helios_aliases(
        self,
        pipeline: HeliosPipeline,
        model_class_name: str,
    ) -> None:
        """Both Helios architecture names resolve the prompt handler registry."""
        od_config = SimpleNamespace(model_class_name=model_class_name)
        coordinator = InteractionCoordinator.build(pipeline, od_config)
        assert coordinator.has_modality("prompt")

        runner = _make_diffusion_model_runner(pipeline=pipeline, model_class_name=model_class_name)
        runner.state_cache["req-1"] = _make_diffusion_request_state()
        runner.submit_interaction("req-1", _prompt_interaction())
        session = runner.state_cache["req-1"].interaction_sessions["prompt"]
        assert isinstance(session, PromptSession)
        assert session.pending_event is not None

    def test_helios_peek_chunk_media_requires_positive_fps(self, pipeline: HeliosPipeline) -> None:
        """Media-timeline peek must raise on missing/invalid fps instead of coercing to 0.0."""
        state = _make_diffusion_request_state(fps=None)
        with pytest.raises(ValueError, match="sampling.fps is required and must be > 0"):
            pipeline.peek_chunk_media(state)

        state.sampling.fps = 0  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(ValueError, match="sampling.fps is required and must be > 0"):
            pipeline.peek_chunk_media(state)

    def test_input_batch_refreshes_prompt_embeds_on_version_change(self) -> None:
        """InputBatch rebuilds prompt_embeds when prompt_update_version changes."""
        state = StepRequestState(
            request_id="req-1",
            sampling=SimpleNamespace(),  # pyright: ignore[reportArgumentType]
            prompt="hello",
        )
        state.prompt_embeds = torch.zeros(1, 2, 3)
        state.latents = torch.zeros(1, 2)
        state.timesteps = torch.tensor([1.0])
        state.interaction_sessions["prompt"] = PromptSession(version=0)

        batch = InputBatch.make_batch([state])
        assert torch.equal(batch.prompt_embeds, torch.zeros(1, 2, 3))  # pyright: ignore[reportArgumentType]

        state.prompt_embeds = torch.ones(1, 2, 3)
        state.interaction_sessions["prompt"] = PromptSession(version=1)
        refreshed = InputBatch.make_batch([state], cached_batch=batch)
        assert torch.equal(refreshed.prompt_embeds, torch.ones(1, 2, 3))  # pyright: ignore[reportArgumentType]

    def test_helios_prompt_handler_enqueue_queues_pending_target(self, pipeline: HeliosPipeline) -> None:
        """Prompt handler queues target embeds without mutating current prompt_embeds."""
        state = _make_diffusion_request_state()

        PromptInteractionHandler.from_pipeline(pipeline).enqueue(
            state,
            event_id="ui-update-1",
            received_at=0.0,
            payload={"prompt": "new scene"},
            transition_chunks=2,
        )

        session = state.interaction_sessions["prompt"]
        assert isinstance(session, PromptSession)
        pending = session.pending_event
        assert pending is not None
        assert torch.equal(pending.target_prompt_embeds, torch.full((1, 4, 2), 2.0))
        assert pending.transition_chunks == 2
        assert torch.equal(state.prompt_embeds, torch.zeros(1, 4, 2))  # pyright: ignore[reportArgumentType]

    def test_helios_prompt_handler_enqueue_rejects_before_initial_generation(self, pipeline: HeliosPipeline) -> None:
        """Reject prompt updates submitted before initial prompt embeds exist."""
        state = _make_diffusion_request_state()
        state.prompt_embeds = None

        with pytest.raises(
            ValueError,
            match="prompt_update is not allowed before initial generation has started",
        ):
            PromptInteractionHandler.from_pipeline(pipeline).enqueue(
                state,
                event_id="ui-update-1",
                received_at=0.0,
                payload={"prompt": "new scene"},
                transition_chunks=2,
            )

        session = state.interaction_sessions.get("prompt")
        assert session is None or (isinstance(session, PromptSession) and session.pending_event is None)

    def test_helios_apply_interaction_at_chunk_boundary_starts_transition(self, pipeline: HeliosPipeline) -> None:
        """At chunk boundary, starts transition state and bumps prompt_update_version."""
        state = _make_diffusion_request_state(fps=None)
        PromptInteractionHandler.from_pipeline(pipeline).enqueue(
            state,
            event_id="ui-update-1",
            received_at=0.0,
            payload={"prompt": "new scene"},
            transition_chunks=2,
        )

        pipeline.apply_interaction_at_chunk_boundary(state)

        session = state.interaction_sessions["prompt"]
        assert isinstance(session, PromptSession)
        assert session.active_event is not None
        assert torch.equal(state.prompt_embeds, torch.ones(1, 4, 2))  # pyright: ignore[reportArgumentType]
        assert session.version == 1
        assert state.interaction_chunk_metadata is not None
        assert state.interaction_chunk_metadata.as_dict() == {
            "started_event_ids": ["ui-update-1"],
            "active_event_ids": ["ui-update-1"],
            "completed_event_ids": [],
        }

    def test_helios_apply_interaction_advances_transition_over_chunks(self, pipeline: HeliosPipeline) -> None:
        """At chunk boundary, interpolates embeds until the target prompt is reached."""
        state = _make_diffusion_request_state(fps=None)
        PromptInteractionHandler.from_pipeline(pipeline).enqueue(
            state,
            event_id="ui-update-1",
            received_at=0.0,
            payload={"prompt": "new scene"},
            transition_chunks=3,
        )
        pipeline.apply_interaction_at_chunk_boundary(state)

        assert torch.allclose(state.prompt_embeds, torch.full((1, 4, 2), 2.0 / 3.0))  # pyright: ignore[reportArgumentType]

        for _ in range(2):
            pipeline.apply_interaction_at_chunk_boundary(state)

        assert torch.allclose(state.prompt_embeds, torch.full((1, 4, 2), 2.0))  # pyright: ignore[reportArgumentType]
        session = state.interaction_sessions["prompt"]
        assert isinstance(session, PromptSession)
        assert session.version == 3

        # Further chunk boundaries after completion must not bump the version.
        pipeline.apply_interaction_at_chunk_boundary(state)
        assert session.version == 3

    def test_stepwise_runner_applies_prompt_update_and_acks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Drive the production runner loop: submit → apply(no media args) → ACK on next chunk.

        Also regresses ``sampling.fps is None`` on the prompt-only path (no peek).
        ACK ids land on the *next* chunk's ``DiffusionOutput`` via
        ``_attach_stepwise_metadata`` consuming the prior boundary metadata.
        """
        import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
        from vllm_omni.diffusion.interaction.mixin import InteractionMixin
        from vllm_omni.diffusion.interaction.types import ChunkMediaSpec

        chunks = [
            DiffusionOutput(output="chunk-0", finished=False, chunk_index=0, total_chunks=2),  # pyright: ignore[reportArgumentType]
            DiffusionOutput(output="chunk-1", finished=True, chunk_index=1, total_chunks=2),  # pyright: ignore[reportArgumentType]
        ]

        class _InteractiveChunkPipeline(InteractionMixin):
            device = torch.device("cpu")
            supports_step_execution = True

            def __init__(self) -> None:
                self._outputs = chunks
                self.decode_calls = 0
                self.transformer = SimpleNamespace(dtype=torch.float32)
                self.encode_prompt = MagicMock(return_value=(torch.full((1, 4, 2), 2.0), None))
                self.prepare_next_chunk = MagicMock()
                od_config = SimpleNamespace(model_class_name="HeliosPipeline")
                self._interaction_coordinator = InteractionCoordinator.build(self, od_config)  # pyright: ignore[reportArgumentType]

            def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
                # Same contract as Helios: never coerce missing fps to 0.0.
                fps = state.sampling.fps
                if fps is None or float(fps) <= 0:
                    raise ValueError(
                        "sampling.fps is required and must be > 0 for interaction modalities "
                        f"that use the chunk media timeline, got {fps!r} "
                        f"(request_id={state.request_id!r})"
                    )
                return ChunkMediaSpec(num_frames=int(state.extra["window_num_frames"]), fps=float(fps))

            def prepare_encode(self, state: StepRequestState) -> StepRequestState:
                state.prompt_embeds = torch.zeros(1, 4, 2)
                state.latents = torch.zeros(1, 1)
                state.timesteps = torch.tensor([1.0, 0.0, 1.0, 0.0])
                state.step_index = 0
                state.step_in_chunk = 0
                state.chunk_num_steps = 2
                state.total_chunks = 2
                state.extra = {"window_num_frames": 8}
                return state

            def denoise_step(self, input_batch: Any, states: Any) -> torch.Tensor:
                del states
                return torch.ones_like(input_batch.latents)

            def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor | None) -> None:
                state.latents = noise_pred
                state.step_index += 1
                state.step_in_chunk += 1

            def post_decode(self, state: StepRequestState) -> DiffusionOutput:
                output = self._outputs[self.decode_calls]
                self.decode_calls += 1
                state.chunk_index += 1
                state.step_index = 0
                state.step_in_chunk = 0
                if not state.request_denoise_completed:
                    state.latents = torch.zeros(1, 1)
                return output

        pipeline = _InteractiveChunkPipeline()
        runner = _make_diffusion_model_runner(pipeline=pipeline)
        runner._supports_step_mode = lambda: DiffusionModelRunner._supports_step_mode(runner)
        runner.od_config.streaming_output = True
        runner.od_config.step_execution = True
        runner.diffusion_kv_backend = SimpleNamespace(remove_diffusion_kv_requests=lambda *_: 0)  # pyright: ignore[reportAttributeAccessIssue]

        sampling = SimpleNamespace(
            generator=None,
            seed=None,
            generator_device=None,
            num_inference_steps=4,
            num_outputs_per_prompt=1,
            max_sequence_length=226,
            fps=None,
        )
        req = SimpleNamespace(
            request_id="req",
            prompt="a prompt",
            sampling_params=sampling,
            kv_sender_info=None,
        )

        @contextmanager
        def _noop_forward_context(*args: Any, **kwargs: Any):
            del args, kwargs
            yield

        monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
        monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)
        monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_reserved", lambda: 0)
        monkeypatch.setattr(model_runner_module.current_omni_platform, "max_memory_allocated", lambda: 0)
        monkeypatch.setattr(model_runner_module.current_omni_platform, "is_available", lambda: False)

        scheduler_output = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(request_id="req", req=req, diffusion_kv_metadata=None)],
            scheduled_cached_reqs=SimpleNamespace(request_ids=[]),
        )
        # Admit request and run first denoise step (no chunk boundary yet).
        DiffusionModelRunner.execute_stepwise(runner, scheduler_output)  # pyright: ignore[reportArgumentType]
        assert "req" in runner.state_cache
        assert runner.state_cache["req"].sampling.fps is None

        runner.submit_interaction("req", _prompt_interaction("new scene", transition_chunks=2))

        cached = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(request_ids=["req"]),
        )
        # Second step completes chunk 0: apply runs (no peek), metadata staged.
        chunk0 = DiffusionModelRunner.execute_stepwise(runner, cached).get_request_output("req")  # pyright: ignore[reportArgumentType]
        assert chunk0 is not None and chunk0.result is not None
        assert chunk0.result.started_event_ids == []
        state = runner.state_cache["req"]
        assert torch.equal(state.prompt_embeds, torch.ones(1, 4, 2))  # pyright: ignore[reportArgumentType]
        assert state.interaction_chunk_metadata is not None
        assert state.interaction_chunk_metadata.started_event_ids == ["ui-update-1"]
        pipeline.prepare_next_chunk.assert_called()  # pyright: ignore[reportAttributeAccessIssue]

        # Next chunk decode attaches prior-boundary ACK ids onto DiffusionOutput.
        DiffusionModelRunner.execute_stepwise(runner, cached)  # pyright: ignore[reportArgumentType]
        chunk1 = DiffusionModelRunner.execute_stepwise(runner, cached).get_request_output("req")  # pyright: ignore[reportArgumentType]
        assert chunk1 is not None and chunk1.result is not None
        assert chunk1.result.started_event_ids == ["ui-update-1"]
        assert chunk1.result.active_event_ids == ["ui-update-1"]
        assert chunk1.finished is True


class TestPromptUpdateIntegration:
    """AsyncOmni prompt update through orchestrator, inline client, and runner."""

    @pytest.mark.asyncio
    async def test_orchestrator_prompt_update_failure_surfaces_non_fatal_error(
        self,
    ) -> None:
        """Worker-side prompt_update rejection is reported without crashing the orchestrator."""

        class _RejectingStagePool:
            async def submit_interaction(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs
                raise ValueError("prompt embeds are not ready")

        output_queue: asyncio.Queue[ErrorMessage] = asyncio.Queue()
        orchestrator = object.__new__(Orchestrator)
        orchestrator.stage_pools = [_RejectingStagePool()]  # pyright: ignore[reportAttributeAccessIssue]
        orchestrator.output_async_queue = output_queue  # pyright: ignore[reportAttributeAccessIssue]
        orchestrator.request_states = {"req-1": OrchestratorRequestState(request_id="req-1")}

        await orchestrator._handle_interaction(
            InteractionMessage(
                request_id="req-1",
                interaction=_prompt_interaction("new scene", event_id="ui-update-1", transition_chunks=2),
            )
        )

        msg = output_queue.get_nowait()
        assert msg == ErrorMessage(
            error="Failed interaction for request req-1: prompt embeds are not ready",
            fatal=False,
            request_id="req-1",
            event_id="ui-update-1",
            stage_id=0,
        )

    @pytest.mark.asyncio
    async def test_runner_prompt_update_failure_surfaces_non_fatal_error(self, pipeline: HeliosPipeline) -> None:
        """Runner-side prompt encode rejection is reported through the orchestrator."""
        pipeline.encode_prompt = MagicMock(  # pyright: ignore[reportAttributeAccessIssue]
            side_effect=ValueError("prompt embeds are not ready")
        )
        runner = _make_diffusion_model_runner(pipeline=pipeline)
        runner.state_cache["req-1"] = _make_diffusion_request_state(request_id="req-1")
        prompt_update_engine = self._PromptUpdateEngine(self._IncrementalStreamingPipeline(), runner)
        inline_client = self._make_inline_pipeline_client(prompt_update_engine)

        output_queue: asyncio.Queue[ErrorMessage] = asyncio.Queue()
        orchestrator = object.__new__(Orchestrator)
        orchestrator.stage_pools = [  # pyright: ignore[reportAttributeAccessIssue]
            StagePool(0, cast(StagePoolClient, inline_client))
        ]
        orchestrator.output_async_queue = output_queue  # pyright: ignore[reportAttributeAccessIssue]
        orchestrator.request_states = {"req-1": OrchestratorRequestState(request_id="req-1")}

        try:
            await orchestrator._handle_interaction(
                InteractionMessage(
                    request_id="req-1",
                    interaction=_prompt_interaction("new scene", event_id="ui-update-1", transition_chunks=2),
                )
            )
        finally:
            inline_client.shutdown()

        pipeline.encode_prompt.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
        msg = output_queue.get_nowait()
        assert msg == ErrorMessage(
            error="Failed interaction for request req-1: prompt embeds are not ready",
            fatal=False,
            request_id="req-1",
            event_id="ui-update-1",
            stage_id=0,
        )

    @pytest.mark.asyncio
    async def test_prompt_update_reaches_runner_from_async_omni(self, pipeline: HeliosPipeline) -> None:
        """Midway prompt update submitted via AsyncOmni reaches the diffusion runner."""
        streaming_pipeline = self._IncrementalStreamingPipeline()
        runner = _make_diffusion_model_runner(pipeline=pipeline)
        prompt_update_engine = self._PromptUpdateEngine(streaming_pipeline, runner)
        inline_client = self._make_inline_pipeline_client(prompt_update_engine)
        fixture = _build_harness([inline_client])
        omni = self._make_async_omni(self._OrchestratorBridgeEngine(fixture))

        generate_task: asyncio.Task[list[OmniRequestOutput]] | None = None
        try:
            generate_task = asyncio.create_task(self._collect_generate_outputs(omni))

            await _wait_for(
                lambda: (
                    len(runner.state_cache) > 0
                    and any(state.external_request_id == "req-omni" for state in omni.request_states.values())
                )
            )
            internal_request_id = next(iter(runner.state_cache))

            await omni.submit_interaction_async(
                "req-omni",
                interaction=_prompt_interaction("new scene", event_id="ui-update-1", transition_chunks=2),
            )

            outputs = await generate_task

            pending = runner.state_cache[internal_request_id].interaction_sessions["prompt"].pending_event
            assert pending is not None
            assert pending.transition_chunks == 2
            assert torch.equal(pending.target_prompt_embeds, torch.full((1, 4, 2), 2.0))
            pipeline.encode_prompt.assert_called_once()  # pyright: ignore[reportAttributeAccessIssue]
            assert [output.custom_output["chunk"] for output in outputs] == [0, 1]
        finally:
            if generate_task is not None and not generate_task.done():
                generate_task.cancel()
                await asyncio.gather(generate_task, return_exceptions=True)
            await self._shutdown_pipeline_omni_harness(omni, fixture, inline_client)

    @staticmethod
    async def _collect_generate_outputs(omni: AsyncOmni) -> list[OmniRequestOutput]:
        outputs: list[OmniRequestOutput] = []
        async for output in omni.generate(
            prompt={"prompt": "a cat"},
            request_id="req-omni",
            sampling_params_list=[OmniDiffusionSamplingParams()],
            output_modalities=["image"],
        ):
            outputs.append(output)
        return outputs

    @classmethod
    def _make_inline_pipeline_client(cls, engine: _PromptUpdateEngine) -> InlineStageDiffusionClient:
        metadata = StageMetadata(
            stage_id=0,
            stage_type="diffusion",
            engine_output_type="image",
            is_comprehension=False,
            requires_multimodal_data=False,
            engine_input_source=[],
            final_output=True,
            final_output_type="image",
            default_sampling_params=OmniDiffusionSamplingParams(),
            custom_process_input_func=None,
            model_stage=None,
            runtime_cfg=None,
        )
        with patch.object(InlineStageDiffusionClient, "_enrich_config"):
            with patch(
                "vllm_omni.diffusion.inline_stage_diffusion_client.DiffusionEngine.make_engine",
                return_value=engine,
            ):
                od_config = MagicMock(spec=OmniDiffusionConfig)
                od_config.streaming_output = True
                return InlineStageDiffusionClient(
                    model="test_model",
                    od_config=od_config,
                    metadata=metadata,
                )

    @staticmethod
    def _make_async_omni(engine: _OrchestratorBridgeEngine) -> AsyncOmni:
        omni = object.__new__(AsyncOmni)
        omni.engine = engine  # pyright: ignore[reportAttributeAccessIssue]
        omni.log_stats = False
        omni._pause_cond = asyncio.Condition()
        omni._paused = False
        omni.request_states = {}
        omni.final_output_task = None
        omni.event_resolver = AsyncEventResolver()
        omni._enable_ar_profiler = False
        omni.prom_metrics = MagicMock()
        omni.mod_metrics = MagicMock()
        omni.resolve_sampling_params_list = lambda params, allow_delta_coercion: params  # pyright: ignore[reportAttributeAccessIssue]
        omni._compute_final_stage_id = lambda output_modalities: 0
        omni._compute_final_output_stage_ids = lambda output_modalities: [0]
        omni.default_sampling_params_list = engine.default_sampling_params_list  # pyright: ignore[reportAttributeAccessIssue]
        omni._log_summary_and_cleanup = lambda request_id: omni.request_states.pop(request_id, None)  # pyright: ignore[reportAttributeAccessIssue]
        return omni

    @staticmethod
    async def _shutdown_pipeline_omni_harness(
        omni: AsyncOmni,
        fixture: OrchestratorFixture,
        inline_client: InlineStageDiffusionClient,
    ) -> None:
        if omni.final_output_task is not None:
            omni.final_output_task.cancel()
            await asyncio.gather(omni.final_output_task, return_exceptions=True)
        inline_client.shutdown()
        fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        await asyncio.to_thread(fixture.thread.join, 5)

    class _IncrementalStreamingPipeline:
        """Returns one streaming chunk per ``step_outputs`` call."""

        supports_step_execution = True

        def __init__(self) -> None:
            self.requests: list[Any] = []
            self._step_index = 0

        def step_outputs(self, request):
            self.requests.append(request)
            if self._step_index == 0:
                self._step_index += 1
                return [DiffusionOutput(output={"chunk": 0}, finished=False)]
            return [DiffusionOutput(output={"chunk": 1}, finished=True)]

    class _OrchestratorBridgeEngine:
        """Minimal AsyncOmni engine facade backed by a live Orchestrator harness."""

        def __init__(self, fixture: OrchestratorFixture) -> None:
            self._fixture = fixture
            self.stage_metadata = [
                StageRuntimeInfo(
                    stage_type="diffusion",
                    final_output=True,
                    final_output_type="image",
                )
            ]
            self.stage_configs: list[Any] = [SimpleNamespace(stage_type="diffusion")]
            self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
            self.num_stages = 1
            self.supported_tasks = ("generate",)
            self._alive = True

        async def add_request_async(
            self,
            *,
            request_id: str,
            prompt: Any,
            sampling_params_list: list[Any],
            final_stage_id: int,
            **kwargs: Any,
        ) -> None:
            self._fixture.request_sync_q.put_nowait(
                StageSubmissionMessage(
                    type="add_request",
                    request_id=request_id,
                    prompt=prompt,
                    original_prompt=prompt,
                    output_prompt_text=None,
                    sampling_params_list=sampling_params_list,
                    final_stage_id=final_stage_id,
                    final_output_stage_ids=kwargs.get("final_output_stage_ids"),
                    preprocess_ms=0.0,
                    request_timestamp=kwargs.get("arrival_time", time.time()),
                    enqueue_ts=time.perf_counter(),
                )
            )

        async def submit_interaction_async(
            self,
            request_id: str,
            *,
            interaction: dict[str, Any],
        ) -> None:
            self._fixture.request_sync_q.put_nowait(
                InteractionMessage(
                    request_id=request_id,
                    interaction=cast(Any, interaction),
                )
            )

        async def try_get_output_async(self) -> Any | None:
            try:
                return self._fixture.output_sync_q.get_nowait()
            except queue.Empty:
                return None

        def get_stage_metadata(self, stage_id: int) -> StageRuntimeInfo:
            return self.stage_metadata[stage_id]

        def is_alive(self) -> bool:
            return self._fixture.thread.is_alive()

        async def abort_async(self, request_ids: list[str]) -> None:
            del request_ids

    class _PromptUpdateEngine:
        """DiffusionEngine stand-in that streams chunks and routes prompt_update to a runner."""

        def __init__(
            self,
            streaming_pipeline: TestPromptUpdateIntegration._IncrementalStreamingPipeline,
            runner: DiffusionModelRunner,
        ) -> None:
            self.streaming_pipeline = streaming_pipeline
            self._runner = runner
            self.executor = SimpleNamespace(
                register_failure_callback=MagicMock(),
                check_health=MagicMock(),
            )

        async def step_streaming(self, request):
            if request.request_id not in self._runner.state_cache:
                state = _make_diffusion_request_state(request_id=request.request_id)
                self._runner.state_cache[request.request_id] = state

            state = self._runner.state_cache[request.request_id]

            outputs = self.streaming_pipeline.step_outputs(request)
            output = outputs[0]
            custom_output = cast(dict[str, Any], output.output) or {}
            yield [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_id,
                    images=list(custom_output.get("images") or []),
                    custom_output=custom_output,
                    finished=output.finished,
                )
            ]

            deadline = time.monotonic() + 5.0
            while (
                not isinstance(state.interaction_sessions.get("prompt"), PromptSession)
                or state.interaction_sessions["prompt"].pending_event is None
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for prompt update during streaming")
                await asyncio.sleep(0.01)

            outputs = self.streaming_pipeline.step_outputs(request)
            output = outputs[0]
            custom_output = cast(dict[str, Any], output.output) or {}
            yield [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_id,
                    images=list(custom_output.get("images") or []),
                    custom_output=custom_output,
                    finished=output.finished,
                )
            ]

        def collective_rpc(self, method, timeout, args, kwargs, unique_reply_rank):
            del timeout, kwargs, unique_reply_rank
            if method == "submit_interaction":
                request_id, interaction = args
                self._runner.submit_interaction(request_id, interaction)
                return None
            raise NotImplementedError(f"collective_rpc not mocked for {method!r}")

        def abort(self, request_id: str) -> None:
            del request_id
