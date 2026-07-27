# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for midway prompt update across runner, batch, pipeline, and entrypoint layers."""

from __future__ import annotations

import asyncio
import queue
import time
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
    pipeline._prepare_next_chunk = MagicMock()
    return pipeline


def _make_diffusion_request_state(*, request_id: str = "req-1") -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(num_outputs_per_prompt=1, max_sequence_length=226),  # pyright: ignore[reportArgumentType]
        prompt="hello",
    )
    state.prompt_embeds = torch.zeros(1, 4, 2)
    state.extra = {}
    return state


def _make_diffusion_model_runner(*, pipeline, streaming_output: bool = True) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.state_cache = {}
    runner.od_config = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        model_class_name="HeliosPipeline",
        streaming_output=streaming_output,
    )
    runner._supports_step_mode = lambda: True
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
        pending = state.extra["pending_prompt_update"]
        assert pending["event_id"] == "ui-update-1"
        assert pending["transition_chunks"] == 2
        assert torch.equal(pending["target_prompt_embeds"], torch.full((1, 4, 2), 2.0))

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
        state.extra["prompt_update_version"] = 0

        batch = InputBatch.make_batch([state])
        assert torch.equal(batch.prompt_embeds, torch.zeros(1, 2, 3))  # pyright: ignore[reportArgumentType]

        state.prompt_embeds = torch.ones(1, 2, 3)
        state.extra["prompt_update_version"] = 1
        refreshed = InputBatch.make_batch([state], cached_batch=batch)
        assert torch.equal(refreshed.prompt_embeds, torch.ones(1, 2, 3))  # pyright: ignore[reportArgumentType]

    def test_helios_prepare_prompt_update_queues_pending_target(self, pipeline: HeliosPipeline) -> None:
        """HeliosPipeline queues target embeds without mutating current prompt_embeds."""
        state = _make_diffusion_request_state()

        pipeline.prepare_prompt_update(state, "new scene", "ui-update-1", transition_chunks=2)

        pending = state.extra["pending_prompt_update"]
        assert torch.equal(pending["target_prompt_embeds"], torch.full((1, 4, 2), 2.0))
        assert pending["transition_chunks"] == 2
        assert torch.equal(state.prompt_embeds, torch.zeros(1, 4, 2))  # pyright: ignore[reportArgumentType]

    def test_helios_prepare_prompt_update_rejects_before_initial_generation(self, pipeline: HeliosPipeline) -> None:
        """Reject prompt updates submitted before initial prompt embeds exist."""
        state = _make_diffusion_request_state()
        state.prompt_embeds = None

        with pytest.raises(
            ValueError,
            match="prompt_update is not allowed before initial generation has started",
        ):
            pipeline.prepare_prompt_update(state, "new scene", "ui-update-1", transition_chunks=2)

        assert "pending_prompt_update" not in state.extra

    def test_helios_apply_prompt_update_at_chunk_boundary_starts_transition(self, pipeline: HeliosPipeline) -> None:
        """At chunk boundary, starts transition state and bumps prompt_update_version."""
        state = _make_diffusion_request_state()
        pipeline.prepare_prompt_update(state, "new scene", "ui-update-1", transition_chunks=2)

        pipeline._apply_prompt_update_at_chunk_boundary(state)

        assert state.extra.get("prompt_update_state") is not None
        assert torch.equal(state.prompt_embeds, torch.ones(1, 4, 2))  # pyright: ignore[reportArgumentType]
        assert state.extra["prompt_update_version"] == 1
        assert state.extra["prompt_update_chunk_metadata"] == {
            "started_event_ids": ["ui-update-1"],
            "active_event_ids": ["ui-update-1"],
            "completed_event_ids": [],
        }

    def test_helios_apply_prompt_update_advances_transition_over_chunks(self, pipeline: HeliosPipeline) -> None:
        """At chunk boundary, interpolates embeds until the target prompt is reached."""
        state = _make_diffusion_request_state()
        pipeline.prepare_prompt_update(state, "new scene", "ui-update-1", transition_chunks=3)
        pipeline._apply_prompt_update_at_chunk_boundary(state)

        assert torch.allclose(state.prompt_embeds, torch.full((1, 4, 2), 2.0 / 3.0))  # pyright: ignore[reportArgumentType]

        for _ in range(2):
            pipeline._apply_prompt_update_at_chunk_boundary(state)

        assert torch.allclose(state.prompt_embeds, torch.full((1, 4, 2), 2.0))  # pyright: ignore[reportArgumentType]
        assert state.extra["prompt_update_version"] == 3

        # Further chunk boundaries after completion must not bump the version.
        pipeline._apply_prompt_update_at_chunk_boundary(state)
        assert state.extra["prompt_update_version"] == 3


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
        """Runner-side prepare_prompt_update rejection is reported through the orchestrator."""
        pipeline.prepare_prompt_update = MagicMock(  # pyright: ignore[reportAttributeAccessIssue, reportMethodAssignment]
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

        pipeline.prepare_prompt_update.assert_called_once_with(  # pyright: ignore[reportAttributeAccessIssue]
            runner.state_cache["req-1"],
            "new scene",
            "ui-update-1",
            2,
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

            pending = runner.state_cache[internal_request_id].extra["pending_prompt_update"]
            assert pending["transition_chunks"] == 2
            assert torch.equal(pending["target_prompt_embeds"], torch.full((1, 4, 2), 2.0))
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
                    batch_size=1,
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
            while "pending_prompt_update" not in state.extra:
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
