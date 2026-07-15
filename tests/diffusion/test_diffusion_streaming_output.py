# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for diffusion pipeline streaming output mode.

Covers:
- Mock pipeline -> StageDiffusionClient (ZMQ subprocess path)
- Mock pipeline -> InlineStageDiffusionClient -> Orchestrator -> AsyncOmni
- Mock pipeline -> InlineStageDiffusionClient -> Orchestrator -> AsyncOmni -> `/v1/realtime/video`
"""

import asyncio
import contextlib
import queue
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import zmq
from fastapi import FastAPI, WebSocket
from pytest_mock import MockerFixture
from starlette.testclient import TestClient
from vllm.utils.network_utils import get_open_zmq_ipc_path

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
from tests.engine.test_orchestrator import OrchestratorFixture, _build_harness, _wait_for
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)
from vllm_omni.engine.async_omni_engine import StageRuntimeInfo
from vllm_omni.engine.messages import ShutdownRequestMessage, StageSubmissionMessage
from vllm_omni.engine.stage_init_utils import StageMetadata
from vllm_omni.entrypoints.async_omni import AsyncEventResolver, AsyncOmni
from vllm_omni.entrypoints.openai.serving_video_output_stream import OmniStreamingVideoOutputHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _StepStreamingPipeline:
    supports_step_execution = True

    def __init__(self, outputs: list[DiffusionOutput]) -> None:
        self.outputs = outputs
        self.requests: list[Any] = []

    def step_outputs(self, request):
        self.requests.append(request)
        return list(self.outputs)


class _FailingStreamingPipeline:
    supports_step_execution = True

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def step_outputs(self, request):
        self.requests.append(request)
        raise RuntimeError("stream failed midway")


class _PipelineBackedEngine:
    """DiffusionEngine stand-in that exposes step-streaming chunks from a mock pipeline."""

    def __init__(self, pipeline: _StepStreamingPipeline | _FailingStreamingPipeline) -> None:
        self.pipeline = pipeline
        self.executor = SimpleNamespace(
            register_failure_callback=MagicMock(),
            check_health=MagicMock(),
        )

    async def step_streaming(self, request):
        try:
            outputs = self.pipeline.step_outputs(request)
        except Exception as exc:
            outputs = [
                _streaming_diffusion_output(chunk=0, finished=False),
                DiffusionOutput(error=str(exc), finished=True),
            ]
        for output in outputs:
            if output.error is not None:
                yield [OmniRequestOutput.from_error(request_id=request.request_id, error_message=output.error)]
                continue
            payload, metadata = _streaming_payload_and_metadata(output)
            stream_metadata = metadata.get("stream", {})
            yield [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_id,
                    images=list(payload.get("image") or []),
                    multimodal_output={"metadata": {"stream": stream_metadata}},
                    finished=output.finished,
                )
            ]

    def abort(self, request_id: str) -> None:
        del request_id


def _streaming_diffusion_output(
    *,
    chunk: int,
    finished: bool,
    images: list | None = None,
) -> DiffusionOutput:
    payload = {"image": images or []}
    metadata = {"stream": {"chunk": chunk}}
    return DiffusionOutput(
        output={
            "payload": payload,
            "metadata": metadata,
        },
        finished=finished,
    )


def _streaming_payload_and_metadata(output: DiffusionOutput) -> tuple[dict, dict]:
    envelope = output.output if isinstance(output.output, dict) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), dict) else {}
    metadata = envelope.get("metadata") if isinstance(envelope.get("metadata"), dict) else {}
    return payload, metadata


class TestPipelineStreamingOutputToStageDiffusionClient:
    """Streaming pipeline output over ZMQ into ``StageDiffusionClient``."""

    @pytest.mark.asyncio
    async def test_streaming_output_reaches_stage_client_from_mock_pipeline(self) -> None:
        """Mock pipeline chunks over ZMQ reach StageDiffusionClient with correct finished flags."""
        pipeline = _StepStreamingPipeline(
            [
                _streaming_diffusion_output(chunk=0, finished=False),
                _streaming_diffusion_output(chunk=1, finished=True),
            ]
        )

        outputs = await self._run_streaming_client_proc(pipeline, request_id="req-stream")

        assert [output.request_id for output in outputs] == ["req-stream", "req-stream"]
        assert [output.finished for output in outputs] == [False, True]
        assert [output.multimodal_output["metadata"]["stream"]["chunk"] for output in outputs] == [0, 1]
        assert pipeline.requests[0].request_id == "req-stream"

    @pytest.mark.asyncio
    async def test_streaming_midway_error_reaches_stage_client_after_prior_chunk(self) -> None:
        """A pipeline error after the first chunk still delivers that chunk then a terminal error output."""
        pipeline = _FailingStreamingPipeline()

        outputs = await self._run_streaming_client_proc(pipeline, request_id="req-error")

        assert len(outputs) == 2
        assert outputs[0].request_id == "req-error"
        assert outputs[0].finished is False
        assert outputs[0].multimodal_output["metadata"]["stream"]["chunk"] == 0
        assert outputs[1].request_id == "req-error"
        assert outputs[1].finished is True
        assert outputs[1].error is not None
        assert "stream failed midway" in outputs[1].error

    @classmethod
    async def _run_streaming_client_proc(
        cls,
        pipeline: _StepStreamingPipeline | _FailingStreamingPipeline,
        *,
        request_id: str,
    ) -> list[OmniRequestOutput]:
        request_address = get_open_zmq_ipc_path()
        response_address = get_open_zmq_ipc_path()
        proc = object.__new__(StageDiffusionProc)
        proc._od_config = SimpleNamespace(streaming_output=True)
        proc._engine = _PipelineBackedEngine(pipeline)
        proc._closed = False
        client = cls._make_client(request_address, response_address)
        proc_task = asyncio.create_task(proc.run_loop(request_address, response_address))
        try:
            await asyncio.sleep(0.05)
            await client.add_request_async(
                request_id,
                "prompt",
                OmniDiffusionSamplingParams(),
            )
            deadline = asyncio.get_running_loop().time() + 2.0
            outputs: list[OmniRequestOutput] = []
            while (not outputs or not outputs[-1].finished) and asyncio.get_running_loop().time() < deadline:
                output = client.get_diffusion_output_nowait()
                if output is not None:
                    outputs.append(output)
                else:
                    await asyncio.sleep(0.01)
            return outputs
        finally:
            try:
                client._request_socket.send(client._encoder.encode({"type": "shutdown"}))
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc_task, timeout=2)
            finally:
                cls._close_client(client)

    @staticmethod
    def _make_client(request_address: str, response_address: str) -> StageDiffusionClient:
        client = object.__new__(StageDiffusionClient)
        client.stage_id = 0
        client.replica_id = 0
        client.final_output = True
        client.final_output_type = "image"
        client.default_sampling_params = {}
        client.requires_multimodal_data = False
        client.custom_process_input_func = None
        client.engine_input_source = []
        client._proc = None
        client._proc_manager = None
        client._owns_process = False
        client._zmq_ctx = zmq.Context()
        client._request_socket = client._zmq_ctx.socket(zmq.PUSH)
        client._request_socket.bind(request_address)
        client._response_socket = client._zmq_ctx.socket(zmq.PULL)
        client._response_socket.bind(response_address)
        client._encoder = OmniMsgpackEncoder()
        client._decoder = OmniMsgpackDecoder()
        client._output_queue: asyncio.Queue[OmniRequestOutput] = asyncio.Queue()
        client._rpc_results = {}
        client._pending_rpcs = set()
        client._tasks = {}
        client._shutting_down = False
        client._engine_dead = False
        return client

    @staticmethod
    def _close_client(client: StageDiffusionClient) -> None:
        client._request_socket.close(linger=0)
        client._response_socket.close(linger=0)
        client._zmq_ctx.term()


class TestPipelineStreamingOutputToEntrypoint:
    """Streaming pipeline output through inline stage, orchestrator, and entrypoints."""

    @pytest.mark.asyncio
    async def test_streaming_output_reaches_async_omni_from_mock_pipeline(self) -> None:
        """Mock pipeline streaming chunks reach AsyncOmni.generate() via inline stage and orchestrator."""
        pipeline = _StepStreamingPipeline(
            [
                _streaming_diffusion_output(chunk=0, finished=False),
                _streaming_diffusion_output(chunk=1, finished=True),
            ]
        )
        inline_client = self._make_inline_pipeline_client(pipeline)
        fixture = _build_harness([inline_client])
        omni = self._make_async_omni(self._OrchestratorBridgeEngine(fixture))

        try:
            outputs: list[OmniRequestOutput] = []
            async for output in omni.generate(
                prompt={"prompt": "a cat"},
                request_id="req-omni",
                sampling_params_list=[OmniDiffusionSamplingParams()],
                output_modalities=["image"],
            ):
                outputs.append(output)

            await _wait_for(lambda: len(pipeline.requests) == 1)
            assert pipeline.requests[0].request_id.startswith("req-omni-")
            assert [output.multimodal_output["metadata"]["stream"]["chunk"] for output in outputs] == [0, 1]
            assert [getattr(output.request_output, "finished", output.finished) for output in outputs] == [False, True]
        finally:
            await self._shutdown_pipeline_omni_harness(omni, fixture, inline_client)

    @pytest.mark.asyncio
    async def test_streaming_output_reaches_api_websocket_from_mock_pipeline(self, mocker: MockerFixture) -> None:
        """Mock pipeline streaming chunks reach the video WebSocket handler through AsyncOmni.generate()."""
        frames = [np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(2)]
        pipeline = _StepStreamingPipeline(
            [
                _streaming_diffusion_output(chunk=0, images=[frames], finished=False),
                _streaming_diffusion_output(chunk=1, images=[frames], finished=True),
            ]
        )
        inline_client = self._make_inline_pipeline_client(pipeline)
        fixture = _build_harness([inline_client])
        bridge_engine = self._OrchestratorBridgeEngine(fixture)
        omni = self._make_async_omni(bridge_engine)

        encoded = [b"pipeline-fmp4-0", b"pipeline-fmp4-1"]

        class FakeStreamingVideoEncoder:
            encode_calls: list[int] = []

            def encode(self, video):
                del video
                idx = len(self.encode_calls)
                self.encode_calls.append(idx)
                return encoded[idx]

            def close(self):
                return b""

        mocker.patch(
            "vllm_omni.entrypoints.openai.serving_video_output_stream.create_streaming_video_encoder",
            lambda *, output_format, fps, video_codec_options=None: FakeStreamingVideoEncoder(),
        )
        mocker.patch(
            "vllm_omni.entrypoints.openai.serving_video_output_stream.get_stage_type",
            return_value="diffusion",
        )
        mocker.patch(
            "vllm_omni.entrypoints.openai.serving_video_output_stream.build_stage_sampling_params_list",
            return_value=[OmniDiffusionSamplingParams()],
        )
        mocker.patch(
            "vllm_omni.entrypoints.openai.serving_video_output_stream.get_default_sampling_params_list",
            return_value=[OmniDiffusionSamplingParams()],
        )

        handler = OmniStreamingVideoOutputHandler(
            engine_client=omni,
            model_name="test-model",
            stage_configs=bridge_engine.stage_configs,
            stall_timeout=5.0,
            start_timeout=5.0,
        )
        app = FastAPI()

        @app.websocket("/v1/realtime/video")
        async def ws_endpoint(websocket: WebSocket):
            await handler.handle_session(websocket)

        try:
            with TestClient(app) as client:
                with client.websocket_connect("/v1/realtime/video") as ws:
                    ws.send_json({"type": "session.start", "prompt": "integration test"})
                    assert ws.receive_json()["type"] == "video.start"
                    assert ws.receive_bytes() == b"pipeline-fmp4-0"
                    assert ws.receive_bytes() == b"pipeline-fmp4-1"
                    done = ws.receive_json()
                    assert done["type"] == "session.done"
                    assert done["chunks"] == 2

            await _wait_for(lambda: len(pipeline.requests) == 1)
            assert pipeline.requests[0].request_id
        finally:
            await self._shutdown_pipeline_omni_harness(omni, fixture, inline_client)

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

    @classmethod
    def _make_inline_pipeline_client(cls, pipeline: _StepStreamingPipeline) -> InlineStageDiffusionClient:
        """Build an inline diffusion stage client whose engine runs the given mock pipeline."""
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
        pipeline_engine = _PipelineBackedEngine(pipeline)
        with patch.object(InlineStageDiffusionClient, "_enrich_config"):
            with patch(
                "vllm_omni.diffusion.inline_stage_diffusion_client.DiffusionEngine.make_engine",
                return_value=pipeline_engine,
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
        omni.engine = engine
        omni.log_stats = False
        omni._pause_cond = asyncio.Condition()
        omni._paused = False
        omni.request_states = {}
        omni.final_output_task = None
        omni.event_resolver = AsyncEventResolver()
        omni._enable_ar_profiler = False
        omni._is_sleeping = False
        omni.prom_metrics = MagicMock()
        omni.mod_metrics = MagicMock()
        omni.resolve_sampling_params_list = lambda params, allow_delta_coercion: params
        omni._compute_final_stage_id = lambda output_modalities: 0
        omni._compute_final_output_stage_ids = lambda output_modalities: [0]
        omni.default_sampling_params_list = engine.default_sampling_params_list
        omni._log_summary_and_cleanup = lambda request_id: omni.request_states.pop(request_id, None)
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


def _make_vllm_config():
    @contextlib.contextmanager
    def set_priority(*args, **kwargs):
        del args, kwargs
        yield

    return SimpleNamespace(
        kernel_config=SimpleNamespace(ir_op_priority=SimpleNamespace(set_priority=set_priority)),
        compilation_config=SimpleNamespace(ir_enable_torch_wrap=True),
    )


class TestSupportedPipelines:
    """Streaming-output protocol checks for supported pipelines."""

    def test_helios_supports_step_execution_for_streaming_output(self) -> None:
        from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
        from vllm_omni.diffusion.models.interface import SupportsStepExecution, supports_step_execution

        # Avoid loading model weights; protocol membership depends on the class contract.
        pipeline = object.__new__(HeliosPipeline)

        assert pipeline.supports_step_execution is True
        assert supports_step_execution(pipeline) is True
        assert isinstance(pipeline, SupportsStepExecution) is True

    def test_load_model_rejects_streaming_output_without_step_execution(self, monkeypatch) -> None:
        class _NoStepPipeline:
            def forward(self): ...

        class _FakeDiffusersPipelineLoader:
            def __init__(self, *args, **kwargs): ...

            def load_model(self, **kwargs):
                return _NoStepPipeline()

        class _FakeDeviceMemoryProfiler:
            consumed_memory = 0

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        runner = object.__new__(DiffusionModelRunner)
        runner.vllm_config = _make_vllm_config()
        runner.od_config = SimpleNamespace(
            enable_cpu_offload=False,
            enable_layerwise_offload=False,
            enforce_eager=True,
            cache_backend=None,
            cache_config=None,
            streaming_output=True,
            step_execution=False,
            model_class_name="NoStepPipeline",
            parallel_config=SimpleNamespace(use_hsdp=False),
        )
        runner.device = torch.device("cpu")
        runner.pipeline = None
        runner.cache_backend = None
        runner.offload_backend = None
        runner.kv_transfer_manager = SimpleNamespace()

        monkeypatch.setattr(model_runner_module, "DiffusersPipelineLoader", _FakeDiffusersPipelineLoader)
        monkeypatch.setattr(model_runner_module, "DeviceMemoryProfiler", _FakeDeviceMemoryProfiler)
        monkeypatch.setattr(model_runner_module, "get_offload_backend", lambda *args, **kwargs: None)
        monkeypatch.setattr(model_runner_module, "get_cache_backend", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match="NoStepPipeline"):
            DiffusionModelRunner.load_model(runner)
