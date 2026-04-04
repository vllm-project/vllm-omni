# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _make_proc(engine: object) -> StageDiffusionProc:
    """Create a lightweight StageDiffusionProc with a test utility executor."""
    proc = StageDiffusionProc(model="mock-model", od_config=Mock())
    proc._engine = engine
    proc._utility_executor = ThreadPoolExecutor(max_workers=2)
    proc._closed = False
    if not hasattr(engine, "close"):
        engine.close = Mock()
    return proc


@pytest.mark.asyncio
async def test_process_request_submits_second_request_while_first_waits() -> None:
    """Ensure proc request submission no longer serializes on one step thread."""
    first_future: Future[DiffusionOutput] = Future()
    second_future: Future[DiffusionOutput] = Future()
    first_submitted = asyncio.Event()
    second_submitted = asyncio.Event()
    submit_calls: list[str] = []

    def prepare_request(request):
        return request, 0.0

    def materialize_outputs(
        request,
        output,
        preprocess_time,
        exec_total_time,
        diffusion_engine_start_time,
    ):
        del preprocess_time, exec_total_time, diffusion_engine_start_time
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request.request_ids[0],
                images=[output.output],
                prompt=request.prompts[0],
            )
        ]

    def submit_request(request):
        req_id = request.request_ids[0]
        submit_calls.append(req_id)
        if req_id == "req-a":
            first_submitted.set()
            return first_future
        second_submitted.set()
        return second_future

    engine = SimpleNamespace(
        _prepare_step_request=prepare_request,
        _materialize_step_outputs=materialize_outputs,
        submit_request=submit_request,
        close=Mock(),
    )
    proc = _make_proc(engine)
    proc._reconstruct_sampling_params = Mock(return_value=OmniDiffusionSamplingParams())

    try:
        task_a = asyncio.create_task(proc._process_request("req-a", "prompt-a", {}))
        await asyncio.wait_for(first_submitted.wait(), timeout=1)

        task_b = asyncio.create_task(proc._process_request("req-b", "prompt-b", {}))
        await asyncio.wait_for(second_submitted.wait(), timeout=1)

        assert submit_calls == ["req-a", "req-b"]
        assert task_a.done() is False

        second_future.set_result(DiffusionOutput(output="image-b"))
        first_future.set_result(DiffusionOutput(output="image-a"))

        result_b = await asyncio.wait_for(task_b, timeout=1)
        result_a = await asyncio.wait_for(task_a, timeout=1)

        assert result_a.request_id == "req-a"
        assert result_b.request_id == "req-b"
        assert result_a.images == ["image-a"]
        assert result_b.images == ["image-b"]
    finally:
        proc.close()


@pytest.mark.asyncio
async def test_process_batch_request_preserves_batch_merge_contract() -> None:
    """Ensure batch postprocessing still merges per-prompt outputs correctly."""
    engine = SimpleNamespace(close=Mock())
    proc = _make_proc(engine)
    proc._reconstruct_sampling_params = Mock(return_value=OmniDiffusionSamplingParams())

    first = OmniRequestOutput.from_diffusion(
        request_id="batch",
        images=["image-1"],
        prompt="prompt-1",
        metrics={"first_metric": 1},
        multimodal_output={"audio": "audio-1"},
        stage_durations={"prepare": 1.0},
        peak_memory_mb=10.0,
    )
    second = OmniRequestOutput.from_diffusion(
        request_id="batch",
        images=["image-2"],
        prompt="prompt-2",
        metrics={"second_metric": 2},
        latents=torch.tensor([2.0]),
        multimodal_output={"mask": "mask-2"},
        final_output_type="audio",
        stage_durations={"decode": 2.0},
        peak_memory_mb=20.0,
    )
    proc._execute_step_request_async = AsyncMock(return_value=[first, second])

    try:
        output = await proc._process_batch_request("batch", ["prompt-1", "prompt-2"], {})

        assert output.request_id == "batch"
        assert output.images == ["image-1", "image-2"]
        assert output.prompt is None
        assert output.metrics == {"first_metric": 1, "second_metric": 2}
        assert output.multimodal_output == {"audio": "audio-1", "mask": "mask-2"}
        assert output.stage_durations == {"prepare": 1.0, "decode": 2.0}
        assert output.peak_memory_mb == 20.0
        assert output.final_output_type == "audio"
        assert torch.equal(output.latents, torch.tensor([2.0]))
    finally:
        proc.close()


@pytest.mark.asyncio
async def test_handle_collective_rpc_uses_submit_rpc_and_merges_lora_ids() -> None:
    """Ensure proc RPC bridging uses the engine future path and keeps semantics."""
    rpc_future: Future[list[list[int] | None]] = Future()
    rpc_future.set_result([[3, 1], [2, 3], None])
    submit_rpc = Mock(return_value=rpc_future)
    engine = SimpleNamespace(submit_rpc=submit_rpc, close=Mock())
    proc = _make_proc(engine)

    try:
        result = await proc._handle_collective_rpc(
            method="list_loras",
            timeout=0.5,
            args=(),
            kwargs={},
        )

        assert result == [1, 2, 3]
        submit_rpc.assert_called_once_with(
            method="list_loras",
            timeout=0.5,
            args=(),
            kwargs={},
            unique_reply_rank=None,
        )
    finally:
        proc.close()
