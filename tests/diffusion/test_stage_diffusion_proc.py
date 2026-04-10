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
    """Build a minimal ``StageDiffusionProc`` wired to a fake engine.

    The production constructor creates a real diffusion engine and helper
    executor during initialization. These tests only need the proc-side bridge
    logic, so they replace the engine with a lightweight test double and keep
    a small utility executor for the CPU-side helper path.

    Args:
        engine: Fake engine object that exposes just the attributes used by the
            proc methods under test.

    Returns:
        A proc instance whose internal engine and executor are safe to use in
        unit tests without starting a real model stack.
    """
    proc = StageDiffusionProc(model="mock-model", od_config=Mock())
    proc._engine = engine
    proc._utility_executor = ThreadPoolExecutor(max_workers=2)
    proc._closed = False
    if not hasattr(engine, "close"):
        engine.close = Mock()
    return proc


@pytest.mark.asyncio
async def test_process_request_submits_second_request_while_first_waits() -> None:
    """Verify request submission is no longer serialized by one worker thread.

    The old bridge path wrapped the entire request lifecycle inside a single
    ``run_in_executor(..., engine.step, ...)`` call. That design meant a first
    request could occupy the only worker thread and prevent a second request
    from even reaching ``engine.submit_request()``.

    This regression test keeps the first engine future unresolved, starts a
    second request, and asserts that both requests are submitted before either
    future completes. That proves the proc bridge now only offloads the CPU
    helper work and lets the engine core loop own request lifetime.
    """
    first_future: Future[DiffusionOutput] = Future()
    second_future: Future[DiffusionOutput] = Future()
    first_submitted = asyncio.Event()
    second_submitted = asyncio.Event()
    submit_calls: list[str] = []

    def prepare_request(request):
        """Return the prepared request unchanged for bridge-only testing."""
        return request, 0.0

    def materialize_outputs(
        request,
        output,
        preprocess_time,
        exec_total_time,
        diffusion_engine_start_time,
    ):
        """Convert one diffusion output into the final request output shape.

        The bridge test does not care about postprocess internals; it only
        needs a deterministic result object so the request tasks can complete
        once the controlled futures are resolved.
        """
        del preprocess_time, exec_total_time, diffusion_engine_start_time
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request.request_ids[0],
                images=[output.output],
                prompt=request.prompts[0],
            )
        ]

    def submit_request(request):
        """Record submission order and hand back a caller-controlled future."""
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
    """Verify batch request merging still matches the orchestrator contract.

    ``_process_batch_request()`` executes a single engine submission and then
    merges the per-prompt ``OmniRequestOutput`` objects into one combined
    response. This test ensures the refactor did not change how images,
    metrics, multimodal payloads, latents, durations, and output type are
    folded back into the batch-level response.
    """
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
    """Verify collective RPC now uses ``submit_rpc`` without changing behavior.

    The proc should no longer route RPCs through a dedicated step executor.
    Instead it should forward them to the engine's future-based RPC bridge and
    preserve method-specific result semantics. ``list_loras`` is a good probe
    because it requires both dispatch correctness and post-processing of
    multi-worker replies into a sorted unique LoRA id list.
    """
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


@pytest.mark.asyncio
async def test_process_batch_request_preserves_parent_request_id_and_kv_sender_info() -> None:
    """Preserve parent request metadata when building batched engine requests."""
    engine = SimpleNamespace(close=Mock())
    proc = _make_proc(engine)
    proc._reconstruct_sampling_params = Mock(return_value=OmniDiffusionSamplingParams())
    captured: dict[str, object] = {}

    async def execute_step_request_async(request):
        """Capture the built engine request and return two prompt results."""
        captured["request"] = request
        return [
            OmniRequestOutput.from_diffusion(
                request_id="req-parent-0",
                images=["img-1"],
                prompt="hello",
            ),
            OmniRequestOutput.from_diffusion(
                request_id="req-parent-1",
                images=["img-2"],
                prompt="world",
            ),
        ]

    proc._execute_step_request_async = AsyncMock(side_effect=execute_step_request_async)

    try:
        result = await proc._process_batch_request(
            request_id="req-parent",
            prompts=["hello", "world"],
            sampling_params_dict={},
            kv_sender_info={0: {"host": "10.0.0.2", "zmq_port": 50151}},
        )

        request = captured["request"]
        assert request.request_id == "req-parent"
        assert request.request_ids == ["req-parent-0", "req-parent-1"]
        assert request.kv_sender_info == {0: {"host": "10.0.0.2", "zmq_port": 50151}}
        assert result.request_id == "req-parent"
        assert result.images == ["img-1", "img-2"]
    finally:
        proc.close()
