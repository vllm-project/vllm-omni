# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_process_batch_request_preserves_parent_request_id_and_kv_sender_info():
    async def run_test():
        captured = {}

        async def step(request):
            captured["request"] = request
            return [
                SimpleNamespace(
                    images=["img-1"],
                    _multimodal_output={},
                    _custom_output={},
                    metrics={},
                    stage_durations={},
                    peak_memory_mb=0.0,
                    latents=None,
                    trajectory_latents=None,
                    trajectory_timesteps=None,
                    trajectory_log_probs=None,
                    trajectory_decoded=None,
                    final_output_type="image",
                ),
                SimpleNamespace(
                    images=["img-2"],
                    _multimodal_output={},
                    _custom_output={},
                    metrics={},
                    stage_durations={},
                    peak_memory_mb=0.0,
                    latents=None,
                    trajectory_latents=None,
                    trajectory_timesteps=None,
                    trajectory_log_probs=None,
                    trajectory_decoded=None,
                    final_output_type="image",
                ),
            ]

        proc = object.__new__(StageDiffusionProc)
        proc._engine = SimpleNamespace(step=step)
        proc._executor = ThreadPoolExecutor(max_workers=1)

        try:
            result = await proc._process_batch_request(
                request_id="req-parent",
                prompts=["hello", "world"],
                sampling_params_dict=asdict(OmniDiffusionSamplingParams()),
                kv_sender_info={0: {"host": "10.0.0.2", "zmq_port": 50151}},
            )
        finally:
            proc._executor.shutdown(wait=True)

        request = captured["request"]
        assert request.request_id == "req-parent"
        assert request.request_ids == ["req-parent-0", "req-parent-1"]
        assert request.kv_sender_info == {0: {"host": "10.0.0.2", "zmq_port": 50151}}
        assert result.request_id == "req-parent"
        assert result.images == ["img-1", "img-2"]

    asyncio.run(run_test())


@dataclass
class MockOmniRequestOutput:
    request_id: str = ""
    status: str = "success"


BASE_HEIGHT = 512
BASE_WIDTH = 512
BASE_INFER_STEPS = 10
DELAY_BASE = 2


class MockDiffusionEngine:
    async def step(self, request):
        def simulate_step_delay(height, width, num_inference_steps) -> float:
            return (height / BASE_HEIGHT) * (width / BASE_WIDTH) * (num_inference_steps / BASE_INFER_STEPS)

        DELAY_BASE = 2
        delay_scale = simulate_step_delay(
            request.sampling_params.height, request.sampling_params.width, request.sampling_params.num_inference_steps
        )
        delay = DELAY_BASE + delay_scale * DELAY_BASE
        await asyncio.sleep(delay)
        return [MockOmniRequestOutput(request_id=request.request_id)]


@pytest.mark.asyncio
async def test_proc_process_request_with_batching_async_output():
    stage_proc = object.__new__(StageDiffusionProc)
    stage_proc._engine = MockDiffusionEngine()

    test_requests = [
        {
            "request_id": "req_1",
            "prompt": "prompt1",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 1},
        },
        {
            "request_id": "req_2",
            "prompt": "prompt2",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 2},
        },
        {
            "request_id": "req_3",
            "prompt": "prompt3",
            "params": {"height": BASE_HEIGHT * 2, "width": BASE_WIDTH * 2, "num_inference_steps": BASE_INFER_STEPS * 3},
        },
    ]

    async def run_task(req_data):
        start_time = time.time()
        result = await stage_proc._process_request(
            request_id=req_data["request_id"], prompt=req_data["prompt"], sampling_params_dict=req_data["params"]
        )
        end_time = time.time()
        return result, end_time - start_time

    coros = [run_task(req) for req in test_requests]
    results = await asyncio.gather(*coros)

    assert len(results) == len(test_requests)
    base_time = DELAY_BASE
    time_gap_std = DELAY_BASE * 2 * 2 * 1  # height/width/steps infer time scale
    eps = 0.5
    for i, (res, elapsed_time) in enumerate(results):
        assert res.request_id == test_requests[i]["request_id"]
        assert isinstance(res, MockOmniRequestOutput)
        time_gap = elapsed_time - base_time
        assert time_gap > time_gap_std - eps and time_gap < time_gap_std + eps
        base_time = elapsed_time
