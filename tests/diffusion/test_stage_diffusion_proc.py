# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_process_batch_request_preserves_parent_request_id_and_kv_sender_info():
    async def run_test():
        captured = {}

        def step(request):
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
