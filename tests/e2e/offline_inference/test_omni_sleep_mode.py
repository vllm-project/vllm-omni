# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end tests for Omni Sleep Mode across various model architectures.
"""

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
MODEL_DIFF = "riverclouds/qwen_image_random"


def get_ack_info(ack, key, default=None):
    if hasattr(ack, key):
        return getattr(ack, key)
    if isinstance(ack, dict):
        return ack.get(key, default)
    return default


def get_dynamic_devices(stage_idx, num_stages, tp_size):
    total_gpus = torch.cuda.device_count()
    gpus_per_stage = tp_size
    start_idx = stage_idx * gpus_per_stage
    if start_idx + gpus_per_stage > total_gpus:
        start_idx = start_idx % total_gpus
    device_ids = [str(start_idx + i) for i in range(gpus_per_stage)]
    return ",".join(device_ids)


# Test 1: Diffusion Model (2-Stage BAGEL)
@pytest.mark.advanced_model
@pytest.mark.omni
@pytest.mark.parametrize("tp_size", [1, 2])
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_diffusion_model_sleep_tp(tp_size):
    num_gpus = torch.cuda.device_count()
    if num_gpus < tp_size:
        pytest.skip(f"Skipping TP={tp_size}")

    engine_args = {
        "model": MODEL,
        "enable_sleep_mode": True,
        "tensor_parallel_size": tp_size,
        "enforce_eager": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
    }

    engine = AsyncOmni(**engine_args, stage_init_timeout=1200)
    try:
        # BAGEL requires 2 params
        diff_sp = OmniDiffusionSamplingParams(num_inference_steps=2, height=256, width=256)
        llm_sp = SamplingParams()

        # Warmup
        async for _ in engine.generate("test", sampling_params=[llm_sp, diff_sp]):
            pass

        # Sleep all
        acks = await engine.sleep(level=2)
        statuses = [get_ack_info(ack, "status") for ack in acks]
        assert all(s == "SUCCESS" for s in statuses), f"Sleep failed. Statuses: {statuses}"

        # Wakeup & Verify
        await engine.wake_up()
        async for _ in engine.generate("verify", sampling_params=[llm_sp, diff_sp]):
            pass

        print(f"Diffusion TP={tp_size} Lifecycle OK")
    finally:
        engine.shutdown()


# Test 2: Multi-stage Manual Config
@pytest.mark.advanced_model
@pytest.mark.omni
@pytest.mark.parametrize("tp_size", [1, 2])
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_multistage_sleep_h100(tp_size):
    num_gpus = torch.cuda.device_count()
    if num_gpus < tp_size * 2:
        pytest.skip("Not enough GPUs")

    stages = []
    for i in range(2):
        devs = get_dynamic_devices(i, 2, tp_size)
        stages.append(
            {
                "stage_id": i,
                "stage_type": "llm" if i == 0 else "diffusion",
                "runtime": {"process": True, "devices": devs},
                "engine_args": {
                    "model": MODEL,
                    "model_stage": "thinker" if i == 0 else "base",
                    "tensor_parallel_size": tp_size,
                    "gpu_memory_utilization": 0.4,
                    "dtype": "bfloat16",
                    "enable_sleep_mode": True,
                    "trust_remote_code": True,
                },
            }
        )

    connectors = [{"src_stage_id": 0, "dst_stage_id": 1, "connector_type": "queue"}]

    engine = AsyncOmni(
        model=MODEL, stages=stages, connectors=connectors, enable_sleep_mode=True, stage_init_timeout=1200
    )
    try:
        sp = OmniDiffusionSamplingParams(num_inference_steps=2)
        async for _ in engine.generate("warmup", sampling_params=[SamplingParams(), sp]):
            pass

        acks = await engine.sleep(stage_ids=[0, 1], level=2)
        assert len(acks) == 2 * tp_size

        await engine.wake_up(stage_ids=[0, 1])
        async for _ in engine.generate("verify", sampling_params=[SamplingParams(), sp]):
            pass
    finally:
        engine.shutdown()


# Test 3: Pure Diffusion Single-Stage
@pytest.mark.advanced_model
@pytest.mark.omni
@pytest.mark.parametrize("tp_size", [1, 2])
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.asyncio
async def test_pure_diffusion_scenario(tp_size):
    engine_args = {
        "model": MODEL_DIFF,
        "enable_sleep_mode": True,
        "tensor_parallel_size": tp_size,
        "enforce_eager": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.5,
    }

    engine = AsyncOmni(**engine_args, stage_init_timeout=1200)
    try:
        await engine.sleep(level=1)
        await engine.wake_up()
        async for _ in engine.generate("test", sampling_params=[SamplingParams()]):
            pass
        print("Pure Diffusion OK")
    finally:
        engine.shutdown()
