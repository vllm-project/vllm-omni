# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate with a Qwen-Image policy delivered as adapter-only CUDA IPC."""

import time
from contextlib import ExitStack

import numpy as np
import pytest
import torch

from tests.e2e.features.rlhf_test.test_weight_transfer_changes_output import (
    _generate_once,
    _get_diffusion_pipeline,
    _image_to_array,
    _sampling_params,
    _spawn_ipc_exporter,
)
from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni


async def _publish(engine: AsyncOmni, factors: list[tuple[str, torch.Tensor]]) -> None:
    payload, exporter = _spawn_ipc_exporter(factors)
    payload.update(peft_config={"r": 1, "lora_alpha": 8, "target_modules": ["to_q"]}, lora_int_id=1)
    try:
        await engine.start_weight_update()
        await engine.update_weights(payload)
        await engine.finish_weight_update()
    finally:
        del payload
        exporter.terminate()
        exporter.join(timeout=10)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.asyncio
async def test_qwen_image_merged_lora_changes_and_restores_generation():
    with ExitStack() as after:
        engine = AsyncOmni(
            model="tiny-random/Qwen-Image",
            enforce_eager=True,
            max_num_seqs=1,
            lora_application="merged",
            weight_transfer_config={"backend": "omni_lora_ipc"},
        )
        after.callback(engine.shutdown)
        prompt = "a beautiful sunset over the ocean with vibrant orange clouds"
        sampling = _sampling_params(seed=42)

        async def generate(request_id: str) -> np.ndarray:
            return _image_to_array(
                await _generate_once(engine, prompt, request_id=request_id, sampling_params=sampling)
            )

        baseline = await generate("merged_lora_baseline")
        pipeline = _get_diffusion_pipeline(engine)
        targets = [
            (name, module)
            for name, module in pipeline.named_modules()
            if name.startswith("transformer.") and name.endswith(".to_qkv") and hasattr(module, "input_size")
        ]
        assert targets, "No Qwen-Image transformer to_qkv projection found"
        name, projection = targets[0]
        weight = projection.weight
        base = weight.detach().clone()
        pointer = weight.data_ptr()
        # Factors ride the checkpoint-side to_q name; the engine packs them
        # onto the fused to_qkv's q rows via stacked_params_mapping.
        checkpoint_name = name.replace(".to_qkv", ".to_q")
        prefix = f"base_model.model.{checkpoint_name}"
        a = torch.ones(1, projection.input_size, dtype=torch.float16)
        q_rows = projection.total_num_heads * projection.head_size
        b = torch.full((q_rows, 1), 32, dtype=torch.float16)
        await engine.init_weight_transfer_engine({"packed": False})
        torch.accelerator.reset_peak_memory_stats()
        memory_before = (torch.accelerator.memory_allocated(), torch.accelerator.memory_reserved())
        update_started = time.perf_counter()
        await _publish(engine, [(f"{prefix}.lora_A.weight", a), (f"{prefix}.lora_B.weight", b)])
        update_seconds = time.perf_counter() - update_started
        memory_after = (
            torch.accelerator.memory_allocated(),
            torch.accelerator.memory_reserved(),
            torch.accelerator.max_memory_allocated(),
            torch.accelerator.max_memory_reserved(),
        )
        print(
            f"Qwen-Image merged IPC HBM before={memory_before} after/peak={memory_after} "
            f"update_seconds={update_seconds:.3f}",
            flush=True,
        )
        assert weight.data_ptr() == pointer
        assert not torch.equal(weight, base)
        merged = await generate("merged_lora_active")
        assert np.any(merged != baseline)

        await _publish(engine, [(f"{prefix}.lora_A.weight", a), (f"{prefix}.lora_B.weight", torch.zeros_like(b))])
        assert weight.data_ptr() == pointer
        torch.testing.assert_close(weight, base, rtol=0, atol=0)
        np.testing.assert_array_equal(await generate("merged_lora_restored"), baseline)
