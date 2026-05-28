# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Regression tests for AsyncOmni collective_rpc in inline diffusion mode.

When AsyncOmni runs a single diffusion stage it activates "inline diffusion
mode", which skips stage worker subprocess creation and therefore never
attaches IPC queues (_in_q / _out_q) to the OmniStage.  Methods like
list_loras(), add_lora(), sleep(), wake_up() all delegate to
collective_rpc(), which must handle this mode correctly instead of
trying to use the non-existent queues.

This is the same code path that verl's vLLMOmniHttpServer.generate()
exercises when it calls ``await self.engine.list_loras()`` before
dispatching a generation request.

Usage:
    pytest tests/e2e/offline_inference/custom_pipeline/test_async_omni_collective_rpc.py -v -s
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import ExitStack

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

MODEL = "tiny-random/Qwen-Image"
CUSTOM_PIPELINE_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob.QwenImagePipelineWithLogProbForTest"
)
WORKER_EXTENSION_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.worker_extension.vLLMOmniColocateWorkerExtensionForTest"
)


def _create_inline_engine() -> AsyncOmni:
    """Create an AsyncOmni instance that uses inline diffusion mode.

    A single diffusion stage triggers inline mode automatically.
    """
    engine = AsyncOmni(
        model=MODEL,
        custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
        worker_extension_cls=WORKER_EXTENSION_CLASS,
        enforce_eager=True,
    )

    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_list_loras_inline_mode():
    """list_loras() must not crash in inline diffusion mode.

    This is the exact call that vLLMOmniHttpServer.generate() makes
    before every generation request.
    """
    with ExitStack() as after:
        engine = _create_inline_engine()
        after.callback(engine.shutdown)

        result = await engine.list_loras()
        assert isinstance(result, list), f"Expected list, got {type(result)}"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_collective_rpc_inline_mode():
    """collective_rpc() must delegate to the inline engine, not stage queues."""
    with ExitStack() as after:
        engine = _create_inline_engine()
        after.callback(engine.shutdown)

        result = await engine.collective_rpc(method="list_loras")
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) == 1, "Inline mode has exactly one stage"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_sleep_wake_up_inline_mode():
    """sleep() and wake_up() must work in inline diffusion mode."""
    with ExitStack() as after:
        engine = _create_inline_engine()
        after.callback(engine.shutdown)

        await engine.sleep(level=1)
        assert await engine.is_sleeping()

        await engine.wake_up()
        assert not await engine.is_sleeping()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_generate_after_list_loras_inline_mode():
    """Full flow: list_loras() then generate(), matching vLLMOmniHttpServer.

    This reproduces the exact sequence that caused the original crash:
    1. list_loras() (was crashing with AssertionError on _out_q)
    2. generate() (should succeed)
    """
    with ExitStack() as after:
        engine = _create_inline_engine()
        after.callback(engine.shutdown)

        # Step 1: list_loras (the call that was crashing)
        loras = await engine.list_loras()
        assert isinstance(loras, list)

        # Step 2: generate (should still work after list_loras)
        sampling_params = OmniDiffusionSamplingParams(
            num_inference_steps=2,
            guidance_scale=0.0,
            height=256,
            width=256,
            seed=42,
        )

        last_output = None
        async for output in engine.generate(
            prompt={"prompt_ids": list(range(50))},
            request_id=f"test_after_lora_{uuid.uuid4().hex[:8]}",
            sampling_params_list=[sampling_params],
            output_modalities=["image"],
        ):
            last_output = output

        assert last_output is not None
        assert isinstance(last_output, OmniRequestOutput)
        assert last_output.images, "Expected at least one generated image"


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_sleep_memory_reclaimed_custom_pipeline():
    """sleep(level=1) must physically reclaim CuMemAllocator-tracked memory for
    custom_pipeline.

    Regression test for: custom pipelines constructed under ``with target_device:``
    (CUDA default-device context) caused safetensors >=0.20.0 to use a
    direct-to-GPU fast path (cudaMalloc via the driver API) that bypasses
    CuMemAllocator, leaving weights invisible to sleep() and pinned in GPU
    memory after the call.

    The fix moves custom_pipeline init outside the CUDA context so all weights
    go through the caching allocator and are therefore fully reclaimed by
    sleep(level=1).  A non-zero ``CuMemAllocator.get_current_usage()`` after
    sleep is the direct signal that the bypass is still occurring.
    """
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
            enable_sleep_mode=True,
        )
        after.callback(engine.shutdown)

        assert not await engine.is_sleeping(), "Engine should be awake after creation"

        # Measure global VRAM before sleep (driver view; includes inline worker
        # thread since inline mode runs in the same process).
        torch.accelerator.synchronize()
        free_before, total = torch.cuda.mem_get_info()
        used_before_gib = (total - free_before) / 1024**3

        # Measure CuMemAllocator-tracked usage before sleep.  In inline mode
        # the worker runs in a thread pool inside this process, so the allocator
        # singleton is shared and can be read directly.
        allocator = None
        tracked_before = 0
        try:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            tracked_before = allocator.get_current_usage()
        except Exception:
            pass

        # Put the engine to sleep; all weights should be offloaded via the pool.
        acks = await engine.sleep(level=1)
        await asyncio.sleep(0.5)  # allow the CUDA driver to settle
        torch.accelerator.synchronize()

        # Measure after sleep.
        free_after, _ = torch.cuda.mem_get_info()
        used_after_gib = (total - free_after) / 1024**3
        drop_gib = used_before_gib - used_after_gib

        # --- Primary assertion: allocator reports zero tracked memory. ---
        # If this fails it means weights were allocated outside the CuMem pool
        # (safetensors direct-to-GPU bypass) — the exact regression this test
        # is designed to catch.
        if allocator is not None:
            tracked_after = allocator.get_current_usage()
            assert tracked_after == 0, (
                f"CuMemAllocator still tracks {tracked_after / 1024**3:.3f} GiB "
                f"after sleep(level=1) on custom_pipeline path "
                f"(was {tracked_before / 1024**3:.3f} GiB before sleep). "
                "Weights were allocated outside the CuMem pool via the "
                "safetensors direct-to-GPU fast path — loader-context fix "
                "may not be applied."
            )

        # --- Secondary assertion: physical VRAM or ACK freed_bytes confirms
        # reclamation at the driver level.
        total_freed_bytes = sum(
            (ack.freed_bytes if hasattr(ack, "freed_bytes") else ack.get("freed_bytes", 0))
            for ack in acks
            if ack is not None
        )
        freed_gib = total_freed_bytes / 1024**3
        assert freed_gib > 0 or drop_gib > 0, (
            f"Expected GPU memory to be reclaimed after sleep(level=1) on "
            f"custom_pipeline + enable_sleep_mode=True. "
            f"CuMemAllocator tracked before={tracked_before / 1024**3:.3f} GiB, "
            f"ACK freed={freed_gib:.3f} GiB, global VRAM drop={drop_gib:.3f} GiB."
        )

        # Engine must report it is sleeping.
        assert await engine.is_sleeping()

        # Wake up and confirm the engine is functional again.
        await engine.wake_up()
        assert not await engine.is_sleeping()
