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
    pytest tests/e2e/features/custom_pipeline/test_async_omni_collective_rpc.py -v -s
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
CUSTOM_PIPELINE_CLASS = "tests.e2e.features.helpers.custom_pipeline.QwenImagePipelineWithLogProbForTest"
WORKER_EXTENSION_CLASS = "tests.e2e.features.helpers.custom_pipeline.vLLMOmniColocateWorkerExtensionForTest"


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
    sleep(level=1). The tracked allocations must transition to the allocator's
    asleep state and report physical reclamation.
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

        # Measure CuMemAllocator-tracked usage before sleep.  In inline / uni
        # mode the worker runs in this process, so the allocator singleton is
        # shared and can be read directly.
        allocator = None
        tracked_before = 0
        tracked_ptrs: set[int] = set()
        try:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            tracked_ptrs = {ptr for ptr, data in allocator.pointer_to_data.items() if not data.is_asleep}
            tracked_before = sum(allocator.pointer_to_data[ptr].handle[1] for ptr in tracked_ptrs)
        except Exception:
            pass

        slept = False
        try:
            # Put the engine to sleep; this engine's weights should be
            # offloaded via the pool.
            acks = await engine.sleep(level=1)
            slept = True
            await asyncio.sleep(0.5)  # allow the CUDA driver to settle
            torch.accelerator.synchronize()

            # Measure after sleep.
            free_after, _ = torch.cuda.mem_get_info()
            used_after_gib = (total - free_after) / 1024**3
            drop_gib = used_before_gib - used_after_gib

            # get_current_usage() includes unmapped sleeping handles, so it
            # does not decrease on sleep. Verify that allocations mapped
            # before the call instead transitioned to the asleep state.
            if allocator is not None:
                assert tracked_before > 0, (
                    "Expected custom_pipeline weights to use the CuMem pool, "
                    "but no mapped allocations were tracked before sleep."
                )
                still_awake = [
                    ptr
                    for ptr in tracked_ptrs
                    if ptr in allocator.pointer_to_data and not allocator.pointer_to_data[ptr].is_asleep
                ]
                assert not still_awake, (
                    f"{len(still_awake)} CuMem allocation(s) remained mapped "
                    "after sleep(level=1) on the custom_pipeline path."
                )

            # Physical VRAM or ACK freed_bytes confirms reclamation at the
            # driver level.
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

            assert await engine.is_sleeping()
            await engine.wake_up()
            slept = False
            assert not await engine.is_sleeping()
        finally:
            if slept:
                await engine.wake_up()
