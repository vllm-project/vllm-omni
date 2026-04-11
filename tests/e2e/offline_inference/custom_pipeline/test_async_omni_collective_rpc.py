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

import uuid
from contextlib import ExitStack

import pytest

from tests.utils import hardware_test
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
