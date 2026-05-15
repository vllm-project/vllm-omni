# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import ExitStack

import pytest

from tests.e2e.offline_inference.custom_pipeline.worker_extension import (
    vLLMOmniColocateWorkerExtensionForTest,
)
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension
from vllm_omni.entrypoints.async_omni import AsyncOmni

CUSTOM_PIPELINE_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob.QwenImagePipelineWithLogProbForTest"
)
WORKER_EXTENSION_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.worker_extension.vLLMOmniColocateWorkerExtensionForTest"
)
MODEL = "tiny-random/Qwen-Image"

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_worker_extension_inheritance():
    assert issubclass(vLLMOmniColocateWorkerExtensionForTest, CustomPipelineWorkerExtension)


@pytest.mark.cpu
def test_worker_extension_test_function():
    assert (
        vLLMOmniColocateWorkerExtensionForTest.test_extension_name() == "vllm-omni-colocate-worker-extension-for-test"
    )


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_collective_rpc_test_extension_name():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        result = await engine.collective_rpc(method="test_extension_name")

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert result == [["vllm-omni-colocate-worker-extension-for-test"]]
