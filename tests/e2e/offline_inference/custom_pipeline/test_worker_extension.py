# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from tests.e2e.offline_inference.custom_pipeline.worker_extension import (
    vLLMOmniColocateWorkerExtensionForTest,
)
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_worker_extension_inheritance():
    assert issubclass(vLLMOmniColocateWorkerExtensionForTest, CustomPipelineWorkerExtension)


def test_worker_extension_test_function():
    assert (
        vLLMOmniColocateWorkerExtensionForTest.test_extension_name() == "vllm-omni-colocate-worker-extension-for-test"
    )
