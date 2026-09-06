# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.worker.cpu_ar_model_runner import CPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_runner() -> CPUARModelRunner:
    runner = object.__new__(CPUARModelRunner)
    model_config = SimpleNamespace(
        engine_output_type="audio",
        async_chunk=True,
        enable_return_routed_experts=False,
    )
    runner.vllm_config = SimpleNamespace(model_config=model_config)
    runner.model_config = model_config
    runner.omni_prefix_cache = None
    runner.speculative_config = None
    runner.use_async_scheduling = True
    runner.model = SimpleNamespace(has_postprocess=False, use_async_omni_output=True)
    return runner


def test_cpu_ar_model_runner_never_uses_async_omni_output():
    runner = _make_runner()

    # Same flags that make GPUARModelRunner._should_use_async_omni_output
    # return True; CPUARModelRunner must still refuse, since building
    # OmniAsyncGPUModelRunnerOutput unconditionally touches torch.cuda APIs
    # that don't exist without a CUDA device.
    assert not CPUARModelRunner._should_use_async_omni_output(runner)
