# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NPU async omni output (OmniAsyncNPUModelRunnerOutput).

Mirrors the GPU tests in tests/worker/test_gpu_ar_model_runner.py but for the
NPU deferred-builder class. Bypass __init__ via object.__new__ so the tests
run on CPU without an NPU device.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms.npu.worker.npu_ar_model_runner import (
    OmniAsyncNPUModelRunnerOutput,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_async_output(
    builder=None,
    *,
    sampled_token_ids=(7,),
    max_gen_len=None,
):
    async_output = object.__new__(OmniAsyncNPUModelRunnerOutput)
    async_output._model_runner_output = None
    async_output._model_runner_output_builder = builder
    async_output._background_thread = None
    async_output._background_exception = None
    async_output._invalid_req_indices = []
    async_output._has_fault = None

    ids = torch.tensor([list(sampled_token_ids)], dtype=torch.long)
    async_output.sampled_token_ids_cpu = ids
    async_output.async_copy_ready_event = SimpleNamespace(synchronize=lambda: None)
    async_output._sampled_token_ids = ids
    async_output._logprobs_tensors = None
    async_output._logprobs_tensors_cpu = None
    async_output._routed_experts = None
    async_output._routed_experts_cpu = None
    async_output.vocab_size = 10
    return async_output


def test_omni_async_npu_model_runner_output_builds_lazily_once():
    calls = []

    def builder():
        calls.append("build")
        return OmniModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0})

    async_output = _make_async_output(builder)

    output = async_output.get_output()

    assert calls == ["build"]
    assert async_output._model_runner_output_builder is None
    assert output.req_ids == ["r1"]
    assert output.sampled_token_ids == [[7]]


def test_omni_async_npu_model_runner_output_reraises_background_exception():
    async_output = object.__new__(OmniAsyncNPUModelRunnerOutput)
    joined = []

    class FakeThread:
        def join(self):
            joined.append("join")

    async_output._background_thread = FakeThread()
    async_output._background_exception = RuntimeError("background failed")
    async_output._model_runner_output = None
    async_output._model_runner_output_builder = lambda: None
    async_output._has_fault = None

    with pytest.raises(RuntimeError, match="background failed"):
        async_output.get_output()

    assert joined == ["join"]
    assert async_output._background_thread is None


def test_omni_async_npu_model_runner_output_builder_runs_once():
    # The builder runs exactly once; get_output() finalizes the sampled token
    # ids via the superclass (AsyncGPUModelRunnerOutput.get_output).
    calls = []

    def builder():
        calls.append("build")
        return OmniModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0})

    async_output = _make_async_output(builder)
    async_output.get_output()

    assert calls == ["build"]
    assert async_output._model_runner_output is not None
