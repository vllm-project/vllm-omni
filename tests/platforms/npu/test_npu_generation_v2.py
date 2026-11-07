# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend generation selection, warmup and platform hook contracts."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from vllm_omni.platforms.npu.worker import npu_generation_worker as worker_module
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_generation_model_runner_v2 import NPUGenerationModelRunnerV2
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("use_v2", [False, True])
def test_worker_selects_stage_runner(monkeypatch, use_v2):
    from vllm_omni.platforms.npu.worker import npu_generation_model_runner_v2 as runner_module

    worker = object.__new__(worker_module.NPUGenerationWorker)
    worker.use_v2_model_runner = use_v2
    worker.vllm_config = Mock()
    worker._init_device = Mock(return_value=torch.device("cpu"))
    worker.model_runner_cls = Mock()
    v2 = Mock()
    monkeypatch.setattr(runner_module, "NPUGenerationModelRunnerV2", v2)
    monkeypatch.setattr(worker_module, "init_workspace_manager", Mock())
    worker.init_device()
    chosen, other = (v2, worker.model_runner_cls) if use_v2 else (worker.model_runner_cls, v2)
    chosen.assert_called_once_with(worker.vllm_config, worker.device)
    other.assert_not_called()
    assert worker.model_runner is chosen.return_value


@pytest.mark.parametrize("use_v2", [False, True])
def test_generation_warmup_avoids_language_model_kernels(monkeypatch, use_v2):
    worker = object.__new__(worker_module.NPUGenerationWorker)
    worker.use_v2_model_runner = use_v2
    worker.model_runner = Mock()
    worker._capture_auxiliary_graphs = Mock()
    parent_warmup = Mock()
    monkeypatch.setattr(OmniNPUWorkerBase, "compile_or_warm_up_model", parent_warmup)
    result = worker.compile_or_warm_up_model()
    if use_v2:
        parent_warmup.assert_not_called()
        worker.model_runner.profile_run.assert_called_once_with()
        worker._capture_auxiliary_graphs.assert_called_once_with()
        assert result.language_model >= 0
        assert result.encoder == 0
    else:
        parent_warmup.assert_called_once_with()
        worker.model_runner.profile_run.assert_not_called()
        assert result is parent_warmup.return_value


def test_generation_uses_ascend_batches_and_shared_lifecycle(monkeypatch):
    runner = object.__new__(NPUGenerationModelRunnerV2)
    prepare = Mock()
    monkeypatch.setattr(NPUModelRunner, "prepare_inputs", prepare)
    scheduler, state, descriptor = Mock(), Mock(), Mock()
    assert runner.prepare_inputs(scheduler, state, descriptor, 0) is prepare.return_value
    prepare.assert_called_once_with(runner, scheduler, state, descriptor, 0)
    runner.input_buffers = Mock()
    make_dummy = Mock()
    monkeypatch.setattr(AscendInputBatch, "make_dummy", make_dummy)
    assert runner._make_dummy_input_batch(2, 32, max_query_len=16, is_padding=False) is make_dummy.return_value
    make_dummy.assert_called_once_with(2, 32, runner.input_buffers, max_query_len=16)
    assert NPUGenerationModelRunnerV2.execute_model is OmniGenerationModelRunner.execute_model
    assert NPUGenerationModelRunnerV2.sample_tokens is OmniGenerationModelRunner.sample_tokens
    assert NPUGenerationModelRunnerV2.profile_run is OmniGenerationModelRunner.profile_run


def test_generation_enters_ascend_forward_context(monkeypatch):
    from vllm_ascend import ascend_forward_context

    runner = object.__new__(NPUGenerationModelRunnerV2)
    runner.model = Mock()
    config = Mock()
    enter = Mock()
    forward_context = Mock(return_value=nullcontext(enter))
    monkeypatch.setattr(ascend_forward_context, "set_ascend_forward_context", forward_context)
    with runner._forward_context(
        None,
        config,
        num_tokens=32,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        slot_mapping=None,
        is_padding=False,
    ) as context:
        assert context is enter
    forward_context.assert_called_once_with(
        None,
        config,
        num_tokens=32,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        model_instance=runner.model,
    )
