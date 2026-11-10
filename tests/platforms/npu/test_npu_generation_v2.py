# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend generation forward-context contract."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode

from vllm_omni.platforms.npu.worker.npu_generation_model_runner_v2 import NPUGenerationModelRunnerV2

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
