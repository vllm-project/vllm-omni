# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Omni NPU worker always initializes the V2 runner."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

from vllm_omni.platforms.npu.worker.base import NPUWorker, OmniNPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("upstream_v2", [False, True])
def test_npu_worker_uses_v2(monkeypatch, upstream_v2):
    config = SimpleNamespace(model_config=SimpleNamespace(), profiler_config=None)

    def init_upstream(worker):
        worker.vllm_config = config
        worker.use_v2_model_runner = upstream_v2

    monkeypatch.setattr(NPUWorker, "__init__", init_upstream)
    worker = OmniNPUWorkerBase()
    assert worker.use_v2_model_runner is True
