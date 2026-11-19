# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The stage runner selection overrides the upstream environment default."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

from vllm_omni.platforms.npu.worker.base import NPUWorker, OmniNPUWorkerBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("upstream_v2", [False, True])
@pytest.mark.parametrize("stage_v2", [False, True, None])
def test_stage_runner_overrides_upstream_default(monkeypatch, upstream_v2, stage_v2):
    model_config = SimpleNamespace()
    if stage_v2 is not None:
        model_config.use_v2_model_runner = stage_v2
    config = SimpleNamespace(model_config=model_config, profiler_config=None)

    def init_upstream(worker):
        worker.vllm_config = config
        worker.use_v2_model_runner = upstream_v2

    monkeypatch.setattr(NPUWorker, "__init__", init_upstream)
    worker = OmniNPUWorkerBase()
    assert worker.use_v2_model_runner is (stage_v2 is True)
