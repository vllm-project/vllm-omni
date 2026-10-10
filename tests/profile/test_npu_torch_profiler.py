# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import SimpleNamespace

import pytest
from vllm.config import ProfilerConfig

from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("record_shapes", [True, False])
def test_npu_profiler_respects_record_shapes(monkeypatch, tmp_path, record_shapes):
    def fake_profile(*, record_shapes=False, **kwargs):
        return SimpleNamespace(record_shapes=record_shapes)

    fake_npu_profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="CPU", NPU="NPU"),
        _ExperimentalConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ExportType=SimpleNamespace(Text="text"),
        ProfilerLevel=SimpleNamespace(Level1=1),
        AiCMetrics=SimpleNamespace(PipeUtilization="pipe"),
        tensorboard_trace_handler=lambda path: path,
        profile=fake_profile,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=fake_npu_profiler))
    wrapper = NPUTorchProfilerWrapper.__new__(NPUTorchProfilerWrapper)
    wrapper._trace_dir = str(tmp_path)
    wrapper._worker_name = "stage0"
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        torch_profiler_record_shapes=record_shapes,
        torch_profiler_with_memory=False,
        torch_profiler_with_stack=False,
    )

    profiler = wrapper._create_profiler(config, ["CPU", "NPU"])

    assert profiler.record_shapes is record_shapes
