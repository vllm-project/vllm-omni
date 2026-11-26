# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Schedule forwarding tests for the NPU profiler wrapper.

Regression guard for the schedule gap noted in #9286: the NPU route built no
``torch_npu.profiler`` schedule, so ``wait_iterations`` / ``warmup_iterations``
/ ``active_iterations`` were silently ignored and ``max_iterations`` was the
only auto-stop knob. These tests install a fake ``torch_npu`` and assert the
schedule object and parameters that ``NPUTorchProfilerWrapper`` builds and
forwards, so the contract is checked on a CPU host without an Ascend device.
"""

from __future__ import annotations

import sys
import types

import pytest

from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_torch_npu(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install a fake ``torch_npu`` and return what its schedule/profile saw."""
    recorded: dict[str, object] = {}
    schedule_sentinel = object()
    recorded["schedule_sentinel"] = schedule_sentinel

    def fake_schedule(**kwargs):
        recorded["schedule_kwargs"] = kwargs
        return schedule_sentinel

    class _RecordingProfile:
        def __init__(self, **kwargs: object) -> None:
            recorded.setdefault("profile_kwargs", []).append(dict(kwargs))

    profiler = types.SimpleNamespace(
        ProfilerActivity=types.SimpleNamespace(CPU="CPU", NPU="NPU"),
        _ExperimentalConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ExportType=types.SimpleNamespace(Text="Text"),
        ProfilerLevel=types.SimpleNamespace(Level1=1),
        AiCMetrics=types.SimpleNamespace(PipeUtilization="PipeUtilization"),
        schedule=fake_schedule,
        profile=_RecordingProfile,
        tensorboard_trace_handler=lambda path: types.SimpleNamespace(path=path),
    )
    module = types.ModuleType("torch_npu")
    module.profiler = profiler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return recorded


def _build_wrapper(tmp_path, **overrides):
    from vllm.config import ProfilerConfig

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        **overrides,
    )
    return NPUTorchProfilerWrapper(config, worker_name="stage_0", local_rank=0)


def test_npu_profiler_omits_schedule_by_default(monkeypatch, tmp_path):
    recorded = _install_fake_torch_npu(monkeypatch)
    _build_wrapper(tmp_path)

    assert "schedule_kwargs" not in recorded
    assert "schedule" not in recorded["profile_kwargs"][0]


def test_npu_profiler_forwards_torch_npu_schedule(monkeypatch, tmp_path):
    recorded = _install_fake_torch_npu(monkeypatch)
    wrapper = _build_wrapper(tmp_path, warmup_iterations=2, active_iterations=3)

    assert recorded["schedule_kwargs"] == {
        "skip_first": 0,
        "wait": 0,
        "warmup": 2,
        "active": 3,
        "repeat": 1,
    }
    assert recorded["profile_kwargs"][0]["schedule"] is recorded["schedule_sentinel"]
    assert wrapper._uses_schedule is True
    # torch_npu's start() does not consume schedule step 0 (verified on
    # Ascend 910B1), so every wait+warmup step is non-active — one more than
    # the torch/CUDA base-class formula of wait+warmup-1.
    assert wrapper._initial_warmup_steps_remaining == 2


def test_npu_profiler_builds_schedule_from_wait_alone(monkeypatch, tmp_path):
    recorded = _install_fake_torch_npu(monkeypatch)
    wrapper = _build_wrapper(tmp_path, wait_iterations=1, active_iterations=1)

    assert recorded["schedule_kwargs"]["wait"] == 1
    assert recorded["schedule_kwargs"]["warmup"] == 0
    # torch_npu's start() does not consume schedule step 0, so the single
    # wait step remains a non-active step() (the torch/CUDA formula gives 0).
    assert wrapper._initial_warmup_steps_remaining == 1
    # The repeat=1 cycle consumes wait + warmup + active step() calls.
    assert wrapper._schedule_cycle_steps == 2
