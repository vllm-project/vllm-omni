# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NPU profiler call wiring.

``NPUTorchProfilerWrapper._create_profiler`` builds the
``torch_npu.profiler.profile`` call. These tests install a fake ``torch_npu``
module and assert the keyword arguments the wrapper forwards, so the contract is
checked on a CPU host without an Ascend device.

Regression guard for #8877: ``torch_profiler_with_stack`` was dropped on the NPU
path -- ``with_stack`` was hardcoded to ``False`` and the flag was only mapped
onto ``with_modules`` -- so NPU traces never carried Python stacks, while the
same configuration produced them on the CUDA path.
"""

from __future__ import annotations

import sys
import types

import pytest

from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_torch_npu(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install a fake ``torch_npu`` and return the kwargs of the profile call."""
    forwarded: dict[str, object] = {}

    class _RecordingProfile:
        def __init__(self, **kwargs: object) -> None:
            forwarded.update(kwargs)

    profiler = types.SimpleNamespace(
        ProfilerActivity=types.SimpleNamespace(CPU="CPU", NPU="NPU"),
        _ExperimentalConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        ExportType=types.SimpleNamespace(Text="Text"),
        ProfilerLevel=types.SimpleNamespace(Level1=1),
        AiCMetrics=types.SimpleNamespace(PipeUtilization="PipeUtilization"),
        profile=_RecordingProfile,
        tensorboard_trace_handler=lambda path: types.SimpleNamespace(path=path),
    )
    module = types.ModuleType("torch_npu")
    module.profiler = profiler  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return forwarded


def _build_wrapper(tmp_path, **overrides):
    from vllm.config import ProfilerConfig

    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(tmp_path),
        **overrides,
    )
    return NPUTorchProfilerWrapper(config, worker_name="stage_0", local_rank=0)


def test_npu_profiler_forwards_with_stack_when_requested(monkeypatch, tmp_path):
    forwarded = _install_fake_torch_npu(monkeypatch)
    _build_wrapper(tmp_path, torch_profiler_with_stack=True)

    assert forwarded["with_stack"] is True


def test_npu_profiler_keeps_with_stack_off_when_disabled(monkeypatch, tmp_path):
    forwarded = _install_fake_torch_npu(monkeypatch)
    _build_wrapper(tmp_path, torch_profiler_with_stack=False)

    assert forwarded["with_stack"] is False


def test_npu_profiler_honours_the_config_default(monkeypatch, tmp_path):
    forwarded = _install_fake_torch_npu(monkeypatch)
    _build_wrapper(tmp_path)

    assert forwarded["with_stack"] is True
    assert forwarded["with_modules"] is True


def test_npu_profiler_forwards_record_shapes_and_memory(monkeypatch, tmp_path):
    forwarded = _install_fake_torch_npu(monkeypatch)
    _build_wrapper(
        tmp_path,
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=True,
    )

    assert forwarded["record_shapes"] is True
    assert forwarded["profile_memory"] is True
