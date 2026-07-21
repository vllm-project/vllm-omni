# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the Step-Audio2 Ascend HiFT patch."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    fake_logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm_logger = types.ModuleType("vllm.logger")
    fake_vllm_logger.init_logger = lambda _name: fake_logger
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_vllm_logger)

    root = next(parent for parent in Path(__file__).resolve().parents if (parent / "vllm_omni").is_dir())
    path = root / "vllm_omni" / "platforms" / "npu" / "models" / "step_audio2_token2wav.py"
    spec = importlib.util.spec_from_file_location("test_step_audio2_token2wav_npu_patch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_f02sine(sine_gen, f0_values: torch.Tensor) -> torch.Tensor:
    rad_values = (f0_values / sine_gen.sampling_rate) % 1
    rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
    rand_ini[:, 0] = 0
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
    rad_values = F.interpolate(
        rad_values.transpose(1, 2),
        scale_factor=1 / sine_gen.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
    phase = F.interpolate(
        phase.transpose(1, 2) * sine_gen.upsample_scale,
        scale_factor=sine_gen.upsample_scale,
        mode="linear",
    ).transpose(1, 2)
    return torch.sin(phase)


def test_even_scale_downsample_matches_linear_interpolate(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)
    x = torch.randn(2, 3, 960)

    expected = F.interpolate(x, scale_factor=1 / 4, mode="linear")
    actual = module._linear_downsample_even_scale(x, 4)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("shape", "scale"), [((1, 2, 12), 3), ((1, 2, 10), 4)])
def test_even_scale_downsample_rejects_unsupported_shapes(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    scale: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    with pytest.raises(ValueError):
        module._linear_downsample_even_scale(torch.randn(shape), scale)


def test_hift_patch_is_exact_idempotent_and_stays_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSineGen:
        sampling_rate = 24000
        upsample_scale = 4
        flag_for_pulse = False

        def __init__(self):
            self.original_calls = 0

        def _f02sine(self, f0_values):
            self.original_calls += 1
            return _reference_f02sine(self, f0_values)

    class FakeHiFT:
        def __init__(self):
            self.m_source = SimpleNamespace(l_sin_gen=FakeSineGen())

        def to(self, *_args, **_kwargs):
            raise AssertionError("HiFT must stay on its accelerator")

    hift = FakeHiFT()
    module.patch_step_audio2_hift_for_npu(hift)
    patched_method = hift.m_source.l_sin_gen._f02sine
    module.patch_step_audio2_hift_for_npu(hift)
    assert hift.m_source.l_sin_gen._f02sine is patched_method

    f0_values = torch.rand(1, 16, 3)
    torch.manual_seed(7)
    expected = _reference_f02sine(hift.m_source.l_sin_gen, f0_values)
    torch.manual_seed(7)
    actual = patched_method(f0_values)

    assert hift.m_source.l_sin_gen.original_calls == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("flag_for_pulse", "upsample_scale", "input_length"),
    [
        (True, 4, 16),
        (False, 3, 12),
        (False, 4.5, 18),
        (False, 4, 10),
    ],
)
def test_hift_patch_delegates_unsupported_cases_to_original_f02sine_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    flag_for_pulse: bool,
    upsample_scale: float,
    input_length: int,
) -> None:
    module = _load_patch_module(monkeypatch)
    calls: list[str] = []

    class FakeSineGen:
        def __init__(self):
            self.flag_for_pulse = flag_for_pulse
            self.upsample_scale = upsample_scale

        def _f02sine(self, f0_values):
            calls.append(f0_values.device.type)
            return f0_values + 1

    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=FakeSineGen()))
    module.patch_step_audio2_hift_for_npu(hift)

    f0_values = torch.randn(1, input_length, 3)
    output = hift.m_source.l_sin_gen._f02sine(f0_values)

    assert calls == ["cpu"]
    assert output.device == f0_values.device
    torch.testing.assert_close(output, f0_values + 1)


@pytest.mark.parametrize("upsample_scale", [0, -2])
def test_hift_patch_rejects_non_positive_scale(
    monkeypatch: pytest.MonkeyPatch,
    upsample_scale: int,
) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeSineGen:
        flag_for_pulse = False

        def __init__(self):
            self.upsample_scale = upsample_scale

        def _f02sine(self, f0_values):
            return f0_values

    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=FakeSineGen()))
    module.patch_step_audio2_hift_for_npu(hift)

    with pytest.raises(ValueError, match="upsample_scale must be positive"):
        hift.m_source.l_sin_gen._f02sine(torch.randn(1, 16, 3))


def test_hift_patch_rejects_causal_sinegen(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)

    class FakeCausalSineGen:
        causal = True

        def _f02sine(self, f0_values):
            return f0_values

    hift = SimpleNamespace(m_source=SimpleNamespace(l_sin_gen=FakeCausalSineGen()))

    with pytest.raises(ValueError, match="only supports non-causal SineGen2"):
        module.patch_step_audio2_hift_for_npu(hift)


def test_hift_patch_reports_incompatible_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_patch_module(monkeypatch)

    with pytest.raises(TypeError, match=r"m_source\.l_sin_gen\._f02sine"):
        module.patch_step_audio2_hift_for_npu(SimpleNamespace())
