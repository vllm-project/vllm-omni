# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the A5 fallback; no NPU kernels are simulated.

Import unrelated activation dependencies as stubs, then exercise the real
HiFTGenerator spectral methods with real torch CPU transforms.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hifigan(monkeypatch):
    snake = ModuleType("vllm_omni.model_executor.models.common.snake_activation")
    setattr(snake, "Snake", torch.nn.Identity)
    monkeypatch.setitem(sys.modules, snake.__name__, snake)
    return load_module(
        "cosyvoice3_hifigan_under_test",
        ROOT / "vllm_omni/model_executor/models/cosyvoice3/code2wav_core/hifigan.py",
    )


@pytest.fixture(params=["HiFTGenerator", "CausalHiFTGenerator"])
def generator(hifigan, request):
    # Only the spectral methods are under test; no model weights are needed.
    cls = getattr(hifigan, request.param)
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.istft_params = {"n_fft": 16, "hop_len": 4}
    module.register_buffer("stft_window", torch.hann_window(16), persistent=False)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_fallback_matches_float32_reference(generator, dtype):
    generator._stft_on_cpu = True
    generator.stft_window = generator.stft_window.to(dtype)
    waveform = torch.linspace(-0.4, 0.4, 128).unsqueeze(0).to(dtype)
    window = generator.stft_window.float()
    expected_spec = torch.stft(waveform.float(), 16, 4, 16, window=window, return_complex=True)
    real, imag = generator._stft(waveform)
    torch.testing.assert_close(real, expected_spec.real.to(dtype))
    torch.testing.assert_close(imag, expected_spec.imag.to(dtype))
    magnitude = expected_spec.abs().to(dtype)
    phase = expected_spec.angle().to(dtype)
    # Cast before complex construction; complex BF16 is not supported by torch.
    spectrum = torch.complex(magnitude.float() * phase.float().cos(), magnitude.float() * phase.float().sin())
    expected_audio = torch.istft(spectrum, 16, 4, 16, window=window).to(dtype)
    actual_audio = generator._istft(magnitude, phase)
    torch.testing.assert_close(actual_audio, expected_audio)
    assert actual_audio.device == waveform.device
    assert actual_audio.dtype == dtype
    assert generator.stft_window.dtype == dtype


def test_native_path_retains_float64_precision(generator):
    assert generator._stft_on_cpu is False
    waveform = torch.linspace(-0.4, 0.4, 128, dtype=torch.float64).unsqueeze(0)
    spec = torch.stft(waveform, 16, 4, 16, window=generator.stft_window, return_complex=True)
    real, imag = generator._stft(waveform)
    torch.testing.assert_close(real, spec.real)
    torch.testing.assert_close(imag, spec.imag)
    torch.testing.assert_close(generator._istft(spec.abs(), spec.angle()), waveform)


@pytest.mark.parametrize("a5", [False, True])
def test_platform_patch_is_a5_only_and_idempotent(monkeypatch, hifigan, a5):
    npu = ModuleType("vllm_omni.platforms.npu")
    setattr(npu, "is_a5", lambda: a5)
    monkeypatch.setitem(sys.modules, npu.__name__, npu)
    monkeypatch.setitem(sys.modules, "vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan", hifigan)
    patch = load_module("cosyvoice3_a5_patch", ROOT / "vllm_omni/platforms/npu/models/cosyvoice3.py")
    patch.apply_cosyvoice3_patches()
    patch.apply_cosyvoice3_patches()
    assert hifigan.HiFTGenerator._stft_on_cpu is a5
    assert hifigan.CausalHiFTGenerator._stft_on_cpu is a5
