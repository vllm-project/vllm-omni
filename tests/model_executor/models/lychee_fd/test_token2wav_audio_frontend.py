# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speaker-prompt mel features use the shared Slaney frontend without librosa."""

from __future__ import annotations

import builtins
import importlib

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.flashcosyvoice.utils import audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def clear_frontend_caches():
    audio.mel_basis.clear()
    audio.hann_window.clear()
    yield
    audio.mel_basis.clear()
    audio.hann_window.clear()


def _prompt_waveform() -> torch.Tensor:
    seconds = torch.arange(9600, dtype=torch.float32) / 24000
    return (
        0.3 * torch.sin(2 * torch.pi * 330 * seconds)
        + 0.1 * torch.sin(2 * torch.pi * 1200 * seconds)
        + 0.04 * torch.sin(2 * torch.pi * 3200 * seconds)
    ).unsqueeze(0)


def test_prompt_frontend_import_and_execution_do_not_require_librosa(monkeypatch):
    original_import = builtins.__import__

    def require_declared_packages(name, *args, **kwargs):
        if name == "librosa" or name.startswith("librosa."):
            raise ModuleNotFoundError("librosa is deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", require_declared_packages)
    importlib.reload(audio)
    features = audio.mel_spectrogram(_prompt_waveform())
    assert features.shape == (1, 80, 20)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert not torch.cuda.is_initialized()


def test_prompt_mel_parameters_and_released_feature_landmarks(monkeypatch):
    calls: list[dict[str, int]] = []
    shared_filter_bank = audio.mel_filter_bank

    def capture_filter_bank(**parameters: int) -> torch.Tensor:
        calls.append(parameters)
        return shared_filter_bank(**parameters)

    monkeypatch.setattr(audio, "mel_filter_bank", capture_filter_bank)
    waveform = _prompt_waveform()
    features = audio.mel_spectrogram(waveform)
    assert calls == [{"sr": 24000, "n_fft": 1920, "n_mels": 80, "fmin": 0, "fmax": 8000}]
    assert audio.mel_basis["8000_cpu"].shape == (80, 961)
    assert features.shape == (1, 80, 20)
    # Small landmarks captured with the released Slaney librosa frontend on
    # the same CPU waveform; allow normal float arithmetic differences.
    expected = torch.tensor([-5.0328636, -6.2233853, -11.1606112, -11.5129251, -11.3123808, -11.5129251])
    torch.testing.assert_close(features[0, [5, 12, 28, 40, 60, 78], 10], expected, rtol=1e-5, atol=1e-4)
    repeated = audio.mel_spectrogram(waveform)
    assert len(calls) == 1
    torch.testing.assert_close(repeated, features)
