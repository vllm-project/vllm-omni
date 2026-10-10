# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import builtins

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd import audio_features
from vllm_omni.model_executor.models.lychee_fd.audio_features import (
    N_MELS,
    WINDOW_SAMPLES,
    alternating_model_positions,
    complete_audio_windows,
    feature_token_count,
    log_mel_spectrogram,
    pad_mel_features,
    valid_mel_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_full_window_has_reference_shape_and_tick_count() -> None:
    mel = log_mel_spectrogram(torch.zeros(WINDOW_SAMPLES))

    assert mel.shape == (N_MELS, 42)
    assert valid_mel_frames(mel.shape[1]) == 40
    assert feature_token_count(mel.shape[1]) == 5
    assert alternating_model_positions(mel.shape[1]) == 10


def test_batched_dynamic_range_is_isolated_per_window() -> None:
    generator = torch.Generator().manual_seed(7)
    loud = torch.randn(WINDOW_SAMPLES, generator=generator)
    quiet = torch.randn(WINDOW_SAMPLES, generator=generator) * 1e-3

    batched = log_mel_spectrogram(torch.stack((loud, quiet)))

    torch.testing.assert_close(batched[0], log_mel_spectrogram(loud))
    torch.testing.assert_close(batched[1], log_mel_spectrogram(quiet))


def test_complete_windows_leave_partial_tail_unconsumed() -> None:
    audio = torch.arange(WINDOW_SAMPLES * 2 + 17)
    windows = complete_audio_windows(audio)

    assert windows.shape == (2, WINDOW_SAMPLES)
    torch.testing.assert_close(windows.reshape(-1), audio[: WINDOW_SAMPLES * 2])


def test_padding_preserves_per_window_valid_lengths() -> None:
    first = torch.zeros(N_MELS, 42)
    second = torch.zeros(N_MELS, 30)

    padded, lengths = pad_mel_features([first, second])

    assert padded.shape == (2, N_MELS, 42)
    assert lengths.tolist() == [40, 28]


@pytest.fixture
def clear_mel_filter_cache():
    audio_features._cpu_mel_filter_bank.cache_clear()
    yield
    audio_features._cpu_mel_filter_bank.cache_clear()


@pytest.mark.parametrize("n_mels", [80, 128])
def test_first_audio_window_does_not_require_librosa(monkeypatch, clear_mel_filter_cache, n_mels):
    original_import = builtins.__import__

    def require_declared_packages(name, *args, **kwargs):
        if name == "librosa" or name.startswith("librosa."):
            raise ModuleNotFoundError("librosa is deliberately absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", require_declared_packages)
    features = log_mel_spectrogram(torch.zeros(WINDOW_SAMPLES), n_mels=n_mels)
    assert features.shape == (n_mels, 42)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    ("n_mels", "bands", "landmarks"),
    [
        (80, [5, 12, 28, 40, 60, 78], [0.4828878, 0.1665400, -0.6923581, -0.6923581, -0.6923581, -0.6923581]),
        (128, [5, 12, 28, 40, 60, 100], [-0.0474788, 1.2620357, -0.3380994, -0.6534697, -0.6534697, -0.6534697]),
    ],
)
def test_slaney_filter_parameters_and_released_feature_landmarks(
    monkeypatch, clear_mel_filter_cache, n_mels, bands, landmarks
):
    calls: list[dict[str, int]] = []
    shared_filter_bank = audio_features.mel_filter_bank

    def capture_filter_bank(**parameters: int) -> torch.Tensor:
        calls.append(parameters)
        return shared_filter_bank(**parameters)

    monkeypatch.setattr(audio_features, "mel_filter_bank", capture_filter_bank)
    seconds = torch.arange(WINDOW_SAMPLES, dtype=torch.float32) / 16000
    waveform = (
        0.3 * torch.sin(2 * torch.pi * 330 * seconds)
        + 0.1 * torch.sin(2 * torch.pi * 1200 * seconds)
        + 0.04 * torch.sin(2 * torch.pi * 3200 * seconds)
    )
    features = log_mel_spectrogram(waveform, n_mels=n_mels)
    assert calls == [{"sr": 16000, "n_fft": 400, "n_mels": n_mels}]
    assert audio_features._cpu_mel_filter_bank(n_mels).shape == (n_mels, 201)
    # CPU landmarks from the released Slaney frontend; allow ordinary float
    # arithmetic differences rather than requiring its exact rounding.
    torch.testing.assert_close(features[bands, 10], torch.tensor(landmarks), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(log_mel_spectrogram(waveform, n_mels=n_mels), features)
    assert len(calls) == 1
