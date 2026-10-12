# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Batch-safe audio feature extraction for Lychee-FD.

The released runtime consumes 16 kHz mono audio in 400 ms windows. Each full
window produces 42 padded STFT frames, 40 valid frames, five encoder/adaptor
vectors and ten alternating model positions.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from vllm_omni.utils.audio import mel_filter_bank

SAMPLE_RATE_HZ = 16_000
WINDOW_MS = 400
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_MS // 1_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
RIGHT_PADDING_SAMPLES = 479


@lru_cache(maxsize=2)
def _cpu_mel_filter_bank(n_mels: int) -> torch.Tensor:
    if n_mels not in (80, 128):
        raise ValueError(f"Lychee-FD supports 80 or 128 mel bins, got {n_mels}")
    # Preserve the released frontend's Slaney scale and area normalization
    # through the shared runtime helper; exact cross-library rounding may vary.
    return mel_filter_bank(sr=SAMPLE_RATE_HZ, n_fft=N_FFT, n_mels=n_mels).contiguous()


@lru_cache(maxsize=1)
def _cpu_hann_window() -> torch.Tensor:
    # The released runtime constructs Hann on CPU before transferring to CUDA.
    # CUDA construction rounds 47 entries differently on the current toolkit,
    # changing a BF16 Mel cell in the actual recorded fixture's third window.
    return torch.hann_window(N_FFT, device="cpu", dtype=torch.float32)


def _as_waveform_batch(audio: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        return audio.unsqueeze(0).float(), True
    if audio.ndim == 2:
        return audio.float(), False
    raise ValueError(f"Expected waveform shape [samples] or [batch, samples], got {tuple(audio.shape)}")


def log_mel_spectrogram(
    audio: torch.Tensor,
    *,
    n_mels: int = N_MELS,
    padding: int = RIGHT_PADDING_SAMPLES,
) -> torch.Tensor:
    """Compute the reference Lychee log-Mel transform.

    For batched input, the dynamic-range maximum is reduced independently for
    every sample. This prevents a loud session from changing another session's
    features when continuous batching is enabled.
    """

    waveforms, squeeze_batch = _as_waveform_batch(audio)
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    if padding:
        waveforms = F.pad(waveforms, (0, padding))

    window = _cpu_hann_window().to(device=waveforms.device, dtype=waveforms.dtype)
    stft = torch.stft(
        waveforms,
        N_FFT,
        HOP_LENGTH,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs().square()
    filters = _cpu_mel_filter_bank(n_mels).to(device=waveforms.device, dtype=magnitudes.dtype)
    mel_spec = torch.matmul(filters, magnitudes)

    log_spec = mel_spec.clamp_min(1e-10).log10()
    sample_max = log_spec.amax(dim=(-2, -1), keepdim=True)
    log_spec = torch.maximum(log_spec, sample_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[0] if squeeze_batch else log_spec


def valid_mel_frames(mel_frames: int) -> int:
    """Remove the two reference padding frames used only by the STFT path."""

    if mel_frames < 2:
        raise ValueError(f"mel_frames must be at least 2, got {mel_frames}")
    return mel_frames - 2


def encoder_feature_length(valid_frames: int) -> int:
    if valid_frames < 0:
        raise ValueError(f"valid_frames must be non-negative, got {valid_frames}")
    return (valid_frames + 1) // 2 // 2


def adaptor_feature_length(encoder_frames: int, *, kernel_size: int = 3, stride: int = 2) -> int:
    if encoder_frames < 0:
        raise ValueError(f"encoder_frames must be non-negative, got {encoder_frames}")
    if kernel_size <= 0 or stride <= 0:
        raise ValueError("kernel_size and stride must be positive")
    return (encoder_frames + 2 - kernel_size) // stride + 1


def feature_token_count(mel_frames: int, *, kernel_size: int = 3, stride: int = 2) -> int:
    return adaptor_feature_length(
        encoder_feature_length(valid_mel_frames(mel_frames)),
        kernel_size=kernel_size,
        stride=stride,
    )


def alternating_model_positions(mel_frames: int) -> int:
    """Return audio-patch/audio-pad positions consumed by one feature window."""

    return feature_token_count(mel_frames) * 2


def complete_audio_windows(audio: torch.Tensor) -> torch.Tensor:
    """Return only complete 400 ms windows; retain any tail in session state."""

    if not torch.is_tensor(audio):
        audio = torch.as_tensor(audio)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono waveform [samples], got {tuple(audio.shape)}")
    window_count = audio.numel() // WINDOW_SAMPLES
    if window_count == 0:
        return audio.new_empty((0, WINDOW_SAMPLES))
    return audio[: window_count * WINDOW_SAMPLES].reshape(window_count, WINDOW_SAMPLES)


def pad_mel_features(features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ``[mel, time]`` features and return valid lengths for the encoder."""

    if not features:
        return torch.empty((0, N_MELS, 0)), torch.empty((0,), dtype=torch.int32)
    if any(feature.ndim != 2 for feature in features):
        raise ValueError(f"Every feature must have shape [{N_MELS}, time]")
    mel_bins = {int(feature.shape[0]) for feature in features}
    if mel_bins != {N_MELS}:
        raise ValueError(f"Every feature must have shape [{N_MELS}, time]")
    lengths = torch.tensor(
        [valid_mel_frames(int(feature.shape[1])) for feature in features],
        dtype=torch.int32,
        device=features[0].device,
    )
    padded = pad_sequence([feature.T for feature in features], batch_first=True, padding_value=0.0)
    return padded.transpose(1, 2), lengths


__all__ = [
    "HOP_LENGTH",
    "N_FFT",
    "N_MELS",
    "RIGHT_PADDING_SAMPLES",
    "SAMPLE_RATE_HZ",
    "WINDOW_MS",
    "WINDOW_SAMPLES",
    "adaptor_feature_length",
    "alternating_model_positions",
    "complete_audio_windows",
    "encoder_feature_length",
    "feature_token_count",
    "log_mel_spectrogram",
    "pad_mel_features",
    "valid_mel_frames",
]
