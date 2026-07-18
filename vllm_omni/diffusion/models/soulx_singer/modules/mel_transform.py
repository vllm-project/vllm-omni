import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm_omni.utils.audio import mel_filter_bank as librosa_mel_fn


def dynamic_range_compression(x, c=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * c)


def dynamic_range_decompression(x, c=1):
    return np.exp(x) / c


def dynamic_range_compression_torch(x, c=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * c)


def dynamic_range_decompression_torch(x, c=1):
    return torch.exp(x) / c


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        mel_basis = {}
        hann_window = {}

        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis = mel.to(torch.float32)
        hann_window = torch.hann_window(win_size)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    def forward(self, y):
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec)

        return spec


def load_mel_spectrogram():
    return load_mel_spectrogram_from_cfg(None)


def _get_from_mapping(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely read a field from a dict/OmegaConf-like object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_mel_spectrogram_from_cfg(audio_cfg: Any | None = None) -> MelSpectrogram:
    """Build MelSpectrogram from `audio_config`-like config.

    Expected keys (either in dict or Hydra/OmegaConf object):
      - hop_size, sample_rate (or sampling_rate), n_fft, num_mels, win_size, fmin, fmax
    """
    # Defaults keep current behavior.
    mel_cfg: dict[str, Any] = {
        "hop_size": _get_from_mapping(audio_cfg, "hop_size", 480),
        "sampling_rate": _get_from_mapping(
            audio_cfg,
            "sampling_rate",
            _get_from_mapping(audio_cfg, "sample_rate", 24000),
        ),
        "n_fft": _get_from_mapping(audio_cfg, "n_fft", 1920),
        "num_mels": _get_from_mapping(audio_cfg, "num_mels", 128),
        "win_size": _get_from_mapping(audio_cfg, "win_size", 1920),
        "fmin": _get_from_mapping(audio_cfg, "fmin", 0),
        "fmax": _get_from_mapping(audio_cfg, "fmax", 12000),
    }

    mel_model = MelSpectrogram(**mel_cfg)
    mel_model.eval()
    return mel_model


class MelSpectrogramEncoder(nn.Module):
    def __init__(self, audio_config: dict | None = None):
        super().__init__()
        self.model = load_mel_spectrogram_from_cfg(audio_config)
        audio_config = audio_config or {}
        self.mel_mean = audio_config.get("mel_mean", -4.92)
        self.mel_var = audio_config.get("mel_var", 8.14)

    def forward(self, x):
        x = self.model(x).transpose(1, 2)
        x = (x - self.mel_mean) / math.sqrt(self.mel_var)
        return x
