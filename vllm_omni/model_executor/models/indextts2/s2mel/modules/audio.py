# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch


def dynamic_range_compression_torch(x, compression=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * compression)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def _hz_to_mel(freqs: np.ndarray | float) -> np.ndarray | float:
    """librosa's default Slaney mel scale (htk=False)."""
    scalar = not hasattr(freqs, "__len__")
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (freqs - f_min) / f_sp

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = freqs >= min_log_hz
    mels[log_t] = min_log_mel + np.log(freqs[log_t] / min_log_hz) / logstep
    return float(mels[0]) if scalar else mels


def _mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
    """Inverse of librosa's default Slaney mel scale (htk=False)."""
    scalar = not hasattr(mels, "__len__")
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return float(freqs[0]) if scalar else freqs


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    """Create a librosa.filters.mel-compatible Slaney filterbank."""
    if fmax is None:
        fmax = float(sr) / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    freqs = _mel_to_hz(mels)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    fdiff = np.diff(freqs)
    ramps = np.subtract.outer(freqs, fft_freqs)
    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{sampling_rate}_{fmax}_{y.device}" not in mel_basis:
        mel = _mel_filterbank(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(sampling_rate) + "_" + str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
