# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Audio utility functions shared across models and entrypoints."""

import numpy as np
import torch
from torchaudio.functional import melscale_fbanks


def mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> torch.Tensor:
    """Compute a mel filterbank matrix.

    Drop-in replacement for ``librosa.filters.mel`` using
    ``torchaudio.functional.melscale_fbanks``.

    Args:
        sr: Sample rate of the audio.
        n_fft: FFT window size.
        n_mels: Number of mel bands.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz). Defaults to ``sr / 2``.

    Returns:
        Tensor of shape ``(n_mels, n_fft // 2 + 1)``.
    """
    if fmax is None:
        fmax = float(sr) / 2.0
    # Use mel_scale='slaney' and norm='slaney' to match librosa's
    # default behaviour (Slaney 1998 frequency mapping with area
    # normalization).
    return melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=float(fmin),
        f_max=float(fmax),
        n_mels=n_mels,
        sample_rate=sr,
        mel_scale="slaney",
        norm="slaney",
    ).T


def peak_normalize(
    audio: np.ndarray,
    db_level: float = -6.0,
) -> np.ndarray:
    """Normalize audio so peak amplitude reaches a target dB level.

    Drop-in replacement for ``sox.Transformer().norm(db_level=...)``.

    Args:
        audio: Input waveform as a 1-D numpy array.
        db_level: Target peak amplitude in dBFS.

    Returns:
        Normalized waveform with the same dtype as *audio*.
    """
    peak = np.abs(audio).max()
    if peak == 0:
        return audio
    target = 10.0 ** (db_level / 20.0)
    return audio * (target / peak)
