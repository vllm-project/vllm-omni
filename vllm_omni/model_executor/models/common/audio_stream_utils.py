# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Common utilities for streaming audio synthesis across models (CosyVoice, StepAudio2, MiniCPM-o).

Provides:
- fade_in_out: Overlap-add cross-fading using a symmetric window to eliminate boundary artifacts.
- build_overlap_window: Generates Hamming/Hanning cross-fade window on the target device.
"""

from __future__ import annotations

import torch


def build_overlap_window(
    window_len: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    window_type: str = "hamming",
) -> torch.Tensor:
    """Build a symmetric cross-fade window of length ``2 * window_len``.

    The first half ramps up (fade-in) and the second half ramps down (fade-out).
    """
    total_len = max(2, int(window_len) * 2)
    if window_type in ("hanning", "hann"):
        return torch.hann_window(total_len, periodic=False, device=device, dtype=dtype)
    else:
        return torch.hamming_window(total_len, periodic=False, device=device, dtype=dtype)


def fade_in_out(
    speech: torch.Tensor,
    previous: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Cross-fade two overlapping waveform segments to eliminate boundary clicks.

    The window is of length ``2 * overlap_len``: the first half ramps up and
    the second half ramps down. The overlap head of ``speech`` is blended with
    the tail of ``previous``.
    """
    if speech is None or previous is None or window is None:
        return speech

    overlap = min(
        int(window.shape[0] // 2),
        int(speech.shape[-1]),
        int(previous.shape[-1]),
    )
    if overlap <= 0:
        return speech

    result = speech.clone()
    window = window.to(device=speech.device, dtype=speech.dtype)
    result[..., :overlap] = result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
    return result
