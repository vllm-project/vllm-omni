# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Literal

import torch
import torch.nn.functional as F


def fit_audio_to_length(
    waveform: torch.Tensor, target_samples: int, short_audio: Literal["trim", "loop", "pad"]
) -> torch.Tensor:
    source_samples = waveform.shape[-1]
    if source_samples == 0:
        raise ValueError("The supplied audio must contain at least one sample.")
    if short_audio not in ("trim", "loop", "pad"):
        raise ValueError(f"Unknown short-audio policy: {short_audio}")
    if source_samples >= target_samples or short_audio == "trim":
        return waveform[..., :target_samples]
    if short_audio == "pad":
        return F.pad(waveform, (0, target_samples - source_samples))
    # 只沿采样轴重复，保留 batch、声道与采样值。
    repeats = (target_samples + source_samples - 1) // source_samples
    return waveform.repeat(1, 1, repeats)[..., :target_samples]
