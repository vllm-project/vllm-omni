# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded reference-audio tensors for Ming's EngineCore and connector handoffs."""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

REFERENCE_AUDIO_FIELDS = (
    "reference_audio_samples",
    "reference_audio_lengths",
    "reference_audio_sample_rates",
)
MAX_REFERENCE_SECONDS = 60
MAX_REFERENCE_SAMPLES = 2_880_000
MAX_REFERENCE_CLIPS = 16


def _validate_size(lengths: list[int], rates: list[int]) -> None:
    """Reject invalid or oversized references before copying their samples."""
    if not lengths or len(lengths) != len(rates) or len(lengths) > MAX_REFERENCE_CLIPS:
        raise ValueError(f"Reference audio requires 1 to {MAX_REFERENCE_CLIPS} clips with matching sample rates")
    if any(length <= 0 or rate <= 0 for length, rate in zip(lengths, rates, strict=True)):
        raise ValueError("Reference audio must be nonempty and have a positive sample rate")
    if (
        sum(lengths) > MAX_REFERENCE_SAMPLES
        or sum(n / sr for n, sr in zip(lengths, rates, strict=True)) > MAX_REFERENCE_SECONDS
    ):
        raise ValueError(
            f"Reference audio exceeds {MAX_REFERENCE_SECONDS} seconds or {MAX_REFERENCE_SAMPLES} samples in total"
        )


def pack_reference_audio(
    references: Iterable[tuple[np.ndarray | torch.Tensor | list[float], int]],
) -> dict[str, torch.Tensor | list[int]]:
    """Pack mono clips into top-level tensor fields supported by EngineCore.

    Preserve clip order and sampling rates, without Python float expansion or
    resampling. Raise ValueError for empty, nonfinite or oversized inputs.
    """
    clips: list[torch.Tensor] = []
    lengths: list[int] = []
    rates: list[int] = []
    for samples, rate in references:
        waveform = torch.as_tensor(samples, dtype=torch.float32, device="cpu").reshape(-1)
        lengths.append(waveform.numel())
        rates.append(int(rate))
        _validate_size(lengths, rates)
        if not np.isfinite(waveform.numpy()).all():
            raise ValueError("Reference audio must contain finite samples")
        clips.append(waveform)
    _validate_size(lengths, rates)
    samples = clips[0].contiguous() if len(clips) == 1 else torch.cat(clips)
    return {
        REFERENCE_AUDIO_FIELDS[0]: samples,
        REFERENCE_AUDIO_FIELDS[1]: lengths,
        REFERENCE_AUDIO_FIELDS[2]: rates,
    }


def has_reference_audio(info: Mapping[str, Any]) -> bool:
    """Recognize packed tensors as well as the existing list-based input."""
    return info.get(REFERENCE_AUDIO_FIELDS[0]) is not None or bool(info.get("reference_audio"))


def unpack_reference_audio(info: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Restore per-clip tensor views at the model boundary; retain list compatibility."""
    samples = info.get(REFERENCE_AUDIO_FIELDS[0])
    if samples is None:
        return info.get("reference_audio")
    lengths = info.get(REFERENCE_AUDIO_FIELDS[1], [])
    rates = info.get(REFERENCE_AUDIO_FIELDS[2], [])
    _validate_size(lengths, rates)
    if not isinstance(samples, torch.Tensor) or samples.ndim != 1 or samples.numel() != sum(lengths):
        raise ValueError("Packed reference audio must be a flat tensor matching its clip lengths")
    return [{"samples": clip, "sample_rate": rate} for clip, rate in zip(samples.split(lengths), rates, strict=True)]
