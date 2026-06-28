# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-modality hash functions for embedding cache key generation (RFC #3427).

All functions return a 16-character hex string (64-bit truncated SHA-256) as
the cache key. 64 bits gives a collision probability of ~1/2^64 per pair —
negligible for any realistic serving workload.

SHA-256 over raw tensor bytes is deterministic, fast (< 0.5 ms for typical
inputs on CPU), and requires no model-specific logic. A perceptual hash would
reduce false misses from minor numeric noise but adds implementation complexity;
this can be layered on later (RFC §Enhancements) if benchmark data justifies it.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _sha256_tensor(t: "torch.Tensor") -> str:
    """Return 16-char hex digest of a tensor's raw bytes (CPU, contiguous)."""
    arr = t.detach().cpu().contiguous()
    # view as uint8 to get raw bytes without interpreter overhead
    raw: bytes = arr.view(-1).numpy().tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    # Truncate to 16 chars (64 bits) — sufficient collision resistance.
    return digest[:16]


def hash_audio_features(input_features: "torch.Tensor") -> str:
    """Hash audio mel-spectrogram features.

    `input_features` is the tensor passed directly to audio_tower —
    shape (batch, mel_bins, time_frames), dtype float32.
    Hash covers the full tensor so that any sample-level difference
    (different audio, same length) produces a distinct key.
    """
    return "a:" + _sha256_tensor(input_features)


def hash_image_pixels(pixel_values: "torch.Tensor") -> str:
    """Hash image pixel values.

    `pixel_values` is the input to the vision tower before patching —
    shape (n_patches, channels, patch_h, patch_w), dtype float16/32.
    """
    return "i:" + _sha256_tensor(pixel_values)


def hash_video_pixels(pixel_values_videos: "torch.Tensor") -> str:
    """Hash video pixel values (all frames concatenated).

    `pixel_values_videos` — shape (total_frames * n_patches, C, H, W).
    """
    return "v:" + _sha256_tensor(pixel_values_videos)
