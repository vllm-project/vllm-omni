# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Little-endian float32 PCM (``pcm_f32le``) helpers.

Plain audio plumbing every model that takes raw PCM needs: decode a base64
chunk, check it is whole finite samples, count and materialise them. No
duplex vocabulary lives here.
"""

from __future__ import annotations

import binascii

import numpy as np
import pybase64 as base64

PCM_F32LE_BYTES_PER_SAMPLE = 4


def decode_pcm_f32le_base64(encoded: object, *, model: str = "audio") -> bytes:
    """Decode base64 ``pcm_f32le`` into raw bytes, refusing partial or non-finite samples.

    ``model`` names the caller in error messages so a client sees which model
    rejected its audio.
    """
    if not isinstance(encoded, str):
        raise ValueError(f"{model} audio must be base64 pcm_f32le")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{model} audio is not valid base64") from exc
    if len(raw) % PCM_F32LE_BYTES_PER_SAMPLE:
        raise ValueError(f"{model} pcm_f32le byte length must be divisible by four")
    samples = np.frombuffer(raw, dtype="<f4")
    if samples.size and not bool(np.isfinite(samples).all()):
        raise ValueError(f"{model} pcm_f32le samples must be finite")
    return raw


def pcm_f32le_sample_count(raw: bytes) -> int:
    return len(raw) // PCM_F32LE_BYTES_PER_SAMPLE


def pcm_f32le_samples(raw: bytes) -> np.ndarray:
    """The samples as a writable, contiguous float32 array (safe for zero-copy tensor views)."""
    return np.ascontiguousarray(np.frombuffer(raw, dtype="<f4"), dtype=np.float32).copy()


__all__ = [
    "PCM_F32LE_BYTES_PER_SAMPLE",
    "decode_pcm_f32le_base64",
    "pcm_f32le_sample_count",
    "pcm_f32le_samples",
]
