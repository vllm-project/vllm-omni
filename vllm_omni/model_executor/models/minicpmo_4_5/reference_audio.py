# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owned reference-waveform snapshots for untyped runner payload transport."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def encode_reference_audio(waveform: torch.Tensor) -> dict[str, Any]:
    """Snapshot mono FP32 PCM without expanding samples into Python scalars.

    ``model_intermediate_buffer`` contains untyped values, so the engine's
    Tensor decode hook cannot restore a Tensor nested in this field. Native
    bytes survive both IPC and shared-connector transport. They also own the
    samples, including when the caller later mutates its input waveform.
    """
    samples = waveform.detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
    return {"format": "pcm_f32le", "data": samples.numpy().astype("<f4", copy=False).tobytes()}


def decode_reference_audio(value: Any) -> torch.Tensor:
    """Read compact PCM or legacy list/Tensor references as an FP32 Tensor."""
    if not isinstance(value, Mapping):
        return torch.as_tensor(value, dtype=torch.float32)
    if value.get("format") != "pcm_f32le":
        raise ValueError("unsupported reference audio payload format")
    data = value.get("data")
    if not isinstance(data, bytes) or len(data) % 4:
        raise ValueError("invalid reference audio PCM payload")
    # The bytes carrier is immutable. Copy once on decode so downstream
    # preprocessing owns a writable waveform rather than aliasing IPC data.
    return torch.from_numpy(np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True))
