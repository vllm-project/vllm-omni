# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validation of one duplex audio append payload.

The runner hands a plugin the append as a mapping ``{format, sample_rate_hz,
audio}`` (``audio`` base64 ``pcm_f32le``). Models with a fixed input rate and
frame size validate it here; the PCM itself is decoded by
``vllm_omni.model_executor.common.audio.pcm``.
"""

from __future__ import annotations

from collections.abc import Mapping

from vllm_omni.model_executor.common.audio.pcm import (
    decode_pcm_f32le_base64,
    pcm_f32le_sample_count,
)


def payload_audio(payload: object) -> object | None:
    """The encoded audio field of an append payload (``audio``, or the older ``data``)."""
    if not isinstance(payload, Mapping):
        return None
    return payload.get("audio") or payload.get("data")


def decode_pcm_f32le_payload(
    payload: object,
    *,
    sample_rate_hz: int,
    exact_samples: int | None = None,
    model: str = "duplex",
) -> bytes:
    """Raw PCM of one append that must be ``pcm_f32le`` at ``sample_rate_hz``.

    ``exact_samples`` pins the unit length (a lockstep model takes exactly one
    frame per append); ``None`` accepts any whole number of samples.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"{model} duplex append payload must be a mapping")
    if payload.get("format") != "pcm_f32le":
        raise ValueError(f"{model} duplex append format must be pcm_f32le")
    if payload.get("sample_rate_hz") != sample_rate_hz:
        raise ValueError(f"{model} duplex append sample_rate_hz must be {sample_rate_hz}")
    raw = decode_pcm_f32le_base64(payload_audio(payload), model=model)
    if exact_samples is not None and pcm_f32le_sample_count(raw) != exact_samples:
        raise ValueError(f"{model} duplex append must contain exactly {exact_samples} samples")
    return raw


def payload_sample_count(payload: object) -> int | None:
    """Sample count of a ``pcm_f32le`` append, or ``None`` when it is not one (lenient, for budgeting)."""
    if not isinstance(payload, Mapping) or payload.get("format") != "pcm_f32le":
        return None
    try:
        raw = decode_pcm_f32le_base64(payload_audio(payload))
    except ValueError:
        return None
    return pcm_f32le_sample_count(raw)


__all__ = ["decode_pcm_f32le_payload", "payload_audio", "payload_sample_count"]
