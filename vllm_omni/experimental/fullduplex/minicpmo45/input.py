from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np

from vllm_omni.experimental.fullduplex.base.pcm_buffer import (
    BasePcmAppendBuffer as MiniCPMO45PcmAppendBuffer,
)
from vllm_omni.experimental.fullduplex.base.pcm_buffer import (
    PcmAppendReservation as MiniCPMO45PcmAppendReservation,
)


def validate_native_ref_audio_config(session_config: dict[str, Any]) -> None:
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    if any(key in session_config for key in ("ref_audio_path", "tts_ref_audio_path")) or any(
        key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")
    ):
        raise ValueError("native duplex ref_audio_path is not accepted; resolve ref_audio in serving first")


def decode_native_ref_audio_from_config(session_config: dict[str, Any]) -> np.ndarray | None:
    validate_native_ref_audio_config(session_config)
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    audio_data = extra_body.get("ref_audio_data")
    if audio_data is None:
        return None
    if not isinstance(audio_data, str):
        raise TypeError("native duplex ref_audio_data must be base64 pcm_f32le")
    fmt = extra_body.get("ref_audio_format") or "pcm_f32le"
    if fmt != "pcm_f32le":
        raise ValueError(f"unsupported native duplex ref_audio_format: {fmt!r}")
    try:
        raw = base64.b64decode(audio_data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid native duplex ref_audio_data") from exc
    if len(raw) % 4:
        raise ValueError("invalid native duplex ref_audio_data length")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


__all__ = [
    "MiniCPMO45PcmAppendReservation",
    "MiniCPMO45PcmAppendBuffer",
    "decode_native_ref_audio_from_config",
    "validate_native_ref_audio_config",
]
