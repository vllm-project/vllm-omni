# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Lychee's released PCM quantization at the model output boundary."""

from __future__ import annotations

import numpy as np
import pybase64 as base64
import torch
from vllm.logger import init_logger

from vllm_omni.engine.duplex.plugin import EncodeAudio

logger = init_logger(__name__)


def make_lychee_audio_encoder(encode_audio: EncodeAudio) -> EncodeAudio:
    """Keep released raw PCM at default speed; delegate other encoding options."""

    def encode(audio_data: object, sample_rate_hz: int, response_format: str, speed: float | None) -> str | None:
        pcm = isinstance(response_format, str) and response_format.lower() in {"pcm", "pcm16"}
        if not pcm or speed not in (None, 1.0):
            return encode_audio(audio_data, sample_rate_hz, response_format, speed)
        if audio_data is None:
            return None
        try:
            if isinstance(audio_data, torch.Tensor):
                waveform = audio_data.detach().cpu().float().numpy()
            else:
                waveform = np.asarray(audio_data, dtype=np.float32)
            # The released Token2wav output clips before scaling and truncates
            # toward zero. Keep explicit little-endian int16 sample storage.
            clipped = np.clip(waveform.reshape(-1), -1.0, 1.0)
            pcm16 = (clipped * 32767.0).astype("<i2")
            return base64.b64encode(pcm16.tobytes()).decode("ascii")
        except Exception:
            logger.exception("Failed to encode Lychee raw PCM output")
            return None

    return encode


__all__ = ["make_lychee_audio_encoder"]
