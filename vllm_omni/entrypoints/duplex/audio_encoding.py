# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Audio encoding for duplex model output (entrypoint layer, injected into ``DuplexOmniEngine``)."""

from __future__ import annotations

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)


def encode_audio(
    audio_data: object,
    sample_rate_hz: int,
    response_format: str,
    speed: float | None,
) -> str | None:
    """Encode a model audio tensor/array into base64 in ``response_format``.

    Moved from the serving runtime bridge; uses the shared ``AudioMixin``
    encoder directly instead of going through the chat service.
    """
    if audio_data is None:
        return None
    try:
        import torch

        from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
        from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio

        if isinstance(audio_data, torch.Tensor):
            audio_tensor = audio_data.detach().cpu().float().numpy()
        else:
            audio_tensor = np.asarray(audio_data, dtype=np.float32)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.reshape(-1)
        audio_response = AudioMixin().create_audio(
            CreateAudio(
                audio_tensor=audio_tensor,
                sample_rate=sample_rate_hz,
                response_format=response_format,
                speed=float(speed) if isinstance(speed, int | float) and speed > 0 else 1.0,
                stream_format="audio",
                base64_encode=True,
            )
        )
        return str(audio_response.audio_data)
    except Exception:
        logger.exception("Failed to encode duplex data-plane audio output")
        return None


__all__ = ["encode_audio"]
