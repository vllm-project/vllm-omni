# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Confucius4-TTS speaker encoder.

The talker conditions on one speaker token produced by an ECAPA-TDNN over the
Wav2Vec2-BERT reference features. The architecture is the shared
:class:`~vllm_omni.model_executor.models.common.ecapa_tdnn.ECAPATimeDelayNet`;
this module only pins the layer shape the Confucius4 checkpoint was trained
with, so ``speaker_encoder.*`` checkpoint keys load without renaming.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.model_executor.models.common.ecapa_tdnn import ECAPATimeDelayNet

from .configuration_confucius4_tts import Confucius4TTSConfig


@dataclass(frozen=True)
class Confucius4SpeakerEncoderConfig:
    """ECAPA-TDNN shape of the Confucius4 talker's speaker encoder.

    Satisfies :class:`~vllm_omni.model_executor.models.common.ecapa_tdnn.ECAPAConfig`.
    ``mel_dim`` is the Wav2Vec2-BERT feature width and ``enc_dim`` the talker
    hidden size; the remaining fields are fixed by the released checkpoint.
    """

    mel_dim: int
    enc_dim: int
    enc_channels: tuple[int, ...] = (512, 512, 512, 512, 1536)
    enc_kernel_sizes: tuple[int, ...] = (5, 3, 3, 3, 1)
    enc_dilations: tuple[int, ...] = (1, 2, 3, 4, 1)
    enc_attention_channels: int = 128
    enc_res2net_scale: int = 8
    enc_se_channels: int = 128


def build_speaker_encoder(config: Confucius4TTSConfig) -> ECAPATimeDelayNet:
    """Build the speaker encoder for a Confucius4 talker config.

    Args:
        config: Model config providing ``speaker_embedding_dim`` (input
            feature width) and ``hidden_size`` (output embedding width).

    Returns:
        An encoder mapping ``(batch, time, speaker_embedding_dim)`` features to
        ``(batch, hidden_size)`` embeddings.
    """
    return ECAPATimeDelayNet(
        Confucius4SpeakerEncoderConfig(
            mel_dim=config.speaker_embedding_dim,
            enc_dim=config.hidden_size,
        )
    )


__all__ = ["Confucius4SpeakerEncoderConfig", "build_speaker_encoder"]
