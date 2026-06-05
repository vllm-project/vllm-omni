# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for higgs-audio v3 (HiggsMultimodalQwen3) in vllm-omni.

V3 uses a Qwen3 backbone (~4B, 36 layers, 2560 hidden, GQA 32/8) with
fused multi-codebook embedding/head. No DualFFN. The audio codec is the
same higgs-audio-v2-tokenizer (8 codebooks, 25fps, 24kHz DAC decoder).

HF checkpoint layout: ``bosonai/higgs-audio-v3-tts-4b`` stores config as
a nested dict with ``audio_encoder_config`` (discrete TTS) and
``text_config`` (Qwen3 backbone).
"""

from __future__ import annotations

from typing import Any

import transformers
from transformers import PretrainedConfig

__all__ = ["HiggsAudioV3Config"]

# Qwen3 checkpoints sometimes ship rope_theta=null; the training default is 1e6.
_QWEN3_ROPE_THETA = 1_000_000


def _build_text_config(raw: Any) -> PretrainedConfig:
    """Realise the text_config sub-dict into a concrete PretrainedConfig."""
    if isinstance(raw, PretrainedConfig):
        return raw
    cfg = dict(raw or {})
    model_type = cfg.get("model_type", "qwen3")
    if model_type == "qwen3" and cfg.get("rope_theta") is None:
        cfg["rope_theta"] = _QWEN3_ROPE_THETA
    try:
        cfg_cls = transformers.CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown text backbone model_type {model_type!r}") from exc
    return cfg_cls(**cfg)


class HiggsAudioV3Config(PretrainedConfig):
    """Typed config for higgs-audio v3 (HiggsMultimodalQwen3).

    Wraps the HF checkpoint's nested structure and exposes audio and text
    sub-configs for the talker and code2wav stages.
    """

    model_type: str = "higgs_multimodal_qwen3"
    is_composition = True

    def __init__(
        self,
        audio_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_token_id: int = -100,
        mel_per_sample: int = 8,
        # Audio codec constants (same as v2)
        num_codebooks: int = 8,
        codebook_size: int = 1026,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        sample_rate: int = 24000,
        frame_rate: int = 25,
        **kwargs: Any,
    ) -> None:
        self.audio_token_id = audio_token_id
        self.mel_per_sample = mel_per_sample

        # Audio encoder config (discrete TTS path)
        if audio_encoder_config is None:
            audio_encoder_config = {
                "encoder_type": "discrete",
                "num_codebooks": num_codebooks,
                "vocab_size": codebook_size,
                "tie_word_embeddings": True,
            }
        self.audio_encoder_config = audio_encoder_config

        # Extract audio constants from encoder config or use defaults
        self.num_codebooks = int(audio_encoder_config.get("num_codebooks", num_codebooks))
        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be > 0, got {self.num_codebooks}")
        self.codebook_size = int(audio_encoder_config.get("vocab_size", codebook_size))
        self.tie_modality_embeddings = bool(audio_encoder_config.get("tie_word_embeddings", True))

        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

        # Build text backbone config
        self.text_config = _build_text_config(text_config)

        # Hidden size for audio modules defaults to text backbone hidden size
        self.audio_hidden_size = int(audio_encoder_config.get("out_dim", self.text_config.hidden_size))

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.text_config

    @property
    def num_real_codes(self) -> int:
        return self.audio_stream_bos_id  # 1024

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size
