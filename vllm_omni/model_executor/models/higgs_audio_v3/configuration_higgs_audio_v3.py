# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration class for higgs-audio v3 (HiggsMultimodalQwen3) in vllm-omni.

V3 uses a Qwen3 backbone (~4B, 36 layers, 2560 hidden, GQA 32/8) with
fused multi-codebook embedding/head. No DualFFN. The audio codec is the
same higgs-audio-v2-tokenizer (8 codebooks, 25fps, 24kHz DAC decoder).

Special token resolution: ``resolve_special_tokens()`` reads the HF tokenizer
from a model directory and populates ``tts_token_id``, ``text_token_id``,
``audio_token_id`` (the continuation/output token), and ``eos_token_id`` so
the talker's LM bias and prompt builder have concrete IDs.
"""

from __future__ import annotations

import os
from typing import Any

import transformers
from transformers import PretrainedConfig

__all__ = ["HiggsAudioV3Config"]

_QWEN3_ROPE_THETA = 1_000_000


def _build_text_config(raw: Any) -> PretrainedConfig:
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
    """Typed config for higgs-audio v3 (HiggsMultimodalQwen3)."""

    model_type: str = "higgs_multimodal_qwen3"
    is_composition = True

    def __init__(
        self,
        audio_encoder_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_token_id: int = -100,
        mel_per_sample: int = 8,
        num_codebooks: int = 8,
        codebook_size: int = 1026,
        audio_stream_bos_id: int = 1024,
        audio_stream_eos_id: int = 1025,
        sample_rate: int = 24000,
        frame_rate: int = 25,
        # Resolved special token IDs (populated by resolve_special_tokens)
        tts_token_id: int | None = None,
        text_token_id: int | None = None,
        audio_continuation_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.audio_token_id = audio_token_id
        self.mel_per_sample = mel_per_sample

        if audio_encoder_config is None:
            audio_encoder_config = {
                "encoder_type": "discrete",
                "num_codebooks": num_codebooks,
                "vocab_size": codebook_size,
                "tie_word_embeddings": True,
            }
        self.audio_encoder_config = audio_encoder_config

        self.num_codebooks = int(audio_encoder_config.get("num_codebooks", num_codebooks))
        if self.num_codebooks <= 0:
            raise ValueError(f"num_codebooks must be > 0, got {self.num_codebooks}")
        self.codebook_size = int(audio_encoder_config.get("vocab_size", codebook_size))
        self.tie_modality_embeddings = bool(audio_encoder_config.get("tie_word_embeddings", True))

        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

        self.text_config = _build_text_config(text_config)
        self.audio_hidden_size = int(audio_encoder_config.get("out_dim", self.text_config.hidden_size))

        # Resolved special token IDs — None until resolve_special_tokens() is called
        self.tts_token_id = tts_token_id
        self.text_token_id = text_token_id
        self.audio_continuation_id = audio_continuation_id

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.text_config

    @property
    def num_real_codes(self) -> int:
        return self.audio_stream_bos_id

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    def resolve_special_tokens(self, model_path: str) -> None:
        """Resolve <|tts|>, <|text|>, <|audio|> and eos token IDs from the HF tokenizer.

        Call after ``from_pretrained()`` with the model directory or HF repo id.
        Populates ``tts_token_id``, ``text_token_id``, ``audio_continuation_id``,
        and ``eos_token_id`` on the config object.
        """
        if not model_path or not os.path.isdir(model_path):
            # Try HF cache resolution
            try:
                from huggingface_hub import try_to_load_from_cache

                cached = try_to_load_from_cache(repo_id=model_path, filename="tokenizer_config.json")
                if isinstance(cached, str) and os.path.isfile(cached):
                    model_path = os.path.dirname(cached)
                else:
                    return
            except Exception:
                return

        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            return

        vocab = dict(tokenizer.get_added_vocab())
        if "<|tts|>" in vocab:
            self.tts_token_id = vocab["<|tts|>"]
        if "<|text|>" in vocab:
            self.text_token_id = vocab["<|text|>"]
        if "<|audio|>" in vocab:
            self.audio_continuation_id = vocab["<|audio|>"]
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            self.eos_token_id = int(tokenizer.eos_token_id)
