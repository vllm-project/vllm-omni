# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ming dense checkpoint config adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from transformers import PretrainedConfig, Qwen2Config

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig

from .constants import (
    AGGREGATOR_HIDDEN_SIZE,
    AUDIO_DUMMY_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_EOS_TOKEN_ID,
    AUDIO_FRAME_HOP,
    AUDIO_START_TOKEN_ID,
    DEFAULT_CFG,
    DEFAULT_SIGMA,
    DEFAULT_TEMPERATURE,
    HISTORY_PATCH_SIZE,
    KEY_CFG,
    KEY_CHUNK_ID,
    KEY_DECODE_STEP,
    KEY_LAST_STOP_PROB,
    KEY_LATENT_HISTORY,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_NEXT_EMBEDS,
    KEY_PROMPT_LATENT_TAIL,
    KEY_PROMPT_LATENTS,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEMPERATURE,
    KEY_TEXT_MODE,
    LATENT_CHUNK_SIZE,
    LATENT_DIM,
    LATENT_LEFT_CONTEXT,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    MAX_DECODE_STEPS,
    PATCH_SIZE,
    SAMPLE_RATE,
    STOP_HEAD_MIN_STEPS,
    STOP_HEAD_THRESHOLD,
    TEXT_EOS_TOKEN_ID,
    VAE_PATCH_SIZE,
    VISION_START_TOKEN_ID,
)
from .validation import _coerce_audio_vae_config, _nested_get, _to_plain_dict, validate_ming_tts_config


def _coerce_qwen2_config(value: Any) -> Qwen2Config:
    if isinstance(value, Qwen2Config):
        return value
    if isinstance(value, PretrainedConfig):
        return Qwen2Config.from_dict(value.to_dict())
    if isinstance(value, dict):
        return Qwen2Config.from_dict(dict(value))
    raise TypeError(f"Unsupported llm_config type for Ming dense config: {type(value)!r}")


class MingDenseConfig(PretrainedConfig):
    # The upstream checkpoint declares model_type="dense". Keep it for HF
    # config compatibility; deploy/ming_tts.yaml selects the vLLM-Omni
    # pipeline via pipeline: ming_tts.
    model_type = "dense"

    def __init__(
        self,
        llm_config: Qwen2Config | dict[str, Any] | None = None,
        ditar_config: dict[str, Any] | None = None,
        aggregator_config: dict[str, Any] | None = None,
        audio_tokenizer_config: AudioVAEConfig | dict[str, Any] | None = None,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.llm_config = _coerce_qwen2_config(llm_config or {})
        self.ditar_config = dict(ditar_config or {})
        self.aggregator_config = dict(aggregator_config or {})
        self.audio_tokenizer_config = _coerce_audio_vae_config(audio_tokenizer_config)
        super().__init__(architectures=architectures, **kwargs)

    def get_text_config(self, decoder: bool = False, **kwargs: Any) -> Qwen2Config:
        del decoder, kwargs
        return self.llm_config


@dataclass
class MingTTSConfig:
    """Flat config object shared by Stage-1 and Stage-2. Build via from_hf_config()."""

    llm_hidden_size: int = LLM_HIDDEN_SIZE
    llm_vocab_size: int = LLM_VOCAB_SIZE
    llm_config: dict[str, Any] = field(default_factory=dict)

    latent_dim: int = LATENT_DIM
    patch_size: int = PATCH_SIZE
    history_patch_size: int = HISTORY_PATCH_SIZE

    ditar_config: dict[str, Any] = field(default_factory=dict)
    aggregator_config: dict[str, Any] = field(default_factory=dict)

    audio_tokenizer_config: AudioVAEConfig | None = None
    vae_patch_size: int = VAE_PATCH_SIZE
    sample_rate: int = SAMPLE_RATE
    audio_frame_hop: int = AUDIO_FRAME_HOP

    cfg: float = DEFAULT_CFG
    sigma: float = DEFAULT_SIGMA
    temperature: float = DEFAULT_TEMPERATURE
    stop_head_min_steps: int = STOP_HEAD_MIN_STEPS
    stop_head_threshold: float = STOP_HEAD_THRESHOLD
    max_decode_steps: int = MAX_DECODE_STEPS

    latent_chunk_size: int = LATENT_CHUNK_SIZE
    latent_left_context: int = LATENT_LEFT_CONTEXT

    text_eos_token_id: int = TEXT_EOS_TOKEN_ID
    audio_dummy_token_id: int = AUDIO_DUMMY_TOKEN_ID
    audio_start_token_id: int = AUDIO_START_TOKEN_ID
    audio_end_token_id: int = AUDIO_END_TOKEN_ID
    audio_eos_token_id: int = AUDIO_EOS_TOKEN_ID

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> MingTTSConfig:
        llm_raw = getattr(hf_config, "llm_config", {}) or {}
        ditar_raw = getattr(hf_config, "ditar_config", {}) or {}
        agg_raw = getattr(hf_config, "aggregator_config", {}) or {}
        atc_raw = getattr(hf_config, "audio_tokenizer_config", None)

        llm_dict = _to_plain_dict(llm_raw)
        ditar = _to_plain_dict(ditar_raw)
        agg = _to_plain_dict(agg_raw)
        ditar.setdefault("attn_backend", "torch")

        atc = _coerce_audio_vae_config(atc_raw)
        atc_enc_latent_dim = _nested_get(atc, "enc_kwargs", "latent_dim", default=LATENT_DIM)
        atc_patch_size = _nested_get(atc, "patch_size", default=VAE_PATCH_SIZE)
        atc_sample_rate = _nested_get(atc, "sample_rate", default=SAMPLE_RATE)

        enc_hop_size = _nested_get(atc, "enc_kwargs", "hop_size", default=AUDIO_FRAME_HOP)

        cfg = cls(
            llm_hidden_size=llm_dict.get("hidden_size", LLM_HIDDEN_SIZE),
            llm_vocab_size=llm_dict.get("vocab_size", LLM_VOCAB_SIZE),
            llm_config=llm_dict,
            latent_dim=atc_enc_latent_dim,
            patch_size=ditar.get("patch_size", PATCH_SIZE),
            history_patch_size=ditar.get("history_patch_size", HISTORY_PATCH_SIZE),
            ditar_config=ditar,
            aggregator_config=agg,
            audio_tokenizer_config=atc,
            vae_patch_size=atc_patch_size,
            sample_rate=atc_sample_rate,
            audio_frame_hop=enc_hop_size if enc_hop_size is not None else AUDIO_FRAME_HOP,
        )
        return cfg

    def validate(self) -> None:
        validate_ming_tts_config(self)


__all__ = [
    "AGGREGATOR_HIDDEN_SIZE",
    "AUDIO_DUMMY_TOKEN_ID",
    "AUDIO_END_TOKEN_ID",
    "AUDIO_EOS_TOKEN_ID",
    "AUDIO_FRAME_HOP",
    "AUDIO_START_TOKEN_ID",
    "DEFAULT_CFG",
    "DEFAULT_SIGMA",
    "DEFAULT_TEMPERATURE",
    "HISTORY_PATCH_SIZE",
    "KEY_CFG",
    "KEY_CHUNK_ID",
    "KEY_DECODE_STEP",
    "KEY_LAST_STOP_PROB",
    "KEY_LATENT_HISTORY",
    "KEY_MAX_DECODE_STEPS",
    "KEY_MIN_DECODE_STEPS",
    "KEY_NEXT_EMBEDS",
    "KEY_PROMPT_LATENT_TAIL",
    "KEY_PROMPT_LATENTS",
    "KEY_REQUEST_ID",
    "KEY_SIGMA",
    "KEY_SPEAKER_EMBEDDING",
    "KEY_TEMPERATURE",
    "KEY_TEXT_MODE",
    "LATENT_CHUNK_SIZE",
    "LATENT_DIM",
    "LATENT_LEFT_CONTEXT",
    "LLM_HIDDEN_SIZE",
    "LLM_VOCAB_SIZE",
    "MAX_DECODE_STEPS",
    "MingDenseConfig",
    "MingTTSConfig",
    "PATCH_SIZE",
    "SAMPLE_RATE",
    "STOP_HEAD_MIN_STEPS",
    "STOP_HEAD_THRESHOLD",
    "TEXT_EOS_TOKEN_ID",
    "VAE_PATCH_SIZE",
    "VISION_START_TOKEN_ID",
]
