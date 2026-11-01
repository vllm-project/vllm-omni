# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint and runtime configuration for inclusionAI Ming-3.0-TTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from transformers import PretrainedConfig
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig

from .constants import (
    AUDIO_END_TOKEN_ID,
    AUDIO_FRAME_HOP,
    AUDIO_PATCH_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    DEFAULT_CFG,
    DEFAULT_SIGMA,
    DEFAULT_TEMPERATURE,
    FLOW_PATCH_SIZE,
    HISTORY_SIZE,
    INITIAL_LATENT_CHUNK_SIZE,
    LATENT_CHUNK_SIZE,
    LATENT_DIM,
    LATENT_LEFT_CONTEXT,
    MAX_DECODE_STEPS,
    MIN_DECODE_STEPS,
    SAMPLE_RATE,
    SPEAKER_EMBEDDING_DIM,
    SPEAKER_START_TOKEN_ID,
    STOP_THRESHOLD,
    TEXT_EOS_TOKEN_ID,
    VAE_PATCH_SIZE,
)


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, PretrainedConfig):
        return value.to_dict()
    return dict(value or {})


def get_ming3_backbone_config(vllm_config: VllmConfig) -> VllmConfig:
    hf_config = vllm_config.model_config.hf_config
    return vllm_config.with_hf_config(getattr(hf_config, "llm_config", hf_config))


def resolve_latent_chunk_sizes(options: Mapping[str, Any]) -> tuple[int, int]:
    """Validate streaming chunk sizes; zero initial size uses the regular size."""
    chunk = int(options.get("latent_chunk_size", LATENT_CHUNK_SIZE))
    initial = int(options.get("initial_latent_chunk_size", INITIAL_LATENT_CHUNK_SIZE))
    if chunk <= 0 or initial < 0 or initial > chunk:
        raise ValueError(f"Invalid Ming3 chunk sizes: initial={initial}, chunk={chunk}")
    return chunk, initial or chunk


class BailingMoeV3Config(PretrainedConfig):
    """Duck-typed config for the upstream Bailing-v3 implementation.

    vLLM's architecture converter currently recognizes the identical
    ``bailing_hybrid`` layout for MLA sizing, while released Ming-3.0-TTS
    metadata calls it ``bailing_moe_v3``.
    """

    model_type = "bailing_hybrid"

    def __init__(self, **kwargs: Any) -> None:
        checkpoint_model_type = kwargs.pop("model_type", "bailing_moe_v3")
        super().__init__(**kwargs)
        self.checkpoint_model_type = checkpoint_model_type


class Ming3TTSConfig(PretrainedConfig):
    model_type = "ming3_tts"
    is_composition = True

    def __init__(
        self,
        llm_config: BailingMoeV3Config | dict[str, Any] | None = None,
        audio_config: AudioVAEConfig | dict[str, Any] | None = None,
        tts_config: dict[str, Any] | None = None,
        aggregator_config: dict[str, Any] | None = None,
        flow_config: dict[str, Any] | None = None,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.tts_config = dict(tts_config or {})
        task_contract = self.tts_config.get("task_contract")
        if task_contract not in (None, "", "lingv3_sft_multitask_v1"):
            raise ValueError(f"Unsupported Ming3 task contract: {task_contract!r}")
        self.llm_config = (
            llm_config if isinstance(llm_config, BailingMoeV3Config) else BailingMoeV3Config(**_plain(llm_config))
        )
        if not task_contract:
            # Legacy checkpoints were validated with the unbounded KDA gate.
            self.llm_config.kda_safe_gate = False
            self.llm_config.kda_lower_bound = None
        self.audio_config = (
            audio_config if isinstance(audio_config, AudioVAEConfig) else AudioVAEConfig(**_plain(audio_config))
        )
        self.aggregator_config = dict(aggregator_config or {})
        self.flow_config = dict(flow_config or {})
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            architectures=architectures or ["Ming3TTSForConditionalGeneration"],
            tie_word_embeddings=False,
            **kwargs,
        )

    def get_text_config(self, decoder: bool = False, **kwargs: Any) -> BailingMoeV3Config:
        del decoder, kwargs
        return self.llm_config


class LingV3TTSConfig(Ming3TTSConfig):
    """Serialized config name retained by the public Ming-3.0-tts-tiny checkpoint."""

    model_type = "lingv3_tts"

    def __init__(self, **kwargs: Any) -> None:
        # Transformers 5.19 generates a dataclass initializer for subclasses
        # without one, bypassing our nested-config normalization.
        super().__init__(**kwargs)


@dataclass
class Ming3TTSRuntimeConfig:
    llm_hidden_size: int
    llm_vocab_size: int
    llm_config: BailingMoeV3Config
    audio_tokenizer_config: AudioVAEConfig
    aggregator_config: dict[str, Any] = field(default_factory=dict)
    ditar_config: dict[str, Any] = field(default_factory=dict)
    latent_dim: int = LATENT_DIM
    patch_size: int = FLOW_PATCH_SIZE
    history_patch_size: int = HISTORY_SIZE
    vae_patch_size: int = VAE_PATCH_SIZE
    sample_rate: int = SAMPLE_RATE
    audio_frame_hop: int = AUDIO_FRAME_HOP
    speaker_embedding_dim: int = SPEAKER_EMBEDDING_DIM
    cfg: float = DEFAULT_CFG
    sigma: float = DEFAULT_SIGMA
    temperature: float = DEFAULT_TEMPERATURE
    stop_head_min_steps: int = MIN_DECODE_STEPS
    stop_head_threshold: float = STOP_THRESHOLD
    max_decode_steps: int = MAX_DECODE_STEPS
    latent_chunk_size: int = LATENT_CHUNK_SIZE
    initial_latent_chunk_size: int = INITIAL_LATENT_CHUNK_SIZE
    latent_left_context: int = LATENT_LEFT_CONTEXT
    text_eos_token_id: int = AUDIO_END_TOKEN_ID
    audio_dummy_token_id: int = AUDIO_PATCH_TOKEN_ID
    audio_start_token_id: int = AUDIO_START_TOKEN_ID
    audio_end_token_id: int = AUDIO_END_TOKEN_ID
    audio_eos_token_id: int = TEXT_EOS_TOKEN_ID
    speaker_placeholder_token_id: int = SPEAKER_START_TOKEN_ID

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> Ming3TTSRuntimeConfig:
        llm = hf_config.llm_config
        tts = dict(getattr(hf_config, "tts_config", {}) or {})
        audio = hf_config.audio_config
        agg = dict(getattr(hf_config, "aggregator_config", {}) or {})
        flow = dict(getattr(hf_config, "flow_config", {}) or {})
        agg.pop("in_channels", None)
        agg.pop("llm_input_dim", None)
        flow.pop("z_channels", None)
        flow.pop("llm_cond_dim", None)
        flow.pop("flow_model_type", None)
        return cls(
            llm_hidden_size=int(llm.hidden_size),
            llm_vocab_size=int(llm.vocab_size),
            llm_config=llm,
            audio_tokenizer_config=audio,
            aggregator_config=agg,
            ditar_config=flow,
            latent_dim=int(tts.get("latent_dim", LATENT_DIM)),
            patch_size=int(tts.get("patch_size", FLOW_PATCH_SIZE)),
            history_patch_size=int(tts.get("history_size", HISTORY_SIZE)),
            vae_patch_size=int(getattr(audio, "patch_size", VAE_PATCH_SIZE)),
            sample_rate=int(tts.get("sample_rate", getattr(audio, "sample_rate", SAMPLE_RATE))),
            audio_frame_hop=int(audio.enc_kwargs.get("hop_size", AUDIO_FRAME_HOP)),
            speaker_embedding_dim=int(tts.get("speaker_embedding_dim", SPEAKER_EMBEDDING_DIM)),
            temperature=float(tts.get("sampling_temperature", DEFAULT_TEMPERATURE)),
            max_decode_steps=int(tts.get("maximum_decode_steps", MAX_DECODE_STEPS)),
            stop_head_min_steps=int(tts.get("minimum_decode_steps", MIN_DECODE_STEPS)),
            stop_head_threshold=float(tts.get("stop_threshold", STOP_THRESHOLD)),
            audio_dummy_token_id=int(tts.get("audio_patch_token_id", AUDIO_PATCH_TOKEN_ID)),
            audio_start_token_id=int(tts.get("audio_start_token_id", AUDIO_START_TOKEN_ID)),
            audio_end_token_id=int(tts.get("audio_end_token_id", AUDIO_END_TOKEN_ID)),
            speaker_placeholder_token_id=int(tts.get("speaker_start_token_id", SPEAKER_START_TOKEN_ID)),
        )

    def validate(self) -> None:
        errors: list[str] = []
        if self.latent_dim != 64:
            errors.append(f"latent_dim must be 64, got {self.latent_dim}")
        if self.patch_size != 1 or self.history_patch_size != 1:
            errors.append(
                f"Ming3 flow patch/history sizes must both be 1, got {self.patch_size}/{self.history_patch_size}"
            )
        if self.vae_patch_size != 4:
            errors.append(f"AudioVAE patch_size must be 4, got {self.vae_patch_size}")
        if self.sample_rate != 44100 or self.audio_frame_hop != 882:
            errors.append(f"expected 44.1kHz/882-hop audio, got {self.sample_rate}/{self.audio_frame_hop}")
        if self.speaker_embedding_dim != 192:
            errors.append(f"speaker embedding dim must be 192, got {self.speaker_embedding_dim}")
        num_layers = int(self.llm_config.num_hidden_layers)
        layer_group_size = int(self.llm_config.layer_group_size)
        mla_layers = sum(
            (layer_idx + 1) % layer_group_size == 0 or layer_idx >= num_layers // layer_group_size * layer_group_size
            for layer_idx in range(num_layers)
        )
        if (num_layers, layer_group_size, num_layers - mla_layers, mla_layers) != (24, 4, 18, 6):
            errors.append(
                "expected a 24-layer Bailing-v3 backbone with an 18-KDA/6-MLA layout, "
                f"got {num_layers - mla_layers}-KDA/{mla_layers}-MLA"
            )
        mla_head_size = int(self.llm_config.kv_lora_rank) + int(self.llm_config.qk_rope_head_dim)
        if mla_head_size != 576:
            errors.append(f"expected MLA head size 576, got {mla_head_size}")
        required_ids = (
            self.audio_dummy_token_id,
            self.audio_start_token_id,
            self.audio_end_token_id,
            self.speaker_placeholder_token_id,
        )
        if required_ids != (157164, 157165, 157166, 157175):
            errors.append(f"unexpected Ming3 special-token IDs: {required_ids}")
        if self.audio_tokenizer_config.semantic_module_kwargs is not None:
            errors.append("semantic AudioVAE module is unsupported")
        if errors:
            raise ValueError("Invalid Ming-3.0-TTS config: " + "; ".join(errors))
