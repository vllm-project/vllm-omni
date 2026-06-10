# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig

from .constants import (
    AGGREGATOR_HIDDEN_SIZE,
    HISTORY_PATCH_SIZE,
    LATENT_DIM,
    LLM_HIDDEN_SIZE,
    LLM_VOCAB_SIZE,
    PATCH_SIZE,
    SAMPLE_RATE,
)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Normalize nested config objects into plain dicts when possible."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if isinstance(obj, PretrainedConfig):
        return obj.to_dict()
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return dict(obj.to_dict())
        except Exception:
            pass
    try:
        return dict(vars(obj))
    except Exception:
        return {}


def _coerce_audio_vae_config(atc_raw: Any) -> AudioVAEConfig | None:
    if atc_raw is None:
        return None
    if isinstance(atc_raw, AudioVAEConfig):
        return atc_raw
    if isinstance(atc_raw, PretrainedConfig):
        atc_dict = atc_raw.to_dict()
    elif isinstance(atc_raw, dict):
        atc_dict = dict(atc_raw)
    elif hasattr(atc_raw, "to_dict") and callable(atc_raw.to_dict):
        atc_dict = dict(atc_raw.to_dict())
    else:
        raise TypeError(f"Unsupported audio_tokenizer_config type for Ming dense config: {type(atc_raw)!r}")

    return AudioVAEConfig(**atc_dict)


def _nested_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested attribute/key access for dicts and config-like objects."""
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur if cur is not None else default


def validate_ming_tts_config(cfg: Any) -> None:
    """Run before GPU allocation/weight loading. Raises ValueError on mismatches."""
    if cfg.audio_dummy_token_id != 151705:
        raise ValueError(
            f"audio_dummy_token_id={cfg.audio_dummy_token_id}, expected 151705 (<audioPatch>). "
            "Wrong tokenizer/checkpoint?"
        )
    if cfg.audio_eos_token_id != 151704:
        raise ValueError(
            f"audio_eos_token_id={cfg.audio_eos_token_id}, expected 151704 (<end_of_audio>). "
            "Wrong tokenizer/checkpoint?"
        )
    if cfg.text_eos_token_id != 151669:
        raise ValueError(
            f"text_eos_token_id={cfg.text_eos_token_id}, expected 151669 (<text_eos>). Wrong tokenizer/checkpoint?"
        )

    if cfg.audio_tokenizer_config is None:
        raise ValueError("audio_tokenizer_config is None. Nested AudioVAE config was not deserialized correctly.")

    if cfg.latent_dim != LATENT_DIM:
        raise ValueError(
            f"latent_dim mismatch: got {cfg.latent_dim}, expected {LATENT_DIM}. "
            "Check audio_tokenizer_config.enc_kwargs.latent_dim."
        )
    if cfg.patch_size != PATCH_SIZE:
        raise ValueError(
            f"patch_size mismatch: got {cfg.patch_size}, expected {PATCH_SIZE}. Check ditar_config.patch_size."
        )
    if cfg.history_patch_size != HISTORY_PATCH_SIZE:
        raise ValueError(
            f"history_patch_size mismatch: got {cfg.history_patch_size}, expected {HISTORY_PATCH_SIZE}. "
            "Check ditar_config.history_patch_size."
        )
    if cfg.llm_hidden_size != LLM_HIDDEN_SIZE:
        raise ValueError(
            f"llm_hidden_size mismatch: got {cfg.llm_hidden_size}, expected {LLM_HIDDEN_SIZE}. "
            "Check llm_config.hidden_size."
        )
    if cfg.llm_vocab_size != LLM_VOCAB_SIZE:
        raise ValueError(f"llm_vocab_size mismatch: got {cfg.llm_vocab_size}, expected {LLM_VOCAB_SIZE}.")
    if cfg.sample_rate != SAMPLE_RATE:
        raise ValueError(f"sample_rate mismatch: got {cfg.sample_rate}, expected {SAMPLE_RATE}.")

    if cfg.vae_patch_size != cfg.patch_size:
        raise ValueError(f"VAE patch size ({cfg.vae_patch_size}) != flow/DiT patch size ({cfg.patch_size}).")

    llm_hidden_from_cfg = cfg.llm_config.get("hidden_size")
    if llm_hidden_from_cfg is not None and llm_hidden_from_cfg != cfg.llm_hidden_size:
        raise ValueError(f"llm_hidden_size ({cfg.llm_hidden_size}) != llm_config.hidden_size ({llm_hidden_from_cfg}).")

    agg_h = cfg.aggregator_config.get("hidden_size")
    dit_h = cfg.ditar_config.get("hidden_size")
    if agg_h is not None and dit_h is not None and agg_h != dit_h:
        raise ValueError(f"aggregator_config.hidden_size ({agg_h}) != ditar_config.hidden_size ({dit_h}).")
    if agg_h is not None and agg_h != AGGREGATOR_HIDDEN_SIZE:
        raise ValueError(f"aggregator hidden_size mismatch: got {agg_h}, expected {AGGREGATOR_HIDDEN_SIZE}.")
    if dit_h is not None and dit_h != AGGREGATOR_HIDDEN_SIZE:
        raise ValueError(f"ditar hidden_size mismatch: got {dit_h}, expected {AGGREGATOR_HIDDEN_SIZE}.")

    atc = cfg.audio_tokenizer_config
    semantic_module_kwargs = _nested_get(atc, "semantic_module_kwargs", default=None)
    if semantic_module_kwargs is not None:
        raise ValueError("Ming dense 0.5B expects audio_tokenizer_config.semantic_module_kwargs to be null.")

    enc_latent = _nested_get(atc, "enc_kwargs", "latent_dim", default=None)
    dec_latent = _nested_get(atc, "dec_kwargs", "latent_dim", default=None)
    if enc_latent is not None and enc_latent != cfg.latent_dim:
        raise ValueError(f"audio enc latent_dim ({enc_latent}) != Ming latent_dim ({cfg.latent_dim}).")
    if dec_latent is not None and dec_latent != cfg.latent_dim:
        raise ValueError(f"audio dec latent_dim ({dec_latent}) != Ming latent_dim ({cfg.latent_dim}).")

    atc_patch = _nested_get(atc, "patch_size", default=None)
    if atc_patch is not None and atc_patch != cfg.vae_patch_size:
        raise ValueError(f"audio_tokenizer_config.patch_size ({atc_patch}) != vae_patch_size ({cfg.vae_patch_size}).")

    atc_sr = _nested_get(atc, "sample_rate", default=None)
    if atc_sr is not None and atc_sr != cfg.sample_rate:
        raise ValueError(f"audio_tokenizer_config.sample_rate ({atc_sr}) != sample_rate ({cfg.sample_rate}).")

    enc_input_dim = _nested_get(atc, "enc_kwargs", "input_dim", default=None)
    enc_hop_size = _nested_get(atc, "enc_kwargs", "hop_size", default=None)
    dec_output_dim = _nested_get(atc, "dec_kwargs", "output_dim", default=None)

    if enc_input_dim is not None and enc_hop_size is not None and enc_input_dim != enc_hop_size:
        raise ValueError(f"AudioVAE encoder input_dim ({enc_input_dim}) != hop_size ({enc_hop_size}).")
    if enc_hop_size is not None and dec_output_dim is not None and enc_hop_size != dec_output_dim:
        raise ValueError(
            f"AudioVAE encoder hop_size ({enc_hop_size}) != decoder output_dim ({dec_output_dim}). "
            "Expected 882 in this checkpoint family."
        )

    if cfg.latent_chunk_size <= 0:
        raise ValueError(f"latent_chunk_size must be > 0, got {cfg.latent_chunk_size}.")
    if cfg.latent_left_context < 0:
        raise ValueError(f"latent_left_context must be >= 0, got {cfg.latent_left_context}.")
    if cfg.max_decode_steps <= 0:
        raise ValueError(f"max_decode_steps must be > 0, got {cfg.max_decode_steps}.")
    if not (0.0 <= cfg.stop_head_threshold <= 1.0):
        raise ValueError(f"stop_head_threshold must be in [0,1], got {cfg.stop_head_threshold}.")
    if cfg.stop_head_min_steps < 0:
        raise ValueError(f"stop_head_min_steps must be >= 0, got {cfg.stop_head_min_steps}.")
