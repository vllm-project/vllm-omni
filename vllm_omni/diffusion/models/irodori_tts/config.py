# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions of this file are derived from Irodori-TTS (MIT),
# Copyright (c) 2026 Aratako.  See the upstream project LICENSE.
"""Metadata-only configuration loading for Irodori-TTS v4 checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Inference-model configuration preserved from the Irodori runtime."""

    latent_dim: int = 128
    latent_patch_size: int = 1
    model_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    mlp_ratio: float = 2.875
    text_mlp_ratio: float | None = 2.6
    speaker_mlp_ratio: float | None = 2.6
    dropout: float = 0.0
    text_vocab_size: int = 102400
    text_tokenizer_repo: str = "sbintuitions/sarashina2.2-0.5b"
    text_encoder_revision: str | None = None
    text_add_bos: bool = True
    text_encoder_type: str = "scratch"
    pretrained_projector_type: str = "linear"
    pretrained_projector_hidden_ratio: float = 2.0
    pretrained_projector_dropout: float = 0.0
    text_dim: int = 1280
    text_layers: int = 14
    text_heads: int = 10
    use_caption_condition: bool = False
    use_speaker_condition: bool | None = None
    caption_vocab_size: int | None = None
    caption_tokenizer_repo: str | None = None
    caption_add_bos: bool | None = None
    caption_dim: int | None = None
    caption_layers: int | None = None
    caption_heads: int | None = None
    caption_mlp_ratio: float | None = None
    speaker_dim: int = 1280
    speaker_layers: int = 14
    speaker_heads: int = 10
    speaker_patch_size: int = 1
    timestep_embed_dim: int = 512
    adaln_rank: int = 256
    norm_eps: float = 1e-5
    use_duration_predictor: bool = False
    duration_aux_dim: int = 14
    duration_hidden_dim: int = 1024
    duration_layers: int = 3
    duration_dropout: float = 0.1
    duration_attention_heads: int = 8
    duration_architecture: str = "token_sum_adarn_zero_no_aux"
    duration_token_init_frames: float = 9.0
    duration_speaker_fusion: str = "adarn_zero"
    duration_caption_fusion: str = "adarn_zero"
    duration_caption_pooling: str = "masked_mean"

    @property
    def patched_latent_dim(self) -> int:
        return self.latent_dim * self.latent_patch_size

    @property
    def speaker_patched_latent_dim(self) -> int:
        return self.patched_latent_dim * self.speaker_patch_size

    @property
    def use_speaker_condition_resolved(self) -> bool:
        if self.use_speaker_condition is None:
            return not bool(self.use_caption_condition)
        return bool(self.use_speaker_condition)

    @property
    def text_mlp_ratio_resolved(self) -> float:
        return self.mlp_ratio if self.text_mlp_ratio is None else float(self.text_mlp_ratio)

    @property
    def speaker_mlp_ratio_resolved(self) -> float:
        return self.mlp_ratio if self.speaker_mlp_ratio is None else float(self.speaker_mlp_ratio)

    @property
    def use_pretrained_text_encoder(self) -> bool:
        return str(self.text_encoder_type).strip().lower() == "pretrained"

    @property
    def caption_vocab_size_resolved(self) -> int:
        return int(self.text_vocab_size if self.caption_vocab_size is None else self.caption_vocab_size)

    @property
    def caption_tokenizer_repo_resolved(self) -> str:
        return self.text_tokenizer_repo if self.caption_tokenizer_repo is None else str(self.caption_tokenizer_repo)

    @property
    def caption_add_bos_resolved(self) -> bool:
        return bool(self.text_add_bos if self.caption_add_bos is None else self.caption_add_bos)

    @property
    def caption_dim_resolved(self) -> int:
        return int(self.text_dim if self.caption_dim is None else self.caption_dim)

    @property
    def caption_layers_resolved(self) -> int:
        return int(self.text_layers if self.caption_layers is None else self.caption_layers)

    @property
    def caption_heads_resolved(self) -> int:
        return int(self.text_heads if self.caption_heads is None else self.caption_heads)

    @property
    def caption_mlp_ratio_resolved(self) -> float:
        return self.text_mlp_ratio_resolved if self.caption_mlp_ratio is None else float(self.caption_mlp_ratio)


@dataclass(frozen=True)
class IrodoriCheckpointConfig:
    """Resolved non-tensor metadata carried by an Irodori checkpoint."""

    model: ModelConfig
    text_encoder_config: dict[str, object]
    max_text_len: int
    max_caption_len: int
    max_ref_seconds: float
    checkpoint_path: str


_CONFIG_META_KEY = "config_json"
_TEXT_ENCODER_CONFIG_META_KEY = "text_encoder_config_json"
_INFERENCE_INT_CONFIG_KEYS = frozenset({"max_text_len", "max_caption_len"})
_INFERENCE_FLOAT_CONFIG_KEYS = frozenset({"ref_max_seconds"})
_TORCHAO_METADATA_KEY = "irodori_quantization_json"


def resolve_irodori_checkpoint(model_or_path: str, revision: str | None = None) -> str:
    """Resolve exactly ``model.safetensors`` without a full Hub snapshot."""

    source = Path(model_or_path).expanduser()
    if source.exists():
        if source.is_dir():
            checkpoint = source / "model.safetensors"
        else:
            checkpoint = source
            if checkpoint.name != "model.safetensors":
                raise ValueError(
                    f"Irodori local checkpoint files must be named 'model.safetensors'; got {checkpoint.name!r}."
                )
        if not checkpoint.is_file():
            raise ValueError(f"Irodori checkpoint not found: {checkpoint}")
        return str(checkpoint)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - vLLM installs this dependency
        raise RuntimeError("huggingface_hub is required to resolve Irodori checkpoints.") from exc
    return hf_hub_download(
        repo_id=model_or_path,
        filename="model.safetensors",
        revision=revision,
    )


def _parse_json_mapping(raw: str | None, *, field: str, path: Path) -> dict[str, Any]:
    if raw is None:
        raise ValueError(f"Missing required metadata field '{field}' in checkpoint: {path}")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in '{field}' metadata for checkpoint: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metadata field '{field}' must decode to an object: {path}")
    return value


def _read_positive_int(value: Any, *, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer in checkpoint metadata: {path}")
    return value


def _read_positive_float(value: Any, *, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number in checkpoint metadata: {path}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number in checkpoint metadata: {path}")
    return value


def _split_flat_checkpoint_config(path: Path, flat_config: dict[str, Any]) -> tuple[dict[str, Any], int, int, float]:
    model_config: dict[str, Any] = {}
    inference_config: dict[str, Any] = {}
    for key, value in flat_config.items():
        if key in _INFERENCE_INT_CONFIG_KEYS | _INFERENCE_FLOAT_CONFIG_KEYS:
            inference_config[key] = value
        else:
            model_config[key] = value
    return (
        model_config,
        _read_positive_int(inference_config.get("max_text_len", 256), name="max_text_len", path=path),
        _read_positive_int(inference_config.get("max_caption_len", 512), name="max_caption_len", path=path),
        _read_positive_float(inference_config.get("ref_max_seconds", 30.0), name="ref_max_seconds", path=path),
    )


def _validate_model_config(config: ModelConfig, *, path: Path) -> None:
    positive_ints = (
        "latent_dim",
        "latent_patch_size",
        "model_dim",
        "num_layers",
        "num_heads",
        "text_dim",
        "text_layers",
        "text_heads",
        "speaker_dim",
        "speaker_layers",
        "speaker_heads",
        "speaker_patch_size",
        "timestep_embed_dim",
        "adaln_rank",
        "duration_aux_dim",
        "duration_hidden_dim",
        "duration_layers",
        "duration_attention_heads",
    )
    for name in positive_ints:
        _read_positive_int(getattr(config, name), name=name, path=path)
    if config.model_dim % config.num_heads:
        raise ValueError("model_dim must be divisible by num_heads in Irodori checkpoint metadata.")
    if config.speaker_dim % config.speaker_heads:
        raise ValueError("speaker_dim must be divisible by speaker_heads in Irodori checkpoint metadata.")
    # v4-Small uses a pretrained ModernBERT backbone plus projectors, so its
    # legacy scratch TextEncoder head fields are deliberately not divisibility
    # constraints (the published 512/10 combination demonstrates this).
    if not config.use_pretrained_text_encoder:
        if config.text_layers > 0 and config.text_dim % config.text_heads:
            raise ValueError("text_dim must be divisible by text_heads in Irodori checkpoint metadata.")
        if config.caption_layers_resolved > 0 and config.caption_dim_resolved % config.caption_heads_resolved:
            raise ValueError("caption_dim must be divisible by caption_heads in Irodori checkpoint metadata.")
    for name in ("mlp_ratio", "norm_eps", "duration_token_init_frames"):
        _read_positive_float(getattr(config, name), name=name, path=path)


def read_irodori_checkpoint_config(path: str | Path) -> IrodoriCheckpointConfig:
    """Read checkpoint metadata while never materialising a model tensor."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise ValueError(f"Irodori checkpoint not found: {checkpoint}")
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - required by vLLM already
        raise RuntimeError("safetensors is required to load Irodori checkpoints.") from exc

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if _TORCHAO_METADATA_KEY in metadata:
        raise ValueError(
            "TorchAO-quantized Irodori checkpoints are not supported. "
            "Use the unquantized v4-Small model.safetensors checkpoint."
        )

    flat_config = _parse_json_mapping(metadata.get(_CONFIG_META_KEY), field=_CONFIG_META_KEY, path=checkpoint)
    text_encoder_config = _parse_json_mapping(
        metadata.get(_TEXT_ENCODER_CONFIG_META_KEY), field=_TEXT_ENCODER_CONFIG_META_KEY, path=checkpoint
    )
    model_config, max_text_len, max_caption_len, max_ref_seconds = _split_flat_checkpoint_config(
        checkpoint, flat_config
    )
    known_fields = {field.name for field in fields(ModelConfig)}
    unknown_fields = sorted(set(model_config) - known_fields)
    if unknown_fields:
        raise ValueError(f"Unknown Irodori model configuration keys: {unknown_fields}")
    try:
        model = ModelConfig(**model_config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Irodori model configuration in checkpoint: {checkpoint}") from exc
    _validate_model_config(model, path=checkpoint)
    return IrodoriCheckpointConfig(
        model=model,
        text_encoder_config=dict(text_encoder_config),
        max_text_len=max_text_len,
        max_caption_len=max_caption_len,
        max_ref_seconds=max_ref_seconds,
        checkpoint_path=str(checkpoint),
    )
