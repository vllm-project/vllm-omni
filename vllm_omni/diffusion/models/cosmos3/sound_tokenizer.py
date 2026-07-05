# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 sound tokenizer integration.

The tokenizer decodes model-generated sound latents for video+audio output. It
does not encode or condition on ``multi_modal_data["audio"]``; request-side
sound generation is enabled by ``generate_sound`` or ``sound_gen``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.progress_bar import _is_rank_zero

from .audio_tokenizer import Cosmos3AVAEAudioTokenizer

logger = init_logger(__name__)

DEFAULT_SOUND_SAMPLE_RATE = 48000
DEFAULT_SOUND_CHANNELS = 2
DEFAULT_SOUND_DIM = 64
DEFAULT_SOUND_HOP_SIZE = 1920
DEFAULT_SOUND_LATENT_FPS = DEFAULT_SOUND_SAMPLE_RATE / DEFAULT_SOUND_HOP_SIZE
DEFAULT_SOUND_NORMALIZE_LATENTS = False
DEFAULT_SOUND_NORMALIZATION_TYPE = "none"
DEFAULT_SOUND_TANH_INPUT_SCALE = 1.5
DEFAULT_SOUND_TANH_OUTPUT_SCALE = 3.5
DEFAULT_SOUND_TANH_CLAMP = 0.995
SOUND_TOKENIZER_COMPONENT_NAME = "sound_tokenizer"
SOUND_TOKENIZER_CHECKPOINT_NAME = "diffusion_pytorch_model.safetensors"


def _pipeline_args(od_config: OmniDiffusionConfig) -> dict[str, Any]:
    return dict(getattr(od_config, "custom_pipeline_args", None) or {})


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        value = config.get(key, None)
        return default if value is None else value
    return getattr(config, key, default)


def _config_path_get(config: Any, *keys: str) -> Any:
    value = config
    for key in keys:
        value = _config_get(value, key, None)
        if value is None:
            return None
    return value


def _sound_tokenizer_config_from(config: Any) -> Any:
    """Return nested ``sound_tokenizer`` config from Cosmos3 config shapes."""
    for path in (
        ("sound_tokenizer",),
        ("model", "config", "sound_tokenizer"),
        ("config", "sound_tokenizer"),
        ("model_config", "sound_tokenizer"),
    ):
        value = _config_path_get(config, *path)
        if value is not None:
            return value
    return None


def _nested_sound_tokenizer_configs(od_config: OmniDiffusionConfig | None) -> tuple[Any, ...]:
    if od_config is None:
        return ()
    configs = []
    for source in (
        getattr(od_config, "model_config", None),
        getattr(od_config, "tf_model_config", None),
    ):
        config = _sound_tokenizer_config_from(source)
        if config is not None:
            configs.append(config)
    return tuple(configs)


def _first_value_from_configs(configs: tuple[Any, ...], keys: tuple[str, ...]) -> Any:
    for config in configs:
        for key in keys:
            value = _config_get(config, key, None)
            if value is not None:
                return value
    return None


def _top_level_model_value(od_config: OmniDiffusionConfig | None, keys: tuple[str, ...]) -> Any:
    if od_config is None:
        return None
    for source in (
        getattr(od_config, "model_config", None),
        getattr(od_config, "tf_model_config", None),
    ):
        for key in keys:
            for path in ((key,), ("model", "config", key), ("config", key), ("model_config", key)):
                value = _config_path_get(source, *path)
                if value is not None:
                    return value
    return None


def _custom_arg_value(args: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = args.get(key)
        if value is not None:
            return value
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_audio_channels(value: Any) -> int:
    if isinstance(value, bool):
        return 2 if value else 1
    if isinstance(value, str) and value.strip().lower() in {
        "1",
        "0",
        "true",
        "false",
        "yes",
        "no",
        "on",
        "off",
    }:
        return 2 if _as_bool(value) else 1
    return int(value)


def _resolve_model_file(path: Any, model_root: str | None) -> str | None:
    if not path:
        return None
    path = str(path)
    if "://" in path or os.path.isabs(path) or os.path.exists(path) or not model_root:
        return path
    return str(Path(model_root) / path)


def _load_sound_tokenizer_component_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Cosmos3 sound tokenizer config must be a JSON object, got {type(config)!r}.")
    return config


def _component_audio_channels(config: dict[str, Any]) -> Any:
    if config.get("dec_out_channels") is not None:
        return config["dec_out_channels"]
    if config.get("audio_channels") is not None:
        return config["audio_channels"]
    if config.get("stereo") is not None:
        return 2 if _as_bool(config["stereo"]) else 1
    return None


def _component_arch_values(config: dict[str, Any]) -> dict[str, Any]:
    values = {
        "sample_rate": config.get("sampling_rate", config.get("sample_rate")),
        "audio_channels": _component_audio_channels(config),
        "io_channels": config.get("vocoder_input_dim", config.get("io_channels", config.get("latent_ch"))),
        "hop_size": config.get("hop_size"),
    }
    return {key: value for key, value in values.items() if value is not None}


def _resolve_arch_value(
    od_config: OmniDiffusionConfig,
    args: dict[str, Any],
    component_values: dict[str, Any],
    *,
    field: str,
    custom_keys: tuple[str, ...],
    nested_keys: tuple[str, ...],
    top_level_keys: tuple[str, ...],
    default: Any,
    cast,
) -> Any:
    custom_value = _custom_arg_value(args, custom_keys)
    component_value = component_values.get(field)
    if component_value is not None:
        resolved = cast(component_value)
        if custom_value is not None and cast(custom_value) != resolved:
            raise ValueError(
                "Conflicting Cosmos3 sound tokenizer architecture override for "
                f"{field}: component config has {resolved!r}, custom args have {cast(custom_value)!r}."
            )
        return resolved

    if custom_value is not None:
        return cast(custom_value)

    nested_value = _first_value_from_configs(_nested_sound_tokenizer_configs(od_config), nested_keys)
    if nested_value is not None:
        return cast(nested_value)

    top_value = _top_level_model_value(od_config, top_level_keys)
    if top_value is not None:
        return cast(top_value)

    return cast(default)


def _resolve_normalization_value(
    od_config: OmniDiffusionConfig,
    args: dict[str, Any],
    *,
    name: str,
    default: Any,
    aliases: tuple[str, ...] = (),
) -> Any:
    keys = (f"sound_{name}", name, *aliases)
    custom_value = _custom_arg_value(args, keys)
    if custom_value is not None:
        return custom_value
    nested_value = _first_value_from_configs(_nested_sound_tokenizer_configs(od_config), (name, *aliases))
    return default if nested_value is None else nested_value


def get_sound_config_value(
    od_config: OmniDiffusionConfig,
    name: str,
    default: Any,
    aliases: tuple[str, ...] = (),
) -> Any:
    # Backward-compatible generic accessor.  Prefer the more specific helpers
    # below for Cosmos3 sound tokenizer fields so precedence stays explicit.
    keys = (name, *aliases)
    for config in (
        _pipeline_args(od_config),
        getattr(od_config, "model_config", None),
        getattr(od_config, "tf_model_config", None),
    ):
        if config is None:
            continue
        for key in keys:
            if hasattr(config, "get"):
                value = config.get(key, None)
            else:
                value = getattr(config, key, None)
            if value is not None:
                return value
    return default


def get_sound_sample_rate(od_config: OmniDiffusionConfig) -> int:
    args = _pipeline_args(od_config)
    return _resolve_arch_value(
        od_config,
        args,
        {},
        field="sample_rate",
        custom_keys=("sound_sample_rate", "sample_rate"),
        nested_keys=("sample_rate", "sampling_rate"),
        top_level_keys=("sound_sample_rate", "sample_rate"),
        default=DEFAULT_SOUND_SAMPLE_RATE,
        cast=int,
    )


def get_sound_channels(od_config: OmniDiffusionConfig) -> int:
    args = _pipeline_args(od_config)
    return _resolve_arch_value(
        od_config,
        args,
        {},
        field="audio_channels",
        custom_keys=("sound_audio_channels", "audio_channels", "stereo"),
        nested_keys=("audio_channels", "dec_out_channels", "stereo"),
        top_level_keys=("sound_audio_channels", "audio_channels", "stereo"),
        default=DEFAULT_SOUND_CHANNELS,
        cast=_as_audio_channels,
    )


def get_sound_hop_size(od_config: OmniDiffusionConfig) -> int:
    args = _pipeline_args(od_config)
    return _resolve_arch_value(
        od_config,
        args,
        {},
        field="hop_size",
        custom_keys=("sound_hop_size", "hop_size"),
        nested_keys=("hop_size",),
        top_level_keys=("sound_hop_size", "hop_size"),
        default=DEFAULT_SOUND_HOP_SIZE,
        cast=int,
    )


class Cosmos3SoundTokenizer:
    """Thin adapter around the local AVAE tokenizer implementation."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.sample_rate = int(getattr(tokenizer, "sample_rate", DEFAULT_SOUND_SAMPLE_RATE))
        self.audio_channels = int(getattr(tokenizer, "audio_channels", DEFAULT_SOUND_CHANNELS))
        self.latent_ch = int(getattr(tokenizer, "latent_ch", DEFAULT_SOUND_DIM))
        self.hop_size = int(getattr(tokenizer, "temporal_compression_factor", DEFAULT_SOUND_HOP_SIZE))
        if self.hop_size <= 0:
            raise ValueError(f"Cosmos3 sound tokenizer hop_size must be positive, got {self.hop_size}.")
        self.latent_fps = float(self.sample_rate) / float(self.hop_size)

    @classmethod
    def from_config(cls, od_config: OmniDiffusionConfig) -> Cosmos3SoundTokenizer:
        args = _pipeline_args(od_config)
        model_path = getattr(od_config, "model", None)
        explicit_avae_path = (
            args.get("sound_tokenizer_path")
            or args.get("avae_path")
            or args.get("cosmos3_avae_path")
            or os.environ.get("COSMOS3_SOUND_TOKENIZER_PATH")
        )
        explicit_config_path = args.get("sound_tokenizer_config_path") or os.environ.get(
            "COSMOS3_SOUND_TOKENIZER_CONFIG_PATH"
        )

        model_root = str(model_path) if model_path and os.path.isdir(model_path) else None
        if model_root is None and model_path and not explicit_avae_path:
            from huggingface_hub import snapshot_download

            model_root = snapshot_download(
                repo_id=str(model_path),
                revision=getattr(od_config, "revision", None),
                allow_patterns=[
                    f"{SOUND_TOKENIZER_COMPONENT_NAME}/config.json",
                    f"{SOUND_TOKENIZER_COMPONENT_NAME}/{SOUND_TOKENIZER_CHECKPOINT_NAME}",
                ],
            )

        if explicit_avae_path:
            avae_path = _resolve_model_file(explicit_avae_path, model_root)
        else:
            tokenizer_dir = Path(model_root) / SOUND_TOKENIZER_COMPONENT_NAME if model_root else None
            candidate = tokenizer_dir / SOUND_TOKENIZER_CHECKPOINT_NAME if tokenizer_dir else None
            avae_path = str(candidate) if candidate and candidate.exists() else None

        if not avae_path:
            raise ValueError(
                "Cosmos3 sound generation was requested, but no AVAE sound "
                "tokenizer checkpoint was provided. Set "
                "custom_pipeline_args['sound_tokenizer_path'] or "
                "COSMOS3_SOUND_TOKENIZER_PATH, or include "
                f"{SOUND_TOKENIZER_COMPONENT_NAME}/{SOUND_TOKENIZER_CHECKPOINT_NAME} under the model path."
            )

        config_path = _resolve_model_file(explicit_config_path, model_root)
        if config_path is None and model_root:
            candidate = Path(model_root) / SOUND_TOKENIZER_COMPONENT_NAME / "config.json"
            config_path = str(candidate) if candidate.exists() else None
        component_config = _load_sound_tokenizer_component_config(config_path)
        component_values = _component_arch_values(component_config)

        sample_rate = _resolve_arch_value(
            od_config,
            args,
            component_values,
            field="sample_rate",
            custom_keys=("sound_sample_rate", "sample_rate"),
            nested_keys=("sample_rate", "sampling_rate"),
            top_level_keys=("sound_sample_rate", "sample_rate"),
            default=DEFAULT_SOUND_SAMPLE_RATE,
            cast=int,
        )
        audio_channels = _resolve_arch_value(
            od_config,
            args,
            component_values,
            field="audio_channels",
            custom_keys=("sound_audio_channels", "audio_channels", "stereo"),
            nested_keys=("audio_channels", "dec_out_channels", "stereo"),
            top_level_keys=("sound_audio_channels", "audio_channels", "stereo"),
            default=DEFAULT_SOUND_CHANNELS,
            cast=_as_audio_channels,
        )
        sound_dim = _resolve_arch_value(
            od_config,
            args,
            component_values,
            field="io_channels",
            custom_keys=("sound_dim", "io_channels", "latent_ch"),
            nested_keys=("io_channels", "vocoder_input_dim", "latent_ch"),
            top_level_keys=("sound_dim",),
            default=DEFAULT_SOUND_DIM,
            cast=int,
        )
        hop_size = _resolve_arch_value(
            od_config,
            args,
            component_values,
            field="hop_size",
            custom_keys=("sound_hop_size", "hop_size"),
            nested_keys=("hop_size",),
            top_level_keys=("sound_hop_size", "hop_size"),
            default=DEFAULT_SOUND_HOP_SIZE,
            cast=int,
        )
        normalize_latents = _as_bool(
            _resolve_normalization_value(
                od_config,
                args,
                name="normalize_latents",
                default=DEFAULT_SOUND_NORMALIZE_LATENTS,
            )
        )
        normalization_type = str(
            _resolve_normalization_value(
                od_config,
                args,
                name="normalization_type",
                default=DEFAULT_SOUND_NORMALIZATION_TYPE,
            )
        )
        tanh_input_scale = float(
            _resolve_normalization_value(
                od_config,
                args,
                name="tanh_input_scale",
                default=DEFAULT_SOUND_TANH_INPUT_SCALE,
            )
        )
        tanh_output_scale = float(
            _resolve_normalization_value(
                od_config,
                args,
                name="tanh_output_scale",
                default=DEFAULT_SOUND_TANH_OUTPUT_SCALE,
            )
        )
        tanh_clamp = float(
            _resolve_normalization_value(
                od_config,
                args,
                name="tanh_clamp",
                default=DEFAULT_SOUND_TANH_CLAMP,
            )
        )
        tokenizer = Cosmos3AVAEAudioTokenizer(
            checkpoint_path=str(avae_path),
            config_path=config_path,
            sample_rate=sample_rate,
            audio_channels=audio_channels,
            io_channels=sound_dim,
            hop_size=hop_size,
            normalize_latents=normalize_latents,
            normalization_type=normalization_type,
            tanh_input_scale=tanh_input_scale,
            tanh_output_scale=tanh_output_scale,
            tanh_clamp=tanh_clamp,
            dtype=getattr(od_config, "dtype", torch.bfloat16),
            device=get_local_device(),
        )
        if _is_rank_zero():
            logger.info(
                "Loaded Cosmos3 AVAE sound tokenizer from %s "
                "(sr=%d, channels=%d, latent_ch=%d, hop=%d, latent_fps=%.3f)",
                avae_path,
                sample_rate,
                audio_channels,
                sound_dim,
                hop_size,
                float(sample_rate) / float(hop_size),
            )
        return cls(tokenizer)

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(self.tokenizer.get_latent_num_samples(num_audio_samples))

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(self.tokenizer.get_audio_num_samples(num_latent_samples))

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode sound latents.

        Args:
            latents: ``[B, C, T]`` or ``[C, T]`` tensor.

        Returns:
            ``[B, audio_channels, N]`` tensor for batched input, or
            ``[audio_channels, N]`` for unbatched input.
        """
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        audio = self.tokenizer.decode(latents)
        audio = audio.clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio
