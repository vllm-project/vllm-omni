from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .models.runtime.config_resolver import (
    resolve_model_local_files_only,
    resolve_model_pretrained_source,
    resolve_tokenizer_source,
    resolve_vq_cfg_block,
    resolve_vq_repo_source,
)

logger = init_logger(__name__)


DETOK_TEXT = 0
DETOK_AUDIO = 1
DETOK_IMAGE = 2


TASK_TO_DETOK = {
    "mmu": DETOK_TEXT,
    "s2t": DETOK_TEXT,
    "mmu_fast": DETOK_TEXT,
    "mmu_fastdllm_v1": DETOK_TEXT,
    "v2t": DETOK_TEXT,
    "t2s": DETOK_AUDIO,
    "t2s_mmu_like": DETOK_AUDIO,
    "t2s_fixed": DETOK_AUDIO,
    "s2s": DETOK_AUDIO,
    "v2s": DETOK_AUDIO,
    "t2i": DETOK_IMAGE,
    "i2i": DETOK_IMAGE,
    "ti2ti": DETOK_IMAGE,
}

DEFAULT_VQ_IMAGE_SOURCE = "snu-aidas/magvitv2"
DEFAULT_VQ_AUDIO_SOURCE = "snu-aidas/emova_speech_tokenizer_vllm"


@dataclass(frozen=True)
class DyninInferSources:
    model_source: str
    tokenizer_source: str
    vq_image_source: str
    vq_audio_source: str
    model_local_files_only: bool
    vq_image_local_files_only: bool
    vq_audio_local_files_only: bool
    config_path: str | None = None

    @property
    def local_files_only(self) -> bool:
        # Backward compatibility for old call sites.
        return self.model_local_files_only


def first_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, list):
        if not value:
            return default
        return value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        if value.numel() == 1:
            return value.item()
        return value
    return value


def get_runtime_info(runtime_additional_information: Any) -> dict[str, Any]:
    if isinstance(runtime_additional_information, list):
        if not runtime_additional_information:
            return {}
        value = runtime_additional_information[0]
        return value if isinstance(value, dict) else {}
    if isinstance(runtime_additional_information, dict):
        return runtime_additional_information
    return {}


def to_token_1d(value: Any, ref_device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]
    if isinstance(value, list):
        if not value:
            device = ref_device or torch.device("cpu")
            return torch.empty(0, dtype=torch.long, device=device)
        if isinstance(value[0], torch.Tensor):
            value = value[0]
        else:
            value = torch.tensor(value[0] if isinstance(value[0], list) else value, dtype=torch.long)
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=torch.long)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    if value.ndim > 1:
        value = value[0]
    if ref_device is not None and value.device != ref_device:
        value = value.to(ref_device)
    return value.to(dtype=torch.long).contiguous()


def _first_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue <= 0:
        return None
    return ivalue


def resolve_hidden_size(
    *,
    vllm_config: VllmConfig,
    model: Any | None = None,
    default: int = 1024,
) -> int:
    if model is not None:
        try:
            embeddings = model.get_input_embeddings()
            weight = getattr(embeddings, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim >= 2:
                hidden_size = _first_positive_int(weight.shape[-1])
                if hidden_size is not None:
                    return hidden_size
        except Exception:
            pass

        model_cfg = getattr(model, "config", None)
        for key in ("hidden_size", "d_model", "n_embd", "dim", "model_dim", "embed_dim"):
            hidden_size = _first_positive_int(getattr(model_cfg, key, None))
            if hidden_size is not None:
                return hidden_size

    config_candidates = [
        getattr(vllm_config.model_config, "hf_config", None),
        getattr(vllm_config.model_config, "hf_text_config", None),
    ]
    for config_obj in config_candidates:
        if config_obj is None:
            continue
        for key in ("hidden_size", "d_model", "n_embd", "dim", "model_dim", "embed_dim"):
            if isinstance(config_obj, dict):
                value = config_obj.get(key)
            else:
                value = getattr(config_obj, key, None)
            hidden_size = _first_positive_int(value)
            if hidden_size is not None:
                return hidden_size

    return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", "", "none", "null"):
        return False
    return default


def _runtime_value(runtime_info: dict[str, Any], key: str) -> Any:
    return first_value(runtime_info.get(key), None)


def _node_value(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        return node.get(key, default)
    except Exception:
        return getattr(node, key, default)


def _runtime_first_value(runtime_info: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _runtime_value(runtime_info, key)
        if value is not None:
            return value
    return None


def _resolve_config_path(vllm_config: VllmConfig, runtime_info: dict[str, Any]) -> str | None:
    runtime_path = _runtime_value(runtime_info, "dynin_config_path")
    if runtime_path:
        return str(runtime_path)

    env_path = os.getenv("DYNIN_CONFIG_PATH")
    if env_path:
        return env_path

    cfg_path = getattr(vllm_config.model_config, "dynin_config_path", None)
    if cfg_path:
        return str(cfg_path)

    bundled = Path(__file__).resolve().parent / "models" / "configs" / "dynin_omni_demo.yaml"
    if bundled.exists():
        return str(bundled)

    return None


@lru_cache(maxsize=16)
def _load_omega_config(config_path: str) -> Any:
    return OmegaConf.load(config_path)


def resolve_dynin_infer_sources(
    *,
    vllm_config: VllmConfig,
    runtime_info: dict[str, Any] | None = None,
) -> DyninInferSources:
    runtime_info = runtime_info or {}

    base_model_source = str(getattr(vllm_config.model_config, "model", ""))
    base_model_path = Path(base_model_source).expanduser()
    local_vllm_model_source = str(base_model_path) if base_model_path.is_dir() else None

    model_source = base_model_source
    tokenizer_source = model_source
    vq_image_source = DEFAULT_VQ_IMAGE_SOURCE
    vq_audio_source = DEFAULT_VQ_AUDIO_SOURCE
    model_local_files_only = False
    vq_image_local_files_only = False
    vq_audio_local_files_only = False

    config_path = _resolve_config_path(vllm_config, runtime_info)
    if config_path:
        config_file = Path(config_path).expanduser()
        if config_file.exists():
            try:
                dynin_cfg = _load_omega_config(str(config_file))
                model_source = resolve_model_pretrained_source(dynin_cfg, default=model_source)
                tokenizer_source = resolve_tokenizer_source(dynin_cfg, default=tokenizer_source)
                model_local_files_only = resolve_model_local_files_only(
                    dynin_cfg,
                    default=model_local_files_only,
                )
                vq_image_cfg = resolve_vq_cfg_block(dynin_cfg, modality="image")
                vq_audio_cfg = resolve_vq_cfg_block(dynin_cfg, modality="audio")
                vq_image_source = resolve_vq_repo_source(vq_image_cfg, default=vq_image_source)
                vq_audio_source = resolve_vq_repo_source(vq_audio_cfg, default=vq_audio_source)
                vq_image_local_files_only = _to_bool(
                    _node_value(vq_image_cfg, "local_files_only", None),
                    default=model_local_files_only,
                )
                vq_audio_local_files_only = _to_bool(
                    _node_value(vq_audio_cfg, "local_files_only", None),
                    default=model_local_files_only,
                )
            except Exception as e:
                logger.warning("Failed to resolve DYNIN inference config from %s: %s", config_file, e)
        else:
            logger.warning("DYNIN config path does not exist: %s", config_file)

    runtime_model_source = _runtime_value(runtime_info, "dynin_model_path")
    if runtime_model_source:
        model_source = str(runtime_model_source)

    runtime_tokenizer_source = _runtime_value(runtime_info, "tokenizer_path")
    if runtime_tokenizer_source:
        tokenizer_source = str(runtime_tokenizer_source)

    runtime_vq_image_source = _runtime_value(runtime_info, "vq_model_image_path")
    if runtime_vq_image_source is None:
        runtime_vq_image_source = _runtime_value(runtime_info, "vq_model_path_image")
    if runtime_vq_image_source:
        vq_image_source = str(runtime_vq_image_source)

    runtime_vq_audio_source = _runtime_value(runtime_info, "vq_model_audio_path")
    if runtime_vq_audio_source is None:
        runtime_vq_audio_source = _runtime_value(runtime_info, "vq_model_path_audio")
    if runtime_vq_audio_source:
        vq_audio_source = str(runtime_vq_audio_source)

    runtime_local_global = _runtime_value(runtime_info, "local_files_only")
    runtime_local_model = _runtime_first_value(
        runtime_info,
        ("model_local_files_only", "local_files_only_model"),
    )
    runtime_local_vq_image = _runtime_first_value(
        runtime_info,
        ("vq_model_image_local_files_only", "local_files_only_vq_image"),
    )
    runtime_local_vq_audio = _runtime_first_value(
        runtime_info,
        ("vq_model_audio_local_files_only", "local_files_only_vq_audio"),
    )

    if runtime_local_global is not None:
        global_local = _to_bool(runtime_local_global, default=False)
        if runtime_local_model is None:
            model_local_files_only = global_local
        if runtime_local_vq_image is None:
            vq_image_local_files_only = global_local
        if runtime_local_vq_audio is None:
            vq_audio_local_files_only = global_local

    if runtime_local_model is not None:
        model_local_files_only = _to_bool(runtime_local_model, default=model_local_files_only)
    if runtime_local_vq_image is not None:
        vq_image_local_files_only = _to_bool(runtime_local_vq_image, default=vq_image_local_files_only)
    if runtime_local_vq_audio is not None:
        vq_audio_local_files_only = _to_bool(runtime_local_vq_audio, default=vq_audio_local_files_only)

    if runtime_local_global is None and runtime_local_model is None and local_vllm_model_source is not None:
        # In vllm-omni stages, vllm_config.model_config.model is typically already
        # a localized model dir. Prefer it over YAML repo_id for Stage-0 init.
        model_local_files_only = True

    if local_vllm_model_source is not None:
        if not runtime_model_source:
            if model_source != local_vllm_model_source:
                logger.info(
                    "DYNIN infer model source overridden to local vLLM model path: %s (from %s)",
                    local_vllm_model_source,
                    model_source,
                )
            model_source = local_vllm_model_source
        if not runtime_tokenizer_source:
            tokenizer_source = local_vllm_model_source

    return DyninInferSources(
        model_source=model_source,
        tokenizer_source=tokenizer_source,
        vq_image_source=vq_image_source,
        vq_audio_source=vq_audio_source,
        model_local_files_only=model_local_files_only,
        vq_image_local_files_only=vq_image_local_files_only,
        vq_audio_local_files_only=vq_audio_local_files_only,
        config_path=config_path,
    )
