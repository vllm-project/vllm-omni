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

from .config_resolver import (
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


_DYNIN_CONFIG_CANDIDATE_RELPATHS = (
    "configs/dynin_omni.yaml",
    # Backward compatibility for older tree layouts.
    "models/configs/dynin_omni.yaml",
    # Some repos may keep the full project path in root.
    "vllm_omni/model_executor/models/dynin_omni/configs/dynin_omni.yaml",
    "dynin_omni.yaml",
)


def _looks_like_hf_repo_id(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    if value.count("/") != 1:
        return False
    org, name = value.split("/", 1)
    return bool(org and name)


def _find_dynin_config_under_root(root: Path) -> Path | None:
    root = root.expanduser()
    for rel_path in _DYNIN_CONFIG_CANDIDATE_RELPATHS:
        candidate = root / rel_path
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_hf_modules_transformers_root() -> Path:
    def _to_transformers_modules_root(path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if resolved.name == "transformers_modules":
            return resolved
        return (resolved / "transformers_modules").resolve()

    modules_root = os.getenv("HF_MODULES_CACHE")
    if modules_root:
        return _to_transformers_modules_root(Path(modules_root))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "modules" / "transformers_modules").resolve()
    try:
        from transformers.utils.hub import HF_MODULES_CACHE

        return _to_transformers_modules_root(Path(HF_MODULES_CACHE))
    except Exception:
        pass
    return (Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules").resolve()


def _resolve_hf_hub_cache_root() -> Path:
    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        env_value = os.getenv(env_name)
        if env_value:
            return Path(env_value).expanduser().resolve()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _iter_hf_module_dirs(repo_id: str) -> list[Path]:
    root = _resolve_hf_modules_transformers_root()
    if not root.is_dir():
        return []

    org, repo = repo_id.split("/", 1)
    org_token = org.replace("-", "_hyphen_")
    repo_token = repo.replace("-", "_hyphen_")
    repo_vllm_token = f"{repo_token}_vllm"

    candidates = [
        root / f"{org_token}_{repo_token}",
        root / f"{org_token}_{repo_vllm_token}",
        root / repo_token,
        root / repo_vllm_token,
        root / org_token / repo_token,
        root / org_token / repo_vllm_token,
    ]
    candidates.extend(root.glob(f"*{repo_token}*"))
    org_dir = root / org_token
    if org_dir.is_dir():
        candidates.extend(org_dir.glob(f"*{repo_token}*"))

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        unique_dirs.append(resolved)
    return unique_dirs


def _iter_hf_snapshot_dirs(repo_id: str) -> list[Path]:
    hub_root = _resolve_hf_hub_cache_root()
    if not hub_root.is_dir():
        return []

    org, repo = repo_id.split("/", 1)
    repo_cache_dir = hub_root / f"models--{org}--{repo}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return []

    candidates: list[Path] = []
    ref_main = repo_cache_dir / "refs" / "main"
    if ref_main.is_file():
        try:
            pinned = ref_main.read_text(encoding="utf-8").strip()
            if pinned:
                pinned_dir = (snapshots_dir / pinned).resolve()
                if pinned_dir.is_dir():
                    candidates.append(pinned_dir)
        except Exception:
            pass

    try:
        snapshot_dirs = sorted(
            (p.resolve() for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception:
        snapshot_dirs = []
    candidates.extend(snapshot_dirs)

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(candidate)
    return unique_dirs


@lru_cache(maxsize=16)
def _resolve_dynin_config_from_hf_repo(repo_id: str) -> str | None:
    if not _looks_like_hf_repo_id(repo_id):
        return None

    for module_dir in _iter_hf_module_dirs(repo_id):
        found = _find_dynin_config_under_root(module_dir)
        if found is not None:
            return str(found)

    for snapshot_dir in _iter_hf_snapshot_dirs(repo_id):
        found = _find_dynin_config_under_root(snapshot_dir)
        if found is not None:
            return str(found)

    return None


def _resolve_config_path(vllm_config: VllmConfig, runtime_info: dict[str, Any]) -> str | None:
    def _resolve_if_exists(path_like: Any, source_name: str) -> str | None:
        if path_like is None:
            return None
        text = str(path_like).strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if path.exists():
            return str(path.resolve())
        logger.warning(
            "DYNIN config path from %s does not exist: %s. Falling back to auto-discovery.",
            source_name,
            path,
        )
        return None

    runtime_path = _runtime_value(runtime_info, "dynin_config_path")
    resolved_runtime_path = _resolve_if_exists(runtime_path, "runtime_info.dynin_config_path")
    if resolved_runtime_path:
        return resolved_runtime_path

    env_path = os.getenv("DYNIN_CONFIG_PATH")
    resolved_env_path = _resolve_if_exists(env_path, "DYNIN_CONFIG_PATH")
    if resolved_env_path:
        return resolved_env_path

    cfg_path = getattr(vllm_config.model_config, "dynin_config_path", None)
    resolved_cfg_path = _resolve_if_exists(cfg_path, "vllm_config.model_config.dynin_config_path")
    if resolved_cfg_path:
        return resolved_cfg_path

    model_source = str(getattr(vllm_config.model_config, "model", "") or "")
    tokenizer_source = str(getattr(vllm_config.model_config, "tokenizer", "") or "")
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    if isinstance(hf_config, dict):
        hf_name_or_path = hf_config.get("_name_or_path", None)
    else:
        hf_name_or_path = getattr(hf_config, "_name_or_path", None)

    # If model/tokenizer source itself is a local directory, prefer local config there first.
    for source in (model_source, tokenizer_source):
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            found = _find_dynin_config_under_root(source_path)
            if found is not None:
                return str(found)

    module_root = Path(__file__).resolve().parent
    bundled_candidates = (
        module_root / "configs" / "dynin_omni.yaml",
        # Backward compatibility if an old tree layout is still present.
        module_root / "models" / "configs" / "dynin_omni.yaml",
    )
    for bundled in bundled_candidates:
        if bundled.exists():
            return str(bundled)

    # As a final fallback, try to resolve config from HF remote code caches/snapshots.
    hf_repo_candidates: list[str] = []
    for source in (model_source, tokenizer_source, hf_name_or_path):
        if not _looks_like_hf_repo_id(source):
            continue
        source_text = str(source)
        if source_text in hf_repo_candidates:
            continue
        hf_repo_candidates.append(source_text)

    for source in hf_repo_candidates:
        if _looks_like_hf_repo_id(source):
            resolved = _resolve_dynin_config_from_hf_repo(source)
            if resolved is not None:
                logger.info("Resolved dynin config from Hugging Face cache for %s: %s", source, resolved)
                return resolved

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
