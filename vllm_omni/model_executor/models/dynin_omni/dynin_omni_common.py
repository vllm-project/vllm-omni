from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
import types
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


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
DEFAULT_MAGVIT_REMOTE_CODE_REPO = "snu-aidas/magvitv2"

DEFAULT_DYNIN_REMOTE_CODE_REPO = "snu-aidas/Dynin-Omni"
_DYNIN_REMOTE_REPO_ENV = "DYNIN_REMOTE_CODE_REPO_ID"
_DYNIN_REMOTE_REV_ENV = "DYNIN_REMOTE_CODE_REVISION"
_DYNIN_REMOTE_LOCAL_ONLY_ENV = "DYNIN_REMOTE_CODE_LOCAL_FILES_ONLY"
_DYNIN_REMOTE_ALLOW_PATTERNS = ("*.py", "*.json", "*.yaml", "*.yml")
_DYNIN_MAGVIT_REMOTE_REPO_ENV = "DYNIN_MAGVIT_REMOTE_CODE_REPO_ID"
_DYNIN_MAGVIT_REMOTE_REV_ENV = "DYNIN_MAGVIT_REMOTE_CODE_REVISION"
_DYNIN_MAGVIT_REMOTE_LOCAL_ONLY_ENV = "DYNIN_MAGVIT_REMOTE_CODE_LOCAL_FILES_ONLY"

_DYNIN_REMOTE_CACHE_LOCK = threading.Lock()
_DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT: dict[str, str] = {}
_DYNIN_REMOTE_ATTR_CACHE: dict[tuple[str, str, str, str | None, bool], Any] = {}


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


@lru_cache(maxsize=16)
def _resolve_dynin_config_from_hf_repo(repo_id: str) -> str | None:
    if not _looks_like_hf_repo_id(repo_id):
        return None
    if snapshot_download is None:
        return None

    # Use huggingface_hub cache resolution directly instead of manually walking
    # HF cache internals. Keep local_files_only=True to preserve prior behavior
    # (cache lookup only; no network fetch during config auto-discovery).
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "allow_patterns": list(_DYNIN_CONFIG_CANDIDATE_RELPATHS),
        "local_files_only": True,
    }
    try:
        snapshot_dir = Path(snapshot_download(**download_kwargs)).expanduser().resolve()
    except Exception:
        return None

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

    # Prefer resolving from HF repo cache/snapshots early when model/tokenizer
    # source is a repo id.
    hf_repo_candidates: list[str] = []
    for source in (model_source, tokenizer_source, hf_name_or_path):
        if not _looks_like_hf_repo_id(source):
            continue
        source_text = str(source)
        if source_text in hf_repo_candidates:
            continue
        hf_repo_candidates.append(source_text)

    for source in hf_repo_candidates:
        resolved = _resolve_dynin_config_from_hf_repo(source)
        if resolved is not None:
            logger.info("Resolved dynin config from Hugging Face cache for %s: %s", source, resolved)
            return resolved

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

    resolver_source: str | None = base_model_source if base_model_source else None
    resolver_local_files_only: bool | None = True if base_model_path.is_dir() else None
    resolve_model_pretrained_source_fn = get_dynin_config_resolver_attr(
        "resolve_model_pretrained_source",
        source=resolver_source,
        local_files_only=resolver_local_files_only,
    )
    resolve_tokenizer_source_fn = get_dynin_config_resolver_attr(
        "resolve_tokenizer_source",
        source=resolver_source,
        local_files_only=resolver_local_files_only,
    )
    resolve_model_local_files_only_fn = get_dynin_config_resolver_attr(
        "resolve_model_local_files_only",
        source=resolver_source,
        local_files_only=resolver_local_files_only,
    )
    resolve_vq_cfg_block_fn = get_dynin_config_resolver_attr(
        "resolve_vq_cfg_block",
        source=resolver_source,
        local_files_only=resolver_local_files_only,
    )
    resolve_vq_repo_source_fn = get_dynin_config_resolver_attr(
        "resolve_vq_repo_source",
        source=resolver_source,
        local_files_only=resolver_local_files_only,
    )

    config_path = _resolve_config_path(vllm_config, runtime_info)
    if config_path:
        config_file = Path(config_path).expanduser()
        if config_file.exists():
            try:
                dynin_cfg = _load_omega_config(str(config_file))
                model_source = resolve_model_pretrained_source_fn(dynin_cfg, default=model_source)
                tokenizer_source = resolve_tokenizer_source_fn(dynin_cfg, default=tokenizer_source)
                model_local_files_only = resolve_model_local_files_only_fn(
                    dynin_cfg,
                    default=model_local_files_only,
                )
                vq_image_cfg = resolve_vq_cfg_block_fn(dynin_cfg, modality="image")
                vq_audio_cfg = resolve_vq_cfg_block_fn(dynin_cfg, modality="audio")
                vq_image_source = resolve_vq_repo_source_fn(vq_image_cfg, default=vq_image_source)
                vq_audio_source = resolve_vq_repo_source_fn(vq_audio_cfg, default=vq_audio_source)
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


def _resolve_dynin_remote_source(source: str | None) -> str:
    if isinstance(source, str):
        stripped = source.strip()
        if stripped:
            source_path = Path(stripped).expanduser()
            if source_path.is_dir():
                return str(source_path.resolve())
            if _looks_like_hf_repo_id(stripped):
                return stripped

    env_repo = os.getenv(_DYNIN_REMOTE_REPO_ENV)
    if _looks_like_hf_repo_id(env_repo):
        return str(env_repo).strip()

    return DEFAULT_DYNIN_REMOTE_CODE_REPO


def _resolve_dynin_remote_revision(revision: str | None) -> str | None:
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    env_revision = os.getenv(_DYNIN_REMOTE_REV_ENV)
    if isinstance(env_revision, str) and env_revision.strip():
        return env_revision.strip()
    return None


def _resolve_dynin_remote_local_files_only(local_files_only: bool | None) -> bool:
    if local_files_only is not None:
        return bool(local_files_only)
    return _to_bool(os.getenv(_DYNIN_REMOTE_LOCAL_ONLY_ENV), default=False)


def _resolve_dynin_remote_snapshot_dir(
    *,
    source: str,
    revision: str | None,
    local_files_only: bool,
) -> str:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return str(source_path.resolve())

    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required to load remote Dynin-Omni code.")

    kwargs: dict[str, Any] = {
        "repo_id": source,
        "repo_type": "model",
        "allow_patterns": list(_DYNIN_REMOTE_ALLOW_PATTERNS),
        "local_files_only": bool(local_files_only),
    }
    if revision is not None:
        kwargs["revision"] = revision

    try:
        return str(snapshot_download(**kwargs))
    except TypeError:
        kwargs.pop("local_files_only", None)
        return str(snapshot_download(**kwargs))


def _ensure_dynin_remote_package(snapshot_dir: str) -> str:
    with _DYNIN_REMOTE_CACHE_LOCK:
        package_name = _DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT.get(snapshot_dir)
        if package_name is not None:
            return package_name

        digest = hashlib.sha1(snapshot_dir.encode("utf-8")).hexdigest()[:12]
        package_name = f"_dynin_hf_remote_{digest}"
        package = types.ModuleType(package_name)
        package.__path__ = [snapshot_dir]  # type: ignore[attr-defined]
        package.__file__ = str(Path(snapshot_dir) / "__init__.py")
        sys.modules.setdefault(package_name, package)
        _DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT[snapshot_dir] = package_name
        return package_name


def _load_dynin_remote_module(
    *,
    module_name: str,
    source: str,
    revision: str | None,
    local_files_only: bool,
):
    snapshot_dir = _resolve_dynin_remote_snapshot_dir(
        source=source,
        revision=revision,
        local_files_only=local_files_only,
    )
    module_path = Path(snapshot_dir) / f"{module_name}.py"
    if not module_path.is_file():
        raise ImportError(
            f"Dynin remote code module '{module_name}.py' not found under '{snapshot_dir}'. source={source!r}"
        )

    package_name = _ensure_dynin_remote_package(snapshot_dir)
    full_name = f"{package_name}.{module_name}"
    existing = sys.modules.get(full_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for '{module_path}'.")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(full_name, None)
        raise
    return module


def get_dynin_remote_attr(
    attr_name: str,
    *,
    module_name: str,
    source: str | None = None,
    revision: str | None = None,
    local_files_only: bool | None = None,
    fallback_module_names: Iterable[str] = (),
    optional: bool = False,
) -> Any | None:
    resolved_source = _resolve_dynin_remote_source(source)
    resolved_revision = _resolve_dynin_remote_revision(revision)
    resolved_local_only = _resolve_dynin_remote_local_files_only(local_files_only)

    module_candidates = [module_name, *[m for m in fallback_module_names if m and m != module_name]]
    last_error: Exception | None = None

    for candidate in module_candidates:
        cache_key = (attr_name, candidate, resolved_source, resolved_revision, resolved_local_only)
        cached = _DYNIN_REMOTE_ATTR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            module = _load_dynin_remote_module(
                module_name=candidate,
                source=resolved_source,
                revision=resolved_revision,
                local_files_only=resolved_local_only,
            )
            if hasattr(module, attr_name):
                value = getattr(module, attr_name)
                _DYNIN_REMOTE_ATTR_CACHE[cache_key] = value
                return value
        except Exception as e:
            last_error = e
            continue

    if optional:
        if last_error is not None:
            logger.debug(
                "Optional Dynin remote attr not found: attr=%s source=%s revision=%s err=%s",
                attr_name,
                resolved_source,
                resolved_revision,
                last_error,
            )
        return None

    raise ImportError(
        f"Failed to resolve '{attr_name}' from remote Dynin code "
        f"(source={resolved_source!r}, revision={resolved_revision!r}, modules={module_candidates})."
    ) from last_error


_DYNIN_MODELING_REMOTE_EXPORTS = {
    "DyninOmniConfig": "DyninOmniConfig",
    "DyninOmniModelLM": "DyninOmniModelLM",
    "VideoTokenMerger": "VideoTokenMerger",
}


def get_dynin_modeling_attr(name: str) -> Any:
    attr_name = _DYNIN_MODELING_REMOTE_EXPORTS.get(name)
    if attr_name is None:
        raise AttributeError(f"Unsupported Dynin modeling export: {name!r}")
    return get_dynin_remote_attr(attr_name, module_name="modeling_dynin_omni")


_DYNIN_SAMPLING_REMOTE_EXPORTS = {
    "log": "log",
    "gumbel_noise": "gumbel_noise",
    "gumbel_sample": "gumbel_sample",
    "top_k": "top_k",
    "mask_by_random_topk": "mask_by_random_topk",
    "cosine_schedule": "cosine_schedule",
    "linear_schedule": "linear_schedule",
    "pow": "pow",
    "sigmoid_schedule": "sigmoid_schedule",
    "get_mask_schedule": "get_mask_schedule",
    "top_k_top_p_filtering": "top_k_top_p_filtering",
}


def get_dynin_sampling_attr(name: str) -> Any:
    attr_name = _DYNIN_SAMPLING_REMOTE_EXPORTS.get(name)
    if attr_name is None:
        raise AttributeError(f"Unsupported Dynin sampling export: {name!r}")
    return get_dynin_remote_attr(attr_name, module_name="sampling")


_DYNIN_CONFIG_RESOLVER_REMOTE_EXPORTS = {
    "resolve_model_pretrained_source": "resolve_model_pretrained_source",
    "resolve_tokenizer_source": "resolve_tokenizer_source",
    "resolve_model_local_files_only": "resolve_model_local_files_only",
    "resolve_vq_cfg_block": "resolve_vq_cfg_block",
    "resolve_vq_repo_source": "resolve_vq_repo_source",
}


def get_dynin_config_resolver_attr(
    name: str,
    *,
    source: str | None = None,
    revision: str | None = None,
    local_files_only: bool | None = None,
) -> Any:
    attr_name = _DYNIN_CONFIG_RESOLVER_REMOTE_EXPORTS.get(name)
    if attr_name is None:
        raise AttributeError(f"Unsupported Dynin config_resolver export: {name!r}")

    if source is not None:
        value = get_dynin_remote_attr(
            attr_name,
            module_name="config_resolver",
            source=source,
            revision=revision,
            local_files_only=local_files_only,
            optional=True,
        )
        if value is not None:
            return value

    return get_dynin_remote_attr(
        attr_name,
        module_name="config_resolver",
        source=DEFAULT_DYNIN_REMOTE_CODE_REPO,
        revision=revision,
        local_files_only=local_files_only,
    )


_DYNIN_MAGVIT_REMOTE_EXPORTS = {
    "VQGANEncoder": "VQGANEncoder",
    "VQGANDecoder": "VQGANDecoder",
    "LFQuantizer": "LFQuantizer",
    "MAGVITv2": "MAGVITv2",
}


def _resolve_magvit_remote_source(source: str | None) -> str:
    if isinstance(source, str):
        stripped = source.strip()
        if stripped:
            source_path = Path(stripped).expanduser()
            if source_path.is_dir():
                return str(source_path.resolve())
            if _looks_like_hf_repo_id(stripped):
                return stripped

    env_repo = os.getenv(_DYNIN_MAGVIT_REMOTE_REPO_ENV)
    if _looks_like_hf_repo_id(env_repo):
        return str(env_repo).strip()

    return DEFAULT_MAGVIT_REMOTE_CODE_REPO


def _resolve_magvit_remote_revision(revision: str | None) -> str | None:
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    env_revision = os.getenv(_DYNIN_MAGVIT_REMOTE_REV_ENV)
    if isinstance(env_revision, str) and env_revision.strip():
        return env_revision.strip()
    return None


def _resolve_magvit_remote_local_files_only(local_files_only: bool | None) -> bool:
    if local_files_only is not None:
        return bool(local_files_only)
    return _to_bool(os.getenv(_DYNIN_MAGVIT_REMOTE_LOCAL_ONLY_ENV), default=False)


def get_dynin_magvit_attr(
    name: str,
    *,
    source: str | None = None,
    revision: str | None = None,
    local_files_only: bool | None = None,
) -> Any:
    attr_name = _DYNIN_MAGVIT_REMOTE_EXPORTS.get(name)
    if attr_name is None:
        raise AttributeError(f"Unsupported Dynin MAGVIT export: {name!r}")

    resolved_source = _resolve_magvit_remote_source(source)
    resolved_revision = _resolve_magvit_remote_revision(revision)
    resolved_local_only = _resolve_magvit_remote_local_files_only(local_files_only)

    value = get_dynin_remote_attr(
        attr_name,
        module_name="modeling_magvitv2",
        source=resolved_source,
        revision=resolved_revision,
        local_files_only=resolved_local_only,
        optional=True,
    )
    if value is not None:
        return value

    # Fallback to default MAGVIT remote repository if caller source does not expose code file.
    if resolved_source != DEFAULT_MAGVIT_REMOTE_CODE_REPO:
        return get_dynin_remote_attr(
            attr_name,
            module_name="modeling_magvitv2",
            source=DEFAULT_MAGVIT_REMOTE_CODE_REPO,
            revision=resolved_revision,
            local_files_only=resolved_local_only,
            optional=False,
        )

    raise ImportError(
        f"Failed to resolve MAGVIT attr '{attr_name}' from source={resolved_source!r} (revision={resolved_revision!r})."
    )
