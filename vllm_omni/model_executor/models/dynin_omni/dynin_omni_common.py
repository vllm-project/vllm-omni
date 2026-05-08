from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
import types
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None


class DetokTarget(IntEnum):
    TEXT = 0
    AUDIO = 1
    IMAGE = 2


TASK_TO_DETOK = {
    "mmu": DetokTarget.TEXT,
    "s2t": DetokTarget.TEXT,
    "mmu_fast": DetokTarget.TEXT,
    "mmu_fastdllm_v1": DetokTarget.TEXT,
    "v2t": DetokTarget.TEXT,
    "t2s": DetokTarget.AUDIO,
    "t2s_mmu_like": DetokTarget.AUDIO,
    "t2s_fixed": DetokTarget.AUDIO,
    "s2s": DetokTarget.AUDIO,
    "v2s": DetokTarget.AUDIO,
    "t2i": DetokTarget.IMAGE,
    "i2i": DetokTarget.IMAGE,
    "ti2ti": DetokTarget.IMAGE,
}

DEFAULT_VQ_IMAGE_SOURCE = "snu-aidas/magvitv2"
DEFAULT_VQ_AUDIO_SOURCE = "snu-aidas/emova_speech_tokenizer_vllm"
DEFAULT_MAGVIT_REMOTE_CODE_REPO = "snu-aidas/magvitv2"
DEFAULT_DYNIN_REMOTE_CODE_REPO = "snu-aidas/Dynin-Omni"
DYNIN_PROMPT_SOURCE_KEY = "dynin_prompt_source"
DYNIN_PROMPT_SOURCE_OFFLINE_PREBUILT = "offline_prebuilt"

DYNIN_TASK_DEFAULT_RUNTIME = {
    "t2t": ("mmu", "mmu", 0, "text"),
    "t2i": ("t2i", "t2i_gen", 2, "image"),
    "t2s": ("t2s_mmu_like", "t2s_gen", 1, "audio"),
    "i2i": ("i2i", "i2i", 2, "image"),
}

DYNIN_TASK_RUNTIME_FALLBACKS: dict[str, dict[str, Any]] = {
    "t2t": {
        "prompt_max_text_len": 1024,
        "max_new_tokens": 1024,
        "steps": 1024,
        "block_length": 16,
        "temperature": 0.0,
        "cfg_scale": 0.0,
    },
    "t2i": {
        "prompt_max_text_len": 128,
        "image_token_count": 1024,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "timesteps": 20,
        "guidance_scale": 3.5,
        "temperature": 1.0,
    },
    "i2i": {
        "prompt_max_text_len": 128,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "timesteps": 64,
        "guidance_scale": 3.5,
        "temperature": 1.0,
        "image_resolution": 336,
        "use_train_i2i_prompt": True,
    },
    "t2s": {
        "runtime_task": "t2s_mmu_like",
        "prompting_task": "t2s_gen",
        "prompt_max_text_len": 1024,
        "t2s_token_length": 512,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "audio_codebook_size": 4096,
        "steps": 512,
        "block_length": 128,
        "temperature": 1.0,
        "cfg_scale": 2.5,
        "t2s_condition": "gender-female_emotion-neutral_speed-normal_pitch-normal",
    },
}

DEFAULT_DYNIN_T2S_INSTRUCTION = "Please read the following text naturally."

DYNIN_SPECIAL_TOKENS = (
    "<|soi|>",
    "<|eoi|>",
    "<|sov|>",
    "<|eov|>",
    "<|t2i|>",
    "<|mmu|>",
    "<|t2v|>",
    "<|v2v|>",
    "<|lvg|>",
    "<|i2i|>",
    "<|ti2ti|>",
    "<|v2t|>",
    "<|v2s|>",
    "<|s2t|>",
    "<|t2s|>",
    "<|s2s|>",
    "<|soa|>",
    "<|eoa|>",
)

_DYNIN_ONLINE_PROMPT_TOKEN_BY_TASK = {
    "t2i": "<|t2i|>",
    "i2i": "<|i2i|>",
    "t2s": "<|t2s|>",
}

_DYNIN_MODALITY_PLACEHOLDERS = (
    "<|soi|><|image|><|eoi|>",
    "<|sov|><|video|><|eov|>",
    "<|soa|><|audio|><|eoa|>",
)

_DYNIN_CONFIG_CANDIDATE_RELPATHS = (
    "configs/dynin_omni.yaml",
    "models/configs/dynin_omni.yaml",
    "vllm_omni/model_executor/models/dynin_omni/configs/dynin_omni.yaml",
    "vllm_omni/model_executor/stage_configs/dynin_omni.yaml",
    "dynin_omni.yaml",
)

_DYNIN_REMOTE_ALLOW_PATTERNS = ("*.py", "*.json", "*.yaml", "*.yml")

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
        return self.model_local_files_only


@dataclass(frozen=True)
class RemoteCodeSettings:
    default_repo: str
    repo_env: str
    revision_env: str
    local_only_env: str


DYNIN_REMOTE_SETTINGS = RemoteCodeSettings(
    default_repo=DEFAULT_DYNIN_REMOTE_CODE_REPO,
    repo_env="DYNIN_REMOTE_CODE_REPO_ID",
    revision_env="DYNIN_REMOTE_CODE_REVISION",
    local_only_env="DYNIN_REMOTE_CODE_LOCAL_FILES_ONLY",
)

MAGVIT_REMOTE_SETTINGS = RemoteCodeSettings(
    default_repo=DEFAULT_MAGVIT_REMOTE_CODE_REPO,
    repo_env="DYNIN_MAGVIT_REMOTE_CODE_REPO_ID",
    revision_env="DYNIN_MAGVIT_REMOTE_CODE_REVISION",
    local_only_env="DYNIN_MAGVIT_REMOTE_CODE_LOCAL_FILES_ONLY",
)


def unwrap_first_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, list):
        return default if not value else value[0]
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        if value.numel() == 1:
            return value.item()
        return value
    return value


def normalize_runtime_info(runtime_additional_information: Any) -> dict[str, Any]:
    if isinstance(runtime_additional_information, list):
        if not runtime_additional_information:
            return {}
        first = runtime_additional_information[0]
        return first if isinstance(first, dict) else {}
    if isinstance(runtime_additional_information, dict):
        return runtime_additional_information
    return {}


def logical_dynin_task(task: Any) -> str:
    task_text = str(unwrap_first_value(task, "") or "").strip().lower()
    if task_text in ("t2s", "t2s_mmu_like", "t2s_fixed"):
        return "t2s"
    if task_text in ("t2i", "i2i"):
        return task_text
    return "t2t"


def dynin_runtime_fallback(task: str, key: str, value: Any = None) -> Any:
    if isinstance(value, str):
        if value.strip() != "":
            return value
    elif value is not None:
        return value
    return DYNIN_TASK_RUNTIME_FALLBACKS.get(task, {}).get(key)


def coerce_token_ids_1d(
    value: Any,
    ref_device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(value, tuple):
        value = value[0]

    if isinstance(value, list):
        if not value:
            device = ref_device or torch.device("cpu")
            return torch.empty(0, dtype=torch.long, device=device)
        if isinstance(value[0], torch.Tensor):
            value = value[0]
        else:
            value = torch.tensor(
                value[0] if isinstance(value[0], list) else value,
                dtype=torch.long,
            )

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
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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

    for config_obj in (
        getattr(vllm_config.model_config, "hf_config", None),
        getattr(vllm_config.model_config, "hf_text_config", None),
    ):
        if config_obj is None:
            continue
        for key in ("hidden_size", "d_model", "n_embd", "dim", "model_dim", "embed_dim"):
            value = config_obj.get(key) if isinstance(config_obj, dict) else getattr(config_obj, key, None)
            hidden_size = _first_positive_int(value)
            if hidden_size is not None:
                return hidden_size

    return default


def build_zero_input_embeddings(
    *,
    input_ids: torch.Tensor,
    hidden_size: int,
    stage_name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if input_ids.ndim == 0:
        shape = (1, hidden_size)
    elif input_ids.ndim == 1:
        shape = (input_ids.shape[0], hidden_size)
    elif input_ids.ndim == 2:
        shape = (input_ids.shape[0], input_ids.shape[1], hidden_size)
    else:
        raise ValueError(f"Unsupported input_ids rank for {stage_name}: {input_ids.ndim}")
    return torch.zeros(shape, dtype=dtype, device=input_ids.device)


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
    return unwrap_first_value(runtime_info.get(key), None)


def _runtime_first_value(runtime_info: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = _runtime_value(runtime_info, key)
        if value is not None:
            return value
    return None


def _node_value(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    try:
        return node.get(key, default)
    except Exception:
        return getattr(node, key, default)


def _looks_like_hf_repo_id(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    if value.count("/") != 1:
        return False
    org, name = value.split("/", 1)
    return bool(org and name)


def _find_dynin_config_under_root(root: Path) -> Path | None:
    for rel_path in _DYNIN_CONFIG_CANDIDATE_RELPATHS:
        candidate = root.expanduser() / rel_path
        if candidate.exists():
            return candidate.resolve()
    return None


@lru_cache(maxsize=16)
def _resolve_dynin_config_from_hf_repo(repo_id: str) -> str | None:
    if not _looks_like_hf_repo_id(repo_id) or snapshot_download is None:
        return None

    try:
        snapshot_dir = (
            Path(
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    allow_patterns=list(_DYNIN_CONFIG_CANDIDATE_RELPATHS),
                    local_files_only=True,
                )
            )
            .expanduser()
            .resolve()
        )
    except Exception:
        return None

    found = _find_dynin_config_under_root(snapshot_dir)
    return str(found) if found is not None else None


def _resolve_existing_path(path_like: Any, source_name: str) -> str | None:
    if path_like is None:
        return None
    text = str(path_like).strip()
    if not text:
        return None

    path = Path(text).expanduser()
    if path.is_file():
        return str(path.resolve())

    logger.warning(
        "DYNIN config path from %s does not exist: %s. Falling back to auto-discovery.",
        source_name,
        path,
    )
    return None


def _resolve_config_path(vllm_config: VllmConfig, runtime_info: dict[str, Any]) -> str | None:
    for value, name in (
        (_runtime_value(runtime_info, "dynin_config_path"), "runtime_info.dynin_config_path"),
        (os.getenv("DYNIN_CONFIG_PATH"), "DYNIN_CONFIG_PATH"),
        (getattr(vllm_config.model_config, "dynin_config_path", None), "vllm_config.model_config.dynin_config_path"),
    ):
        resolved = _resolve_existing_path(value, name)
        if resolved:
            return resolved

    model_source = str(getattr(vllm_config.model_config, "model", "") or "")
    tokenizer_source = str(getattr(vllm_config.model_config, "tokenizer", "") or "")
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    hf_name_or_path = (
        hf_config.get("_name_or_path") if isinstance(hf_config, dict) else getattr(hf_config, "_name_or_path", None)
    )

    hf_repo_candidates: list[str] = []
    for source in (model_source, tokenizer_source, hf_name_or_path):
        if not _looks_like_hf_repo_id(source):
            continue
        source = str(source)
        if source not in hf_repo_candidates:
            hf_repo_candidates.append(source)

    for source in hf_repo_candidates:
        resolved = _resolve_dynin_config_from_hf_repo(source)
        if resolved is not None:
            logger.info("Resolved dynin config from Hugging Face cache for %s: %s", source, resolved)
            return resolved

    for source in (model_source, tokenizer_source):
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            found = _find_dynin_config_under_root(source_path)
            if found is not None:
                return str(found)

    module_root = Path(__file__).resolve().parent
    for bundled in (
        module_root / "configs" / "dynin_omni.yaml",
        module_root / "models" / "configs" / "dynin_omni.yaml",
        module_root.parent / "stage_configs" / "dynin_omni.yaml",
    ):
        if bundled.exists():
            return str(bundled)

    return None


@lru_cache(maxsize=16)
def _load_omega_config(config_path: str) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(
            f"omegaconf is required to load Dynin config files. Install it to read config: {config_path}"
        ) from e
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
                model_source = resolve_model_pretrained_source_fn(
                    dynin_cfg,
                    default=model_source,
                )
                tokenizer_source = resolve_tokenizer_source_fn(
                    dynin_cfg,
                    default=tokenizer_source,
                )
                model_local_files_only = resolve_model_local_files_only_fn(
                    dynin_cfg,
                    default=model_local_files_only,
                )
                vq_image_cfg = resolve_vq_cfg_block_fn(dynin_cfg, modality="image")
                vq_audio_cfg = resolve_vq_cfg_block_fn(dynin_cfg, modality="audio")
                vq_image_source = resolve_vq_repo_source_fn(
                    vq_image_cfg,
                    default=vq_image_source,
                )
                vq_audio_source = resolve_vq_repo_source_fn(
                    vq_audio_cfg,
                    default=vq_audio_source,
                )
                vq_image_local_files_only = _to_bool(
                    _node_value(vq_image_cfg, "local_files_only", None),
                    default=model_local_files_only,
                )
                vq_audio_local_files_only = _to_bool(
                    _node_value(vq_audio_cfg, "local_files_only", None),
                    default=model_local_files_only,
                )
            except Exception as e:
                logger.warning(
                    "Failed to resolve DYNIN inference config from %s: %s",
                    config_file,
                    e,
                )
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
        model_local_files_only = _to_bool(
            runtime_local_model,
            default=model_local_files_only,
        )
    if runtime_local_vq_image is not None:
        vq_image_local_files_only = _to_bool(
            runtime_local_vq_image,
            default=vq_image_local_files_only,
        )
    if runtime_local_vq_audio is not None:
        vq_audio_local_files_only = _to_bool(
            runtime_local_vq_audio,
            default=vq_audio_local_files_only,
        )

    if runtime_local_global is None and runtime_local_model is None and local_vllm_model_source is not None:
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


def _resolve_remote_source(source: str | None, settings: RemoteCodeSettings) -> str:
    if isinstance(source, str):
        stripped = source.strip()
        if stripped:
            source_path = Path(stripped).expanduser()
            if source_path.is_dir():
                return str(source_path.resolve())
            if _looks_like_hf_repo_id(stripped):
                return stripped

    env_repo = os.getenv(settings.repo_env)
    if _looks_like_hf_repo_id(env_repo):
        return str(env_repo).strip()

    return settings.default_repo


def _resolve_remote_revision(revision: str | None, settings: RemoteCodeSettings) -> str | None:
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    env_revision = os.getenv(settings.revision_env)
    if isinstance(env_revision, str) and env_revision.strip():
        return env_revision.strip()
    return None


def _resolve_remote_local_only(local_files_only: bool | None, settings: RemoteCodeSettings) -> bool:
    if local_files_only is not None:
        return bool(local_files_only)
    return _to_bool(os.getenv(settings.local_only_env), default=False)


def _resolve_remote_snapshot_dir(
    *,
    source: str,
    revision: str | None,
    local_files_only: bool,
) -> str:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return str(source_path.resolve())

    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is required to load remote code.")

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


def _ensure_remote_package(snapshot_dir: str) -> str:
    with _DYNIN_REMOTE_CACHE_LOCK:
        existing = _DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT.get(snapshot_dir)
        if existing is not None:
            return existing

        digest = hashlib.sha1(snapshot_dir.encode("utf-8")).hexdigest()[:12]
        package_name = f"_dynin_hf_remote_{digest}"

        package = types.ModuleType(package_name)
        package.__path__ = [snapshot_dir]  # type: ignore[attr-defined]
        package.__file__ = str(Path(snapshot_dir) / "__init__.py")

        sys.modules.setdefault(package_name, package)
        _DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT[snapshot_dir] = package_name
        return package_name


def _load_remote_module(
    *,
    module_name: str,
    source: str,
    revision: str | None,
    local_files_only: bool,
):
    snapshot_dir = _resolve_remote_snapshot_dir(
        source=source,
        revision=revision,
        local_files_only=local_files_only,
    )

    module_path = Path(snapshot_dir) / f"{module_name}.py"
    if not module_path.is_file():
        raise ImportError(f"Remote code module '{module_name}.py' not found under '{snapshot_dir}'. source={source!r}")

    package_name = _ensure_remote_package(snapshot_dir)
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


def resolve_remote_attr(
    attr_name: str,
    *,
    module_name: str,
    settings: RemoteCodeSettings,
    source: str | None = None,
    revision: str | None = None,
    local_files_only: bool | None = None,
    fallback_module_names: Iterable[str] = (),
    optional: bool = False,
) -> Any | None:
    resolved_source = _resolve_remote_source(source, settings)
    resolved_revision = _resolve_remote_revision(revision, settings)
    resolved_local_only = _resolve_remote_local_only(local_files_only, settings)

    module_candidates = [module_name, *[m for m in fallback_module_names if m and m != module_name]]
    last_error: Exception | None = None

    for candidate in module_candidates:
        cache_key = (attr_name, candidate, resolved_source, resolved_revision, resolved_local_only)
        cached = _DYNIN_REMOTE_ATTR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            module = _load_remote_module(
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

    if optional:
        if last_error is not None:
            logger.debug(
                "Optional remote attr not found: attr=%s source=%s revision=%s err=%s",
                attr_name,
                resolved_source,
                resolved_revision,
                last_error,
            )
        return None

    raise ImportError(
        f"Failed to resolve '{attr_name}' from remote code "
        f"(source={resolved_source!r}, revision={resolved_revision!r}, modules={module_candidates})."
    ) from last_error


_DYNIN_MODELING_REMOTE_EXPORTS = {
    "DyninOmniConfig": "DyninOmniConfig",
    "DyninOmniModelLM": "DyninOmniModelLM",
    "VideoTokenMerger": "VideoTokenMerger",
}

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

_DYNIN_CONFIG_RESOLVER_REMOTE_EXPORTS = {
    "resolve_model_pretrained_source": "resolve_model_pretrained_source",
    "resolve_tokenizer_source": "resolve_tokenizer_source",
    "resolve_model_local_files_only": "resolve_model_local_files_only",
    "resolve_vq_cfg_block": "resolve_vq_cfg_block",
    "resolve_vq_repo_source": "resolve_vq_repo_source",
}

_DYNIN_MAGVIT_REMOTE_EXPORTS = {
    "VQGANEncoder": "VQGANEncoder",
    "VQGANDecoder": "VQGANDecoder",
    "LFQuantizer": "LFQuantizer",
    "MAGVITv2": "MAGVITv2",
}


def _get_export_attr(
    name: str,
    export_map: dict[str, str],
    *,
    module_name: str,
    settings: RemoteCodeSettings,
    source: str | None = None,
    revision: str | None = None,
    local_files_only: bool | None = None,
    optional: bool = False,
) -> Any | None:
    attr_name = export_map.get(name)
    if attr_name is None:
        raise AttributeError(f"Unsupported export: {name!r}")

    return resolve_remote_attr(
        attr_name,
        module_name=module_name,
        settings=settings,
        source=source,
        revision=revision,
        local_files_only=local_files_only,
        optional=optional,
    )


def get_dynin_modeling_attr(name: str) -> Any:
    return _get_export_attr(
        name,
        _DYNIN_MODELING_REMOTE_EXPORTS,
        module_name="modeling_dynin_omni",
        settings=DYNIN_REMOTE_SETTINGS,
    )


def get_dynin_sampling_attr(name: str) -> Any:
    return _get_export_attr(
        name,
        _DYNIN_SAMPLING_REMOTE_EXPORTS,
        module_name="sampling",
        settings=DYNIN_REMOTE_SETTINGS,
    )


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
        value = resolve_remote_attr(
            attr_name,
            module_name="config_resolver",
            settings=DYNIN_REMOTE_SETTINGS,
            source=source,
            revision=revision,
            local_files_only=local_files_only,
            optional=True,
        )
        if value is not None:
            return value

    return resolve_remote_attr(
        attr_name,
        module_name="config_resolver",
        settings=DYNIN_REMOTE_SETTINGS,
        source=DEFAULT_DYNIN_REMOTE_CODE_REPO,
        revision=revision,
        local_files_only=local_files_only,
        optional=False,
    )


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

    value = resolve_remote_attr(
        attr_name,
        module_name="modeling_magvitv2",
        settings=MAGVIT_REMOTE_SETTINGS,
        source=source,
        revision=revision,
        local_files_only=local_files_only,
        optional=True,
    )
    if value is not None:
        return value

    resolved_source = _resolve_remote_source(source, MAGVIT_REMOTE_SETTINGS)
    resolved_revision = _resolve_remote_revision(revision, MAGVIT_REMOTE_SETTINGS)
    resolved_local_only = _resolve_remote_local_only(local_files_only, MAGVIT_REMOTE_SETTINGS)

    if resolved_source != DEFAULT_MAGVIT_REMOTE_CODE_REPO:
        return resolve_remote_attr(
            attr_name,
            module_name="modeling_magvitv2",
            settings=MAGVIT_REMOTE_SETTINGS,
            source=DEFAULT_MAGVIT_REMOTE_CODE_REPO,
            revision=resolved_revision,
            local_files_only=resolved_local_only,
            optional=False,
        )

    raise ImportError(
        f"Failed to resolve MAGVIT attr '{attr_name}' from source={resolved_source!r} (revision={resolved_revision!r})."
    )


def build_dynin_chat_prompt(content: str) -> str:
    return (
        f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def extract_dynin_user_prompt_text(decoded_prompt: str) -> str:
    text = str(decoded_prompt or "")
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    user_marker = "<|start_header_id|>user<|end_header_id|>"
    end_header_marker = "<|end_header_id|>"
    eot_marker = "<|eot_id|>"

    if assistant_marker in text:
        text = text.rsplit(assistant_marker, 1)[0]
    if eot_marker in text:
        text = text.rsplit(eot_marker, 1)[0]
    if user_marker in text:
        text = text.rsplit(user_marker, 1)[-1]
    if end_header_marker in text:
        text = text.split(end_header_marker, 1)[-1]
    return text.strip()


def normalize_dynin_online_prompt_text(task: str, decoded_prompt: str) -> str:
    text = extract_dynin_user_prompt_text(decoded_prompt)
    if not text:
        text = str(decoded_prompt or "")

    for placeholder in _DYNIN_MODALITY_PLACEHOLDERS:
        text = text.replace(placeholder, " ")

    task_token = _DYNIN_ONLINE_PROMPT_TOKEN_BY_TASK.get(task)
    if task_token:
        text = text.replace(task_token, " ", 1)

    text = " ".join(text.split()).strip()

    if task == "t2s":
        if not text:
            text = "Hello. This is a default text-to-speech sample."
        text = build_dynin_chat_prompt(f"{DEFAULT_DYNIN_T2S_INSTRUCTION}\n{text}")
    elif task in {"t2i", "i2i"} and not text:
        text = "A high quality detailed image."

    return text


def infer_dynin_online_task(
    *,
    decoded_prompt: str,
    has_image: bool = False,
    has_audio: bool = False,
    has_video: bool = False,
) -> str:
    prompt = str(decoded_prompt or "")
    if "<|i2i|>" in prompt:
        return "i2i"
    if "<|t2i|>" in prompt and not has_audio and not has_video:
        return "t2i"
    if "<|t2s|>" in prompt and not has_audio and not has_video:
        return "t2s"
    return "t2t"


def build_dynin_prompt_payload(
    *,
    task: str,
    text: str,
    image_tokens: torch.Tensor | None,
    image_placeholder_tokens: int,
    audio_placeholder_tokens: int,
    image_token_offset: int,
    mask_token_id: int,
    use_train_i2i_prompt: bool,
) -> tuple[Any, str]:
    _, prompting_task, _, _ = DYNIN_TASK_DEFAULT_RUNTIME[task]

    if task == "t2t":
        payload = ([[]], [build_dynin_chat_prompt(text)])
        return payload, prompting_task

    if task == "t2i":
        image_placeholder = torch.full(
            (1, int(image_placeholder_tokens)),
            fill_value=int(mask_token_id),
            dtype=torch.long,
        )
        payload = ([text], image_placeholder)
        return payload, prompting_task

    if task == "i2i":
        if image_tokens is None:
            raise ValueError("i2i requires image tokens")
        src = image_tokens.view(1, -1).long() + int(image_token_offset)
        target_len = int(image_placeholder_tokens) if image_placeholder_tokens > 0 else int(src.shape[1])
        image_placeholder = torch.full(
            (1, target_len),
            fill_value=int(mask_token_id),
            dtype=torch.long,
        )
        if use_train_i2i_prompt:
            labels_placeholder = torch.full(
                (1, target_len),
                fill_value=-100,
                dtype=torch.long,
            )
            payload = ([text], src, image_placeholder, labels_placeholder)
            return payload, "i2i"
        payload = ([text], src, image_placeholder)
        return payload, "i2i_gen"

    if task == "t2s":
        audio_placeholder = torch.full(
            (1, int(audio_placeholder_tokens)),
            fill_value=int(mask_token_id),
            dtype=torch.long,
        )
        payload = ([text], audio_placeholder)
        return payload, prompting_task

    raise ValueError(f"Unsupported Dynin online bootstrap task: {task}")


def _wrap_runtime_field(value: Any) -> list[Any]:
    return [value]


def build_dynin_online_runtime_info(
    *,
    task: str,
    text_vocab_size: int,
    infer_sources: DyninInferSources,
    dynin_config_path: str | None = None,
    prompting_input: Any | None = None,
    attention_mask: list[int] | None = None,
    prompt_length: int | None = None,
    uncond_prompting_input: Any | None = None,
    image_token_count: int = 0,
    t2s_token_length: int | None = None,
    use_train_i2i_prompt: bool | None = None,
) -> dict[str, Any]:
    runtime_task, prompting_task, detok_id, _ = DYNIN_TASK_DEFAULT_RUNTIME[task]

    prompt_max_text_len = int(dynin_runtime_fallback(task, "prompt_max_text_len", None) or 1024)
    max_new_tokens = int(dynin_runtime_fallback(task, "max_new_tokens", None) or 256)
    steps = int(dynin_runtime_fallback(task, "steps", None) or 256)
    block_length = int(dynin_runtime_fallback(task, "block_length", None) or 2)
    temperature = float(dynin_runtime_fallback(task, "temperature", None) or 0.0)
    cfg_scale = float(dynin_runtime_fallback(task, "cfg_scale", None) or 0.0)
    remasking = str(dynin_runtime_fallback(task, "remasking", None) or "low_confidence")
    timesteps = int(dynin_runtime_fallback(task, "timesteps", None) or 20)
    guidance_scale = float(dynin_runtime_fallback(task, "guidance_scale", None) or 0.0)
    mask_token_id = int(dynin_runtime_fallback(task, "mask_token_id", None) or 126336)
    codebook_size = int(dynin_runtime_fallback(task, "codebook_size", None) or 8192)
    audio_codebook_size = int(dynin_runtime_fallback(task, "audio_codebook_size", None) or 4096)
    image_resolution = int(dynin_runtime_fallback(task, "image_resolution", None) or 336)
    if image_token_count <= 0 and task in {"t2i", "i2i"}:
        fallback_count = dynin_runtime_fallback(task, "image_token_count", None)
        if fallback_count is not None:
            image_token_count = int(fallback_count)
        else:
            image_token_count = max(1, (image_resolution // 16) ** 2)

    if t2s_token_length is None:
        t2s_token_length = int(dynin_runtime_fallback(task, "t2s_token_length", None) or 383)
    t2s_condition = str(
        dynin_runtime_fallback(
            task,
            "t2s_condition",
            None,
        )
        or "gender-female_emotion-neutral_speed-normal_pitch-normal"
    )
    if use_train_i2i_prompt is None:
        use_train_i2i_prompt = bool(dynin_runtime_fallback(task, "use_train_i2i_prompt", task == "i2i"))

    runtime_info: dict[str, Any] = {
        "task": _wrap_runtime_field(runtime_task),
        "prompting_task": _wrap_runtime_field(prompting_task),
        "detok_id": _wrap_runtime_field(int(detok_id)),
        "prompt_max_text_len": _wrap_runtime_field(prompt_max_text_len),
        "prompting_max_text_len": _wrap_runtime_field(prompt_max_text_len),
        "cond_dropout_prob": _wrap_runtime_field(0.0),
        "prompting_cond_dropout_prob": _wrap_runtime_field(0.0),
        "tokenizer_path": _wrap_runtime_field(str(infer_sources.tokenizer_source)),
        "text_vocab_size": _wrap_runtime_field(int(text_vocab_size)),
        "model_local_files_only": _wrap_runtime_field(bool(infer_sources.model_local_files_only)),
        "max_new_tokens": _wrap_runtime_field(int(t2s_token_length if task == "t2s" else max_new_tokens)),
        "steps": _wrap_runtime_field(steps),
        "block_length": _wrap_runtime_field(block_length),
        "temperature": _wrap_runtime_field(temperature),
        "cfg_scale": _wrap_runtime_field(cfg_scale),
        "remasking": _wrap_runtime_field(remasking),
        "mask_id": _wrap_runtime_field(mask_token_id),
        "mask_token_id": _wrap_runtime_field(mask_token_id),
        "codebook_size": _wrap_runtime_field(codebook_size),
        "audio_codebook_size": _wrap_runtime_field(audio_codebook_size),
        "timesteps": _wrap_runtime_field(timesteps),
        "guidance_scale": _wrap_runtime_field(guidance_scale),
        "noise_type": _wrap_runtime_field("mask"),
        "noise_schedule_name": _wrap_runtime_field("cosine"),
        "noise_schedule_params": _wrap_runtime_field({}),
        "seq_len": _wrap_runtime_field(int(image_token_count)),
        "condition": _wrap_runtime_field(t2s_condition),
        "t2s_condition": _wrap_runtime_field(t2s_condition),
        "vq_model_image_path": _wrap_runtime_field(str(infer_sources.vq_image_source)),
        "vq_model_image_local_files_only": _wrap_runtime_field(bool(infer_sources.vq_image_local_files_only)),
        "vq_model_audio_path": _wrap_runtime_field(str(infer_sources.vq_audio_source)),
        "vq_model_audio_local_files_only": _wrap_runtime_field(bool(infer_sources.vq_audio_local_files_only)),
        "image_resolution": _wrap_runtime_field(image_resolution),
        "t2s_token_length": _wrap_runtime_field(int(t2s_token_length)),
        "use_train_i2i_prompt": _wrap_runtime_field(bool(use_train_i2i_prompt)),
    }

    if dynin_config_path:
        runtime_info["dynin_config_path"] = _wrap_runtime_field(str(dynin_config_path))
    if prompting_input is not None:
        runtime_info["prompting_input"] = _wrap_runtime_field(prompting_input)
    if uncond_prompting_input is not None:
        runtime_info["uncond_prompting_input"] = _wrap_runtime_field(uncond_prompting_input)
    if attention_mask:
        runtime_info["attention_mask"] = _wrap_runtime_field(list(attention_mask))
    if prompt_length is None and attention_mask:
        prompt_length = len(attention_mask)
    if prompt_length is not None:
        runtime_info["prompt_length"] = _wrap_runtime_field(int(prompt_length))

    return runtime_info
