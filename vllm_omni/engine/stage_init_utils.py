# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Stage initialization helpers for vLLM-Omni multi-stage runtime.

Extracts orchestration-level init logic (config extraction, plugin loading,
multiprocessing setup, device mapping, device locking, engine args building)
out of StageEngineCoreClient into reusable functions.
"""

from __future__ import annotations

import copy
import fcntl
import importlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

import regex as re
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.repo_utils import hf_api
from vllm.transformers_utils.runai_utils import is_runai_obj_uri
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.executor import Executor

from vllm_omni.config.omni_config import (
    _CACHE_STAGE_ENGINE_FIELD_MAP,
    _DIFFUSION_CACHE_STAGE_ENGINE_FIELD_MAP,
    _DIFFUSION_LOAD_STAGE_ENGINE_FIELD_MAP,
    _DIFFUSION_PARALLEL_CONFIG_ENGINE_FIELDS,
    _DIFFUSION_SCHEDULER_STAGE_ENGINE_FIELD_MAP,
    _LOAD_STAGE_ENGINE_FIELD_MAP,
    _PARALLEL_CONFIG_ENGINE_FIELD_MAP,
    _SCHEDULER_STAGE_ENGINE_FIELD_MAP,
    BaseVllmOmniStageConfig,
    VllmOmniDiffusionStageConfig,
)
from vllm_omni.config.stage_config import StageType
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.entrypoints.utils import filter_dataclass_kwargs, resolve_model_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams
from vllm_omni.inputs.preprocess import OmniInputPreprocessor
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.inc_config import OmniINCConfig

logger = init_logger(__name__)


@dataclass
class ReplicaInitPlan:
    """One concrete replica startup unit within a logical stage."""

    replica_id: int
    num_replicas: int
    launch_mode: str
    stage_cfg: Any
    metadata: Any
    stage_connector_spec: dict[str, Any]
    omni_kv_connector: tuple[dict[str, Any] | None, str | None, str | None]
    stage_vllm_config: Any | None = None
    executor_class: type | None = None
    engine_args_dict: dict[str, Any] | None = None


@dataclass
class LogicalStageInitPlan:
    """Startup plan for one logical stage."""

    stage_idx: int
    stage_id: int
    replicas: list[ReplicaInitPlan]


def _missing_stage_subdirs(base: str, subdirs: Sequence[str]) -> list[str]:
    """Return the entries of ``subdirs`` that are not directories under ``base``."""
    return [subdir for subdir in subdirs if not os.path.isdir(os.path.join(base, subdir))]


# Artifacts that make a snapshot subfolder trustworthy. A directory that
# exists but holds none of these is an interrupted download, not a snapshot;
# treating it as complete strips the Hub fallback vLLM would need later.
_WEIGHT_ARTIFACT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.gguf")
# Vocabulary-bearing files. Configs and chat templates are small and download
# first, so their presence alone cannot distinguish a tokenizer folder from an
# interrupted download.
_TOKENIZER_ARTIFACT_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
)
# HF sharded checkpoints name their pieces `<stem>-NNNNN-of-NNNNN.<ext>` and
# always ship an index; a shard-named file without one is a partial download.
_SHARD_NAME_RE = re.compile(r"-\d+-of-\d+\.(safetensors|bin)$")


def _indexed_shards_complete(folder: Path) -> bool | None:
    """Check sharded weights against their index; ``None`` when no index exists."""
    indexes = list(folder.rglob("*.index.json"))
    if not indexes:
        return None
    for index in indexes:
        try:
            weight_map = json.loads(index.read_text()).get("weight_map") or {}
        except (OSError, ValueError):
            return False
        shards = set(weight_map.values())
        if not shards or any(not (index.parent / shard).is_file() for shard in shards):
            return False
    return True


def _subdir_is_populated(base: str, subdir: str, needs_weights: bool) -> bool:
    folder = Path(base) / subdir
    if not folder.is_dir():
        return False
    if needs_weights:
        indexed = _indexed_shards_complete(folder)
        if indexed is not None:
            return indexed
        weights = [path for pattern in _WEIGHT_ARTIFACT_PATTERNS for path in folder.rglob(pattern)]
        if not weights:
            return False
        return not any(_SHARD_NAME_RE.search(path.name) for path in weights)
    return any((folder / name).is_file() for name in _TOKENIZER_ARTIFACT_NAMES)


def _incomplete_stage_subdirs(
    base: str,
    subdirs: Sequence[str],
    weight_subdirs: Collection[str] = (),
) -> list[str]:
    """Return the entries of ``subdirs`` without a POPULATED directory under ``base``.

    Hub-snapshot paths use this stricter check: ``os.path.isdir`` alone accepts
    a subfolder holding only config.json from an interrupted download, and the
    warm-cache early return would then convert the Hub ID into a local path
    vLLM cannot fetch missing weights for. ``weight_subdirs`` names the entries
    that must contain a weight artifact, not merely any file.
    """
    return [
        subdir for subdir in subdirs if not _subdir_is_populated(base, subdir, needs_weights=subdir in weight_subdirs)
    ]


def _resolve_model_to_local_path(
    model: str,
    required_subdirs: Sequence[str] = (),
    weight_subdirs: Collection[str] = (),
    *,
    revision: str | None = None,
    download_dir: str | None = None,
) -> str:
    """Resolve an HF Hub model ID to a local path that holds ``required_subdirs``.

    ``snapshot_download(local_files_only=True)`` returns the snapshot root as
    soon as *any* file of the repo is cached, even when the subfolders this
    stage needs were never materialized. Joining a stage subdir onto such a
    root produces a path that exists nowhere, and upstream ``EngineArgs``
    forwards a non-directory ``model`` to HuggingFace as a repo id, which fails
    with an ``HFValidationError`` about the cache path. Verify the subfolders
    here and pull just the missing ones, so the join always lands on a real
    directory or raises an error that names what is missing.

    ``revision`` and ``download_dir`` mirror the engine args of the same name:
    once the repo ID is replaced by a local path, downstream ModelConfig can no
    longer correct either, so they must shape the snapshot selection here.
    """
    if os.path.isdir(model):
        return model

    # Keep the warm-cache path offline-friendly: no Hub round trip when the
    # stage's subfolders are already there.
    try:
        cached_root: str | None = hf_api().snapshot_download(
            model, local_files_only=True, revision=revision, cache_dir=download_dir
        )
    except Exception:
        cached_root = None
    if cached_root is not None and not _incomplete_stage_subdirs(cached_root, required_subdirs, weight_subdirs):
        return cached_root

    # Cold cache, or a snapshot root whose stage subfolders were never (or
    # only partially) downloaded: pull exactly the subfolders this stage asked
    # for. snapshot_download resumes a partial subfolder for free.
    allow_patterns = [f"{subdir.strip('/')}/*" for subdir in required_subdirs] or None
    try:
        resolved = hf_api().snapshot_download(
            model, allow_patterns=allow_patterns, revision=revision, cache_dir=download_dir
        )
    except Exception as exc:
        raise RuntimeError(
            f"[stage_init] Could not resolve {model!r} to a local snapshot containing "
            f"{sorted(required_subdirs)}: the download failed and "
            + (
                f"the cached snapshot {cached_root!r} is missing or incomplete for "
                f"{sorted(_incomplete_stage_subdirs(cached_root, required_subdirs, weight_subdirs))}."
                if cached_root is not None
                else "nothing is cached locally."
            )
        ) from exc

    missing = _incomplete_stage_subdirs(resolved, required_subdirs, weight_subdirs)
    if missing:
        raise RuntimeError(
            f"[stage_init] Snapshot {resolved!r} for {model!r} has no populated {sorted(missing)} "
            "subfolder; the stage cannot be initialized from it."
        )
    return resolved


def _resolve_model_tokenizer_paths(model: str, engine_args: dict[str, Any]) -> str:
    """Apply model_subdir/tokenizer_subdir indirections from stage engine args."""
    model_subdir = engine_args.pop("model_subdir", None)
    tokenizer_subdir = engine_args.pop("tokenizer_subdir", None)
    if model_subdir is None and tokenizer_subdir is None:
        return model

    revision = engine_args.get("revision")
    tokenizer_revision = engine_args.get("tokenizer_revision")
    download_dir = engine_args.get("download_dir")
    # A tokenizer pinned to a different revision cannot come from the model's
    # snapshot; resolve it against its own. An empty subdir means the snapshot
    # root and still needs its own revision.
    split_tokenizer = tokenizer_subdir is not None and tokenizer_revision is not None and tokenizer_revision != revision

    required_subdirs = [subdir for subdir in (model_subdir, tokenizer_subdir) if subdir]
    model_required = [subdir for subdir in (model_subdir,) if subdir] if split_tokenizer else required_subdirs
    weight_subdirs = frozenset(subdir for subdir in (model_subdir,) if subdir)
    if is_runai_obj_uri(model):
        # Object-storage URIs stay opaque until each stage builds its own
        # ModelConfig, so the joins below are resolved by vLLM's streamer
        # rather than by the local filesystem.
        resolved_base = model
        tokenizer_base = model
    else:
        resolved_base = _resolve_model_to_local_path(
            model, model_required, weight_subdirs, revision=revision, download_dir=download_dir
        )
        # Reachable for a local model directory; the Hub branch above has
        # already failed closed on a missing subfolder.
        missing = _missing_stage_subdirs(resolved_base, model_required)
        if missing:
            raise RuntimeError(
                f"[stage_init] Model directory {resolved_base!r} has no {sorted(missing)} "
                "subfolder; the stage cannot be initialized from it."
            )
        if split_tokenizer:
            # An empty subdir targets the snapshot root, which cannot be
            # subset by allow_patterns; resolve the whole revision.
            tokenizer_required = [tokenizer_subdir] if tokenizer_subdir else []
            tokenizer_base = _resolve_model_to_local_path(
                model, tokenizer_required, revision=tokenizer_revision, download_dir=download_dir
            )
            missing = _missing_stage_subdirs(tokenizer_base, tokenizer_required)
            if missing:
                raise RuntimeError(
                    f"[stage_init] Tokenizer directory {tokenizer_base!r} has no {sorted(missing)} "
                    "subfolder; the stage cannot be initialized from it."
                )
            if not tokenizer_required and not _subdir_is_populated(tokenizer_base, "", False):
                # An empty subdir means the tokenizer lives at the snapshot
                # root, so there is no subfolder for the check above to look
                # at and any resolved root would otherwise pass. Require the
                # vocabulary artifacts themselves.
                raise RuntimeError(
                    f"[stage_init] Tokenizer directory {tokenizer_base!r} holds none of "
                    f"{list(_TOKENIZER_ARTIFACT_NAMES)}; the stage cannot be initialized from it."
                )
        else:
            tokenizer_base = resolved_base

    if model_subdir:
        model = os.path.join(resolved_base, model_subdir)
        logger.info("[stage_init] Using model subdirectory: %s", model)

    if tokenizer_subdir is not None:
        tokenizer_path = os.path.join(tokenizer_base, tokenizer_subdir) if tokenizer_subdir else tokenizer_base
        engine_args["tokenizer"] = tokenizer_path
        logger.info("[stage_init] Using tokenizer from: %s", tokenizer_path)
    elif model_subdir and "tokenizer" not in engine_args:
        # Keep legacy behavior: model in subdir, tokenizer defaults to base path.
        engine_args["tokenizer"] = resolved_base
        logger.info("[stage_init] Using tokenizer from base model path: %s", resolved_base)

    return model


def _resolve_model_path(model: str, engine_args: dict[str, Any]) -> str:
    """Apply a model-owned path resolver from stage engine args."""
    resolver_path = engine_args.pop("model_path_resolver", None)
    if resolver_path is None:
        return model
    resolver = _resolve_omni_metadata_hook(str(resolver_path))
    if resolver is None:
        return model
    return str(
        resolver(
            model,
            engine_args.get("revision"),
            engine_args.get("task_type"),
        )
    )


def apply_cli_tokenizer(
    engine_args: dict[str, Any],
    *,
    cli_tokenizer: str | None,
    stage_defines_tokenizer: bool,
) -> None:
    """Forward CLI tokenizer unless the stage config defines its own."""
    if cli_tokenizer is None or stage_defines_tokenizer:
        return
    engine_args["tokenizer"] = cli_tokenizer


def terminate_alive_proc(proc, timeout=5):
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.kill()


def set_death_signal(sig: int) -> None:
    """Best-effort parent-death signal for Linux subprocesses."""
    try:
        import ctypes
        import platform

        if platform.system() != "Linux":
            return
        ctypes.CDLL("libc.so.6").prctl(1, sig)
    except Exception:
        pass


def patch_generation_config_if_needed(model_config: Any) -> None:
    """Guard InputProcessor init for models whose config lacks model_type."""
    try:
        model_config.try_get_generation_config()
    except Exception:
        model_config.try_get_generation_config = lambda: {}


def resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    """Resolve worker_cls from worker_type for non-diffusion stages."""
    worker_type = engine_args.get("worker_type", None)
    if not worker_type:
        return
    worker_cls = engine_args.get("worker_cls")
    if worker_cls is not None and worker_cls != "auto":
        return

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")


def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    """Read *key* from *obj* regardless of whether it's a dict or object."""
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tp_size_for_stage(stage_configs: Sequence[Any], stage_id: Any) -> int | None:
    """Resolve tensor_parallel_size for *stage_id* from the loaded stage configs."""
    id_strs = {str(stage_id)}
    try:
        id_strs.add(str(int(stage_id)))
    except (TypeError, ValueError):
        pass

    for stage_cfg in stage_configs:
        if str(getattr(stage_cfg, "stage_id", None)) not in id_strs:
            continue
        engine_args = getattr(stage_cfg, "engine_args", None)
        if engine_args is None:
            return 1
        parallel_config = _get_attr_or_item(engine_args, "parallel_config")
        if parallel_config is not None:
            tp = _get_attr_or_item(parallel_config, "tensor_parallel_size", 1)
        else:
            tp = _get_attr_or_item(engine_args, "tensor_parallel_size", 1)
        try:
            return max(1, int(tp))
        except (TypeError, ValueError):
            return 1
    return None


def _inject_inferred_kv_tp_topology(
    omni_kv: Any,
    stage_id: int,
    stage_configs: Sequence[Any],
    engine_input_source: Sequence[int] | None = None,
) -> None:
    """Infer adjacent-stage TP topology and inject it into omni_kv_config.

    This keeps heterogeneous TP working without requiring user-authored
    rank_mapping blocks in config files.
    """
    if omni_kv is None:
        return

    if hasattr(omni_kv, "get"):
        need_send = bool(omni_kv.get("need_send_cache", False))
        need_recv = bool(omni_kv.get("need_recv_cache", False))
        omni_from_stage = omni_kv.get("omni_from_stage")
        omni_to_stage = omni_kv.get("omni_to_stage")
        rank_mapping = omni_kv.get("rank_mapping")
    else:
        need_send = bool(getattr(omni_kv, "need_send_cache", False))
        need_recv = bool(getattr(omni_kv, "need_recv_cache", False))
        omni_from_stage = getattr(omni_kv, "omni_from_stage", None)
        omni_to_stage = getattr(omni_kv, "omni_to_stage", None)
        rank_mapping = getattr(omni_kv, "rank_mapping", None)

    if not need_send and not need_recv:
        return

    current_tp = _tp_size_for_stage(stage_configs, stage_id)
    if current_tp is None:
        return

    peer_stage_id = None
    from_tp = None
    to_tp = None
    if str(omni_from_stage) == str(stage_id):
        peer_stage_id = omni_to_stage
        from_tp = current_tp
        to_tp = _tp_size_for_stage(stage_configs, peer_stage_id)
    elif str(omni_to_stage) == str(stage_id):
        peer_stage_id = omni_from_stage
        from_tp = _tp_size_for_stage(stage_configs, peer_stage_id)
        to_tp = current_tp
    elif need_recv and engine_input_source:
        peer_stage_id = engine_input_source[0]
        from_tp = _tp_size_for_stage(stage_configs, peer_stage_id)
        to_tp = current_tp

    if from_tp is None or to_tp is None:
        return

    if not isinstance(rank_mapping, dict):
        rank_mapping = {}
    rank_mapping.setdefault("from_tp", int(from_tp))
    rank_mapping.setdefault("to_tp", int(to_tp))

    if hasattr(omni_kv, "__setitem__"):
        omni_kv["rank_mapping"] = rank_mapping
    else:
        setattr(omni_kv, "rank_mapping", rank_mapping)


def inject_kv_stage_info(stage_cfg: Any, stage_id: int, stage_configs: Sequence[Any] | None = None) -> None:
    """Inject stage_id, engine_input_source, and inferred TP topology into omni_kv_config.

    When *stage_configs* is provided, also infers from_tp/to_tp for
    heterogeneous TP topologies so the KV transfer manager can compute
    rank mappings automatically.
    """
    try:
        engine_args = stage_cfg.engine_args
        if hasattr(engine_args, "get"):
            omni_kv = engine_args.get("omni_kv_config", None)
        else:
            omni_kv = getattr(engine_args, "omni_kv_config", None)

        if omni_kv is None:
            return

        if hasattr(omni_kv, "setdefault"):
            omni_kv.setdefault("stage_id", stage_id)
        elif hasattr(omni_kv, "__setitem__"):
            if "stage_id" not in omni_kv:
                omni_kv["stage_id"] = stage_id

        engine_input_source = getattr(stage_cfg, "engine_input_source", None)
        if engine_input_source is not None:
            if hasattr(omni_kv, "setdefault"):
                omni_kv.setdefault("engine_input_source", list(engine_input_source))
            elif hasattr(omni_kv, "__setitem__") and "engine_input_source" not in omni_kv:
                omni_kv["engine_input_source"] = list(engine_input_source)

        if stage_configs:
            _inject_inferred_kv_tp_topology(
                omni_kv,
                stage_id=stage_id,
                stage_configs=stage_configs,
                engine_input_source=engine_input_source,
            )
    except Exception as e:
        logger.debug("Failed to inject stage info into omni_kv_config: %s", e)


def inject_omni_kv_connector_config(
    engine_args_dict: dict[str, Any],
    omni_kv_connector: tuple[dict[str, Any] | None, str | None, str | None],
    stage_id: int,
) -> None:
    """Inject resolved connector config into a stage engine-args dict."""
    omni_conn_cfg, omni_from, omni_to = omni_kv_connector
    if not omni_conn_cfg:
        return

    omni_kv = engine_args_dict.get("omni_kv_config") or {}
    if not isinstance(omni_kv, dict):
        omni_kv = dict(omni_kv)
    omni_kv["connector_config"] = omni_conn_cfg
    omni_kv["omni_from_stage"] = omni_from
    omni_kv["omni_to_stage"] = omni_to
    omni_kv.setdefault("stage_id", stage_id)
    engine_args_dict["omni_kv_config"] = omni_kv


@dataclass
class StageMetadata:
    """Lightweight stage attributes extracted from stage_config."""

    stage_id: int
    stage_type: Literal["llm", "diffusion"]
    engine_output_type: str | None
    is_comprehension: bool
    requires_multimodal_data: bool
    engine_input_source: list[int]
    final_output: bool
    final_output_type: str | None
    default_sampling_params: OmniSamplingParams
    custom_process_input_func: Callable | None
    model_stage: str | None
    runtime_cfg: Any
    prompt_transform_func: Callable | None = None
    prompt_expand_func: Callable | None = None
    cfg_kv_collect_func: Callable | None = None
    # Multi-replica: replica_id distinguishes replicas of the same stage.
    # For single-replica stages this defaults to 0.
    replica_id: int = 0


def _apply_rocm_attention_backend(
    engine_args: dict[str, Any],
    stage_type: str | StageType,
) -> None:
    """Preserve Omni's ROCm attention-backend compatibility default."""
    if (
        not current_omni_platform.is_rocm()
        or stage_type == StageType.DIFFUSION
        or engine_args.get("attention_backend") is not None
    ):
        return

    from vllm._aiter_ops import rocm_aiter_ops

    if rocm_aiter_ops.is_enabled():
        engine_args["attention_backend"] = "ROCM_AITER_FA"
    # Before vLLM v0.19.0, the default attention backend is TRITON_ATTN for ROCm.
    # Since vLLM v0.19.0, the default attention backend is ROCM_ATTN for ROCm.
    # However, the compatibility of ROCM_ATTN with Omni is not guaranteed.
    # Therefore, we still use TRITON_ATTN as the default attention backend,
    # when the selected_backend is not specified.
    engine_args["attention_backend"] = "TRITON_ATTN"


def extract_legacy_stage_metadata(stage_config: Any) -> StageMetadata:
    """Extract metadata through the active production legacy path.

    Keep production callers on this path until RFC #4021 migrates the
    engine-argument and stage-init consumers together.
    """
    stage_id: int = stage_config.stage_id
    stage_type: Literal["llm", "diffusion"] = _get_attr_or_item(stage_config, "stage_type", "llm")
    engine_args = stage_config.engine_args

    _apply_rocm_attention_backend(engine_args, stage_type)

    runtime_cfg = stage_config.runtime
    engine_input_source: list[int] = _get_attr_or_item(stage_config, "engine_input_source", [])
    final_output: bool = stage_config.final_output
    final_output_type: str | None = stage_config.final_output_type

    default_sp = _to_dict(_get_attr_or_item(stage_config, "default_sampling_params", {}))
    # A pooling stage carries its task via default_pooling_params, set where the
    # stage is declared.
    default_pp = _to_dict(_get_attr_or_item(stage_config, "default_pooling_params", {}))
    # A pooling stage is an LLM stage run with runner="pooling" (vLLM's
    # is_pooling_model signal); pick params by that signal, not execution_type.
    is_pooling = str(engine_args.get("runner", "")).lower() == "pooling"
    default_params: OmniSamplingParams | PoolingParams
    if stage_type == "diffusion":
        default_params = OmniDiffusionSamplingParams(**default_sp)
    elif is_pooling:
        default_params = PoolingParams(**default_pp)
    else:  # generative llm: ar / generation
        default_params = SamplingParams(**default_sp)

    custom_process_input_func: Callable | None = None
    _cpif_path = _get_attr_or_item(stage_config, "custom_process_input_func")
    if _cpif_path:
        mod_path, fn_name = _cpif_path.rsplit(".", 1)
        custom_process_input_func = getattr(importlib.import_module(mod_path), fn_name)

    prompt_transform_func: Callable | None = None
    _ptf_path = _get_attr_or_item(stage_config, "prompt_transform_func")
    if _ptf_path:
        _mod, _fn = _ptf_path.rsplit(".", 1)
        prompt_transform_func = getattr(importlib.import_module(_mod), _fn)

    prompt_expand_func: Callable | None = None
    _pef_path = _get_attr_or_item(stage_config, "prompt_expand_func")
    if _pef_path:
        _mod, _fn = _pef_path.rsplit(".", 1)
        prompt_expand_func = getattr(importlib.import_module(_mod), _fn)

    cfg_kv_collect_func: Callable | None = None
    _ckf_path = _get_attr_or_item(stage_config, "cfg_kv_collect_func")
    if _ckf_path:
        _mod, _fn = _ckf_path.rsplit(".", 1)
        cfg_kv_collect_func = getattr(importlib.import_module(_mod), _fn)

    model_stage = engine_args.get("model_stage")

    if stage_type == "diffusion":
        return StageMetadata(
            stage_id=stage_id,
            stage_type="diffusion",
            engine_output_type=None,
            is_comprehension=False,
            requires_multimodal_data=False,
            engine_input_source=engine_input_source,
            final_output=final_output,
            final_output_type=final_output_type,
            default_sampling_params=default_params,
            custom_process_input_func=custom_process_input_func,
            model_stage=model_stage,
            runtime_cfg=runtime_cfg,
            prompt_transform_func=prompt_transform_func,
            cfg_kv_collect_func=cfg_kv_collect_func,
        )

    engine_output_type = engine_args.get("engine_output_type")
    is_comprehension = stage_config.is_comprehension
    requires_multimodal_data = getattr(runtime_cfg, "requires_multimodal_data", False)

    return StageMetadata(
        stage_id=stage_id,
        stage_type=stage_type,
        engine_output_type=engine_output_type,
        is_comprehension=is_comprehension,
        requires_multimodal_data=requires_multimodal_data,
        engine_input_source=engine_input_source,
        final_output=final_output,
        final_output_type=final_output_type,
        default_sampling_params=default_params,
        custom_process_input_func=custom_process_input_func,
        model_stage=model_stage,
        runtime_cfg=runtime_cfg,
        prompt_transform_func=prompt_transform_func,
        prompt_expand_func=prompt_expand_func,
    )


def extract_stage_metadata(stage_config: Any) -> StageMetadata:
    """Preserve the legacy one-argument API for external callers."""
    return extract_legacy_stage_metadata(stage_config)


def _resolve_omni_metadata_hook(path: str | None) -> Callable | None:
    if not path:
        return None
    module_path, function_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), function_name)


def extract_stage_metadata_from_omni_stage_config(
    stage_config: BaseVllmOmniStageConfig,
) -> StageMetadata:
    """Project one typed stage config into metadata for a future cutover.

    This projection is not used by production startup yet. Current replica
    layout, engine-argument, remote-diffusion, and platform setup paths still
    require the legacy StageConfig/OmegaConf shape.
    """
    stage_type: Literal["llm", "diffusion"] = "diffusion" if stage_config.stage_type == StageType.DIFFUSION else "llm"
    sampling_params_cls = SamplingParams if stage_type == "llm" else OmniDiffusionSamplingParams
    sampling_params: OmniSamplingParams = sampling_params_cls(
        **(stage_config.model_config.default_sampling_params or {})
    )
    custom_process_input_func = _resolve_omni_metadata_hook(stage_config.custom_process_input_func)

    if stage_type == "diffusion":
        return StageMetadata(
            stage_id=stage_config.stage_id,
            stage_type="diffusion",
            engine_output_type=None,
            is_comprehension=False,
            requires_multimodal_data=False,
            engine_input_source=stage_config.input_sources,
            final_output=stage_config.final_output,
            final_output_type=stage_config.final_output_type,
            default_sampling_params=sampling_params,
            custom_process_input_func=custom_process_input_func,
            model_stage=stage_config.model_stage,
            runtime_cfg=stage_config.runtime_config,
            prompt_transform_func=_resolve_omni_metadata_hook(stage_config.prompt_transform_func),
            cfg_kv_collect_func=_resolve_omni_metadata_hook(stage_config.cfg_kv_collect_func),
        )

    return StageMetadata(
        stage_id=stage_config.stage_id,
        stage_type="llm",
        engine_output_type=stage_config.engine_output_type,
        is_comprehension=stage_config.is_comprehension,
        requires_multimodal_data=stage_config.requires_multimodal_data,
        engine_input_source=stage_config.input_sources,
        final_output=stage_config.final_output,
        final_output_type=stage_config.final_output_type,
        default_sampling_params=sampling_params,
        custom_process_input_func=custom_process_input_func,
        model_stage=stage_config.model_stage,
        runtime_cfg=stage_config.runtime_config,
        prompt_transform_func=_resolve_omni_metadata_hook(stage_config.prompt_transform_func),
        prompt_expand_func=_resolve_omni_metadata_hook(stage_config.prompt_expand_func),
    )


def prepare_engine_environment() -> None:
    """One-time global setup: load plugins, set multiprocessing spawn method."""
    from vllm_omni.plugins import load_omni_general_plugins

    load_omni_general_plugins()

    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        logger.info("[stage_init] Set VLLM_WORKER_MULTIPROC_METHOD=spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def _maybe_set_qwen3_omni_moe_env(engine_args_dict: dict[str, Any]) -> None:
    if (
        engine_args_dict.get("model_arch") == "Qwen3OmniMoeForConditionalGeneration"
        and "VLLM_USE_FLASHINFER_MOE_FP16" not in os.environ
    ):
        os.environ["VLLM_USE_FLASHINFER_MOE_FP16"] = "0"
        logger.info("[stage_init] Set VLLM_USE_FLASHINFER_MOE_FP16=0 for Qwen3-Omni stage")


def split_devices_for_replicas(
    devices_str: str | None,
    num_replicas: int,
    devices_per_replica: int,
    stage_id: int,
) -> list[str | None]:
    """Split a devices string into per-replica subsets.

    The result always has one entry per replica, because callers index it by
    replica id. When ``devices_str`` is ``None`` the stage declares no explicit
    placement, so every replica gets ``None`` and inherits the launcher's
    ``CUDA_VISIBLE_DEVICES``.

    When ``num_replicas`` is 1, returns ``[devices_str]`` unchanged.
    Otherwise, two YAML shapes are accepted:

    1. **Legacy / pool mode** — ``len(devices) == num_replicas * devices_per_replica``:
       the string enumerates the full per-stage pool. Each replica gets
       ``devices_per_replica`` consecutive entries. The values are logical indices
       into the launcher's ``CUDA_VISIBLE_DEVICES``.

       ``split_devices_for_replicas("1,2,3,4", 2, 2, 1) → ["1,2", "3,4"]``

    2. **Template mode** — ``len(devices) == devices_per_replica``: the YAML declares
       a single per-replica template (the same shape one replica would
       use), and is **replica-count-independent**. Each replica r gets the offsets
       ``[r*devices_per_replica + a for a in template]`` of the launcher's
       ``CUDA_VISIBLE_DEVICES``. The template's entries must lie in
       ``[0, devices_per_replica)``.

       ``split_devices_for_replicas("0,1", 2, 2, 1) → ["0,1", "2,3"]``
       ``split_devices_for_replicas("0,1", 4, 2, 1) → ["0,1", "2,3", "4,5", "6,7"]``

       This lets the same ``devices: "0,1"`` YAML work for any
       ``--omni-dp-size-local``: the launcher's CVD scales, the YAML
       does not.

    Any other length raises ``ValueError`` (the two modes are
    length-disjoint for ``num_replicas > 1``).
    """
    if devices_str is None:
        # No explicit placement: hand back one empty slot per replica, matching
        # ``get_headless_replica_devices``. Returning a single-element list here
        # made callers index past the end for every replica after the first.
        return [None] * max(1, num_replicas)

    if num_replicas <= 1:
        return [devices_str]

    device_list = [d.strip() for d in devices_str.split(",") if d.strip()]

    if len(device_list) == num_replicas * devices_per_replica:
        return [
            ",".join(device_list[r * devices_per_replica : (r + 1) * devices_per_replica]) for r in range(num_replicas)
        ]

    if len(device_list) == devices_per_replica:
        try:
            offsets = [int(a) for a in device_list]
        except ValueError as e:
            raise ValueError(f"Stage {stage_id}: template-mode devices must be ints, got {devices_str!r}") from e
        bad = [a for a in offsets if not (0 <= a < devices_per_replica)]
        if bad:
            raise ValueError(
                f"Stage {stage_id}: template-mode device offset(s) {bad} "
                f"out of range [0, {devices_per_replica}); devices={devices_str!r}"
            )
        return [",".join(str(r * devices_per_replica + a) for a in offsets) for r in range(num_replicas)]

    raise ValueError(
        f"Stage {stage_id}: devices={devices_str!r} has {len(device_list)} id(s); "
        f"need either {devices_per_replica} (per-replica template) or "
        f"{num_replicas * devices_per_replica} (pool / legacy). "
        f"num_replicas={num_replicas}, devices_per_replica={devices_per_replica}."
    )


def get_stage_tp_size(stage_cfg: Any) -> int:
    """Extract tensor_parallel_size from a stage config object."""
    engine_args = getattr(stage_cfg, "engine_args", {})
    if hasattr(engine_args, "get"):
        return int(engine_args.get("tensor_parallel_size", 1) or 1)
    return int(getattr(engine_args, "tensor_parallel_size", 1) or 1)


def _get_local_llm_parallel_sizes(
    stage_cfg: Any,
    engine_args: Any | None = None,
) -> tuple[int, int, int]:
    """Return ``(tp, local_dp, pp)`` for one local LLM replica.

    ``data_parallel_size`` is cluster-wide, whereas ``runtime.devices`` is
    local to this process.  Prefer an explicitly resolved
    ``data_parallel_size_local`` (including zero for a head process that owns
    no local engines), and only fall back to the global DP width when it is
    unset.
    """
    if engine_args is None:
        engine_args = getattr(stage_cfg, "engine_args", {})
    tp_size = int(_get_attr_or_item(engine_args, "tensor_parallel_size", 1) or 1)
    pp_size = int(_get_attr_or_item(engine_args, "pipeline_parallel_size", 1) or 1)
    local_dp_size = _get_attr_or_item(engine_args, "data_parallel_size_local", None)
    if local_dp_size is None:
        local_dp_size = _get_attr_or_item(engine_args, "data_parallel_size", 1)
    return tp_size, int(local_dp_size if local_dp_size is not None else 1), pp_size


def get_stage_devices_per_replica(stage_cfg: Any, engine_args: Any | None = None) -> int:
    """Return the number of devices consumed by one replica of *stage_cfg*."""
    if engine_args is None:
        engine_args = getattr(stage_cfg, "engine_args", {})
    if getattr(stage_cfg, "stage_type", "llm") == "diffusion":
        parallel_config = _get_attr_or_item(engine_args, "parallel_config")
        if parallel_config is None:
            return 1

        world_size = _get_attr_or_item(parallel_config, "world_size")
        if world_size is not None:
            return max(1, int(world_size))

        try:
            from vllm_omni.diffusion.data import DiffusionParallelConfig

            return max(1, int(DiffusionParallelConfig.from_dict(_to_dict(parallel_config)).world_size))
        except Exception:
            return 1

    tp_size, local_dp_size, pp_size = _get_local_llm_parallel_sizes(stage_cfg, engine_args)
    return tp_size * max(1, local_dp_size) * pp_size


def compute_replica_layout(
    stage_configs: Sequence[Any],
    *,
    allow_zero: bool = False,
) -> tuple[list[int], dict[int, list[str | None]]]:
    """Compute per-stage replica counts and device assignments.

    Args:
        stage_configs: per-stage config objects with a ``runtime`` sub-config
            exposing ``num_replicas`` and ``devices``.
        allow_zero: when True, ``num_replicas == 0`` is honored (used by
            single-stage / head-distributed mode for non-self stages that
            will be filled dynamically by remote registrations); when False
            (default), the count is clamped to at least 1.

    Returns:
        replicas_per_stage: num_replicas per logical stage.
        replica_devices_map: stage_idx -> per-replica device strings
            (only for stages with num_replicas > 1).
    """
    replicas_per_stage: list[int] = []
    for stage_cfg in stage_configs:
        runtime_cfg = getattr(stage_cfg, "runtime", {})
        num_replicas = int(
            runtime_cfg.get("num_replicas", 1)
            if hasattr(runtime_cfg, "get")
            else getattr(runtime_cfg, "num_replicas", 1)
        )
        if num_replicas < 0:
            raise ValueError(f"num_replicas must be >= 0, got {num_replicas}")
        replicas_per_stage.append(num_replicas if allow_zero else max(1, num_replicas))

    replica_devices_map: dict[int, list[str | None]] = {}
    for stage_id, stage_cfg in enumerate(stage_configs):
        num_replicas = replicas_per_stage[stage_id]
        if num_replicas <= 1:
            continue
        runtime_cfg = getattr(stage_cfg, "runtime", {})
        devices_str = (
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "devices", None)
        )
        devices_per_replica = get_stage_devices_per_replica(stage_cfg)
        replica_devices_map[stage_id] = split_devices_for_replicas(
            devices_str,
            num_replicas,
            devices_per_replica,
            stage_id,
        )
        logger.info(
            "[stage_init] Stage %s: %d replicas, devices_per_replica=%d, devices split: %s",
            stage_id,
            num_replicas,
            devices_per_replica,
            replica_devices_map[stage_id],
        )

    return replicas_per_stage, replica_devices_map


def setup_stage_devices(stage_id: int, runtime_cfg: Any) -> None:
    """Device mapping via set_stage_devices for a single stage."""
    physical_devices = set_stage_devices(
        stage_id,
        runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
    )
    # Only log if we actually set the env vars in the stage
    if physical_devices:
        logger.info(
            "[stage_init] Stage-%s set runtime devices: %s",
            stage_id,
            physical_devices,
        )


@contextmanager
def stage_runtime_setup(stage_id: int, runtime_cfg: Any) -> Generator[None, None, None]:
    """Apply per-stage ``runtime.env`` and ``runtime.devices`` for the context.

    Restores ``runtime.env`` on exit. Device visibility restore remains the
    caller's responsibility (e.g. ``AsyncOmniEngine`` saves/restores the
    platform device-control env var around this block).
    """
    with stage_runtime_env(stage_id, runtime_cfg):
        setup_stage_devices(stage_id, runtime_cfg)
        yield


@contextmanager
def stage_runtime_env(stage_id: int, runtime_cfg: Any) -> Generator[None, None, None]:
    """Apply per-stage ``runtime.env`` for the duration of the context."""
    if runtime_cfg is None:
        runtime_cfg = {}
    elif not isinstance(runtime_cfg, dict):
        runtime_cfg = cast(dict[str, Any], _to_dict(runtime_cfg))

    raw_env = runtime_cfg.get("env")
    if raw_env is None:
        yield
        return
    if isinstance(raw_env, dict):
        runtime_env = cast(dict[str, Any], raw_env)
    else:
        runtime_env = cast(dict[str, Any], _to_dict(raw_env))
        if not runtime_env:
            logger.warning(
                "[stage_init] Stage-%s ignored runtime.env with unsupported type %s",
                stage_id,
                type(raw_env).__name__,
            )
            yield
            return

    previous_env: dict[str, str | None] = {}
    for key, value in runtime_env.items():
        env_key = str(key)
        previous_env[env_key] = os.environ.get(env_key)
        os.environ[env_key] = str(value)

    if previous_env:
        logger.info("[stage_init] Stage-%s applied runtime env keys: %s", stage_id, sorted(previous_env))
    try:
        yield
    finally:
        for key, old_value in previous_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _project_omni_config_fields(
    config: Any,
    *,
    field_map: Mapping[str, str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Copy defined typed config fields into backend adapter kwargs."""
    projected: dict[str, Any] = {}
    for config_field in fields(config):
        name = config_field.name
        if name in exclude or (field_map is not None and name not in field_map):
            continue
        value = getattr(config, name)
        if value is not None:
            projected[field_map.get(name, name) if field_map is not None else name] = copy.deepcopy(value)
    return projected


def _project_upstream_config_fields(
    config: Any,
    field_map: Mapping[str, str],
) -> dict[str, Any]:
    """Project every explicit upstream input, including newly added fields."""
    explicit_fields: frozenset[str] = getattr(config, "_omni_explicit_fields", frozenset())
    unprojected_fields = explicit_fields - frozenset(field_map)
    if unprojected_fields:
        names = ", ".join(sorted(unprojected_fields))
        raise ValueError(f"{type(config).__name__} has explicit field(s) with no EngineArgs projection: {names}")
    return _project_omni_config_fields(
        config,
        field_map=field_map,
        exclude=frozenset(field_map) - explicit_fields,
    )


def _project_omni_stage_engine_args(
    stage_config: BaseVllmOmniStageConfig,
) -> dict[str, Any]:
    """Read backend inputs from one structured stage config."""
    engine_args: dict[str, Any] = {}
    is_diffusion = isinstance(stage_config, VllmOmniDiffusionStageConfig)
    if is_diffusion:
        diffusion_stage = cast(VllmOmniDiffusionStageConfig, stage_config)
        engine_args.update(_project_omni_config_fields(diffusion_stage.diffusion_config))

    for config, excluded_fields in (
        (
            stage_config.model_config,
            frozenset({"default_sampling_params", "has_sampling_extra_args"}),
        ),
        (
            stage_config.runtime_config,
            frozenset({"devices", "num_replicas", "env", "num_gpus"}),
        ),
    ):
        engine_args.update(
            _project_omni_config_fields(
                config,
                exclude=excluded_fields,
            )
        )

    if is_diffusion:
        for config, field_map in (
            (stage_config.load_config, _DIFFUSION_LOAD_STAGE_ENGINE_FIELD_MAP),
            (stage_config.cache_config, _DIFFUSION_CACHE_STAGE_ENGINE_FIELD_MAP),
            (stage_config.scheduler_config, _DIFFUSION_SCHEDULER_STAGE_ENGINE_FIELD_MAP),
        ):
            engine_args.update(_project_omni_config_fields(config, field_map=field_map))
    else:
        for config, field_map in (
            (stage_config.load_config, _LOAD_STAGE_ENGINE_FIELD_MAP),
            (stage_config.cache_config, _CACHE_STAGE_ENGINE_FIELD_MAP),
            (stage_config.scheduler_config, _SCHEDULER_STAGE_ENGINE_FIELD_MAP),
        ):
            engine_args.update(_project_upstream_config_fields(config, field_map))

    for name in ("compilation_config", "profiler_config"):
        value = getattr(stage_config, name)
        if value is not None:
            engine_args[name] = copy.deepcopy(value)

    # The legacy builder always emits this key, including for pipelines such
    # as Audex that intentionally defer architecture discovery to HF config.
    engine_args["model_arch"] = copy.deepcopy(stage_config.model_config.model_arch)

    topology = stage_config.stage_pipeline_config
    topology_engine_args = {
        "model_stage": stage_config.model_stage,
        "worker_type": stage_config.worker_type,
        "scheduler_cls": stage_config.scheduler_cls,
        "hf_config_name": stage_config.hf_config_name,
        "engine_output_type": stage_config.engine_output_type,
        "custom_process_next_stage_input_func": stage_config.custom_process_next_stage_input_func,
        "model_path_resolver": topology.model_path_resolver,
        "retains_state_across_chunks": topology.retains_state_across_chunks,
    }
    engine_args.update(
        {name: copy.deepcopy(value) for name, value in topology_engine_args.items() if value is not None}
    )

    connector_config = stage_config.connector_config
    engine_args["async_chunk"] = connector_config.async_chunk
    if connector_config.omni_kv_config is not None:
        engine_args["omni_kv_config"] = copy.deepcopy(connector_config.omni_kv_config)

    if is_diffusion:
        engine_args["parallel_config"] = _project_omni_config_fields(
            stage_config.parallel_config,
            field_map={name: name for name in _DIFFUSION_PARALLEL_CONFIG_ENGINE_FIELDS},
            exclude=frozenset({"world_size"}),
        )
    else:
        engine_args.update(
            _project_upstream_config_fields(
                stage_config.parallel_config,
                _PARALLEL_CONFIG_ENGINE_FIELD_MAP,
            )
        )

    quantization_config = stage_config.quantization_config
    if quantization_config is not None:
        quantization_key = (
            "quantization" if isinstance(quantization_config, str) and not is_diffusion else "quantization_config"
        )
        engine_args[quantization_key] = copy.deepcopy(quantization_config)

    return engine_args


def _sampling_extra_args_keys(default_sampling_params: Any) -> tuple[str, ...]:
    """Key names of a stage's default sampling ``extra_args``, sorted.

    Only the keys travel into the engine config: engine-core code needs to know
    which request-shaping conventions a stage uses (e.g. CFG request pairing)
    before any request exists, while the values stay a serving-layer concern.
    """
    extra_args = _to_dict(default_sampling_params or {}).get("extra_args") or {}
    try:
        return tuple(sorted(str(key) for key in extra_args))
    except TypeError:
        return ()


def _finalize_engine_args_dict(
    engine_args_dict: dict[str, Any],
    *,
    stage_type: str | StageType,
    stage_id: int,
    model: str,
    stage_connector_spec: dict[str, Any] | None,
    cli_tokenizer: str | None,
    has_sampling_extra_args: bool,
    sampling_extra_args_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Apply representation-independent engine adapter behavior."""
    pipeline_model_root = model
    model = engine_args_dict.pop("model", None) or model
    stage_defines_tokenizer = (
        engine_args_dict.get("tokenizer") is not None or engine_args_dict.get("tokenizer_subdir") is not None
    )
    model = _resolve_model_path(model, engine_args_dict)
    audex_stage = str(engine_args_dict.get("model_stage") or "")
    if audex_stage == "audex_xcodec":
        # TTA stage 1 decodes with the external XCodec1 checkpoint, not a
        # subfolder of the Audex repo: the source comes from XCODEC1_PATH,
        # the stage yaml's own ``model`` entry, or the default HF repo — the
        # pipeline-level model (the Audex root) is never a valid source.
        from vllm_omni.model_executor.models.audex.checkpoint import (
            ensure_audex_snapshot,
            ensure_xcodec1_snapshot,
        )

        stage_model = None if model == pipeline_model_root else model
        model = ensure_xcodec1_snapshot(os.environ.get("XCODEC1_PATH") or stage_model)
        if not stage_defines_tokenizer:
            # XCodec1 ships no tokenizer files; borrow the thinker's (same
            # workaround as the TTS decoder stage).
            audex_root = ensure_audex_snapshot(pipeline_model_root, profile="tta")
            engine_args_dict["tokenizer"] = os.path.join(audex_root, "checkpoint_folder_audiogen")
            stage_defines_tokenizer = True
    elif audex_stage.startswith("audex"):
        # Audex users pass the HF repo ROOT; make sure the required snapshot
        # subset exists locally BEFORE subdir resolution, otherwise the
        # subdirs get joined onto the raw repo id on a fresh cache. The
        # profile keeps TTS-only deployments from pulling the full-checkpoint
        # extras.
        from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

        if audex_stage == "audex_omni":
            audex_profile = "full"
        elif audex_stage == "audex_tta_thinker":
            audex_profile = "tta"
        else:
            audex_profile = "tts"
        model = ensure_audex_snapshot(model, profile=audex_profile)
    model = _resolve_model_tokenizer_paths(model, engine_args_dict)
    if engine_args_dict.get("model_stage") in ("audex_thinker", "audex_tta_thinker"):
        # Audex ships its thinker weights deduplicated into a sibling folder;
        # replicate the official prepare-script symlink on first use. The TTA
        # thinker loads the same checkpoint_folder_audiogen checkpoint.
        from vllm_omni.model_executor.models.audex.checkpoint import ensure_audiogen_weights

        ensure_audiogen_weights(model)
    apply_cli_tokenizer(
        engine_args_dict,
        cli_tokenizer=cli_tokenizer,
        stage_defines_tokenizer=stage_defines_tokenizer,
    )
    engine_args_dict["model"] = model
    # Stage id must come from stage config instead of inherited CLI kwargs
    # (e.g. `--stage-id` defaulting to None).
    engine_args_dict["stage_id"] = stage_id
    if stage_connector_spec:
        engine_args_dict["stage_connector_spec"] = dict(stage_connector_spec or {})

    is_diffusion = stage_type == StageType.DIFFUSION
    if is_diffusion:
        from vllm_omni.diffusion.data import parse_attention_config

        if engine_args_dict.get("diffusion_attention_config") is not None:
            engine_args_dict["diffusion_attention_config"] = parse_attention_config(
                engine_args_dict["diffusion_attention_config"],
            )
    else:
        resolve_worker_cls(engine_args_dict)

    if engine_args_dict.get("worker_type") == "generation":
        # Non-AR generation stages (e.g. code2wav) do not benefit from
        # prefix caching and can expose hybrid KV-cache layouts that vLLM's
        # prefix-cache coordinator does not handle.
        engine_args_dict.setdefault("disable_hybrid_kv_cache_manager", True)
        engine_args_dict.setdefault("enable_prefix_caching", False)

    engine_args_dict["has_sampling_extra_args"] = has_sampling_extra_args
    engine_args_dict["sampling_extra_args_keys"] = sampling_extra_args_keys

    # TODO: Remove this after the performance regression is fixed
    # Set VLLM_USE_FLASHINFER_MOE_FP16=0 for Qwen3-Omni to avoid performance regression
    _maybe_set_qwen3_omni_moe_env(engine_args_dict)
    return engine_args_dict


def build_legacy_engine_args_dict(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    cli_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Implement engine-argument building for the legacy stage representation."""
    engine_args_dict = _to_dict(stage_config.engine_args)
    # Legacy configs can materialize an omitted optional TP size as None.
    # Remove it from the detached adapter dict so the backend default applies
    # without mutating stage_config.engine_args.
    if engine_args_dict.get("tensor_parallel_size") is None:
        engine_args_dict.pop("tensor_parallel_size", None)

    default_sp = _to_dict(_get_attr_or_item(stage_config, "default_sampling_params", {}))
    return _finalize_engine_args_dict(
        engine_args_dict,
        stage_type=_get_attr_or_item(stage_config, "stage_type", "llm"),
        stage_id=stage_config.stage_id,
        model=model,
        stage_connector_spec=stage_connector_spec,
        cli_tokenizer=cli_tokenizer,
        has_sampling_extra_args=bool(default_sp.get("extra_args")),
        sampling_extra_args_keys=_sampling_extra_args_keys(default_sp),
    )


def build_engine_args_dict(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    cli_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Build engine arguments through the stable production entry point.

    Production inputs still use the legacy stage representation. Keep that
    compatibility choice behind this function so callers do not bind directly
    to a representation-specific implementation.
    """
    return build_legacy_engine_args_dict(
        stage_config,
        model,
        stage_connector_spec=stage_connector_spec,
        cli_tokenizer=cli_tokenizer,
    )


def build_engine_args_dict_from_omni_stage_config(
    stage_config: BaseVllmOmniStageConfig,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    cli_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Project one typed stage config into backend engine arguments.

    This projection is prepared for the RFC #4021 stage-init cutover. Current
    production startup reaches the legacy implementation through
    ``build_engine_args_dict`` while strategy and startup-plan inputs still
    use the legacy representation.
    """
    engine_args_dict = _project_omni_stage_engine_args(stage_config)
    _apply_rocm_attention_backend(engine_args_dict, stage_config.stage_type)
    return _finalize_engine_args_dict(
        engine_args_dict,
        stage_type=stage_config.stage_type,
        stage_id=stage_config.stage_id,
        model=model,
        stage_connector_spec=stage_connector_spec,
        cli_tokenizer=cli_tokenizer,
        has_sampling_extra_args=stage_config.model_config.has_sampling_extra_args,
        sampling_extra_args_keys=_sampling_extra_args_keys(stage_config.model_config.default_sampling_params),
    )


def _count_stage_devices(devices: Any) -> int | None:
    if devices is None:
        return None
    if isinstance(devices, (list, tuple)):
        return len(devices)
    values = [device for device in str(devices).split(",") if device.strip()]
    return len(values) or None


def _check_stage_device_layout(stage_config: Any, engine_args_dict: dict[str, Any]) -> None:
    """Fail early when a stage's world size cannot fit its assigned ``devices``.

    Re-runs :func:`check_device_layout` (normally only reached on the
    ``--strategy-config`` path) against the fully resolved per-stage layout, so
    an inconsistent ``tensor_parallel_size`` vs ``devices`` (issue #5003) is
    reported here with a clear message instead of surfacing later as an opaque
    worker-side ``local rank ... out of bounds`` assertion.
    """
    from vllm_omni.config.composable_parallel import StrategyApplyError, check_device_layout

    runtime = getattr(stage_config, "runtime", None)
    devices = _get_attr_or_item(runtime, "devices", None) if runtime is not None else None
    if devices is None:
        # No explicit placement -> vLLM assigns devices itself; nothing to check.
        return

    num_replicas = _get_attr_or_item(runtime, "num_replicas", 1) if runtime is not None else 1
    stage_id = getattr(stage_config, "stage_id", "?")
    tp_size, local_dp_size, pp_size = _get_local_llm_parallel_sizes(stage_config, engine_args_dict)
    if local_dp_size == 0:
        # This process hosts no local DP engines, so its local device list does
        # not describe the cluster-wide DP layout and must not be validated.
        return

    try:
        check_device_layout(
            devices,
            tensor_parallel_size=tp_size,
            data_parallel_size=local_dp_size,
            pipeline_parallel_size=pp_size,
            num_replicas=int(num_replicas or 1),
            role=f"stage-{stage_id}",
        )
    except StrategyApplyError as e:
        message = (
            f"Stage {stage_id}: device layout is inconsistent — {e} "
            "Set devices and the per-stage TP, local DP, PP, and replica counts "
            "so the declared device count matches the local world size."
        )
        device_count = _count_stage_devices(devices)
        world_without_tp = local_dp_size * pp_size
        valid_without_tp = {world_without_tp, int(num_replicas or 1) * world_without_tp}
        if tp_size > 1 and device_count in valid_without_tp:
            message += (
                " This layout is consistent with issue #5003: a top-level "
                "--tensor-parallel-size is applied to every stage, but each stage's "
                "`devices` is not adjusted automatically. Pass --stage-overrides "
                "to set tensor_parallel_size and devices together on every stage, "
                "so single-GPU stages get tensor_parallel_size=1, e.g. "
                '\'{"0": {"tensor_parallel_size": 4, "devices": "0,1,2,3"}, '
                '"1": {"tensor_parallel_size": 1, "devices": "0"}, '
                '"2": {"tensor_parallel_size": 1, "devices": "1"}}\'. '
                "Or omit the top-level --tensor-parallel-size and set it only in "
                "stage-0's override."
            )
        raise ValueError(message) from e


def build_vllm_config(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    engine_args_dict: dict[str, Any] | None = None,
    headless: bool = False,
) -> tuple[Any, type]:
    """Build engine args, then create VllmConfig and executor_class.

    Returns:
        (vllm_config, executor_class)
    """
    if engine_args_dict is None:
        engine_args_dict = build_engine_args_dict(
            stage_config,
            model,
            stage_connector_spec=stage_connector_spec,
        )

    filtered_engine_args_dict = filter_dataclass_kwargs(OmniEngineArgs, engine_args_dict)

    # _to_dict serializes dataclass fields (e.g. StructuredOutputsConfig) into
    # plain dicts.  When OmniEngineArgs is instantiated with the dict, these
    # fields remain dicts instead of being reconstructed as dataclass objects.
    # Later, EngineArgs.create_engine_config() does
    #   self.structured_outputs_config.reasoning_parser = ...
    # which fails on a plain dict.  Reconstruct the dataclass here.
    soc = filtered_engine_args_dict.get("structured_outputs_config")
    if isinstance(soc, dict):
        from vllm.config import StructuredOutputsConfig

        filtered_engine_args_dict["structured_outputs_config"] = StructuredOutputsConfig(**soc)

    omni_engine_args = OmniEngineArgs(**filtered_engine_args_dict)

    # Guard against a per-stage world size that its assigned ``devices`` cannot
    # satisfy (issue #5003). A top-level ``--tensor-parallel-size`` is broadcast
    # to every stage, but ``devices`` is not, so a stage can end up with e.g.
    # tensor_parallel_size=4 while still holding a single-GPU deploy default.
    # Without --strategy-config the strategy-path device check never runs, so
    # the mismatch used to surface only as an opaque worker-side assertion
    # ("DP adjusted local rank N is out of bounds for M devices."). Re-run the
    # same check here, before workers spawn, to fail early with a clear message.
    _check_stage_device_layout(stage_config, filtered_engine_args_dict)

    # Multi-stage pipelines (qwen3_tts code2wav, etc.) set max_model_len
    # larger than HF max_position_embeddings by design. vLLM's validator
    # rejects that without the env flag.
    if filtered_engine_args_dict.get("max_model_len") is not None and not os.environ.get(
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
    ):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        logger.debug(
            "Auto-set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for stage %s (max_model_len=%s).",
            stage_config.stage_id,
            filtered_engine_args_dict["max_model_len"],
        )

    vllm_config = omni_engine_args.create_engine_config(
        usage_context=UsageContext.LLM_CLASS,
        headless=headless,
    )
    executor_class = Executor.get_class(vllm_config)

    # Upgrade vanilla INCConfig to OmniINCConfig for multi-stage models.
    upgraded = OmniINCConfig.maybe_upgrade(vllm_config.quant_config)
    if upgraded is not vllm_config.quant_config:
        vllm_config = replace(vllm_config, quant_config=upgraded)

    custom_voice_dir = engine_args_dict.get("custom_voice_dir")
    if custom_voice_dir:
        setattr(vllm_config.model_config.hf_config, "custom_voice_dir", custom_voice_dir)

    return vllm_config, executor_class


def build_llm_stage_output_processor(
    plan: LogicalStageInitPlan,
    stage_vllm_config: Any,
    log_stats: bool = False,
) -> Any | None:
    """Build one output processor per logical LLM stage.

    ``log_stats`` controls whether the processor populates per-request
    IterationStats (consumed by the Prometheus wrap). Default False matches
    the upstream MultimodalOutputProcessor default and respects the
    --log-stats CLI flag plumbed through AsyncOmniEngine.
    """

    metadata = plan.replicas[0].metadata
    if stage_vllm_config.model_config.skip_tokenizer_init:
        tokenizer = None
    else:
        tokenizer = cached_tokenizer_from_config(
            model_config=stage_vllm_config.model_config,
        )
    return MultimodalOutputProcessor(
        tokenizer=tokenizer,
        log_stats=log_stats,
        engine_core_output_type=metadata.engine_output_type,
    )


class _TokenOnlyRenderer(BaseRenderer):
    """Renderer for stages that explicitly skip tokenizer initialization."""

    def render_messages(self, messages, params):
        raise ValueError(
            "Chat messages are unavailable when skip_tokenizer_init=True; submit prompt_token_ids or prompt_embeds"
        )


def _build_token_only_renderer(stage_vllm_config: Any) -> BaseRenderer:
    return _TokenOnlyRenderer(stage_vllm_config, tokenizer=None)


def build_stage0_input_processor(stage_vllm_config: Any) -> InputProcessor:
    """Build the shared stage-0 input processor."""

    patch_generation_config_if_needed(stage_vllm_config.model_config)
    if bool(getattr(stage_vllm_config.model_config, "skip_tokenizer_init", False)):
        input_processor = InputProcessor(
            vllm_config=stage_vllm_config,
            renderer=_build_token_only_renderer(stage_vllm_config),
        )
    else:
        input_processor = InputProcessor(vllm_config=stage_vllm_config)
    input_processor.input_preprocessor = OmniInputPreprocessor(
        vllm_config=stage_vllm_config,
        renderer=input_processor.renderer,
    )
    return input_processor


def _cleanup_stale_lock_if_dead(lock_file: str) -> bool:
    """If *lock_file* exists and its recorded PID is dead, unlink the file.

    Returns ``True`` if the stale lock was cleaned up (caller should retry),
    ``False`` otherwise (lock holder appears alive, or file could not be read).
    """
    try:
        with open(lock_file) as fh:
            content = fh.read().strip()
        if not content:
            return False
        pid = int(content)
    except (OSError, ValueError):
        return False

    # Check whether the PID is still alive.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        # PID does not exist — stale lock.
        logger.info(
            "Removing stale device lock %s (PID %s is dead)",
            lock_file,
            pid,
        )
        try:
            os.unlink(lock_file)
            return True
        except OSError:
            logger.debug("Failed to unlink stale lock %s", lock_file)
            return False
    except PermissionError:
        # PID exists but we cannot signal it (different user) — treat as alive.
        return False

    # PID is alive — legitimate lock holder.
    return False


def acquire_device_locks(
    stage_id: int,
    engine_args_dict: dict[str, Any],
    stage_init_timeout: int,
) -> list[int]:
    """Acquire exclusive file locks on devices needed by this stage.

    Returns list of lock file descriptors that must be released after init.
    """
    lock_fds: list[int] = []
    try:
        # Get parallel sizes
        if "parallel_config" in engine_args_dict:
            pc = engine_args_dict["parallel_config"]
            tensor_parallel_size = pc.get("tensor_parallel_size", 1)
            pipeline_parallel_size = pc.get("pipeline_parallel_size", 1)
            data_parallel_size = pc.get("data_parallel_size", 1)
            prefill_context_parallel_size = pc.get("prefill_context_parallel_size", 1)
            sequence_parallel_size = pc.get("sequence_parallel_size", 1)
            cfg_parallel_size = pc.get("cfg_parallel_size", 1)
        else:
            tensor_parallel_size = engine_args_dict.get("tensor_parallel_size", 1)
            pipeline_parallel_size = engine_args_dict.get("pipeline_parallel_size", 1)
            data_parallel_size = engine_args_dict.get("data_parallel_size", 1)
            prefill_context_parallel_size = engine_args_dict.get("prefill_context_parallel_size", 1)
            sequence_parallel_size = 1
            cfg_parallel_size = 1

        num_devices_per_stage = (
            tensor_parallel_size
            * pipeline_parallel_size
            * data_parallel_size
            * prefill_context_parallel_size
            * sequence_parallel_size
            * cfg_parallel_size
        )

        # Get physical device IDs
        device_control_env = current_omni_platform.device_control_env_var
        visible_devices_str = os.environ.get(device_control_env)
        physical_devices: list[int] = []

        if visible_devices_str:
            try:
                physical_devices = [int(x.strip()) for x in visible_devices_str.split(",") if x.strip()]
            except (ValueError, IndexError):
                pass

        if not physical_devices:
            num_devices = current_omni_platform.get_device_count()
            physical_devices = list(range(num_devices))

        if len(physical_devices) < num_devices_per_stage:
            raise RuntimeError(
                f"Stage {stage_id} requires {num_devices_per_stage} device(s) based on parallel_config, "
                f"but only {len(physical_devices)} device(s) are available: {physical_devices}"
            )

        num_devices_to_lock = num_devices_per_stage
        devices_to_lock = sorted(physical_devices[:num_devices_to_lock])

        logger.debug(
            "Parallel config: TP=%d, PP=%d, DP=%d, PCP=%d, SP=%d, CFG=%d; will lock %d devices: %s",
            tensor_parallel_size,
            pipeline_parallel_size,
            data_parallel_size,
            prefill_context_parallel_size,
            sequence_parallel_size,
            cfg_parallel_size,
            num_devices_to_lock,
            devices_to_lock,
        )

        # Acquire locks
        wait_start = time.time()
        for device_id in devices_to_lock:
            lock_file = f"/tmp/vllm_omni_device_{device_id}_init.lock"
            lock_acquired = False
            already_cleaned_stale = False  # only try stale cleanup once per device

            while not lock_acquired:
                try:
                    lock_fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        os.ftruncate(lock_fd, 0)
                        os.write(lock_fd, f"{os.getpid()}\n".encode())
                        os.fsync(lock_fd)
                        lock_acquired = True
                        lock_fds.append(lock_fd)
                        logger.debug("Acquired exclusive lock for device %s", device_id)
                    except BlockingIOError:
                        os.close(lock_fd)
                        # Detect and clean stale locks from dead processes.
                        if not already_cleaned_stale:
                            already_cleaned_stale = True
                            if _cleanup_stale_lock_if_dead(lock_file):
                                continue  # retry flock immediately
                        if time.time() - wait_start > stage_init_timeout:
                            logger.warning(
                                "Timeout waiting for device %s initialization lock, proceeding anyway",
                                device_id,
                            )
                            break
                        time.sleep(0.01)
                except OSError as e:
                    logger.debug(
                        "Failed to acquire lock for device %s: %s, continuing anyway",
                        device_id,
                        e,
                    )
                    try:
                        os.close(lock_fd)
                    except (OSError, NameError):
                        pass
                    break

    except Exception as e:
        logger.debug(
            "[Stage-%s] Failed to set up sequential initialization lock: %s",
            stage_id,
            e,
        )

    return lock_fds


def release_device_locks(lock_fds: list[int]) -> None:
    """Release file locks acquired by acquire_device_locks."""
    for lock_fd in lock_fds:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            logger.debug("Released initialization lock (fd=%s)", lock_fd)
        except (OSError, ValueError):
            pass


def load_omni_transfer_config_for_model(model: str, config_path: str | None) -> Any:
    """Load omni transfer config from an explicit path or resolved model config.

    Resolves ``base_config`` inheritance (CI overlay → base deploy YAML) so
    that connectors defined in the base config are visible to the transfer
    config parser.
    """
    from vllm_omni.distributed.omni_connectors import load_omni_transfer_config

    try:
        resolved_config_path = config_path or resolve_model_config_path(model)
        if resolved_config_path is None:
            return None
        from vllm_omni.config.stage_config import resolve_deploy_yaml

        resolved_dict = resolve_deploy_yaml(resolved_config_path)
        return load_omni_transfer_config(config_dict=resolved_dict)
    except Exception as e:
        logger.warning("[stage_init] Failed to load transfer config: %s", e)
        return None


def get_stage_connector_spec(
    omni_transfer_config: Any,
    stage_id: int,
    async_chunk: bool,
) -> dict[str, Any]:
    """Return the first connector spec for a stage data-plane edge."""
    from vllm_omni.distributed.omni_connectors import get_stage_connector_config

    stage_connectors_cfg = get_stage_connector_config(omni_transfer_config, stage_id)
    for cfg in stage_connectors_cfg.values():
        return dict(cfg.get("spec", {}))

    # A producer does not consume connector data itself. Keep its connector
    # for both async-chunk and terminal full-payload sends, but mark it
    # sender-only so the scheduler does not park orchestrator-provided inputs
    # waiting for an upstream payload.
    target_stage = str(stage_id)
    for (from_stage, _to_stage), spec in getattr(omni_transfer_config, "connectors", {}).items():
        if from_stage == target_stage:
            extra = dict(spec.extra or {})
            extra.setdefault("role", "sender")
            return {"name": spec.name, "extra": extra}
    return {}


def build_diffusion_config(
    model: str,
    stage_cfg: Any,
    metadata: StageMetadata,
) -> Any:
    """Build diffusion config for a stage."""

    engine_args_dict = build_engine_args_dict(stage_cfg, model)
    od_config = OmniDiffusionConfig.from_kwargs(**engine_args_dict)

    num_devices_per_stage = od_config.parallel_config.world_size
    device_control_env = current_omni_platform.device_control_env_var
    visible_devices_str = os.environ.get(device_control_env) if device_control_env else None
    physical_devices: list[str | int]
    if visible_devices_str:
        physical_devices = [device.strip() for device in visible_devices_str.split(",") if device.strip()]
    else:
        physical_devices = list(range(current_omni_platform.get_device_count()))

    if len(physical_devices) < num_devices_per_stage:
        raise ValueError(
            f"Stage {metadata.stage_id} requires {num_devices_per_stage} device(s) based on parallel_config, "
            f"but {len(physical_devices)} device(s) are available: {physical_devices}"
        )

    od_config.num_gpus = num_devices_per_stage
    if metadata.cfg_kv_collect_func is not None:
        od_config.cfg_kv_collect_func = metadata.cfg_kv_collect_func
    return od_config


def initialize_diffusion_stage(
    stage_id: int,
    model: str,
    stage_cfg: Any,
    metadata: StageMetadata,
    stage_init_timeout: int,
    batch_size: int = 1,
    use_inline: bool = False,
) -> Any:
    """Build a diffusion stage client.

    Args:
        model: Model name or path.
        stage_cfg: Stage configuration.
        metadata: Extracted stage metadata.
        stage_init_timeout: Timeout in seconds for stage initialization handshake
        batch_size: Client-side request batch width. Does not set scheduler
            ``max_num_seqs``; pass ``--max-num-seqs`` or stage YAML for that.
            Forwarded to ``StageDiffusionClient``.
        use_inline: If True, uses the inline diffusion client instead of subprocess.
    """
    from vllm_omni.diffusion.stage_diffusion_client import create_diffusion_client

    od_config = build_diffusion_config(model, stage_cfg, metadata)
    return create_diffusion_client(model, od_config, metadata, stage_init_timeout, batch_size, use_inline)


def _stage_declares_cfg_pairs(model_config: Any) -> bool:
    """Whether this stage submits classifier-free-guidance request pairs.

    Two independent declarations, because a model may own either side of the
    mechanism without the other:

    * a CFG logits processor is configured (blending implies pairing), or
    * the stage's default sampling ``extra_args`` carry ``cfg_role``, which is
      how a model declares paired requests without owning a logits processor.
      Only the *key set* survives into the engine config (see
      ``sampling_extra_args_keys``); the values stay in the serving layer.
    """
    processors = getattr(model_config, "logits_processors", None) or []
    if any("CFGLogitsProcessor" in getattr(proc, "__name__", str(proc)) for proc in processors):
        return True
    return "cfg_role" in (getattr(model_config, "sampling_extra_args_keys", None) or ())


def maybe_apply_cfg_scheduler_patches(vllm_config: Any) -> None:
    """Install the CFG pairing scheduler patches for CFG-configured engines.

    Must run in the engine-core process BEFORE ``Scheduler`` is constructed:
    the patch wraps ``Scheduler.__init__`` to add the pair registry, so a
    scheduler built earlier would never become pair-aware. Gated so non-CFG
    stages stay untouched.
    """
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None or not _stage_declares_cfg_pairs(model_config):
        return

    from vllm_omni.model_executor.models.common.cfg_pairing import apply_cfg_patches

    apply_cfg_patches()

    from vllm.v1.core.sched.scheduler import Scheduler

    if not getattr(Scheduler.schedule, "_cfg_pairing_patched", False):
        raise RuntimeError("CFG pairing scheduler patches failed to install before Scheduler construction")


# Name kept for callers written against the Audex-only gate.
maybe_apply_audex_cfg_patches = maybe_apply_cfg_scheduler_patches
