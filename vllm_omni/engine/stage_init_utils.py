"""
Stage initialization helpers for vLLM-Omni multi-stage runtime.

Extracts orchestration-level init logic (config extraction, plugin loading,
multiprocessing setup, device mapping, device locking, engine args building)
out of StageEngineCoreClient into reusable functions.
"""

from __future__ import annotations

import fcntl
import importlib
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.executor import Executor

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.entrypoints.utils import filter_dataclass_kwargs, resolve_model_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams
from vllm_omni.inputs.preprocess import OmniInputPreprocessor
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


@dataclass
class LogicalStageInitPlan:
    """Startup plan for one logical stage."""

    stage_idx: int
    configured_stage_id: int
    replicas: list[ReplicaInitPlan]


def _resolve_model_to_local_path(model: str) -> str:
    """Resolve an HF Hub model ID to a local cache path."""
    if os.path.isdir(model):
        return model

    try:
        from huggingface_hub import snapshot_download

        # Keep init path resolution offline-friendly.
        return snapshot_download(model, local_files_only=True)
    except Exception:
        logger.warning(
            "[stage_init] Could not resolve %s to local snapshot; using as-is",
            model,
        )
        return model


def _resolve_model_tokenizer_paths(model: str, engine_args: dict[str, Any]) -> str:
    """Apply model_subdir/tokenizer_subdir indirections from stage engine args."""
    model_subdir = engine_args.pop("model_subdir", None)
    tokenizer_subdir = engine_args.pop("tokenizer_subdir", None)
    if model_subdir is None and tokenizer_subdir is None:
        return model

    resolved_base = _resolve_model_to_local_path(model)

    if model_subdir:
        model = os.path.join(resolved_base, model_subdir)
        logger.info("[stage_init] Using model subdirectory: %s", model)

    if tokenizer_subdir is not None:
        tokenizer_path = os.path.join(resolved_base, tokenizer_subdir) if tokenizer_subdir else resolved_base
        engine_args["tokenizer"] = tokenizer_path
        logger.info("[stage_init] Using tokenizer from: %s", tokenizer_path)
    elif model_subdir and "tokenizer" not in engine_args:
        # Keep legacy behavior: model in subdir, tokenizer defaults to base path.
        engine_args["tokenizer"] = resolved_base
        logger.info("[stage_init] Using tokenizer from base model path: %s", resolved_base)

    return model


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
    prompt_expand_func: Callable | None = None
    cfg_kv_collect_func: Callable | None = None
    # Multi-replica: replica_id distinguishes replicas of the same stage.
    # For single-replica stages this defaults to 0.
    replica_id: int = 0


def extract_stage_metadata(stage_config: Any) -> StageMetadata:
    """Pure data extraction from a stage_config object."""
    stage_id: int = stage_config.stage_id
    stage_type: Literal["llm", "diffusion"] = getattr(stage_config, "stage_type", "llm")
    engine_args = stage_config.engine_args

    if current_omni_platform.is_rocm():
        if stage_type != "diffusion" and engine_args.get("attention_backend") is None:
            from vllm._aiter_ops import rocm_aiter_ops

            if rocm_aiter_ops.is_enabled():
                engine_args["attention_backend"] = "ROCM_AITER_FA"
            # Before vLLM v0.19.0, the default attention backend is TRITON_ATTN for ROCm.
            # Since vLLM v0.19.0, the default attention backend is ROCM_ATTN for ROCm.
            # However, the compatibility of ROCM_ATTN with Omni is not guaranteed.
            # Therefore, we still use TRITON_ATTN as the default attention backend,
            # when the selected_backend is not specified.
            engine_args["attention_backend"] = "TRITON_ATTN"

    runtime_cfg = getattr(stage_config, "runtime", {})
    engine_input_source: list[int] = getattr(stage_config, "engine_input_source", [])
    final_output: bool = getattr(stage_config, "final_output", False)
    final_output_type: str | None = getattr(stage_config, "final_output_type", None)

    default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
    SPClass = SamplingParams if stage_type == "llm" else OmniDiffusionSamplingParams
    default_sampling_params: OmniSamplingParams = SPClass(**default_sp)

    custom_process_input_func: Callable | None = None
    _cpif_path = getattr(stage_config, "custom_process_input_func", None)
    if _cpif_path:
        mod_path, fn_name = _cpif_path.rsplit(".", 1)
        custom_process_input_func = getattr(importlib.import_module(mod_path), fn_name)

    prompt_expand_func: Callable | None = None
    _pef_path = getattr(stage_config, "prompt_expand_func", None)
    if _pef_path:
        _mod, _fn = _pef_path.rsplit(".", 1)
        prompt_expand_func = getattr(importlib.import_module(_mod), _fn)

    cfg_kv_collect_func: Callable | None = None
    _ckf_path = getattr(stage_config, "cfg_kv_collect_func", None)
    if _ckf_path:
        _mod, _fn = _ckf_path.rsplit(".", 1)
        cfg_kv_collect_func = getattr(importlib.import_module(_mod), _fn)

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
            default_sampling_params=default_sampling_params,
            custom_process_input_func=custom_process_input_func,
            model_stage=None,
            runtime_cfg=runtime_cfg,
            cfg_kv_collect_func=cfg_kv_collect_func,
        )

    model_stage = getattr(engine_args, "model_stage", None)
    engine_output_type = getattr(engine_args, "engine_output_type", None)
    is_comprehension = getattr(stage_config, "is_comprehension", False)
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
        default_sampling_params=default_sampling_params,
        custom_process_input_func=custom_process_input_func,
        model_stage=model_stage,
        runtime_cfg=runtime_cfg,
        prompt_expand_func=prompt_expand_func,
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


def split_devices_for_replicas(
    devices_str: str | None,
    num_replicas: int,
    tp_size: int,
    stage_id: int,
) -> list[str]:
    """Split a devices string into per-replica subsets.

    When ``num_replicas`` is 1, returns ``[devices_str]`` unchanged.
    Otherwise, the total number of device IDs must equal
    ``num_replicas * tp_size``; each replica gets ``tp_size`` consecutive
    device IDs.

    Example::

        split_devices_for_replicas("1,2,3,4", num_replicas=2, tp_size=2, stage_id=1)
        # → ["1,2", "3,4"]
    """
    if num_replicas <= 1 or devices_str is None:
        return [devices_str] if devices_str is not None else [devices_str]

    device_list = [d.strip() for d in devices_str.split(",") if d.strip()]
    required = num_replicas * tp_size
    if len(device_list) != required:
        raise ValueError(
            f"Stage {stage_id}: num_replicas={num_replicas}, "
            f"tensor_parallel_size={tp_size} requires "
            f"{required} devices, got {len(device_list)}: {devices_str}"
        )

    result: list[str] = []
    for r in range(num_replicas):
        chunk = device_list[r * tp_size : (r + 1) * tp_size]
        result.append(",".join(chunk))
    return result


def get_stage_tp_size(stage_cfg: Any) -> int:
    """Extract tensor_parallel_size from a stage config object."""
    engine_args = getattr(stage_cfg, "engine_args", {})
    if hasattr(engine_args, "get"):
        return int(engine_args.get("tensor_parallel_size", 1) or 1)
    return int(getattr(engine_args, "tensor_parallel_size", 1) or 1)


def get_stage_devices_per_replica(stage_cfg: Any) -> int:
    """Return the number of devices consumed by one replica of *stage_cfg*."""
    if getattr(stage_cfg, "stage_type", "llm") != "diffusion":
        return get_stage_tp_size(stage_cfg)

    parallel_config = _get_attr_or_item(getattr(stage_cfg, "engine_args", {}), "parallel_config")
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


def compute_replica_layout(
    stage_configs: Sequence[Any],
) -> tuple[list[int], dict[int, list[str]]]:
    """Compute per-stage replica counts and device assignments.

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
        replicas_per_stage.append(max(1, num_replicas))

    replica_devices_map: dict[int, list[str]] = {}
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


def build_engine_args_dict(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    cli_tokenizer: str | None = None,
) -> dict[str, Any]:
    """Build the normalized engine args dict for one stage."""
    engine_args = stage_config.engine_args
    # HACK (Alex) Tensor parallel size should not be passed as None;
    # remove it if this is the case so that we fall back to default
    # creation from vLLM's engine args.
    # NOTE: This will be fixed more generically in ongoing work for engine arg filtering.
    if "tensor_parallel_size" in engine_args and engine_args["tensor_parallel_size"] is None:
        del engine_args["tensor_parallel_size"]

    stage_type = getattr(stage_config, "stage_type", "llm")
    stage_id = stage_config.stage_id

    engine_args_dict = _to_dict(engine_args)
    stage_defines_tokenizer = (
        engine_args_dict.get("tokenizer") is not None or engine_args_dict.get("tokenizer_subdir") is not None
    )
    model = _resolve_model_tokenizer_paths(model, engine_args_dict)
    apply_cli_tokenizer(
        engine_args_dict,
        cli_tokenizer=cli_tokenizer,
        stage_defines_tokenizer=stage_defines_tokenizer,
    )
    engine_args_dict["model"] = model
    # Stage id must come from stage config instead of inherited CLI kwargs
    # (e.g. `--stage-id` defaulting to None).
    engine_args_dict["stage_id"] = stage_id
    if engine_args_dict.get("async_chunk", False):
        engine_args_dict["stage_connector_spec"] = dict(stage_connector_spec or {})

    if stage_type == "diffusion":
        from vllm_omni.diffusion.data import parse_attention_config

        if engine_args_dict.get("diffusion_attention_config") is not None:
            engine_args_dict["diffusion_attention_config"] = parse_attention_config(
                engine_args_dict.get("diffusion_attention_config"),
            )

    if stage_type != "diffusion":
        resolve_worker_cls(engine_args_dict)

    if engine_args_dict.get("worker_type") == "generation":
        # Non-AR generation stages (e.g. code2wav) do not benefit from
        # prefix caching and can expose hybrid KV-cache layouts that vLLM's
        # prefix-cache coordinator does not handle.
        engine_args_dict.setdefault("disable_hybrid_kv_cache_manager", True)
        engine_args_dict.setdefault("enable_prefix_caching", False)

    # Check whether the stage's default_sampling_params defines extra_args.
    default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
    engine_args_dict["has_sampling_extra_args"] = bool(default_sp.get("extra_args"))

    return engine_args_dict


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
    omni_engine_args = OmniEngineArgs(**filtered_engine_args_dict)

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

    return vllm_config, executor_class


def build_llm_stage_output_processor(plan: LogicalStageInitPlan, stage_vllm_config: Any) -> Any | None:
    """Build one output processor per logical LLM stage."""

    metadata = plan.replicas[0].metadata
    if stage_vllm_config.model_config.skip_tokenizer_init:
        tokenizer = None
    else:
        tokenizer = cached_tokenizer_from_config(
            model_config=stage_vllm_config.model_config,
        )
    return MultimodalOutputProcessor(
        tokenizer=tokenizer,
        log_stats=False,
        engine_core_output_type=metadata.engine_output_type,
    )


def build_stage0_input_processor(stage_vllm_config: Any) -> InputProcessor:
    """Build the shared stage-0 input processor."""

    patch_generation_config_if_needed(stage_vllm_config.model_config)
    input_processor = InputProcessor(vllm_config=stage_vllm_config)
    input_processor.input_preprocessor = OmniInputPreprocessor(
        vllm_config=stage_vllm_config,
        renderer=input_processor.renderer,
    )
    return input_processor


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


def acquire_diffusion_device_locks(
    stage_id: int,
    od_config: Any,
    stage_init_timeout: int,
) -> list[int]:
    """Acquire init locks for the GPU set used by a diffusion stage.

    Diffusion stages express their device count via ``OmniDiffusionConfig``'s
    ``parallel_config.world_size`` rather than the LLM-style
    ``tensor_parallel_size`` knob, so adapt to the shape that
    ``acquire_device_locks`` understands.
    """
    parallel_config = getattr(od_config, "parallel_config", None)
    world_size = getattr(parallel_config, "world_size", 1)
    try:
        world_size = max(1, int(world_size))
    except (TypeError, ValueError):
        world_size = 1

    return acquire_device_locks(
        stage_id,
        {"tensor_parallel_size": world_size},
        stage_init_timeout,
    )


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
    """Return the first connector spec for the stage when async chunking is enabled."""
    from vllm_omni.distributed.omni_connectors import get_stage_connector_config

    if not async_chunk:
        return {}

    stage_connectors_cfg = get_stage_connector_config(omni_transfer_config, stage_id)
    for cfg in stage_connectors_cfg.values():
        return dict(cfg.get("spec", {}))
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
        batch_size: Maximum number of requests to batch together in the
            diffusion engine.  Passed through to ``StageDiffusionClient``
            and ultimately to ``AsyncOmni``.
        use_inline: If True, uses the inline diffusion client instead of subprocess.
    """
    from vllm_omni.diffusion.stage_diffusion_client import create_diffusion_client

    od_config = build_diffusion_config(model, stage_cfg, metadata)
    return create_diffusion_client(model, od_config, metadata, stage_init_timeout, batch_size, use_inline)
