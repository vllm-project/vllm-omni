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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.executor import Executor

from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints.stage_utils import _to_dict, set_stage_devices
from vllm_omni.entrypoints.utils import filter_dataclass_kwargs, resolve_model_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

logger = init_logger(__name__)


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


def resolve_worker_cls(engine_args: dict[str, Any]) -> None:
    """Resolve worker_cls from worker_type for non-diffusion stages."""
    worker_type = engine_args.get("worker_type", None)
    if not worker_type:
        return
    worker_cls = engine_args.get("worker_cls")
    if worker_cls is not None and worker_cls != "auto":
        return

    from vllm_omni.platforms import current_omni_platform

    worker_type = str(worker_type).lower()
    if worker_type == "ar":
        engine_args["worker_cls"] = current_omni_platform.get_omni_ar_worker_cls()
    elif worker_type == "generation":
        engine_args["worker_cls"] = current_omni_platform.get_omni_generation_worker_cls()
    else:
        raise ValueError(f"Unknown worker_type: {worker_type}")


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


@dataclass
class StartedLlmStage:
    """Resources for an LLM stage that has completed startup."""

    stage_id: int
    metadata: Any
    vllm_config: Any
    executor_class: type
    engine_manager: Any
    coordinator: Any
    addresses: Any


def extract_stage_metadata(stage_config: Any) -> StageMetadata:
    """Pure data extraction from a stage_config object."""
    stage_id: int = stage_config.stage_id
    stage_type: Literal["llm", "diffusion"] = getattr(stage_config, "stage_type", "llm")
    engine_args = stage_config.engine_args
    runtime_cfg = getattr(stage_config, "runtime", {})
    engine_input_source: list[int] = getattr(stage_config, "engine_input_source", [])
    final_output: bool = getattr(stage_config, "final_output", False)
    final_output_type: str | None = getattr(stage_config, "final_output_type", None)

    default_sp = _to_dict(getattr(stage_config, "default_sampling_params", {}))
    SPClass = SamplingParams if stage_type == "llm" else OmniDiffusionSamplingParams
    default_sampling_params: OmniSamplingParams = SPClass(**default_sp)

    custom_process_input_func: Callable | None = None
    if hasattr(stage_config, "custom_process_input_func"):
        mod_path, fn_name = stage_config.custom_process_input_func.rsplit(".", 1)
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


def setup_stage_devices(stage_id: int, runtime_cfg: Any) -> None:
    """Device mapping via set_stage_devices for a single stage."""
    try:
        from vllm_omni.platforms import current_omni_platform

        device_type = current_omni_platform.device_type
        set_stage_devices(
            stage_id,
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
            device_type=device_type,
        )
        logger.info(
            "[stage_init] Stage-%s set devices for %s, runtime devices: %s",
            stage_id,
            device_type,
            runtime_cfg.get("devices") if hasattr(runtime_cfg, "get") else None,
        )
    except Exception as e:
        logger.warning("Device setup failed for stage %s: %s", stage_id, e)


def build_engine_args_dict(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the normalized engine args dict for one stage."""
    engine_args = stage_config.engine_args
    stage_type = getattr(stage_config, "stage_type", "llm")
    stage_id = stage_config.stage_id

    engine_args_dict = _to_dict(engine_args)
    model = _resolve_model_tokenizer_paths(model, engine_args_dict)
    engine_args_dict["model"] = model
    # Stage id must come from stage config instead of inherited CLI kwargs
    # (e.g. `--stage-id` defaulting to None).
    engine_args_dict["stage_id"] = stage_id
    if engine_args_dict.get("async_chunk", False):
        engine_args_dict["stage_connector_spec"] = dict(stage_connector_spec or {})

    if stage_type != "diffusion":
        resolve_worker_cls(engine_args_dict)

    return engine_args_dict


def build_vllm_config(
    stage_config: Any,
    model: str,
    stage_connector_spec: dict[str, Any] | None = None,
    engine_args_dict: dict[str, Any] | None = None,
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
    vllm_config = omni_engine_args.create_engine_config(usage_context=UsageContext.LLM_CLASS)
    executor_class = Executor.get_class(vllm_config)

    return vllm_config, executor_class


def acquire_device_locks(
    stage_id: int,
    engine_args_dict: dict[str, Any],
    stage_init_timeout: int = 300,
) -> list[int]:
    """Acquire exclusive file locks on devices needed by this stage.

    Returns list of lock file descriptors that must be released after init.
    """
    lock_fds: list[int] = []
    try:
        from vllm_omni.platforms import current_omni_platform

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

        num_devices_to_lock = min(num_devices_per_stage, len(physical_devices))
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


def load_omni_transfer_config_for_model(model: str, config_path: str | None) -> Any:
    """Load omni transfer config from an explicit path or resolved model config."""
    from vllm_omni.distributed.omni_connectors import load_omni_transfer_config

    try:
        resolved_config_path = config_path or resolve_model_config_path(model)
        return load_omni_transfer_config(resolved_config_path)
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


def initialize_diffusion_stage(
    model: str,
    stage_cfg: Any,
    metadata: StageMetadata,
    batch_size: int = 1,
) -> Any:
    """Build a diffusion stage client.

    Args:
        model: Model name or path.
        stage_cfg: Stage configuration.
        metadata: Extracted stage metadata.
        batch_size: Maximum number of requests to batch together in the
            diffusion engine.  Passed through to ``StageDiffusionClient``
            and ultimately to ``AsyncOmniDiffusion``.
    """
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient

    od_config = OmniDiffusionConfig.from_kwargs(
        model=model,
        **_to_dict(stage_cfg.engine_args),
    )
    if metadata.cfg_kv_collect_func is not None:
        od_config.cfg_kv_collect_func = metadata.cfg_kv_collect_func
    return StageDiffusionClient(model, od_config, metadata, batch_size=batch_size)


def close_started_llm_stage(started: StartedLlmStage) -> None:
    """Close managers owned by a launched stage that never attached."""
    resources = (
        ("engine manager", started.engine_manager),
        ("coordinator", started.coordinator),
    )
    for resource_name, resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as cleanup_error:
            logger.warning(
                "[stage_init] Failed to close launched %s for stage %s: %s",
                resource_name,
                started.stage_id,
                cleanup_error,
            )


def finalize_initialized_stages(
    stage_clients: list[Any | None],
    input_processor: InputProcessor | None,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """Validate successful init and build runtime metadata lists."""
    if any(stage_client is None for stage_client in stage_clients):
        raise RuntimeError("Stage initialization completed with missing stage clients")

    initialized_stage_clients = [stage_client for stage_client in stage_clients if stage_client is not None]
    default_sampling_params_list = [stage_client.default_sampling_params for stage_client in initialized_stage_clients]
    stage_metadata = [
        {
            "final_output": stage_client.final_output,
            "final_output_type": stage_client.final_output_type,
            "stage_type": stage_client.stage_type,
        }
        for stage_client in initialized_stage_clients
    ]

    if not isinstance(input_processor, InputProcessor):
        has_llm_stage = any(metadata.get("stage_type") != "diffusion" for metadata in stage_metadata)
        if has_llm_stage:
            raise RuntimeError("Failed to initialize stage-0 InputProcessor for LLM pipeline")

    return initialized_stage_clients, default_sampling_params_list, stage_metadata


def cleanup_failed_stage_initialization(
    stage_clients: list[Any | None],
    started_llm_stages: list[StartedLlmStage],
) -> None:
    """Shutdown attached stages and close any launched-but-unattached engines."""
    for cleanup_stage_id, stage_client in reversed(list(enumerate(stage_clients))):
        if stage_client is None:
            continue
        try:
            stage_client.shutdown()
        except Exception as cleanup_error:
            logger.warning(
                "[stage_init] Failed to shutdown initialized stage %s after init failure: %s",
                cleanup_stage_id,
                cleanup_error,
            )

    for started in reversed(started_llm_stages):
        if stage_clients[started.stage_id] is not None:
            continue
        close_started_llm_stage(started)
