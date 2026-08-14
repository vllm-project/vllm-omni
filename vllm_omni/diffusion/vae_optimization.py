# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability-gated production VAE runtime optimizations.

The implementation intentionally lives outside model adapters.  Pipelines opt
in through architecture metadata, while this module owns validation, runtime
wrapping, diagnostics, failure fallback, and bounded compilation.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

VaeMode = Literal["auto", "true", "false"]
VaeProfile = Literal["safe", "optimized", "diagnostic", "student"]


@dataclass(frozen=True)
class VaeOptimizationCapabilities:
    tiled_decode: bool = False
    stacked_tiles: bool = False
    compilation: bool = False
    spatial_sharding: bool = False
    independent_process_group: bool = False


@dataclass(frozen=True)
class ResolvedVaeOptimization:
    profile: VaeProfile
    stack_tiling: VaeMode
    compile: VaeMode
    diagnostics: bool
    compile_max_shape_buckets: int


_NO_VAE_OPTIMIZATIONS = VaeOptimizationCapabilities()
_WAN_VAE_CAPABILITIES = VaeOptimizationCapabilities(
    tiled_decode=True,
    spatial_sharding=True,
)
_VAE_CAPABILITIES: dict[str, VaeOptimizationCapabilities] = {
    "MiniMaxH3Pipeline": VaeOptimizationCapabilities(
        tiled_decode=True,
        stacked_tiles=True,
        compilation=True,
        spatial_sharding=False,
        independent_process_group=True,
    ),
    "MiniMaxH3ModularPipeline": VaeOptimizationCapabilities(
        tiled_decode=True,
        stacked_tiles=True,
        compilation=True,
        spatial_sharding=False,
        independent_process_group=True,
    ),
    "WanPipeline": _WAN_VAE_CAPABILITIES,
    "WanImageToVideoPipeline": _WAN_VAE_CAPABILITIES,
    "WanVACEPipeline": _WAN_VAE_CAPABILITIES,
    "WanS2VPipeline": _WAN_VAE_CAPABILITIES,
    "WanT2VDMD2Pipeline": _WAN_VAE_CAPABILITIES,
    "WanI2VDMD2Pipeline": _WAN_VAE_CAPABILITIES,
}

_PROFILE_DEFAULTS: dict[str, tuple[VaeMode, VaeMode, bool]] = {
    "safe": ("false", "false", False),
    "optimized": ("auto", "auto", False),
    "diagnostic": ("false", "false", True),
}


def get_vae_optimization_capabilities(model_class_name: str) -> VaeOptimizationCapabilities:
    """Return explicitly declared VAE capabilities for an architecture."""

    return _VAE_CAPABILITIES.get(model_class_name, _NO_VAE_OPTIMIZATIONS)


def supports_independent_vae_process_group(model_class_name: str) -> bool:
    return get_vae_optimization_capabilities(model_class_name).independent_process_group


def _normalize_mode(value: str | bool | None, default: VaeMode, field_name: str) -> VaeMode:
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if value not in ("auto", "true", "false"):
        raise ValueError(f"{field_name} must be one of 'auto', 'true', or 'false', got {value!r}")
    return value


def resolve_vae_optimization(config: Any) -> ResolvedVaeOptimization:
    """Resolve profile defaults and reject unsupported explicit settings."""

    profile = getattr(config, "vae_optimization_profile", "safe")
    if profile == "student":
        raise ValueError(
            "vae_optimization_profile='student' requires a model-specific post-trained decoder artifact; "
            "MiniMax-H3 published decoder checkpoints are not drop-in replacements"
        )
    if profile not in _PROFILE_DEFAULTS:
        raise ValueError(
            f"vae_optimization_profile must be one of 'safe', 'optimized', 'diagnostic', or 'student', got {profile!r}"
        )

    default_stack, default_compile, diagnostics = _PROFILE_DEFAULTS[profile]
    stack_tiling = _normalize_mode(getattr(config, "vae_stack_tiling", None), default_stack, "vae_stack_tiling")
    compile_mode = _normalize_mode(getattr(config, "vae_compile", None), default_compile, "vae_compile")
    if profile == "safe" and (stack_tiling != "false" or compile_mode != "false"):
        raise ValueError(
            "vae_optimization_profile='safe' requires vae_stack_tiling=false and vae_compile=false; "
            "use profile='optimized' or 'diagnostic' for fast paths"
        )

    architecture = str(getattr(config, "model_class_name", "") or "")
    capabilities = get_vae_optimization_capabilities(architecture)
    if stack_tiling == "true" and not capabilities.stacked_tiles:
        raise ValueError(f"{architecture or 'this pipeline'} does not declare stacked-tile VAE support")
    if stack_tiling == "auto" and not capabilities.stacked_tiles:
        stack_tiling = "false"
    if compile_mode == "true" and not capabilities.compilation:
        raise ValueError(f"{architecture or 'this pipeline'} does not declare VAE compilation support")
    if compile_mode == "auto" and not capabilities.compilation:
        compile_mode = "false"

    parallel_mode = getattr(getattr(config, "parallel_config", None), "vae_parallel_mode", "tile")
    if parallel_mode.startswith("spatial_shard") and not capabilities.spatial_sharding:
        raise ValueError(f"{architecture or 'this pipeline'} does not declare VAE {parallel_mode!r} support")

    if stack_tiling != "false" and not capabilities.tiled_decode:
        raise ValueError(f"{architecture or 'this pipeline'} does not declare tiled VAE decode support")

    max_buckets = int(getattr(config, "vae_compile_max_shape_buckets", 4))
    if max_buckets < 1:
        raise ValueError(f"vae_compile_max_shape_buckets must be >= 1, got {max_buckets}")

    return ResolvedVaeOptimization(
        profile=profile,
        stack_tiling=stack_tiling,
        compile=compile_mode,
        diagnostics=diagnostics or bool(getattr(config, "enable_diffusion_pipeline_profiler", False)),
        compile_max_shape_buckets=max_buckets,
    )


def prepare_vae_optimization_config(config: Any) -> ResolvedVaeOptimization:
    """Resolve startup settings before model construction.

    Diagnostic profiling must be enabled before a pipeline constructor installs
    its regular stage profiler.  Stacked tiling also implies native tiling.
    """

    settings = resolve_vae_optimization(config)
    config.vae_stack_tiling = settings.stack_tiling
    config.vae_compile = settings.compile
    if settings.diagnostics:
        config.enable_diffusion_pipeline_profiler = True
    if settings.stack_tiling != "false":
        config.vae_use_tiling = True
    logger.info(
        "Resolved VAE optimization profile=%s stack_tiling=%s compile=%s diagnostics=%s",
        settings.profile,
        settings.stack_tiling,
        settings.compile,
        settings.diagnostics,
    )
    return settings


def _synchronize_for_diagnostics() -> None:
    if current_omni_platform.is_available():
        current_omni_platform.synchronize()


def _record_duration(pipeline: Any, metric: str, duration: float) -> None:
    lock = getattr(pipeline, "_profiler_lock", None)
    if lock is None:
        lock = Lock()
        pipeline._profiler_lock = lock
    if not hasattr(pipeline, "_stage_durations"):
        pipeline._stage_durations = {}
    with lock:
        pipeline._stage_durations[metric] = pipeline._stage_durations.get(metric, 0.0) + duration
    timing = {
        "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
        "metric": metric,
        "duration_s": duration,
    }
    logger.info("[VAE component timing] %s", json.dumps(timing, separators=(",", ":")))


def _install_timing_wrapper(pipeline: Any, owner: Any, method_name: str, metric: str) -> None:
    method = getattr(owner, method_name, None)
    if not callable(method) or getattr(method, "_vllm_omni_vae_timing", False):
        return

    @functools.wraps(method)
    def timed(*args: Any, **kwargs: Any) -> Any:
        _synchronize_for_diagnostics()
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            _synchronize_for_diagnostics()
            _record_duration(pipeline, metric, time.perf_counter() - started)

    timed._vllm_omni_vae_timing = True  # type: ignore[attr-defined]
    setattr(owner, method_name, timed)


def _install_tile_task_timing_wrapper(pipeline: Any, remote_model: Any) -> None:
    method = getattr(remote_model, "_run_tile_tasks", None)
    if not callable(method) or getattr(method, "_vllm_omni_vae_timing", False):
        return

    @functools.wraps(method)
    def timed(*args: Any, **kwargs: Any) -> Any:
        forward_fn = args[2] if len(args) > 2 else kwargs.get("forward_fn")
        function_name = str(getattr(forward_fn, "__name__", ""))
        metric = "video_vae.tile_encode" if "encode" in function_name else "video_vae.tile_decode"
        _synchronize_for_diagnostics()
        started = time.perf_counter()
        try:
            return method(*args, **kwargs)
        finally:
            _synchronize_for_diagnostics()
            _record_duration(pipeline, metric, time.perf_counter() - started)

    timed._vllm_omni_vae_timing = True  # type: ignore[attr-defined]
    remote_model._run_tile_tasks = timed


def _install_component_observability(pipeline: Any) -> None:
    video_vae = getattr(pipeline, "video_vae", None)
    audio_vae = getattr(pipeline, "audio_vae", None)
    if video_vae is not None:
        _install_timing_wrapper(pipeline, video_vae, "decode_latent", "video_vae.decode_latent")
        _install_timing_wrapper(pipeline, video_vae, "encode_image", "video_vae.encode_image")
        _install_timing_wrapper(pipeline, video_vae, "encode_video", "video_vae.encode_video")
        remote_model = getattr(video_vae, "model", None)
        if remote_model is not None:
            _install_timing_wrapper(pipeline, remote_model, "tiled_decode", "video_vae.tiled_decode")
            _install_timing_wrapper(pipeline, remote_model, "tiled_encode", "video_vae.tiled_encode")
            _install_tile_task_timing_wrapper(pipeline, remote_model)
            _install_timing_wrapper(
                pipeline,
                remote_model,
                "_all_gather_tiled_results",
                "video_vae.tile_communication",
            )
    if audio_vae is not None:
        _install_timing_wrapper(pipeline, audio_vae, "decode_latent", "audio_vae.decode_latent")
        _install_timing_wrapper(pipeline, audio_vae, "encode_waveform", "audio_vae.encode_waveform")


def _latent_sha256(latent: torch.Tensor) -> str:
    """Fingerprint canonical tensor metadata and bytes for diagnostics."""

    payload = latent.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    metadata = json.dumps(
        {"dtype": str(latent.dtype), "shape": list(latent.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _install_decode_metadata_wrapper(
    pipeline: Any,
    video_vae: Any,
    settings: ResolvedVaeOptimization,
) -> None:
    method = getattr(video_vae, "decode_latent", None)
    if not callable(method) or getattr(method, "_vllm_omni_vae_metadata", False):
        return
    calls = 0

    @functools.wraps(method)
    def observed(latent: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        cold = calls == 0
        calls += 1
        # Capture the exact decoder input before any model or fallback path has
        # a chance to mutate a caller-owned tensor.
        latent_sha256 = _latent_sha256(latent)
        result = method(latent, *args, **kwargs)
        controller = getattr(pipeline, "_vae_stack_tiling_controller", None)
        stack_state = getattr(controller, "last_decision", None) or {
            "stacked": False,
            "tile_count": None,
            "decision": "disabled",
            "fallback": False,
            "fallback_count": 0,
        }
        remote_model = getattr(video_vae, "model", None)
        ratio = int(getattr(remote_model, "vae_ratio", 1))
        diagnostics = {
            "component": "video_vae",
            "event": "decode",
            "latent_shape": list(latent.shape),
            "latent_sha256": latent_sha256,
            "resolution": [int(latent.shape[-2]) * ratio, int(latent.shape[-1]) * ratio],
            "frame_count": int(result.shape[-3]) if isinstance(result, torch.Tensor) and result.ndim == 5 else None,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "parallel_size": int(getattr(video_vae, "parallel_size", 1)),
            "profile": settings.profile,
            "stack_tiling_mode": settings.stack_tiling,
            "stacked": bool(stack_state["stacked"]),
            "tile_count": stack_state["tile_count"],
            "decision": stack_state["decision"],
            "fallback": bool(stack_state["fallback"]),
            "fallback_count": int(stack_state["fallback_count"]),
            "cold": cold,
        }
        pipeline._vae_last_diagnostics = diagnostics
        logger.info("[VAE diagnostics] %s", json.dumps(diagnostics, separators=(",", ":")))
        return result

    observed._vllm_omni_vae_metadata = True  # type: ignore[attr-defined]
    video_vae.decode_latent = observed


class _StackedTileController:
    """Select stacked tiles and retry only pre-collective allocation failures."""

    _MIN_FREE_BYTES = 2 * 1024**3

    def __init__(self, video_vae: Any, mode: VaeMode) -> None:
        self.video_vae = video_vae
        self.remote_model = video_vae.model
        self.mode = mode
        self.fallbacks = 0
        self._collective_entries = 0
        self._collective_marker_available = False
        self.last_decision: dict[str, Any] | None = None

    def _install_collective_marker(self) -> None:
        """Track whether a decode entered tile communication before failing.

        Replaying a decode after a collective has started is unsafe: peers may
        have completed a different number of collectives and NCCL may already
        have marked the process group unusable.  The marker lets allocation
        failures that happened strictly before communication retry locally,
        while collective and other runtime failures propagate to the worker's
        normal request/process cleanup path.
        """

        collective = getattr(self.remote_model, "_all_gather_tiled_results", None)
        if not callable(collective):
            return
        if getattr(collective, "_vllm_omni_vae_collective_marker", False):
            # A previously configured controller already marks this method.
            # Treat it as unavailable to this controller: its private counter
            # cannot prove where this request failed, so an OOM must propagate.
            return

        @functools.wraps(collective)
        def marked_collective(*args: Any, **kwargs: Any) -> Any:
            self._collective_entries += 1
            return collective(*args, **kwargs)

        marked_collective._vllm_omni_vae_collective_marker = True  # type: ignore[attr-defined]
        self.remote_model._all_gather_tiled_results = marked_collective
        self._collective_marker_available = True

    def _tile_count(self, latent: torch.Tensor) -> int:
        ratio = int(self.remote_model.vae_ratio)
        height = int(latent.shape[-2]) * ratio
        width = int(latent.shape[-1]) * ratio
        y_idx, _, _ = self.remote_model.split_tiles(height, is_decoder=True)
        x_idx, _, _ = self.remote_model.split_tiles(width, is_decoder=True)
        return len(y_idx) * len(x_idx)

    def _has_memory_headroom(self, latent: torch.Tensor, local_tiles: int) -> bool:
        if latent.device.type == "cpu" or not current_omni_platform.is_available():
            return True
        free_bytes = current_omni_platform.get_free_memory(latent.device)
        ratio = int(self.remote_model.vae_ratio)
        frames = max(1, int(latent.shape[-3]) * 4)
        output_bytes = (
            int(latent.shape[0])
            * 3
            * frames
            * int(latent.shape[-2])
            * ratio
            * int(latent.shape[-1])
            * ratio
            * latent.element_size()
        )
        estimated_working_set = output_bytes * max(4, local_tiles)
        return free_bytes >= max(self._MIN_FREE_BYTES, estimated_working_set)

    def should_stack(self, latent: torch.Tensor) -> tuple[bool, int, str]:
        if latent.ndim != 5 or any(int(size) <= 0 for size in latent.shape):
            if self.mode == "true":
                raise ValueError(f"stacked VAE tiling requires a non-empty 5D latent, got {tuple(latent.shape)}")
            return False, 0, "invalid_latent_shape"
        tile_count = self._tile_count(latent)
        parallel_size = max(1, int(getattr(self.video_vae, "parallel_size", 1)))
        local_tiles = math.ceil(tile_count / parallel_size)
        if local_tiles < 2:
            return False, tile_count, "fewer_than_two_local_tiles"
        if not self._has_memory_headroom(latent, local_tiles):
            return False, tile_count, "insufficient_memory_headroom"
        return True, tile_count, "validated"

    def wrap(self) -> None:
        self._install_collective_marker()
        original = self.video_vae.decode_latent

        @functools.wraps(original)
        def decode_with_stacked_tiles(latent: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            stack, tile_count, decision = self.should_stack(latent)
            previous = self.remote_model.stack_tiling
            self.remote_model.stack_tiling = stack
            fallback = False
            collective_entries_before = self._collective_entries
            try:
                try:
                    return original(latent, *args, **kwargs)
                except torch.OutOfMemoryError as exc:
                    parallel_size = max(1, int(getattr(self.video_vae, "parallel_size", 1)))
                    collective_started = self._collective_entries != collective_entries_before
                    can_prove_pre_collective = parallel_size == 1 or self._collective_marker_available
                    if not stack or collective_started or not can_prove_pre_collective:
                        raise
                    fallback = True
                    self.fallbacks += 1
                    self.remote_model.stack_tiling = False
                    if latent.device.type != "cpu":
                        torch.accelerator.empty_cache()
                    logger.warning(
                        "Stacked VAE tiles failed (%s); retrying this request with sequential tiles.",
                        exc,
                    )
                    return original(latent, *args, **kwargs)
            finally:
                self.remote_model.stack_tiling = previous
                self.last_decision = {
                    "stacked": stack,
                    "tile_count": tile_count,
                    "decision": decision,
                    "fallback": fallback,
                    "fallback_count": self.fallbacks,
                }
                logger.debug(
                    "VAE stacked-tile decision: %s",
                    json.dumps(self.last_decision, separators=(",", ":")),
                )

        decode_with_stacked_tiles._vllm_omni_stack_tiling = True  # type: ignore[attr-defined]
        self.video_vae.decode_latent = decode_with_stacked_tiles


def configure_vae_runtime(pipeline: nn.Module, config: Any) -> ResolvedVaeOptimization:
    """Apply validated runtime behavior after pipeline construction."""

    settings = resolve_vae_optimization(config)
    capabilities = get_vae_optimization_capabilities(str(getattr(config, "model_class_name", "") or ""))
    video_vae = getattr(pipeline, "video_vae", None)
    if settings.stack_tiling != "false":
        remote_model = getattr(video_vae, "model", None)
        required = ("stack_tiling", "split_tiles", "vae_ratio")
        missing = [name for name in required if remote_model is None or not hasattr(remote_model, name)]
        if missing:
            if settings.stack_tiling == "true":
                raise ValueError(f"video VAE is missing stacked-tile runtime attributes: {', '.join(missing)}")
            logger.info("VAE stacked tiling auto mode unavailable at runtime; using sequential tiles (%s).", missing)
        elif getattr(getattr(video_vae, "decode_latent", None), "_vllm_omni_stack_tiling", False):
            logger.debug("VAE stacked tiling is already configured; skipping duplicate wrapper installation.")
        else:
            controller = _StackedTileController(video_vae, settings.stack_tiling)
            controller.wrap()
            pipeline._vae_stack_tiling_controller = controller

    if settings.diagnostics and capabilities.tiled_decode:
        _install_component_observability(pipeline)
        if video_vae is not None:
            _install_decode_metadata_wrapper(pipeline, video_vae, settings)
    pipeline._vae_optimization_settings = settings
    return settings


def _shape_bucket(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    def describe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return (tuple(value.shape), str(value.dtype), value.device.type)
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(describe(item) for item in value))
        if isinstance(value, dict):
            return ("dict", tuple((key, describe(item)) for key, item in sorted(value.items())))
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return type(value).__qualname__

    return tuple(describe(value) for value in args) + tuple(
        (key, describe(value)) for key, value in sorted(kwargs.items())
    )


def _bounded_compiled_forward(
    forward: Any,
    *,
    module_name: str,
    max_buckets: int,
) -> Any:
    compiled: dict[tuple[Any, ...], Any] = {}
    failed: set[tuple[Any, ...]] = set()
    attempted: set[tuple[Any, ...]] = set()

    @functools.wraps(forward)
    def bounded(*args: Any, **kwargs: Any) -> Any:
        bucket = _shape_bucket(args, kwargs)
        if bucket in failed or (bucket not in attempted and len(attempted) >= max_buckets):
            return forward(*args, **kwargs)
        try:
            if bucket not in compiled:
                attempted.add(bucket)
                compiled[bucket] = torch.compile(forward, dynamic=False)
                logger.info("VAE compile created shape bucket %d/%d for %s", len(attempted), max_buckets, module_name)
            return compiled[bucket](*args, **kwargs)
        except Exception as exc:
            failed.add(bucket)
            compiled.pop(bucket, None)
            if current_omni_platform.is_available():
                torch.accelerator.empty_cache()
            logger.warning(
                "VAE compile failed for %s bucket=%s (%s); retrying eagerly and keeping this bucket eager.",
                module_name,
                bucket,
                exc,
            )
            return forward(*args, **kwargs)

    bounded._vllm_omni_vae_compiled = True  # type: ignore[attr-defined]
    bounded._vllm_omni_compiled_buckets = compiled  # type: ignore[attr-defined]
    bounded._vllm_omni_failed_buckets = failed  # type: ignore[attr-defined]
    bounded._vllm_omni_attempted_buckets = attempted  # type: ignore[attr-defined]
    return bounded


def setup_vae_compile(pipeline: nn.Module) -> int:
    """Compile stable video-decoder regions with bounded eager fallback."""

    settings = getattr(pipeline, "_vae_optimization_settings", None)
    if settings is None or settings.compile == "false":
        return 0
    if not current_omni_platform.supports_torch_inductor():
        message = f"platform {current_omni_platform.get_torch_device()} does not support VAE torch.compile"
        if settings.compile == "true":
            raise ValueError(message)
        logger.info("%s; using eager decode.", message)
        return 0
    video_vae = getattr(pipeline, "video_vae", None)
    decoder = getattr(getattr(video_vae, "model", None), "decoder", None)
    if decoder is None:
        if settings.compile == "true":
            raise ValueError("VAE compilation was requested but the video decoder is unavailable")
        logger.info("VAE compile auto mode unavailable because the video decoder is missing; using eager decode.")
        return 0

    count = 0
    for name, module in decoder.named_modules():
        if module.__class__.__name__ != "TransformerBlock":
            continue
        if getattr(module.forward, "_vllm_omni_vae_compiled", False):
            continue
        module.forward = _bounded_compiled_forward(
            module.forward,
            module_name=f"video_vae.decoder.{name}",
            max_buckets=settings.compile_max_shape_buckets,
        )
        count += 1
    if count == 0:
        message = "VAE compile found no stable TransformerBlock decoder regions"
        if settings.compile == "true":
            raise ValueError(message)
        logger.info("%s; using eager decode.", message)
        return 0
    logger.info(
        "VAE bounded regional compile installed for %d decoder block(s), max_shape_buckets=%d.",
        count,
        settings.compile_max_shape_buckets,
    )
    return count


def finalize_vae_stage_durations(stage_durations: dict[str, float]) -> dict[str, float]:
    """Add a non-negative tile-merge estimate from nested diagnostic timers."""

    result = dict(stage_durations)
    tiled = float(result.get("video_vae.tiled_decode", 0.0))
    tile_compute = float(result.get("video_vae.tile_decode", 0.0))
    communication = float(result.get("video_vae.tile_communication", 0.0))
    if tiled:
        result["video_vae.tile_merge"] = max(0.0, tiled - tile_compute - communication)
    return result


__all__ = [
    "ResolvedVaeOptimization",
    "VaeOptimizationCapabilities",
    "configure_vae_runtime",
    "finalize_vae_stage_durations",
    "get_vae_optimization_capabilities",
    "prepare_vae_optimization_config",
    "resolve_vae_optimization",
    "setup_vae_compile",
    "supports_independent_vae_process_group",
]
