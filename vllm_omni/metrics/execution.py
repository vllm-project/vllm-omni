# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request execution summary (``--enable-execution-summary``): one ``[OmniExecution]`` JSON line
per finished request with what each diffusion stage ran and the runtime it ran with."""

from __future__ import annotations

import dataclasses
import enum
import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.logger import init_logger

from vllm_omni.outputs.output_metadata import valid_frame_counts

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

SCHEMA_VERSION = 1
EXECUTION_METRICS_KEY = "execution"
_HF_SNAPSHOT = re.compile(r"models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})")
_PARALLEL_FIELDS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    "sequence_parallel_size",
    "ulysses_degree",
    "ring_degree",
    "cfg_parallel_size",
    "vae_patch_parallel_size",
    "use_hsdp",
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value if value is None or isinstance(value, (str, bool, int, float)) else str(value)


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def log_execution(event: str, payload: dict[str, Any]) -> None:
    logger.info("[OmniExecution] %s", json.dumps({"schema_version": SCHEMA_VERSION, "event": event, **payload}))


def num_tensor_frames(output: DiffusionOutput | None) -> int:
    size = _tensor_size(output)
    return size[2] if size else 0


def _tensor_size(output: DiffusionOutput | None) -> tuple[int, int, int] | None:
    media = getattr(output, "media", None)
    if media is None:
        return None
    from vllm_omni.diffusion.media import VideoTensorLayout

    shape = media.video.tensor.shape
    if media.video.spec.layout is VideoTensorLayout.BCTHW:
        _, _, frames, height, width = shape
    else:
        _, frames, height, width, _ = shape
    return int(width), int(height), int(frames)


def _video_metadata(output: Any) -> Mapping[str, Any]:
    metadata = getattr(output, "multimodal_output", None) or {}
    video = (metadata.get("metadata") or {}).get("video") if isinstance(metadata, Mapping) else None
    return video if isinstance(video, Mapping) else {}


def _output_fps(output: DiffusionOutput, formatted_outputs: Sequence[Any] | None) -> float | None:
    metadata = getattr(output.media, "metadata", None) or {}
    fps = (metadata.get("video") or {}).get("fps", metadata.get("fps"))
    return next((f for f in [fps, *(_video_metadata(o).get("fps") for o in formatted_outputs or ())] if f), None)


def _output_size(formatted_outputs: Sequence[Any] | None) -> tuple[int | None, int | None, int | None]:
    """(width, height, num_frames) of post-processed outputs: images, encoder frame counts or the MP4 itself."""
    for out in formatted_outputs or ():
        images = getattr(out, "images", None) or []
        for image in images:
            if isinstance(getattr(image, "size", None), tuple):
                return int(image.size[0]), int(image.size[1]), None
        counts = _video_metadata(out).get("frame_counts")
        if valid_frame_counts(counts, len(images)):
            return None, None, counts[0]
        if images and isinstance(images[0], (bytes, bytearray)):
            from vllm_omni.diffusion.utils.media_utils import count_mp4_frames

            return None, None, count_mp4_frames(bytes(images[0]))
    return None, None, None


def diffusion_facts(
    request: OmniDiffusionRequest,
    output: DiffusionOutput,
    formatted_outputs: Sequence[Any] | None = None,
    od_config: OmniDiffusionConfig | None = None,
    num_streamed_frames: int | None = None,
) -> dict[str, Any]:
    """What a finished request ran; values the pipeline generated or resolved win over the request's."""
    sp = request.sampling_params
    reported = output.execution_facts or {}
    out_w, out_h, out_frames = _tensor_size(output) or _output_size(formatted_outputs)
    extra = sp.extra_args or {}
    shifts = (reported.get("flow_shift"), extra.get("flow_shift"), getattr(od_config, "flow_shift", None))
    flow_shift = next((v for v in shifts if v is not None), None)
    cfg_ran = sp.guidance_scale_provided and reported.get("cfg") is not False
    lora = sp.lora_request if isinstance(sp.lora_request, list) else [sp.lora_request]
    static_lora = getattr(od_config, "lora_path", None)
    lora_names = [str(getattr(r, "lora_name", r)) for r in lora if r is not None]
    lora_names += [str(p) for p in (static_lora if isinstance(static_lora, list) else [static_lora]) if p]
    facts = {
        "width": out_w or reported.get("width") or sp.width,
        "height": out_h or reported.get("height") or sp.height,
        "num_frames": num_streamed_frames or out_frames or sp.num_frames,
        "fps": sp.fps if sp.fps is not None else _output_fps(output, formatted_outputs),
        "seed": sp.seed,
        "num_inference_steps": sp.num_inference_steps,
        "completed_steps": output.num_completed_steps,
        "cached_steps": output.num_cached_steps,
        # None when left to the model default or the pipeline runs without CFG.
        "guidance_scale": sp.guidance_scale if cfg_ran else None,
        "flow_shift": flow_shift,
        "num_outputs_per_prompt": sp.num_outputs_per_prompt,
        "lora": lora_names or None,
    }
    for key, value in reported.items():
        facts.setdefault(key, value)
    return facts


def execution_facts(
    request: OmniDiffusionRequest,
    output: DiffusionOutput,
    formatted_outputs: Sequence[Any] | None,
    od_config: OmniDiffusionConfig,
    engine_runtime: dict[str, Any] | None,
    num_streamed_frames: int | None = None,
) -> dict[str, Any]:
    """The JSON-ready facts a diffusion stage attaches to a finished request's metrics."""
    diffusion = diffusion_facts(request, output, formatted_outputs, od_config, num_streamed_frames)
    runtime = {**(engine_runtime or {}), **(output.execution_runtime or {})}
    return _jsonable({"diffusion": diffusion, "runtime": runtime or None})


def _model_identity(model: Any) -> tuple[Any, str | None]:
    """(hub repo id, snapshot commit) when ``model`` (or a symlink in it) resolves into a hub snapshot."""
    if not isinstance(model, str) or not os.path.isdir(model):
        return model, None
    for path in [model, *(os.path.join(model, n) for n in sorted(os.listdir(model))[:16])]:
        match = _HF_SNAPSHOT.search(os.path.realpath(path))
        if match:
            return f"{match[1]}/{match[2]}", match[3]
    return model, None


def engine_runtime(od_config: OmniDiffusionConfig) -> dict[str, Any] | None:
    """Runtime facts the engine knows from its config; computed once at startup."""
    try:
        import vllm

        import vllm_omni

        model, snapshot = _model_identity(od_config.model)
        quant = od_config.quantization_config
        parallel = od_config.parallel_config
        cache_backend = od_config.cache_backend
        return {
            "vllm_omni_version": vllm_omni.__version__,
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "model": model,
            "model_revision": od_config.revision or snapshot,
            "model_class_name": od_config.model_class_name,
            "configured_dtype": _dtype_name(od_config.dtype),
            "quantization": getattr(quant, "get_name", lambda: quant)(),
            "cache_backend": cache_backend,
            "cache_config": od_config.cache_config if cache_backend not in (None, "none") else None,
            "parallel_config": {f: getattr(parallel, f, None) for f in _PARALLEL_FIELDS} if parallel else None,
            "max_num_seqs": od_config.max_num_seqs,
        }
    except Exception:
        logger.warning("Failed to collect the execution summary runtime", exc_info=True)
        return None


def _weight_dtypes(pipeline: torch.nn.Module) -> dict[str, dict[str, int]]:
    """Weight elements per dtype of each component; persistent buffers count, as packed quantized weights are."""
    seen: set[int] = set()
    components: dict[str, dict[str, int]] = {}
    for name, child in pipeline.named_children():
        counts: dict[str, int] = {}
        for module in child.modules():
            buffers = [b for n, b in module._buffers.items() if n not in module._non_persistent_buffers_set]
            for tensor in [*module._parameters.values(), *buffers]:
                if tensor is not None and id(tensor) not in seen:
                    seen.add(id(tensor))
                    dtype = _dtype_name(tensor.dtype)
                    counts[dtype] = counts.get(dtype, 0) + tensor.numel()
        if counts:
            components[name] = counts
    return components


def worker_runtime(pipeline: torch.nn.Module | None) -> dict[str, Any] | None:
    """Runtime facts the worker knows after loading: device, attention backends, weight dtypes."""
    if pipeline is None:
        return None
    try:
        from vllm_omni.diffusion.attention.layer import Attention
        from vllm_omni.platforms import current_omni_platform

        attention: dict[str, dict[str, int]] = {}
        for module in pipeline.modules():
            if isinstance(module, Attention):
                backend = getattr(module, "attn_backend", None)
                name = backend.get_name() if backend else type(getattr(module, "attention", None)).__name__
                role = attention.setdefault(str(getattr(module, "role", None)), {})
                role[name] = role.get(name, 0) + 1
        facts = {
            "device": current_omni_platform.get_device_name(),
            "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            "attention": attention,
            "weight_dtypes": _weight_dtypes(pipeline),
        }
        # Facts only the pipeline knows (e.g. compute precision, checkpoint-shipped quantization).
        if callable(report := getattr(pipeline, "execution_runtime_facts", None)):
            facts.update(report())
        return facts
    except Exception:
        logger.warning("Failed to collect the execution summary runtime", exc_info=True)
        return None


def stage_entry(stage_stats: Any, execution: Mapping[str, Any] | None) -> dict[str, Any]:
    entry = {
        "stage_id": getattr(stage_stats, "stage_id", None),
        "replica_id": getattr(stage_stats, "replica_id", None),
        "output_type": getattr(stage_stats, "final_output_type", None),
        "batch_size": getattr(stage_stats, "batch_size", None),
    }
    if execution is not None:
        entry.update(execution)
    else:
        entry["num_tokens_in"] = getattr(stage_stats, "num_tokens_in", None)
        entry["num_tokens_out"] = getattr(stage_stats, "num_tokens_out", None)
    if getattr(stage_stats, "audio_duration_s", 0):
        entry["audio"] = {"duration_s": stage_stats.audio_duration_s, "sample_rate": stage_stats.audio_sample_rate}
    return entry
