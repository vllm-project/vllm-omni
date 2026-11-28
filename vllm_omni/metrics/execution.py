# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execution summary: what each request actually ran (``--enable-execution-summary``).

``[OmniTiming]`` reports how long a request took; ``[OmniExecution]`` reports
what it ran, so a benchmark harness can check a run against its declared
workload. Each line is one JSON object with ``schema_version`` and ``event``:

- ``request``: once per finished request, next to ``[OmniTiming]``. Each
  diffusion stage reports what the request ran (``diffusion``) and how the
  stage ran it (``runtime``: versions, model revision, dtype, quantization,
  resolved attention backends, cache and parallel config), so a single line
  verifies a request without correlating startup logs.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.outputs.output_metadata import valid_frame_counts

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

SCHEMA_VERSION = 1
# Key under which a diffusion stage attaches its facts to the output metrics.
EXECUTION_METRICS_KEY = "execution"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def log_execution(event: str, payload: dict[str, Any]) -> None:
    logger.info("[OmniExecution] %s", json.dumps({"schema_version": SCHEMA_VERSION, "event": event, **payload}))


# ----------------------------------------------------------------------------- diffusion facts


def _tensor_size(output: DiffusionOutput | None) -> tuple[int, int, int] | None:
    """(width, height, num_frames) of a typed video output, before frame interpolation."""
    media = getattr(output, "media", None)
    if media is None:
        return None
    from vllm_omni.diffusion.media import VideoTensorLayout

    shape = tuple(media.video.tensor.shape)
    if media.video.spec.layout is VideoTensorLayout.BCTHW:
        _, _, num_frames, height, width = shape
    else:
        _, num_frames, height, width, _ = shape
    return int(width), int(height), int(num_frames)


def num_tensor_frames(output: DiffusionOutput | None) -> int:
    size = _tensor_size(output)
    return size[2] if size is not None else 0


def _video_metadata(request_output: Any) -> Mapping[str, Any]:
    """The ``metadata["video"]`` a model postprocessor attached to its multimodal output."""
    multimodal_output = getattr(request_output, "multimodal_output", None)
    metadata = multimodal_output.get("metadata") if isinstance(multimodal_output, Mapping) else None
    video = metadata.get("video") if isinstance(metadata, Mapping) else None
    return video if isinstance(video, Mapping) else {}


def _output_size(formatted_outputs: Sequence[Any] | None) -> tuple[int | None, int | None, int | None, str] | None:
    """(width, height, num_frames, size_source) read from post-processed outputs."""
    for request_output in formatted_outputs or ():
        images = getattr(request_output, "images", None) or []
        for image in images:
            size = getattr(image, "size", None)  # PIL images; encoded video bytes have none
            if isinstance(size, tuple) and len(size) == 2:
                return int(size[0]), int(size[1]), None, "image"
        frame_counts = _video_metadata(request_output).get("frame_counts")
        if valid_frame_counts(frame_counts, len(images)):
            return None, None, frame_counts[0], "metadata"
    return None


def _output_fps(output: DiffusionOutput | None, formatted_outputs: Sequence[Any] | None) -> float | None:
    metadata = getattr(getattr(output, "media", None), "metadata", None) or {}
    video = metadata.get("video")
    fps = video.get("fps") if isinstance(video, Mapping) else metadata.get("fps")
    for request_output in formatted_outputs or ():
        if fps is not None:
            break
        fps = _video_metadata(request_output).get("fps")
    return fps


def _lora(lora_request: Any) -> list[str] | None:
    if lora_request is None:
        return None
    requests = lora_request if isinstance(lora_request, list) else [lora_request]
    return [str(getattr(item, "lora_name", item)) for item in requests]


def diffusion_facts(
    request: OmniDiffusionRequest,
    output: DiffusionOutput,
    formatted_outputs: Sequence[Any] | None = None,
    od_config: OmniDiffusionConfig | None = None,
    num_streamed_frames: int | None = None,
) -> dict[str, Any]:
    """What one finished diffusion request ran.

    Values come from what the pipeline generated or resolved where available,
    so requests that leave parameters to the model defaults still report the
    values that ran. ``size_source`` names where ``width``/``height`` come
    from: ``tensor`` (typed video tensor), ``image``, ``pipeline`` (the size
    the pipeline reported, e.g. for video it encoded in the worker) or
    ``request``; ``frames_source`` does the same for ``num_frames`` (``tensor``,
    ``metadata`` for the encoder's frame counts, ``stream`` or ``request``).
    ``num_scheduled_steps`` and ``num_completed_steps`` are the denoising
    steps the model runner counted (``None`` without a progress signal).
    ``guidance_scale`` is ``None`` unless the request set it.
    """
    sp = request.sampling_params
    reported = getattr(output, "execution_facts", None) or {}
    width, height, size_source = sp.width, sp.height, "request"
    num_frames, frames_source = sp.num_frames, "request"
    output_size = _output_size(formatted_outputs)
    tensor_size = _tensor_size(output)
    if tensor_size is not None:
        width, height, num_frames = tensor_size
        size_source = frames_source = "tensor"
    else:
        if output_size is not None and output_size[0] is not None:
            width, height, size_source = output_size[0], output_size[1], output_size[3]
        elif reported.get("width") is not None and reported.get("height") is not None:
            width, height, size_source = reported["width"], reported["height"], "pipeline"
        if output_size is not None and output_size[2] is not None:
            num_frames, frames_source = output_size[2], output_size[3]
    if num_streamed_frames:
        # Streamed requests return their video in chunks; report the whole clip.
        num_frames, frames_source = num_streamed_frames, "stream"
    flow_shift = reported.get("flow_shift")
    if flow_shift is None:
        flow_shift = (sp.extra_args or {}).get("flow_shift")
    if flow_shift is None and od_config is not None:
        flow_shift = getattr(od_config, "flow_shift", None)
    facts = {
        "width": width,
        "height": height,
        "size_source": size_source,
        "num_frames": num_frames,
        "frames_source": frames_source,
        "fps": sp.fps if sp.fps is not None else _output_fps(output, formatted_outputs),
        "seed": sp.seed,
        "num_inference_steps": sp.num_inference_steps,
        "num_scheduled_steps": output.num_scheduled_steps,
        "num_completed_steps": output.num_completed_steps,
        "guidance_scale": sp.guidance_scale if sp.guidance_scale_provided else None,
        "flow_shift": flow_shift,
        "num_outputs_per_prompt": sp.num_outputs_per_prompt,
        "attention_schedule": _jsonable(sp.attention_schedule),
        "lora": _lora(sp.lora_request),
    }
    # Other values the pipeline resolved (e.g. ``audio_flow_shift``).
    for key, value in reported.items():
        facts.setdefault(key, _jsonable(value))
    return facts


# ----------------------------------------------------------------------------- runtime facts

# ``<cache>/models--<org>--<name>/snapshots/<commit>`` of a Hugging Face hub download.
_HF_SNAPSHOT = re.compile(r"models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})")


def _model_identity(model: Any) -> tuple[Any, str | None]:
    """(hub repo id, snapshot commit) when ``model`` resolves into a hub cache snapshot.

    Serving wrappers often pass a local directory of symlinks into the snapshot,
    so the directory and its entries are resolved.
    """
    if not isinstance(model, str) or not os.path.isdir(model):
        return model, None
    candidates = [model]
    try:
        candidates += [os.path.join(model, name) for name in sorted(os.listdir(model))[:16]]
    except OSError:
        pass
    for path in candidates:
        match = _HF_SNAPSHOT.search(os.path.realpath(path))
        if match is not None:
            return f"{match.group(1)}/{match.group(2)}", match.group(3)
    return model, None


def engine_runtime(od_config: OmniDiffusionConfig) -> dict[str, Any] | None:
    """Configuration part of a diffusion stage's ``runtime`` facts; computed once at startup."""
    try:
        return _engine_runtime(od_config)
    except Exception:
        logger.warning("Failed to collect the execution runtime config", exc_info=True)
        return None


def _engine_runtime(od_config: OmniDiffusionConfig) -> dict[str, Any]:
    import vllm

    import vllm_omni

    model, snapshot = _model_identity(getattr(od_config, "model", None))
    quantization = getattr(od_config, "quantization_config", None)
    if hasattr(quantization, "get_name"):
        quantization = quantization.get_name()
    parallel = getattr(od_config, "parallel_config", None)
    parallel_fields = (
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
    return {
        "vllm_omni_version": vllm_omni.__version__,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "model": model,
        "model_revision": getattr(od_config, "revision", None) or snapshot,
        "model_class_name": getattr(od_config, "model_class_name", None),
        "dtype": str(getattr(od_config, "dtype", None)).removeprefix("torch."),
        "quantization": _jsonable(quantization),
        "cache_backend": getattr(od_config, "cache_backend", None),
        "parallel_config": (
            {name: getattr(parallel, name, None) for name in parallel_fields} if parallel is not None else None
        ),
        "max_num_seqs": getattr(od_config, "max_num_seqs", None),
    }


def worker_runtime(pipeline: torch.nn.Module | None) -> dict[str, Any] | None:
    """Worker part of a diffusion stage's ``runtime`` facts: device and resolved attention backends.

    Computed once after the model loads; ``attention`` counts the backend each
    attention layer resolved, per role.
    """
    if pipeline is None:
        return None
    try:
        from vllm_omni.diffusion.attention.layer import Attention
        from vllm_omni.platforms import current_omni_platform

        backends: dict[str, dict[str, int]] = {}
        for module in pipeline.modules():
            if not isinstance(module, Attention):
                continue
            backend_cls = getattr(module, "attn_backend", None)
            # Model-owned kernels (``custom_attention``) bypass backend selection.
            if backend_cls is not None:
                name = backend_cls.get_name()
            else:
                name = type(getattr(module, "attention", None)).__name__
            counts = backends.setdefault(str(getattr(module, "role", None)), {})
            counts[name] = counts.get(name, 0) + 1
        return {
            "device": current_omni_platform.get_device_name(),
            "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
            "attention": backends,
        }
    except Exception:
        logger.warning("Failed to collect the attention backends", exc_info=True)
        return None


# ----------------------------------------------------------------------------- request event


def stage_entry(stage_stats: Any, execution: Mapping[str, Any] | None) -> dict[str, Any]:
    """One stage of a ``request`` event: common fields plus its modality facts.

    A diffusion stage adds the ``diffusion`` and ``runtime`` facts its engine
    attached to the output; token counts only apply to the other stages.
    """
    entry: dict[str, Any] = {
        "stage_id": getattr(stage_stats, "stage_id", None),
        "replica_id": getattr(stage_stats, "replica_id", None),
        "output_type": getattr(stage_stats, "final_output_type", None),
        "batch_size": getattr(stage_stats, "batch_size", None),
    }
    if execution is not None:
        entry.update({key: _jsonable(value) for key, value in execution.items()})
    else:
        entry["num_tokens_in"] = getattr(stage_stats, "num_tokens_in", None)
        entry["num_tokens_out"] = getattr(stage_stats, "num_tokens_out", None)
    if getattr(stage_stats, "audio_duration_s", 0):
        entry["audio"] = {
            "duration_s": getattr(stage_stats, "audio_duration_s", None),
            "sample_rate": getattr(stage_stats, "audio_sample_rate", None),
        }
    return entry
