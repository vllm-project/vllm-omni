# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execution summary: what each request actually ran (``--enable-execution-summary``).

``[OmniTiming]`` reports how long a request took; ``[OmniExecution]`` reports
what it ran, so a benchmark harness can check a run against its declared
workload. Each line is one JSON object with ``schema_version`` and ``event``:

- ``request``: once per finished request, next to ``[OmniTiming]``; common
  fields per stage plus ``diffusion`` or ``audio`` fields;
- ``config``: once per diffusion stage at startup;
- ``attention``: once per diffusion stage, the backends its layers resolved.
"""

from __future__ import annotations

import dataclasses
import enum
import json
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

    Size comes from what the pipeline generated where the output allows it;
    ``size_source`` names the source: ``tensor`` (typed video tensor),
    ``image``, ``metadata`` (frame count reported with video the worker
    already encoded; width and height from the request) or ``request``.
    ``num_scheduled_steps`` and ``num_completed_steps`` are the denoising
    steps the model runner counted (``None`` without a progress signal).
    ``guidance_scale`` and ``flow_shift`` are ``None`` when neither the request
    nor the server set them and the model default applied.
    """
    sp = request.sampling_params
    width, height, num_frames, size_source = sp.width, sp.height, sp.num_frames, "request"
    tensor_size = _tensor_size(output)
    if tensor_size is not None:
        (width, height, num_frames), size_source = tensor_size, "tensor"
    elif (output_size := _output_size(formatted_outputs)) is not None:
        out_width, out_height, out_frames, size_source = output_size
        width = out_width if out_width is not None else width
        height = out_height if out_height is not None else height
        num_frames = out_frames if out_frames is not None else num_frames
    if num_streamed_frames:
        # Streamed requests return their video in chunks; report the whole clip.
        num_frames = num_streamed_frames
    flow_shift = (sp.extra_args or {}).get("flow_shift")
    if flow_shift is None and od_config is not None:
        flow_shift = getattr(od_config, "flow_shift", None)
    return {
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "size_source": size_source,
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


# ----------------------------------------------------------------------------- startup events


def log_config(od_config: OmniDiffusionConfig) -> None:
    """``config`` event: what a diffusion stage runs with, once at engine startup."""
    try:
        import vllm

        import vllm_omni

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
        log_execution(
            "config",
            {
                "vllm_omni_version": vllm_omni.__version__,
                "vllm_version": vllm.__version__,
                "torch_version": torch.__version__,
                "model": getattr(od_config, "model", None),
                "model_revision": getattr(od_config, "revision", None),
                "model_class_name": getattr(od_config, "model_class_name", None),
                "dtype": str(getattr(od_config, "dtype", None)),
                "quantization": _jsonable(quantization),
                "attention_backend": getattr(
                    getattr(od_config.diffusion_attention_config, "default", None), "backend", None
                ),
                "cache_backend": getattr(od_config, "cache_backend", None),
                "parallel_config": (
                    {name: getattr(parallel, name, None) for name in parallel_fields} if parallel is not None else None
                ),
                "enforce_eager": getattr(od_config, "enforce_eager", None),
                "step_execution": getattr(od_config, "step_execution", None),
                "max_num_seqs": getattr(od_config, "max_num_seqs", None),
            },
        )
    except Exception:
        logger.warning("Failed to log the execution config", exc_info=True)


def log_attention(pipeline: torch.nn.Module | None) -> None:
    """``attention`` event: the backend each attention layer resolved, per role (first rank only)."""
    if pipeline is None or (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
        return
    try:
        from vllm_omni.diffusion.attention.layer import Attention
        from vllm_omni.platforms import current_omni_platform

        backends: dict[str, dict[str, int]] = {}
        num_layers = 0
        for module in pipeline.modules():
            if not isinstance(module, Attention):
                continue
            num_layers += 1
            backend_cls = getattr(module, "attn_backend", None)
            # Model-owned kernels (``custom_attention``) bypass backend selection.
            if backend_cls is not None:
                name = backend_cls.get_name()
            else:
                name = type(getattr(module, "attention", None)).__name__
            counts = backends.setdefault(str(getattr(module, "role", None)), {})
            counts[name] = counts.get(name, 0) + 1
        log_execution(
            "attention",
            {
                "device": current_omni_platform.get_device_name(),
                "world_size": torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1,
                "num_layers": num_layers,
                "backends": backends,
            },
        )
    except Exception:
        logger.warning("Failed to log the attention backends", exc_info=True)


# ----------------------------------------------------------------------------- request event


def stage_entry(stage_stats: Any, diffusion: Mapping[str, Any] | None) -> dict[str, Any]:
    """One stage of a ``request`` event: common fields plus its modality fields."""
    entry: dict[str, Any] = {
        "stage_id": getattr(stage_stats, "stage_id", None),
        "replica_id": getattr(stage_stats, "replica_id", None),
        "output_type": getattr(stage_stats, "final_output_type", None),
        "batch_size": getattr(stage_stats, "batch_size", None),
        "num_tokens_in": getattr(stage_stats, "num_tokens_in", None),
        "num_tokens_out": getattr(stage_stats, "num_tokens_out", None),
    }
    if diffusion is not None:
        entry["diffusion"] = dict(diffusion)
    if getattr(stage_stats, "audio_duration_s", 0):
        entry["audio"] = {
            "duration_s": getattr(stage_stats, "audio_duration_s", None),
            "sample_rate": getattr(stage_stats, "audio_sample_rate", None),
        }
    return entry
