# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM-Omni pipeline for MiniMax H3 FL2VA and Ref2VA partitions."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.cache.cachedit import (
    CacheDiTBackend,
    RequestScopedCacheDiTRuntime,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_world_group,
    init_world_group,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.offloader import (
    BoundedAllocatorCache,
    OffloadPlan,
    apply_sequential_offload,
    remove_sequential_offload,
    sequential_offload_component,
)
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError, client_error_from_metadata
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    resolve_minimax_h3_partition,
)
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MiniMaxH3TextConditioning,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    load_minimax_h3_images as _load_images,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    minimax_h3_multi_image_presentation,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
    minimax_h3_text_only_ids,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    resolve_minimax_h3_aspect_ratio as _resolve_minimax_h3_aspect_ratio,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    resolve_minimax_h3_output_canvas as _resolve_output_canvas,
)
from vllm_omni.model_executor.models.minimax_h3.preprocessing import (
    resolve_minimax_h3_reference_image_shape as _reference_image_shape,
)
from vllm_omni.model_executor.models.minimax_h3.reference_video import (
    MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY,
    deserialize_prepared_reference_videos,
    load_audio_file,
    load_video_audio,
    load_video_frames,
    prepare_reference_videos,
    sample_reference_video_frames,
    validate_reference_audio_files,
    validate_reference_audio_waveforms,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import (
    resolve_component_quant_config as _resolve_component_quant_config,
)
from vllm_omni.quantization.component_config import (
    resolve_encoder_quant_config as _resolve_encoder_quant_config,
)

from .batched_packing import minimax_h3_batched_forward_kwargs
from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
    minimax_h3_prepare_denoise_rows,
    minimax_h3_publish_denoise_progress,
)
from .encoder import MiniMaxH3Qwen3VLEncoder
from .lora import load_minimax_h3_turbo_lora
from .minimax_h3_transformer import (
    MiniMaxH3Attention,
    MiniMaxH3DiTModel,
    _attention_isolates_packed_requests,
)
from .npu.lora import (
    MINIMAX_H3_NATIVE_INFERENCE_STEPS,
    load_minimax_h3_native_lora,
)
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .quality_policy import MINIMAX_H3_GENERIC_CACHE_KEY, MiniMaxH3QualityPolicy
from .scheduling_minimax_h3_euler_ancestral import (
    minimax_h3_euler_eta0_step,
    minimax_h3_rf_v_to_x0,
)
from .time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
    minimax_h3_time_shift_sigmas,
)
from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    from vllm_omni.lora.request import LoRARequest

MINIMAX_H3_FPS = 24
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
MINIMAX_H3_OUTPUT_SHORT_EDGE = 768
MINIMAX_H3_OUTPUT_MAX_PIXELS = 768 * 1344
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32
MINIMAX_H3_SUPPORTED_ASPECT_RATIOS = {
    "21:9": 21.0 / 9.0,
    "16:9": 16.0 / 9.0,
    "4:3": 4.0 / 3.0,
    "1:1": 1.0,
    "3:4": 3.0 / 4.0,
    "9:16": 9.0 / 16.0,
}
MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES = 30 * 1024 * 1024
MINIMAX_H3_REFERENCE_IMAGE_FORMATS = frozenset({"jpeg", "png", "webp", "heic", "heif"})
MINIMAX_H3_MIN_OUTPUT_SECONDS = 4.0
MINIMAX_H3_MAX_OUTPUT_SECONDS = 15.0
MINIMAX_H3_TURBO_SIGMA_POINTS = 5
MINIMAX_H3_TURBO_VIDEO_SHIFT = 6.0
MINIMAX_H3_TURBO_AUDIO_SHIFT = 3.0
MINIMAX_H3_DOWNLOAD_PATTERNS = [
    "FL2VA/**",
    "Ref2VA/model_index.json",
    "Ref2VA/transformer/**",
]
MINIMAX_H3_TASK_DOWNLOAD_PATTERNS = {
    "fl2va": ["FL2VA/**"],
    "ref2va": ["Ref2VA/**"],
}
MINIMAX_H3_DIFFUSION_DOWNLOAD_PATTERNS = {
    "fl2va": [
        "FL2VA/model_index.json",
        "FL2VA/transformer/**",
        "FL2VA/video_vae/**",
        "FL2VA/audio_vae/**",
    ],
    "ref2va": [
        "Ref2VA/model_index.json",
        "Ref2VA/transformer/**",
        "Ref2VA/video_vae/**",
        "Ref2VA/audio_vae/**",
    ],
    "combined": [
        "FL2VA/model_index.json",
        "FL2VA/transformer/**",
        "FL2VA/video_vae/**",
        "FL2VA/audio_vae/**",
        "Ref2VA/model_index.json",
        "Ref2VA/transformer/**",
    ],
}


def _resolve_minimax_h3_text_encoder_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    resolved = _resolve_component_quant_config(quant_config, "text_encoder")
    return _resolve_encoder_quant_config(resolved)


def _minimax_h3_partition_for_task(
    task_type: str | None,
    model: str | None = None,
) -> str:
    return resolve_minimax_h3_partition(model or "", task_type, auto_partition="combined")


def _resolve_minimax_h3_model_root(
    model: str,
    revision: str | None,
    partition: str,
    *,
    load_text_encoder: bool,
) -> Path:
    path = Path(model)
    if path.is_dir():
        if path.name in {"FL2VA", "Ref2VA"}:
            return path.parent
        return path
    if load_text_encoder:
        allow_patterns = (
            MINIMAX_H3_DOWNLOAD_PATTERNS if partition == "combined" else MINIMAX_H3_TASK_DOWNLOAD_PATTERNS[partition]
        )
    else:
        allow_patterns = MINIMAX_H3_DIFFUSION_DOWNLOAD_PATTERNS[partition]
    return Path(
        download_weights_from_hf_specific(
            model_name_or_path=model,
            cache_dir=None,
            allow_patterns=allow_patterns,
            revision=revision,
            require_all=True,
        )
    )


# Keys of ``_prepare_request_inputs`` that feed ``diffuse`` / ``_build_denoise_inputs``.
_MINIMAX_H3_DENOISE_INPUT_KEYS = (
    "task",
    "text_embeddings",
    "text_tags",
    "seed",
    "latent_t",
    "latent_h",
    "latent_w",
    "audio_t",
    "num_frames",
    "num_steps",
    "video_shift",
    "audio_shift",
    "base_schedule",
    "visual_condition",
    "visual_condition_shape",
    "audio_condition",
    "ref_audio_t",
    "ref_blocks",
    "visual_condition_shapes",
    "audio_condition_lengths",
    "keyframe_frame_indices",
)

# ``StepRequestState.extra`` keys owned by the step-execution path.
_STEP_BRANCH = "minimax_h3_branch"
_STEP_AUDIO_ROWS = "minimax_h3_audio_rows"
_STEP_AUDIO_NOISE_PRED = "minimax_h3_audio_noise_pred"
_STEP_SIGMAS_VIDEO = "minimax_h3_sigmas_video"
_STEP_SIGMAS_AUDIO = "minimax_h3_sigmas_audio"
_STEP_COND_ANCHOR = "minimax_h3_cond_anchor"
_STEP_AUDIO_ANCHOR = "minimax_h3_audio_anchor"
_STEP_SHAPE = "minimax_h3_shape"
_STEP_TRANSFORMER = "minimax_h3_transformer"


def _minimax_h3_step_schedule(state: StepRequestState) -> dict[str, float]:
    """Return the sigma/timestep values this request needs for its current step.

    Mirrors the per-iteration arithmetic of ``minimax_h3_denoise_loop`` so step
    mode and request mode advance identically.
    """
    step = int(state.step_index)
    sigmas_video = state.extra[_STEP_SIGMAS_VIDEO]
    sigmas_audio = state.extra[_STEP_SIGMAS_AUDIO]
    sigma_video = float(sigmas_video[step])
    sigma_audio = float(sigmas_audio[step])
    t_video = 1.0 - sigma_video
    t_audio = 1.0 - sigma_audio
    return {
        "sigma_video": sigma_video,
        "sigma_video_next": float(sigmas_video[step + 1]),
        "sigma_audio": sigma_audio,
        "sigma_audio_next": float(sigmas_audio[step + 1]),
        "t_video": t_video,
        "t_audio": t_audio,
        "imgvid_cond_timestep": max(t_video, MINIMAX_H3_IMGVID_COND_TIMESTEP),
        "audio_ref_cond_timestep": max(t_audio, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
    }


def _read_base_schedule(release: Mapping[str, Any]) -> DMD2SigmaSchedule | None:
    """Read a partition's distilled schedule. An absent key means legacy uniform."""
    return DMD2SigmaSchedule.from_metadata(release)


def resolve_minimax_h3_diffusion_model_path(
    model: str,
    revision: str | None,
    task_type: str | None,
) -> str:
    """Resolve a repository root or Hub ID to its startup partition."""
    partition = (
        "combined"
        if str(task_type or "").lower() == "combined"
        else resolve_minimax_h3_partition(model, task_type, auto_partition="fl2va")
    )
    model_root = _resolve_minimax_h3_model_root(
        model,
        revision,
        partition,
        load_text_encoder=False,
    )
    if partition == "combined":
        return str(model_root)
    subdir = "Ref2VA" if partition == "ref2va" else "FL2VA"
    return str(model_root / subdir)


def _minimax_h3_post_process(output, output_type: str = "np"):
    """Convert the joint video/audio output without capturing worker state.

    The callable crosses the multiprocessing result queue, so it must remain a
    module-level function that the standard pickle module can resolve.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return output
    video, audio = output
    if output_type == "latent":
        return output
    if output_type == "np":
        video = video.detach().float().cpu().permute(0, 2, 3, 4, 1).clamp(0, 1).numpy()
        audio = audio.detach().float().cpu().numpy()
        video = [sample for sample in video]
    return {
        "video": video,
        "audio": audio,
        "audio_sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE,
        "fps": MINIMAX_H3_FPS,
    }


def _register_dlo_component_cache(cache: BoundedAllocatorCache, *components: Any) -> None:
    for component in components:
        if component is not None:
            component.set_omni_component_cache(cache)


def get_minimax_h3_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config
    return _minimax_h3_post_process


def _align_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _load_image(value: Any) -> Image.Image:
    images = _load_images(value)
    if len(images) != 1:
        raise OmniClientError(f"MiniMax H3 expected one image, got {len(images)}")
    return images[0]


def _load_audio(value: Any) -> tuple[torch.Tensor, int]:
    if isinstance(value, (list, tuple)) and not (len(value) == 2 and isinstance(value[1], (int, np.integer))):
        audios = _load_audios(value)
        if len(audios) != 1:
            raise OmniClientError(f"MiniMax H3 expected one audio, got {len(audios)}")
        return audios[0]
    if isinstance(value, (str, os.PathLike)):
        return load_audio_file(str(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        waveform, sample_rate = value
        waveform = torch.as_tensor(waveform).float()
        return waveform, int(sample_rate)
    if isinstance(value, dict):
        waveform = value.get("waveform", value.get("array"))
        sample_rate = value.get("sample_rate", value.get("sampling_rate"))
        if waveform is not None and sample_rate is not None:
            return torch.as_tensor(waveform).float(), int(sample_rate)
    raise OmniClientError("MiniMax H3 audio input must be a path, (waveform, sample_rate), or a waveform mapping")


def _load_audios(value: Any) -> list[tuple[torch.Tensor, int]]:
    if isinstance(value, (list, tuple)) and not (len(value) == 2 and isinstance(value[1], (int, np.integer))):
        if not value:
            raise OmniClientError("MiniMax H3 audio input must not be empty")
        return [_load_audio(item) for item in value]
    return [_load_audio(value)]


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, bool):
        raise OmniClientError(f"{name} must be an integer or a list of integers")
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
        if not result:
            raise OmniClientError(f"{name} must not be empty")
        if any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) for item in result):
            raise OmniClientError(f"{name} must contain only integers")
        return [int(item) for item in result]
    raise OmniClientError(f"{name} must be an integer or a list of integers")


def _resolve_fl2va_keyframe_indices(extra: Mapping[str, Any], image_count: int) -> list[int]:
    target = extra.get("target")
    target = target if isinstance(target, Mapping) else {}
    raw = extra.get("frame_indices", extra.get("frame_index"))
    if raw is None:
        raw = target.get("frame_indices", target.get("frame_index"))
    if raw is None:
        raw_indices = [0] if image_count == 1 else [0, -1]
    else:
        raw_indices = _as_int_list(raw, name="frame_indices")
    if len(raw_indices) != image_count:
        raise OmniClientError(
            f"MiniMax H3 FL2VA requires one frame index per image: got {raw_indices!r} for {image_count} image(s)"
        )
    if tuple(raw_indices) not in ((0,), (-1,), (0, -1)):
        raise OmniClientError("MiniMax H3 FL2VA frame_indices must be [0], [-1], or [0, -1]")
    return raw_indices


def _reuse_prepared_reference_videos(
    prepared: list[dict[str, Any]] | None,
    *,
    expected_count: int,
) -> list[dict[str, Any]] | None:
    if prepared is None:
        return None
    if len(prepared) != expected_count:
        raise OmniClientError("MiniMax H3 prepared-reference-video count does not match the request")
    for item in prepared:
        if not os.path.isfile(item["prepared_path"]):
            raise OmniClientError(f"MiniMax H3 prepared reference video is unavailable: {item['prepared_path']}")
    return prepared


def _validate_ref2va_reference_counts(
    image_count: int,
    video_count: int,
    audio_count: int,
) -> None:
    """Validate the official Ref2VA reference-count contract."""
    if image_count < 0 or video_count < 0 or audio_count < 0:
        raise OmniClientError("MiniMax H3 reference counts must be non-negative")
    if image_count + video_count == 0:
        raise OmniClientError("ref2va requires at least one image or video reference")
    if image_count > 9:
        raise OmniClientError("ref2va accepts at most 9 image references")
    if video_count > 3:
        raise OmniClientError("ref2va accepts at most 3 video references")
    if audio_count > 3:
        raise OmniClientError("ref2va accepts at most 3 standalone audio references")
    if image_count + video_count + audio_count > 12:
        raise OmniClientError("ref2va accepts at most 12 total references")


def _resolve_minimax_h3_num_outputs(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise OmniClientError("MiniMax H3 num_outputs_per_prompt must be an integer in [1, 10]")
    value = int(value)
    if not 1 <= value <= 10:
        raise OmniClientError(f"MiniMax H3 num_outputs_per_prompt must be in [1, 10], got {value}")
    return value


def _minimax_h3_output_seeds(seed: int, num_outputs: int) -> list[int]:
    return [int(seed) + output_index for output_index in range(int(num_outputs))]


def _validate_reference_image(image: Image.Image) -> None:
    width, height = image.size
    if min(width, height) < 256 or max(width, height) > 5760:
        raise OmniClientError(
            f"MiniMax H3 reference image dimensions must be in [256, 5760] pixels, got {width}x{height}"
        )
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"MiniMax H3 reference image aspect ratio must be in [0.4, 2.5], got {width}x{height}")


def _dit_rank_world() -> tuple[Any, int, int]:
    if not dist.is_initialized():
        return None, 0, 1
    group = get_world_group().device_group
    return group, dist.get_rank(group), dist.get_world_size(group)


def _broadcast_rank0_exception(exc: Exception | None) -> None:
    """Synchronize a rank-0-only exception across every DiT rank.

    H3 reference-video preparation runs only on rank 0; the other DiT ranks
    return ``None`` without touching disk. When rank 0 raises inside that
    path it exits :meth:`prepare_encode` before reaching the downstream
    ``dist.broadcast`` calls, and non-zero ranks then hang on those
    collectives forever. Every rank calls this helper right after the
    rank-0-only work, before any subsequent collective, so all ranks either
    raise the same error together or all continue.
    """
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if exc is not None:
            raise exc
        return
    if rank == 0:
        if exc is None:
            payload: list[Any] = [None]
        else:
            payload = [
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "status_code": getattr(exc, "status_code", None),
                    "error_type": getattr(exc, "error_type", None),
                }
            ]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0, group=group)
    info = payload[0]
    if info is None:
        return
    if rank == 0:
        assert exc is not None
        raise exc
    # Rebuild a matching client-facing error on non-zero ranks so the runner's
    # per-request try/except records the same 4xx status as rank 0. The exact
    # subclass need not survive the wire; the message and status suffice.
    status_code = info.get("status_code")
    error_type = info.get("error_type")
    message = f"[rank 0] {info['type']}: {info['message']}"
    if status_code is not None:
        raise client_error_from_metadata(
            message,
            status_code=int(status_code),
            error_type=error_type,
        )
    raise RuntimeError(message)


def _broadcast_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if tensor is None:
            raise ValueError("source tensor is required for single-rank execution")
        return tensor.to(device=device, dtype=dtype)

    shape = torch.zeros(5, dtype=torch.long, device=device)
    if rank == 0:
        if tensor is None:
            raise ValueError("rank 0 must provide a tensor to broadcast")
        shape[0] = tensor.ndim
        shape[1 : tensor.ndim + 1] = torch.tensor(
            tensor.shape,
            device=device,
        )
    dist.broadcast(shape, src=0, group=group)
    ndim = int(shape[0].item())
    tensor_shape = tuple(int(v) for v in shape[1 : ndim + 1].tolist())
    if rank == 0:
        output = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        output = torch.empty(tensor_shape, device=device, dtype=dtype)
    dist.broadcast(output, src=0, group=group)
    return output


class _SingleRankEncoderGroup:
    """Lightweight encoder group for ``text_encoder_tp_size == 1``.

    Avoids creating a distributed ``GroupCoordinator`` with a single-member
    rank set, which would assert on every other DiT rank that is not part of
    the group.  The pipeline and encoder only use the attributes below, and
    all ``world_size == 1`` code paths short-circuit before any collective.
    """

    world_size: int = 1
    ranks: list[int] = [0]

    def __init__(self, rank: int) -> None:
        self.rank_in_group = 0 if rank == 0 else -1
        self.device_group = None


class MiniMaxH3Pipeline(
    nn.Module,
    DenoiseProgressMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
):
    """CFG-distilled joint video/audio generation for MiniMax H3."""

    supports_step_execution: ClassVar[bool] = True

    _dit_modules: ClassVar[list[str]] = ["transformer", "transformers_ref"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        offload_submodules={"token_refiner": "blocks"},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
        on_demand_component_paths=frozenset({"text_encoder", "video_vae", "audio_vae"}),
    )
    _PROFILER_TARGETS: ClassVar[list[str]] = [
        "_prepare_reference_videos",
        "encode_prompt",
        "_encode_visual_conditions",
        "_encode_reference_audio_conditions",
        "diffuse",
        "decode",
        "prepare_encode",
        "denoise_step",
        "post_decode",
    ]
    dummy_run_num_frames: ClassVar[int] = 0
    # Only distilled releases pin a schedule, so the default keeps the legacy
    # uniform path available to partially constructed pipelines.
    _base_schedule_by_partition: ClassVar[Mapping[str, DMD2SigmaSchedule | None]] = {}

    def _load_diffusion_lora_adapter(
        self,
        *,
        lora_request: LoRARequest,
        lora_path: str | Path,
        dtype: torch.dtype,
    ) -> tuple[LoRAModel, PEFTHelper] | None:
        # A cache eviction may be followed by a different adapter reusing the
        # same client-supplied ID. Every real load replaces the classification.
        self._turbo_lora_adapter_ids.discard(lora_request.lora_int_id)
        self._native_lora_adapter_ids.discard(lora_request.lora_int_id)
        self._lora_sigma_schedules.pop(lora_request.lora_int_id, None)
        od_config = getattr(self, "od_config", None)
        offload_modes = []
        if getattr(od_config, "enable_cpu_offload", False):
            offload_modes.append("model-level CPU offload (--enable-cpu-offload)")
        if getattr(od_config, "enable_layerwise_offload", False):
            offload_modes.append("layerwise offload (--enable-layerwise-offload)")
        loaded = load_minimax_h3_turbo_lora(
            partition=self.partition,
            lora_request=lora_request,
            lora_path=lora_path,
            dtype=dtype,
            unsupported_offload_mode=" or ".join(offload_modes) or None,
        )
        if loaded is not None:
            self._turbo_lora_adapter_ids.add(lora_request.lora_int_id)
            return loaded

        # Selection is by the artifact's safetensors ``key_format``, not by the
        # running platform: the native loader is checkpoint-format parsing with
        # no ``torch_npu`` dependency, so it needs no ``current_omni_platform``
        # dispatch and binds the same adapter on NPU, CUDA and CPU.
        native_loaded = load_minimax_h3_native_lora(
            partition=self.partition,
            lora_request=lora_request,
            lora_path=lora_path,
            dtype=dtype,
            unsupported_offload_mode=" or ".join(offload_modes) or None,
        )
        if native_loaded is not None:
            lora_model, peft_helper, sigma_schedule = native_loaded
            self._native_lora_adapter_ids.add(lora_request.lora_int_id)
            self._lora_sigma_schedules[lora_request.lora_int_id] = sigma_schedule
            return lora_model, peft_helper
        return None

    def _validate_diffusion_lora_binding(
        self,
        *,
        lora_model: LoRAModel,
        bound_lora_names: frozenset[str],
    ) -> None:
        if lora_model.id in self._turbo_lora_adapter_ids:
            missing = sorted(set(lora_model.loras) - bound_lora_names)
            if missing:
                raise ValueError(
                    "MiniMax-H3 Turbo LoRA binding is incomplete: "
                    f"bound={len(bound_lora_names)}/{len(lora_model.loras)}, missing={missing[:5]}"
                )
            return
        if lora_model.id not in self._native_lora_adapter_ids:
            return
        missing = sorted(set(lora_model.loras) - bound_lora_names)
        if missing:
            raise ValueError(
                "MiniMax-H3 native LoRA binding is incomplete: "
                f"bound={len(bound_lora_names)}/{len(lora_model.loras)}, missing={missing[:5]}"
            )

    def _has_active_turbo_lora(self, sampling: Any) -> bool:
        lora_request = sampling.lora_request
        return (
            lora_request is not None
            and not math.isclose(0.0, float(sampling.lora_scale))
            and lora_request.lora_int_id in self._turbo_lora_adapter_ids
        )

    def _has_active_native_lora(self, sampling: Any) -> bool:
        lora_request = sampling.lora_request
        return (
            lora_request is not None
            and not math.isclose(0.0, float(sampling.lora_scale))
            and lora_request.lora_int_id in self._native_lora_adapter_ids
        )

    def _validate_native_sampling(self, sampling: Any, *, task: str) -> None:
        if task != "t2va":
            raise OmniClientError("MiniMax-H3 native LoRA supports T2VA requests only")
        # Derive the expected count from the adapter's own schedule so the
        # message can never disagree with the schedule the denoise loop runs.
        schedule = self._lora_sigma_schedules.get(sampling.lora_request.lora_int_id)
        expected_steps = MINIMAX_H3_NATIVE_INFERENCE_STEPS if schedule is None else schedule.num_inference_steps
        # Only request mode can take the count from the adapter schedule: step
        # mode admits the request in ``StepScheduler``, which reads
        # ``num_inference_steps`` off it before any pipeline hook runs. Reject
        # omission there rather than advertise a contract that would either fail
        # admission or disagree with the denoise loop.
        od_config = getattr(self, "od_config", None)
        omission_allowed = not getattr(od_config, "step_execution", False)
        or_omitted = " or omitted" if omission_allowed else ""
        sigma_steps = sampling.num_inference_steps
        if sigma_steps is None:
            if omission_allowed:
                return
            raise OmniClientError(
                f"MiniMax-H3 native LoRA requires an explicit num_inference_steps={expected_steps} "
                "under step execution, because the step scheduler derives the total step count from "
                "the request before the adapter schedule is known"
            )
        if int(sigma_steps) == expected_steps + 1:
            raise OmniClientError(
                "MiniMax-H3 native LoRA uses the distilled interval-count contract; "
                f"num_inference_steps must be {expected_steps}{or_omitted}, not {expected_steps + 1}"
            )
        if int(sigma_steps) != expected_steps:
            raise OmniClientError(
                f"MiniMax-H3 native LoRA requires num_inference_steps={expected_steps} "
                f"(one denoiser evaluation per sigma interval){or_omitted}"
            )

    def _sigma_schedule_for_request(self, sampling: Any, task: str) -> DMD2SigmaSchedule | None:
        lora_request = sampling.lora_request
        if (
            lora_request is not None
            and not math.isclose(0.0, float(sampling.lora_scale))
            and lora_request.lora_int_id in self._lora_sigma_schedules
        ):
            adapter_schedule = self._lora_sigma_schedules[lora_request.lora_int_id]
            checkpoint_schedule = self._base_schedule_for_task(task)
            if checkpoint_schedule is not None:
                raise OmniClientError(
                    "MiniMax-H3 native LoRA cannot be activated on a checkpoint that already pins base_schedule"
                )
            return adapter_schedule
        return self._base_schedule_for_task(task)

    def _validate_turbo_sampling(self, sampling: Any) -> None:
        extra = sampling.extra_args or {}
        sigma_points = sampling.num_inference_steps
        if sigma_points != MINIMAX_H3_TURBO_SIGMA_POINTS:
            raise OmniClientError(
                "MiniMax-H3 Turbo requires num_inference_steps=5 (five sigma points produce four denoiser evaluations)"
            )
        try:
            video_shift = float(extra.get("flow_shift", self.default_video_shift))
        except (TypeError, ValueError) as exc:
            raise OmniClientError(f"MiniMax-H3 Turbo requires flow_shift={MINIMAX_H3_TURBO_VIDEO_SHIFT:g}") from exc
        if not math.isclose(video_shift, MINIMAX_H3_TURBO_VIDEO_SHIFT):
            raise OmniClientError(f"MiniMax-H3 Turbo requires flow_shift={MINIMAX_H3_TURBO_VIDEO_SHIFT:g}")
        try:
            audio_shift = float(extra.get("audio_flow_shift", self.default_audio_shift))
        except (TypeError, ValueError) as exc:
            raise OmniClientError(
                f"MiniMax-H3 Turbo requires audio_flow_shift={MINIMAX_H3_TURBO_AUDIO_SHIFT:g}"
            ) from exc
        if not math.isclose(audio_shift, MINIMAX_H3_TURBO_AUDIO_SHIFT):
            raise OmniClientError(f"MiniMax-H3 Turbo requires audio_flow_shift={MINIMAX_H3_TURBO_AUDIO_SHIFT:g}")

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        """Adopt runner-installed generic Cache-DiT for request transitions."""

        self._cache_dit_runtime.adopt(
            backend,
            installation_key=MINIMAX_H3_GENERIC_CACHE_KEY,
        )

    def is_cache_dit_enabled(self) -> bool:
        """Return the request-scoped Cache-DiT installation state."""

        return self._cache_dit_runtime.is_enabled

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        if int(self.parallel_config.cfg_parallel_size) != 1:
            raise ValueError("MiniMax-H3 is CFG-distilled and has no negative branch; cfg_parallel_size must be 1")
        self.device = get_local_device()
        self.load_text_encoder = od_config.model_loaded["text_encoder"]
        self.partition = _minimax_h3_partition_for_task(
            getattr(od_config, "task_type", None),
            str(od_config.model),
        )
        self._turbo_lora_adapter_ids: set[int] = set()
        self._native_lora_adapter_ids: set[int] = set()
        self._lora_sigma_schedules: dict[int, DMD2SigmaSchedule] = {}
        model_root = _resolve_minimax_h3_model_root(
            str(od_config.model),
            od_config.revision,
            self.partition,
            load_text_encoder=self.load_text_encoder,
        )
        model_path = model_root / ("Ref2VA" if self.partition == "ref2va" else "FL2VA")
        model_index = json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
        release = model_index.get("_minimax_h3") or {}
        partition = str(release.get("partition", "")).lower()
        expected_partition = "ref2va" if self.partition == "ref2va" else "fl2va"
        if partition != expected_partition:
            raise ValueError(f"invalid MiniMax-H3 {expected_partition} partition at {model_path}")

        supported_tasks = {str(task).lower() for task in release.get("tasks", [])}
        if not supported_tasks:
            supported_tasks = {"ref2va"} if partition == "ref2va" else {"t2va", "fl2va"}
        ref2va_model_path = None
        if self.partition == "combined":
            ref2va_model_path = model_root / "Ref2VA"
            ref2va_index_path = ref2va_model_path / "model_index.json"
            if not ref2va_index_path.is_file():
                raise ValueError(f"Ref2VA partition not found at {ref2va_model_path}")
            ref2va_index = json.loads(ref2va_index_path.read_text(encoding="utf-8"))
            ref2va_release = ref2va_index.get("_minimax_h3") or {}
            if str(ref2va_release.get("partition", "")).lower() != "ref2va":
                raise ValueError(f"invalid MiniMax-H3 ref2va partition at {ref2va_model_path}")
            supported_tasks.update(str(task).lower() for task in ref2va_release.get("tasks", ["ref2va"]))

        self.supported_tasks = frozenset(supported_tasks)
        shifts = release.get("sigma_shift_scales") or {}
        self.default_video_shift = float(shifts.get("video", 12.0))
        self.default_audio_shift = float(shifts.get("audio", 3.0))
        # Distilled releases pin their own few-step rectified-flow positions; the
        # uniform schedule derived from num_inference_steps does not match what
        # such a checkpoint was trained on. Each partition carries its own
        # contract, so a distilled FL2VA must not drag Ref2VA onto its schedule.
        self._base_schedule_by_partition = {expected_partition: _read_base_schedule(release)}
        if ref2va_model_path is not None:
            self._base_schedule_by_partition["ref2va"] = _read_base_schedule(ref2va_release)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(model_path),
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        self._dit_modules = ["transformer"]
        if ref2va_model_path is not None:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=str(ref2va_model_path),
                    subfolder="transformer",
                    revision=od_config.revision,
                    prefix="transformers_ref.",
                    fall_back_to_pt=False,
                )
            )
            self._dit_modules.append("transformers_ref")
        transformer_quant_config = _resolve_component_quant_config(
            od_config.quantization_config,
            "transformer",
        )
        self.transformer = MiniMaxH3DiTModel(
            od_config,
            quant_config=transformer_quant_config,
        )
        if ref2va_model_path is not None:
            self.transformers_ref = MiniMaxH3DiTModel(
                od_config,
                quant_config=transformer_quant_config,
            )

        if self.load_text_encoder:
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                str(model_path),
                subfolder="tokenizer",
                local_files_only=os.path.isdir(model_path),
            )
            self.processor = Qwen3VLProcessor.from_pretrained(
                str(model_path),
                subfolder="processor",
                local_files_only=os.path.isdir(model_path),
            )
        else:
            self.tokenizer = None
            self.processor = None

        _, rank, dit_world = _dit_rank_world()
        self._dit_rank = rank
        if self.load_text_encoder:
            text_encoder_tp_size = int(getattr(self.parallel_config, "text_encoder_tp_size", 1))
            if text_encoder_tp_size < 1:
                raise ValueError(f"text_encoder_tp_size must be >= 1, got {text_encoder_tp_size}")
            if text_encoder_tp_size > dit_world:
                raise ValueError(
                    f"text_encoder_tp_size must not exceed the DiT group size ({dit_world}), got {text_encoder_tp_size}"
                )
            # The Qwen3-VL text model uses 64 attention heads / 8 KV heads.
            if 64 % text_encoder_tp_size or 8 % text_encoder_tp_size:
                raise ValueError(
                    "text_encoder_tp_size must divide both Qwen3-VL "
                    f"num_attention_heads (64) and num_key_value_heads (8), "
                    f"got {text_encoder_tp_size}"
                )
            self.text_encoder_tp_size = text_encoder_tp_size
            self.text_encoder_group = self._build_text_encoder_group(text_encoder_tp_size)
            self.text_encoder = MiniMaxH3Qwen3VLEncoder(
                os.path.join(model_path, "text_encoder"),
                device=self.device,
                load_model=rank < text_encoder_tp_size,
                encoder_group=self.text_encoder_group,
                quant_config=_resolve_minimax_h3_text_encoder_quant_config(od_config.quantization_config),
            )
            if rank < text_encoder_tp_size:
                self.weights_sources.append(
                    DiffusersPipelineLoader.ComponentSource(
                        model_or_path=str(model_path),
                        subfolder="text_encoder",
                        revision=od_config.revision,
                        prefix="text_encoder.",
                        fall_back_to_pt=False,
                    )
                )
        else:
            self.text_encoder_tp_size = 0
            self.text_encoder_group = None
            self.text_encoder = None
            self._encoder_modules = []
        stage_components = bool(
            od_config.enable_layerwise_offload or getattr(od_config, "enable_distributed_layerwise_offload", False)
        )
        component_load_device = torch.device("cpu") if stage_components else self.device
        self.video_vae = MiniMaxH3VideoVAE(
            os.path.join(model_path, "video_vae"),
            device=self.device,
            load_device=component_load_device,
        )
        self.audio_vae = MiniMaxH3AudioVAE(
            os.path.join(model_path, "audio_vae"),
            device=self.device,
            load_device=component_load_device,
        )
        # Registry-side VAE patch-parallel discovery uses ``pipeline.vae``.
        self.vae = self.video_vae

        self._dlo_component_cache = None
        if getattr(od_config, "enable_distributed_layerwise_offload", False):
            self._dlo_component_cache = BoundedAllocatorCache(self.device)
            _register_dlo_component_cache(
                self._dlo_component_cache,
                self.text_encoder,
                self.video_vae,
                self.audio_vae,
            )

        self._quality_policy = MiniMaxH3QualityPolicy(od_config)
        self._cache_dit_runtime = RequestScopedCacheDiTRuntime(self)

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=(od_config.enable_diffusion_pipeline_profiler)
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def source_prefix(item: tuple[str, torch.Tensor]) -> str:
            name, _ = item
            prefix = name.partition(".")[0] + "."
            if prefix in {"transformer.", "transformers_ref.", "text_encoder."}:
                return prefix
            raise ValueError(f"unexpected MiniMax-H3 weight {name!r}")

        loaded_with_prefix: set[str] = set()
        loaded_prefixes: set[str] = set()
        for prefix, grouped_weights in groupby(weights, key=source_prefix):
            if prefix in loaded_prefixes:
                raise ValueError(f"MiniMax-H3 weight source {prefix!r} is not contiguous")
            loaded_prefixes.add(prefix)
            component = getattr(self, prefix.removesuffix("."))
            loaded = component.load_weights((name[len(prefix) :], tensor) for name, tensor in grouped_weights)
            if prefix != "text_encoder.":
                component.post_load_weights()
            loaded_with_prefix.update(prefix + name for name in loaded)
        # Both VAEs load eagerly in ``__init__`` rather than through
        # ``weights_sources``. The text encoder uses the shared component
        # loader so online quantization and offload processing follow the same
        # path as the DiT.
        for component_name in ("video_vae", "audio_vae"):
            component = getattr(self, component_name)
            if component is None:
                continue
            loaded_with_prefix.update(f"{component_name}.{name}" for name, _ in component.named_parameters())
        return loaded_with_prefix

    def _transformer_for_task(self, task: str) -> MiniMaxH3DiTModel:
        if task == "ref2va" and hasattr(self, "transformers_ref"):
            return self.transformers_ref
        return self.transformer

    def _base_schedule_for_task(self, task: str) -> DMD2SigmaSchedule | None:
        """Return the distilled schedule of the partition that serves ``task``."""
        partition = "ref2va" if task == "ref2va" else "fl2va"
        return self._base_schedule_by_partition.get(partition)

    def _resolve_task(
        self,
        requested: str | None,
        multi_modal_data: dict[str, Any],
        *,
        has_turbo_lora: bool = False,
        has_native_lora: bool = False,
    ) -> str:
        if requested is None:
            # A Ref2VA-only startup has no FL2VA transformer; preserve its
            # historical implicit default even for image-only references.
            if self.partition == "ref2va":
                requested = "ref2va"
            elif multi_modal_data.get("video") is not None or multi_modal_data.get("audio") is not None:
                requested = "ref2va"
            elif multi_modal_data.get("image") is not None:
                requested = "fl2va"
            else:
                requested = "t2va"
        task = str(requested).lower()
        if task not in self.supported_tasks:
            raise OmniClientError(
                f"checkpoint partition {self.partition!r} supports {sorted(self.supported_tasks)}, got task={task!r}"
            )
        if task == "ref2va" and has_turbo_lora:
            raise OmniClientError("MiniMax-H3 Turbo LoRA supports T2VA/FL2VA requests only")
        if has_native_lora and task != "t2va":
            raise OmniClientError("MiniMax-H3 native LoRA supports T2VA requests only")
        return task

    def _resolve_shape(
        self,
        task: str,
        sampling: Any,
        image: Image.Image | None,
    ) -> tuple[int, int, int, int, int]:
        fps = int(sampling.fps or MINIMAX_H3_FPS)
        if fps != MINIMAX_H3_FPS:
            raise OmniClientError(f"MiniMax H3 output fps is fixed at {MINIMAX_H3_FPS}")
        extra = sampling.extra_args or {}
        target = extra.get("target")
        if target is not None and not isinstance(target, Mapping):
            raise OmniClientError("MiniMax H3 extra_args['target'] must be an object")
        target = target if isinstance(target, Mapping) else {}
        duration = target.get("duration_seconds", extra.get("duration_seconds", extra.get("duration")))
        if duration is not None:
            if isinstance(duration, bool):
                raise OmniClientError(f"MiniMax H3 output duration must be in [4, 15] seconds, got {duration!r}")
            try:
                duration = float(duration)
            except (TypeError, ValueError) as exc:
                raise OmniClientError(
                    f"MiniMax H3 output duration must be in [4, 15] seconds, got {duration!r}"
                ) from exc
            if (
                not math.isfinite(duration)
                or not MINIMAX_H3_MIN_OUTPUT_SECONDS <= duration <= MINIMAX_H3_MAX_OUTPUT_SECONDS
            ):
                raise OmniClientError(f"MiniMax H3 output duration must be in [4, 15] seconds, got {duration}")
            requested_frames = int(round(duration * fps))
        elif int(sampling.num_frames or 1) > 1:
            requested_frames = int(sampling.num_frames)
        else:
            requested_frames = 124 if task == "ref2va" else 209
            duration = requested_frames / fps
        if not MINIMAX_H3_MIN_OUTPUT_SECONDS <= requested_frames / fps <= MINIMAX_H3_MAX_OUTPUT_SECONDS:
            raise OmniClientError(
                f"MiniMax H3 output duration must be in [4, 15] seconds, got {requested_frames / fps:.3f}"
            )
        num_frames = minimax_h3_align_frame_count(requested_frames)

        height = sampling.height
        width = sampling.width
        aspect_ratio = target.get("aspect_ratio", extra.get("aspect_ratio"))
        raw_short_edge = target.get("short_edge", extra.get("short_edge", MINIMAX_H3_OUTPUT_SHORT_EDGE))
        if isinstance(raw_short_edge, bool) or not isinstance(raw_short_edge, (int, np.integer)):
            raise OmniClientError(
                f"MiniMax H3 target.short_edge must be {MINIMAX_H3_OUTPUT_SHORT_EDGE}, got {raw_short_edge!r}"
            )
        short_edge = int(raw_short_edge)

        aspect_ratio = _resolve_minimax_h3_aspect_ratio(
            task,
            aspect_ratio,
            image,
        )
        if not 0.25 <= aspect_ratio <= 4.0:
            raise OmniClientError(f"MiniMax H3 canvas aspect ratio must be in [1:4, 4:1], got {aspect_ratio}")

        if height is None or width is None:
            height, width = _resolve_output_canvas(aspect_ratio, short_edge)
        height = int(height) // 32 * 32
        width = int(width) // 32 * 32
        if min(height, width) <= 0:
            raise OmniClientError(f"invalid MiniMax H3 canvas {width}x{height}")
        if width > 4 * height or height > 4 * width:
            raise OmniClientError("MiniMax H3 canvas aspect ratio must be in [1:4, 4:1]")

        latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)
        audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / fps)
        return height, width, num_frames, latent_t, audio_t

    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image: Image.Image | None = None,
        images: list[Image.Image] | None = None,
        prepared_videos: list[dict[str, Any]] | None = None,
        condition_labels: list[tuple[str, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, rank, _ = _dit_rank_world()
        hidden = None
        tags = None
        ids = None
        vision_kwargs: dict[str, torch.Tensor] = {}
        images = list(images) if images is not None else ([image] if image is not None else [])
        if rank == 0:
            if task == "t2va":
                ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
                tags = torch.ones(ids.shape[0], dtype=torch.long)
                vision_kwargs = {}
            else:
                image_token_counts: list[int] = []
                if images:
                    vision = self.processor.image_processor(
                        images=images,
                        return_tensors="pt",
                    )
                    image_grid = vision["image_grid_thw"]
                    merge = int(self.processor.image_processor.merge_size) ** 2
                    image_token_counts = [int(grid.prod().item()) // merge for grid in image_grid]
                    vision_kwargs.update(
                        {
                            "pixel_values": vision["pixel_values"],
                            "image_grid_thw": image_grid,
                        }
                    )

                video_block_counts: list[list[int]] = []
                video_block_timestamps: list[list[float]] = []
                if prepared_videos:
                    videos = []
                    sampled_videos = []
                    for index, item in enumerate(prepared_videos):
                        sampled = sample_reference_video_frames(item["prepared_path"])
                        videos.append(np.stack(sampled["frames"]))
                        sampled_videos.append(sampled)
                    vision = self.processor.video_processor(
                        videos=videos,
                        do_sample_frames=False,
                        return_tensors="pt",
                    )
                    video_grid = vision["video_grid_thw"]
                    merge = int(self.processor.image_processor.merge_size) ** 2
                    for index, sampled in enumerate(sampled_videos):
                        blocks = int(video_grid[index, 0])
                        per_block = int(video_grid[index, 1]) * int(video_grid[index, 2]) // merge
                        timestamps = sampled["block_timestamps"]
                        if len(timestamps) != blocks:
                            raise ValueError(
                                f"video block count mismatch: processor={blocks}, timestamps={len(timestamps)}"
                            )
                        video_block_counts.append([per_block] * blocks)
                        video_block_timestamps.append(timestamps)
                    vision_kwargs.update(
                        {
                            "pixel_values_videos": vision["pixel_values_videos"],
                            "video_grid_thw": video_grid,
                        }
                    )

                if not images and not prepared_videos:
                    raise OmniClientError(f"{task} requires an image or video condition")
                if condition_labels is None:
                    condition_labels = []
                    for image_index in range(1, len(images) + 1):
                        condition_labels.append(("image", image_index))
                    audio_index = 0
                    for video_index, item in enumerate(prepared_videos or (), start=1):
                        if item["input_has_audio"]:
                            audio_index += 1
                            condition_labels.append(("audio", audio_index))
                        condition_labels.append(("video", video_index))

                if task == "fl2va":
                    if prepared_videos:
                        raise OmniClientError("fl2va does not accept video conditions")
                    ids, tags = minimax_h3_multi_image_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts,
                    )
                elif prepared_videos:
                    ids, tags = minimax_h3_ref2va_video_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts or None,
                        video_block_token_counts=video_block_counts,
                        video_block_timestamps=video_block_timestamps,
                    )
                else:
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=condition_labels,
                        image_token_count=image_token_counts or None,
                    )

            logger.info(
                "MiniMax H3 %s Qwen presentation: %d tokens%s",
                task,
                int(ids.shape[0]),
                (
                    f", {len(images)} reference images"
                    + (f", {len(prepared_videos)} reference videos" if prepared_videos else "")
                    if images
                    else (f", {len(prepared_videos)} reference videos" if prepared_videos else "")
                ),
            )

        if rank < self.text_encoder_tp_size:
            # Distribute the encode inputs from the DiT main rank to the other
            # encoder TP ranks, then run the distributed encode on all of them.
            ids = self._distribute_encode_inputs(ids, vision_kwargs)
            hidden = self._encode_text_hidden(ids, vision_kwargs)

        hidden = _broadcast_tensor(
            hidden,
            dtype=torch.bfloat16,
            device=self.device,
        )
        tags = _broadcast_tensor(
            tags,
            dtype=torch.long,
            device=self.device,
        )
        return hidden, tags

    def _build_text_encoder_group(self, text_encoder_tp_size: int) -> Any:
        """Create the encoder tensor-parallel process group.

        The encoder group covers the first ``text_encoder_tp_size`` DiT ranks
        (the DiT group is always global ranks ``[0, dit_world)``).  Every rank
        participates in ``new_group`` so the collective completes; ranks
        outside the group never run encoder collectives.  For a single-rank
        encoder we return a lightweight placeholder so non-encoder ranks do
        not need to join a ``GroupCoordinator`` that would assert on ranks
        outside the group.
        """
        if text_encoder_tp_size == 1:
            return _SingleRankEncoderGroup(rank=self._dit_rank)
        ranks = list(range(text_encoder_tp_size))
        return init_world_group(
            ranks=ranks,
            local_rank=envs.LOCAL_RANK,
            backend=current_omni_platform.dist_backend,
        )

    def _encoder_group_broadcast_tensor(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a tensor from encoder rank 0 over the encoder TP group."""
        group = self.text_encoder_group
        if group.world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)

        shape = torch.zeros(8, dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            if tensor is None:
                raise ValueError("encoder rank 0 must provide a tensor to broadcast")
            shape[0] = tensor.ndim
            shape[1 : tensor.ndim + 1] = torch.tensor(tensor.shape, device=device)
        torch.distributed.broadcast(shape, src=group.ranks[0], group=group.device_group)
        ndim = int(shape[0].item())
        tensor_shape = tuple(int(value) for value in shape[1 : ndim + 1].tolist())
        if group.rank_in_group == 0:
            output = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            output = torch.empty(tensor_shape, device=device, dtype=dtype)
        torch.distributed.broadcast(output, src=group.ranks[0], group=group.device_group)
        return output

    def _distribute_encode_inputs(
        self,
        ids: torch.Tensor | None,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fan out encode inputs from encoder rank 0 to the encoder TP ranks.

        Mutates ``vision_kwargs`` in place so every encoder rank ends up with
        the same vision tensors, and returns the broadcast ``input_ids``.
        """
        keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        key_dtypes = {
            "pixel_values": torch.bfloat16,
            "pixel_values_videos": torch.bfloat16,
            "image_grid_thw": torch.long,
            "video_grid_thw": torch.long,
        }
        group = self.text_encoder_group
        device = self.device
        if group.world_size == 1:
            if ids is None:
                raise ValueError("encoder rank 0 must produce input ids")
            return ids.to(device=device, dtype=torch.long)

        mask = torch.zeros(len(keys), dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            for index, key in enumerate(keys):
                mask[index] = 1 if key in vision_kwargs else 0
        torch.distributed.broadcast(mask, src=group.ranks[0], group=group.device_group)

        if group.rank_in_group == 0:
            ids = self._encoder_group_broadcast_tensor(ids, dtype=torch.long, device=device)
        else:
            ids = self._encoder_group_broadcast_tensor(None, dtype=torch.long, device=device)
        for index, key in enumerate(keys):
            if mask[index].item() == 0:
                continue
            source = vision_kwargs.get(key) if group.rank_in_group == 0 else None
            vision_kwargs[key] = self._encoder_group_broadcast_tensor(
                source,
                dtype=key_dtypes[key],
                device=device,
            )
        return ids

    def _prepare_reference_videos(
        self,
        values: Any,
        *,
        target_frame_count: int,
        workdir: str,
        start_time_seconds: Any = None,
    ) -> list[dict[str, Any]] | None:
        _, rank, _ = _dit_rank_world()
        if rank != 0:
            return None
        return prepare_reference_videos(
            values,
            target_frame_count=target_frame_count,
            workdir=workdir,
            start_time_seconds=start_time_seconds,
        )

    def _encode_text_hidden(
        self,
        input_ids: torch.Tensor,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if getattr(self, "_model_cpu_offload_modules", None):
            # Invoke nn.Module.__call__ so the generic model-level offloader
            # swaps the resident DiT and encoder.
            return self.text_encoder(input_ids, **vision_kwargs)

        if self.od_config.enable_layerwise_offload or getattr(
            self.od_config, "enable_distributed_layerwise_offload", False
        ):
            # Layerwise DiT offload already provides the low-residency encoder
            # phase used by the checkpoint reference.
            with self._component_on_device(self.text_encoder):
                return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

        # Keep both Qwen and DiT resident across requests. Moving either model
        # here makes encoder latency include a tens-of-gigabytes PCIe transfer,
        # which defeats the no-offload contract.
        self.text_encoder.load_to_device()
        return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

    def _uses_manual_component_offload(self) -> bool:
        od_config = getattr(self, "od_config", None)
        return bool(
            getattr(od_config, "enable_layerwise_offload", False)
            or getattr(od_config, "enable_distributed_layerwise_offload", False)
        )

    def enable_omni_model_cpu_offload(
        self,
        *,
        device: torch.device,
        pin_memory: bool,
        use_hsdp: bool,
    ) -> None:
        if getattr(self, "_model_cpu_offload_modules", None):
            return

        components = ModuleDiscovery.discover(self)
        dits = components.dits
        stages = [*components.encoders, *components.vaes]
        modules = [*dits, *stages]
        apply_sequential_offload(
            dit_modules=dits,
            encoder_modules=stages,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
            offload_initial_dits=True,
        )

        self._model_cpu_offload_modules = modules
        logger.info(
            "MiniMax-H3 model-level CPU offload enabled for %d DiT(s), text encoder, video VAE, and audio VAE",
            len(dits),
        )

    def disable_omni_model_cpu_offload(self) -> None:
        modules = getattr(self, "_model_cpu_offload_modules", None)
        if not modules:
            return
        remove_sequential_offload(modules)
        self._model_cpu_offload_modules = []

    @contextmanager
    def _component_on_device(self, component: nn.Module):
        if getattr(self, "_model_cpu_offload_modules", None):
            with sequential_offload_component(component):
                yield
            return
        staged = self._uses_manual_component_offload()
        try:
            if staged:
                component.load_to_device()
            yield
        except BaseException:
            if staged:
                try:
                    component.offload_to_cpu()
                except BaseException:
                    logger.exception("Failed to release %s after component failure", component.__class__.__name__)
                cache = getattr(self, "_dlo_component_cache", None)
                if cache is not None:
                    try:
                        cache.release_if_needed(force=True)
                    except BaseException:
                        logger.exception("Failed to release retained allocator cache after component failure")
            raise
        else:
            if staged:
                try:
                    component.offload_to_cpu()
                except BaseException:
                    cache = getattr(self, "_dlo_component_cache", None)
                    if cache is not None:
                        try:
                            cache.release_if_needed(force=True)
                        except BaseException:
                            logger.exception("Failed to release retained allocator cache after offload failure")
                    raise

    def _encode_visual_conditions(
        self,
        images: list[Image.Image],
        prepared_videos: list[dict[str, Any]] | None,
        *,
        video_count: int,
    ) -> tuple[torch.Tensor | None, list[tuple[int, int, int]]]:
        rows: list[torch.Tensor] = []
        shapes: list[tuple[int, int, int]] = []
        _, rank, _ = _dit_rank_world()
        # Keep image and video references in one residency window when both
        # appear in a request; otherwise the video branch would reload the VAE.
        needs_video_vae = video_count > 0 or (rank == 0 and bool(images))
        video_vae_context = self._component_on_device(self.video_vae) if needs_video_vae else nullcontext()
        with video_vae_context:
            if images:
                image_rows = None
                if rank == 0:
                    image_rows = torch.cat([self.video_vae.encode_image(image) for image in images])
                rows.append(
                    _broadcast_tensor(
                        image_rows,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                shapes.extend((1, image.height // 16, image.width // 16) for image in images)
            if video_count:
                video_rows, video_shapes = self._encode_video_conditions_resident(
                    prepared_videos,
                    count=video_count,
                )
                rows.append(video_rows)
                shapes.extend(video_shapes)
        return (torch.cat(rows) if rows else None), shapes

    def _encode_audio_conditions_resident(
        self,
        audios: list[tuple[torch.Tensor, int]],
        *,
        max_duration_seconds: float | None = None,
    ) -> tuple[torch.Tensor | None, list[int]]:
        if not audios:
            return None, []
        if max_duration_seconds is not None:
            max_duration_seconds = float(max_duration_seconds)
            if max_duration_seconds <= 0:
                raise ValueError("max_duration_seconds must be positive")
        _, rank, _ = _dit_rank_world()
        rows = None
        lengths = torch.zeros(len(audios), dtype=torch.long, device=self.device)
        if rank == 0:
            bounded_audios = []
            for waveform, sample_rate in audios:
                if max_duration_seconds is not None:
                    max_samples = int(round(max_duration_seconds * int(sample_rate)))
                    waveform = waveform[..., :max_samples]
                bounded_audios.append((waveform, sample_rate))
            encoded = [self.audio_vae.encode_waveform(*audio) for audio in bounded_audios]
            rows = torch.cat([item[0] for item in encoded])
            lengths = torch.tensor(
                [int(item[1]) for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(lengths, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [int(value) for value in lengths.tolist()],
        )

    def _encode_video_conditions_resident(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        count: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        group, rank, world_size = _dit_rank_world()
        distributed_encode = self.video_vae.is_distributed_enabled()
        if distributed_encode:
            # Native tiled encode uses collectives, so every VPP rank must
            # enter each reference encode in the same input order.
            prepared_videos_list = [prepared_videos]
            dist.broadcast_object_list(
                prepared_videos_list,
                src=0,
                group=group,
                device=self.device,
            )
            prepared_videos = prepared_videos_list[0]

        rows = None
        shapes = torch.zeros((count, 3), dtype=torch.long, device=self.device)
        if rank == 0 or distributed_encode:
            if prepared_videos is None or len(prepared_videos) != count:
                raise ValueError("reference-video preparation is incomplete")
            encoded = [
                self.video_vae.encode_video(load_video_frames(item["prepared_path"])) for item in prepared_videos
            ]
            rows = torch.cat([item[0] for item in encoded])
            shapes = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        if distributed_encode:
            return (
                rows.to(device=self.device, dtype=torch.float32),
                [tuple(int(value) for value in item) for item in shapes.tolist()],
            )

        if world_size > 1:
            dist.broadcast(shapes, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [tuple(int(value) for value in item) for item in shapes.tolist()],
        )

    def _encode_video_audio_conditions_resident(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
    ) -> tuple[torch.Tensor | None, list[int]]:
        _, rank, _ = _dit_rank_world()
        count = sum(has_audio)
        if count == 0:
            return None, []
        rows = None
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        if rank == 0:
            if prepared_videos is None:
                raise ValueError("rank 0 reference-video preparation is incomplete")
            encoded = [
                self.audio_vae.encode_waveform(
                    *load_video_audio(
                        item["original_path"],
                        start_time_seconds=float(item.get("start_time_seconds", 0.0)),
                        duration_seconds=item.get(
                            "audio_duration_seconds",
                            item.get("duration_seconds"),
                        ),
                    )
                )
                for item in prepared_videos
                if item["input_has_audio"]
            ]
            rows = torch.cat([item[0] for item in encoded])
            lengths = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(lengths, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [int(value) for value in lengths.tolist()],
        )

    def _encode_reference_audio_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
        standalone_audios: list[tuple[torch.Tensor, int]],
        max_duration_seconds: float,
    ) -> tuple[torch.Tensor | None, list[int], torch.Tensor | None, list[int]]:
        # Embedded and standalone audio are consecutive direct Audio-VAE
        # calls. Keep the component resident across both paths.
        needs_audio_vae = any(has_audio) or bool(standalone_audios)
        audio_vae_context = self._component_on_device(self.audio_vae) if needs_audio_vae else nullcontext()
        with audio_vae_context:
            embedded_condition, embedded_lengths = self._encode_video_audio_conditions_resident(
                prepared_videos,
                has_audio=has_audio,
            )
            external_condition, external_lengths = self._encode_audio_conditions_resident(
                standalone_audios,
                max_duration_seconds=max_duration_seconds,
            )
        return (
            embedded_condition,
            embedded_lengths,
            external_condition,
            external_lengths,
        )

    def _initial_noise(
        self,
        *,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_generator = torch.Generator(device="cpu").manual_seed(seed)
        video = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=video_generator,
            dtype=torch.float32,
        )
        video_rows = minimax_h3_patchify_video_latent(
            video,
            patch_size=(1, 2, 2),
        )
        audio_generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_rows = torch.randn(
            audio_t * 2,
            32,
            generator=audio_generator,
            dtype=torch.float32,
        )
        return video_rows, audio_rows

    @contextmanager
    def _resident_dit_layers_on_device(self, *, enabled: bool = True):
        controller = getattr(self, "_dlo_residency_controller", None)
        if controller is not None and enabled:
            controller.load_resident_layers()
        try:
            yield
        finally:
            if controller is not None and enabled:
                controller.offload_resident_layers()

    def _build_denoise_inputs(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        base_schedule: Sequence[float] | None,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Build the packed layout, initial rows, anchors, and sigma schedules.

        Shared by request-mode :meth:`diffuse` and step-mode
        :meth:`prepare_encode` so both paths start from identical state.
        """
        initial_video, initial_audio = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None or ref_audio_t is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [
                    {"kind": "image", "latent_h": ref_h, "latent_w": ref_w},
                    {"kind": "audio", "ref_audio_t": ref_audio_t},
                ]
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        else:
            packed = minimax_h3_packed_sequence(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=task == "fl2va",
                keyframe_frame_indices=keyframe_frame_indices if task == "fl2va" else None,
                frame_count=num_frames if task == "fl2va" else None,
            )

        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )

        visual_anchor = visual_condition
        if visual_anchor is not None:
            condition_shapes = visual_condition_shapes
            if condition_shapes is None and visual_condition_shape is not None:
                condition_shapes = [visual_condition_shape]
            if not condition_shapes:
                raise ValueError("visual condition shape is missing")
            visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_anchor,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )
            full_video = torch.zeros(
                branch.img_pos.shape[0],
                96,
                dtype=torch.float32,
            )
            full_video[branch.update_mask] = initial_video
            initial_video = full_video

        audio_anchor = audio_condition
        if audio_anchor is not None:
            condition_audio_t = audio_condition_lengths
            if condition_audio_t is None and ref_audio_t is not None:
                condition_audio_t = [ref_audio_t]
            if not condition_audio_t:
                raise ValueError("reference audio length is missing")
            audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
                audio_anchor,
                condition_audio_t=condition_audio_t,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )
            full_audio = torch.zeros(
                branch.audio_pos.shape[0],
                32,
                dtype=torch.float32,
            )
            full_audio[branch.audio_update_mask] = initial_audio
            initial_audio = full_audio

        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=video_shift,
            base_schedule=base_schedule,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=audio_shift,
            base_schedule=base_schedule,
        )
        return {
            "branch": branch,
            # The request-mode loop moves these onto the device itself; step mode
            # keeps them resident across steps, so normalize once for both.
            "video_rows": initial_video.to(device=self.device, dtype=torch.float32),
            "audio_rows": initial_audio.to(device=self.device, dtype=torch.float32),
            "cond_anchor": (
                None if visual_anchor is None else visual_anchor.to(device=self.device, dtype=torch.float32)
            ),
            "audio_anchor": (
                None if audio_anchor is None else audio_anchor.to(device=self.device, dtype=torch.float32)
            ),
            "sigmas_video": video_sigmas,
            "sigmas_audio": audio_sigmas,
        }

    def _unpack_denoised_rows(
        self,
        branch: MiniMaxH3DenoiseBranch,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        *,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the target rows and unpack them back into VAE latents."""
        target_video = video_rows[branch.update_mask_dev]
        video_latent = minimax_h3_unpatchify_video_tokens(
            target_video,
            latent_shape=(
                latent_t,
                latent_h // 2,
                latent_w // 2,
                24,
            ),
            patch_size=(1, 2, 2),
        )
        target_audio = audio_rows[branch.audio_update_mask_dev]
        audio_latent = minimax_h3_unpack_audio_tokens(
            target_audio,
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        return video_latent, audio_latent

    def diffuse(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        base_schedule: Sequence[float] | None,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._build_denoise_inputs(
            task=task,
            text_embeddings=text_embeddings,
            text_tags=text_tags,
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=num_steps,
            video_shift=video_shift,
            audio_shift=audio_shift,
            base_schedule=base_schedule,
            visual_condition=visual_condition,
            visual_condition_shape=visual_condition_shape,
            audio_condition=audio_condition,
            ref_audio_t=ref_audio_t,
            ref_blocks=ref_blocks,
            visual_condition_shapes=visual_condition_shapes,
            audio_condition_lengths=audio_condition_lengths,
            keyframe_frame_indices=keyframe_frame_indices,
        )
        branch = inputs["branch"]
        transformer = self._transformer_for_task(task)
        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):
            with self.progress_bar(total=len(inputs["sigmas_video"]) - 1) as progress:
                video_rows, audio_rows = minimax_h3_denoise_loop(
                    model=transformer,
                    positive=branch,
                    initial_video_rows=inputs["video_rows"],
                    initial_audio_rows=inputs["audio_rows"],
                    keyframe_cond_rows=inputs["cond_anchor"],
                    audio_ref_rows=inputs["audio_anchor"],
                    sigmas_video=inputs["sigmas_video"],
                    sigmas_audio=inputs["sigmas_audio"],
                    device=self.device,
                    imgvid_cond_noise_aug_for_inference=(MINIMAX_H3_IMGVID_COND_TIMESTEP),
                    audio_cond_noise_aug_for_inference=(MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                    on_step=lambda step, video, audio: progress.update(),
                )

        return self._unpack_denoised_rows(
            branch,
            video_rows,
            audio_rows,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )

    def decode(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._component_on_device(self.video_vae):
            with current_omni_platform.create_autocast_context(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=True,
            ):
                video = self.video_vae.decode_latent(video_latent)
        video = video[..., :height, :width].contiguous()
        with self._component_on_device(self.audio_vae):
            audio = self.audio_vae.decode_latent(audio_latent)
        return video, audio

    @staticmethod
    def _extract_prompt(raw_prompt: Any) -> tuple[str, dict[str, Any]]:
        """Split a request prompt into its text and multimodal parts."""
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
            multi_modal_data: dict[str, Any] = {}
        else:
            prompt = str(raw_prompt.get("prompt") or "")
            multi_modal_data = raw_prompt.get("multi_modal_data") or {}
        if not prompt:
            raise OmniClientError("MiniMax H3 requires a non-empty prompt")
        return prompt, multi_modal_data

    @staticmethod
    def _extract_text_conditioning(raw_prompt: Any) -> MiniMaxH3TextConditioning | None:
        if isinstance(raw_prompt, str):
            return None
        additional_information = raw_prompt.get("additional_information") or {}
        text_encoder_output = additional_information.get("text_encoder_output")
        if text_encoder_output is None:
            return None
        if not isinstance(text_encoder_output, Mapping):
            raise OmniClientError("text_encoder_output must be a mapping")
        try:
            return MiniMaxH3TextConditioning.from_payload(text_encoder_output)
        except ValueError as exc:
            raise OmniClientError(str(exc)) from exc

    @staticmethod
    def _extract_prepared_reference_videos(raw_prompt: Any) -> list[dict[str, Any]] | None:
        if isinstance(raw_prompt, str):
            return None
        additional_information = raw_prompt.get("additional_information") or {}
        meta = additional_information.get("meta") or {}
        descriptor = meta.get(MINIMAX_H3_PREPARED_REFERENCE_VIDEOS_KEY)
        if descriptor is None:
            return None
        if not isinstance(descriptor, str):
            raise OmniClientError("MiniMax H3 prepared-reference-video descriptor must be a string")
        try:
            _, videos = deserialize_prepared_reference_videos(descriptor)
        except ValueError as exc:
            raise OmniClientError(str(exc)) from exc
        return videos

    def _prepare_request_inputs(
        self,
        *,
        prompt: str,
        multi_modal_data: dict[str, Any],
        sampling: Any,
        text_conditioning: MiniMaxH3TextConditioning | None = None,
        prepared_reference_videos: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Resolve the task and output shape, then run every request-level encode.

        Shared by request-mode :meth:`forward` and step-mode
        :meth:`prepare_encode`; the returned mapping feeds :meth:`diffuse` and
        :meth:`_build_denoise_inputs` unchanged.
        """
        quality = sampling.quality
        logger.debug("MiniMax H3 request quality=%s", quality)
        extra = sampling.extra_args or {}
        has_turbo_lora = self._has_active_turbo_lora(sampling)
        has_native_lora = self._has_active_native_lora(sampling)
        task = self._resolve_task(
            extra.get("task"),
            multi_modal_data,
            has_turbo_lora=has_turbo_lora,
            has_native_lora=has_native_lora,
        )
        if has_turbo_lora:
            self._validate_turbo_sampling(sampling)
        if has_native_lora:
            self._validate_native_sampling(sampling, task=task)

        raw_image = multi_modal_data.get("image")
        raw_videos = multi_modal_data.get("video")
        raw_audio = multi_modal_data.get("audio")
        images = _load_images(raw_image) if raw_image is not None else []
        video_values = list(raw_videos) if isinstance(raw_videos, (list, tuple)) else raw_videos
        audio_values = list(raw_audio) if isinstance(raw_audio, (list, tuple)) else raw_audio

        if task == "t2va" and (images or raw_videos is not None or raw_audio is not None):
            raise OmniClientError("t2va does not accept image, video, or audio conditions")
        if task == "fl2va":
            if not images:
                raise OmniClientError("fl2va requires multi_modal_data.image")
            if len(images) > 2:
                raise OmniClientError("fl2va accepts at most first and last images")
            if raw_videos is not None or raw_audio is not None:
                raise OmniClientError("fl2va accepts image keyframes only")
        if task == "ref2va":
            video_count = (
                len(video_values) if isinstance(video_values, (list, tuple)) else int(video_values is not None)
            )
            audio_is_waveform_pair = (
                isinstance(raw_audio, (list, tuple))
                and len(raw_audio) == 2
                and isinstance(raw_audio[1], (int, np.integer))
            )
            audio_count = (
                len(audio_values)
                if isinstance(audio_values, (list, tuple)) and not audio_is_waveform_pair
                else int(raw_audio is not None)
            )
            _validate_ref2va_reference_counts(len(images), video_count, audio_count)
        elif raw_videos is not None:
            raise OmniClientError(f"{task} does not accept a video condition")

        image = images[0] if images else None
        height, width, num_frames, latent_t, audio_t = self._resolve_shape(task, sampling, image)
        if task == "fl2va":
            for item in images:
                _validate_reference_image(item)
            prepared_images = [item.resize((width, height), Image.Resampling.LANCZOS) for item in images]
            keyframe_frame_indices = _resolve_fl2va_keyframe_indices(extra, len(images))
        elif task == "ref2va":
            prepared_images = []
            for item in images:
                ref_width, ref_height = _reference_image_shape(item)
                prepared_images.append(item.resize((ref_width, ref_height), Image.Resampling.LANCZOS))
            keyframe_frame_indices = None
        else:
            prepared_images = []
            keyframe_frame_indices = None

        visual_condition = None
        visual_shape = None
        visual_shapes = None
        audio_condition = None
        ref_audio_t = None
        audio_lengths = None
        ref_blocks = None
        with tempfile.TemporaryDirectory(prefix="minimax_h3_ref2va_") as workdir:
            prepared_videos = None
            has_audio: list[bool] = []
            video_count = 0
            if raw_videos is not None:
                video_count = len(raw_videos) if isinstance(raw_videos, (list, tuple)) else 1
                # File-based reference-video prep runs only on rank 0; other
                # ranks return None without touching disk. If rank 0 raises
                # (e.g. invalid file, unsupported codec) it must not exit
                # ``prepare_encode`` before the downstream broadcasts below --
                # non-zero ranks would then deadlock on them forever. Capture
                # the exception here and let every rank agree on the outcome
                # before starting any subsequent collective.
                prep_error: Exception | None = None
                try:
                    _, rank, _ = _dit_rank_world()
                    if prepared_reference_videos is not None:
                        if rank == 0:
                            prepared_videos = _reuse_prepared_reference_videos(
                                prepared_reference_videos,
                                expected_count=video_count,
                            )
                    else:
                        prepared_videos = self._prepare_reference_videos(
                            raw_videos,
                            target_frame_count=num_frames,
                            workdir=workdir,
                            start_time_seconds=extra.get("start_time_seconds"),
                        )
                except Exception as exc:
                    prep_error = exc
                _broadcast_rank0_exception(prep_error)
                has_audio_tensor = torch.zeros(
                    video_count,
                    dtype=torch.long,
                    device=self.device,
                )
                _, rank, world_size = _dit_rank_world()
                if rank == 0:
                    has_audio_tensor = torch.tensor(
                        [int(item["input_has_audio"]) for item in prepared_videos or []],
                        dtype=torch.long,
                        device=self.device,
                    )
                if world_size > 1:
                    dist.broadcast(
                        has_audio_tensor,
                        src=0,
                        group=get_world_group().device_group,
                    )
                has_audio = [bool(value) for value in has_audio_tensor.tolist()]

            if raw_audio is not None:
                validate_reference_audio_files(raw_audio)
            standalone_audios = _load_audios(raw_audio) if raw_audio is not None else []
            validate_reference_audio_waveforms(standalone_audios)
            condition_labels: list[tuple[str, int]] = []
            for image_index in range(1, len(prepared_images) + 1):
                condition_labels.append(("image", image_index))
            audio_index = 0
            for video_index, item in enumerate(prepared_videos or (), start=1):
                if item["input_has_audio"]:
                    audio_index += 1
                    condition_labels.append(("audio", audio_index))
                condition_labels.append(("video", video_index))
            for _ in standalone_audios:
                audio_index += 1
                condition_labels.append(("audio", audio_index))

            if text_conditioning is not None:
                text_embeddings = text_conditioning.hidden_states.to(
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                text_tags = text_conditioning.token_tags.to(
                    device=self.device,
                    dtype=torch.long,
                )
            elif getattr(self, "text_encoder", None) is not None:
                text_embeddings, text_tags = self.encode_prompt(
                    task=task,
                    prompt=prompt,
                    images=prepared_images,
                    prepared_videos=prepared_videos,
                    condition_labels=condition_labels if task == "ref2va" else None,
                )
            else:
                raise OmniClientError(
                    "MiniMax H3 diffusion stage requires text_encoder_output when text_encoder is not loaded"
                )

            # ``prepared_videos`` is intentionally ``None`` on non-zero DiT
            # ranks; the distributed video encoder broadcasts the prepared
            # metadata inside ``_encode_visual_conditions``.  Use the global
            # video count here so video + standalone-audio Ref2VA requests do
            # not look like audio-only requests on those ranks.
            if video_count or prepared_images:
                visual_condition, visual_shapes = self._encode_visual_conditions(
                    prepared_images,
                    prepared_videos,
                    video_count=video_count,
                )
                (
                    embedded_audio_condition,
                    embedded_audio_lengths,
                    external_audio_condition,
                    external_audio_lengths,
                ) = self._encode_reference_audio_conditions(
                    prepared_videos,
                    has_audio=has_audio,
                    standalone_audios=standalone_audios,
                    max_duration_seconds=float(num_frames) / float(sampling.fps or MINIMAX_H3_FPS),
                )
                audio_parts = [
                    item for item in (embedded_audio_condition, external_audio_condition) if item is not None
                ]
                audio_condition = torch.cat(audio_parts) if audio_parts else None
                audio_lengths = embedded_audio_lengths + external_audio_lengths
                ref_blocks = []
                image_shapes = visual_shapes[: len(prepared_images)]
                video_shapes = visual_shapes[len(prepared_images) :]
                for shape in image_shapes:
                    ref_blocks.append(
                        {
                            "kind": "image",
                            "latent_h": shape[1],
                            "latent_w": shape[2],
                        }
                    )
                audio_iterator = iter(embedded_audio_lengths)
                for shape, contributes_audio in zip(video_shapes, has_audio, strict=True):
                    ref_audio = next(audio_iterator) if contributes_audio else 0
                    ref_blocks.append(
                        {
                            "kind": "video_audio" if ref_audio else "video",
                            "ref_audio_t": ref_audio,
                            "latent_t": shape[0],
                            "latent_h": shape[1],
                            "latent_w": shape[2],
                        }
                    )
                for ref_audio_t in external_audio_lengths:
                    ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t})
            elif standalone_audios:
                raise OmniClientError("standalone audio references require a Ref2VA visual reference")

            if visual_shapes and len(visual_shapes) == 1:
                visual_shape = visual_shapes[0]
            if audio_lengths:
                if any(length < 80 or length > 600 for length in audio_lengths):
                    raise OmniClientError("MiniMax H3 audio references must each be between 2 and 15 seconds")
                if sum(audio_lengths) > 600:
                    raise OmniClientError("MiniMax H3 audio references must be at most 15 seconds in total")
                if len(audio_lengths) == 1:
                    ref_audio_t = audio_lengths[0]

        seed = int(sampling.seed if sampling.seed is not None else 42)
        sigma_schedule = self._sigma_schedule_for_request(sampling, task)
        if sigma_schedule is None:
            base_schedule = None
            num_steps = int(sampling.num_inference_steps or 50)
        else:
            # The schedule lists sigma boundaries; the denoise loop runs one
            # step per interval, and that count is what requests and Cache-DiT
            # speak in.
            base_schedule = sigma_schedule.base_schedule
            num_steps = sigma_schedule.num_inference_steps
            requested_steps = sampling.num_inference_steps
            if requested_steps is not None and int(requested_steps) != num_steps:
                raise OmniClientError(
                    "this MiniMax H3 checkpoint pins a distilled sigma schedule; num_inference_steps "
                    f"must be {num_steps} or omitted, got {int(requested_steps)}"
                )
        video_shift = float(extra.get("flow_shift", self.default_video_shift))
        audio_shift = float(extra.get("audio_flow_shift", self.default_audio_shift))
        quality_plan = self._quality_policy.resolve(
            quality=quality,
            num_inference_steps=num_steps,
            extra_args=extra,
        )
        self._cache_dit_runtime.prepare(quality_plan.cache_dit)
        num_outputs = _resolve_minimax_h3_num_outputs(sampling.num_outputs_per_prompt)
        return {
            "task": task,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "latent_t": latent_t,
            "latent_h": height // 16,
            "latent_w": width // 16,
            "audio_t": audio_t,
            "text_embeddings": text_embeddings,
            "text_tags": text_tags,
            "visual_condition": visual_condition,
            "visual_condition_shape": visual_shape,
            "audio_condition": audio_condition,
            "ref_audio_t": ref_audio_t,
            "ref_blocks": ref_blocks,
            "visual_condition_shapes": visual_shapes,
            "audio_condition_lengths": audio_lengths,
            "keyframe_frame_indices": keyframe_frame_indices,
            "seed": seed,
            "num_steps": num_steps,
            "video_shift": video_shift,
            "audio_shift": audio_shift,
            "base_schedule": base_schedule,
            "num_outputs": num_outputs,
        }

    @staticmethod
    def _denoise_kwargs(context: dict[str, Any]) -> dict[str, Any]:
        """Select the denoise-input arguments from a prepared request context."""
        return {key: context[key] for key in _MINIMAX_H3_DENOISE_INPUT_KEYS}

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise OmniClientError("MiniMax H3 supports one request at a time")
        raw_prompt = request.prompts[0]
        prompt, multi_modal_data = self._extract_prompt(raw_prompt)
        context = self._prepare_request_inputs(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
            sampling=request.sampling_params,
            text_conditioning=self._extract_text_conditioning(raw_prompt),
            prepared_reference_videos=self._extract_prepared_reference_videos(raw_prompt),
        )
        denoise_kwargs = self._denoise_kwargs(context)
        num_outputs = context["num_outputs"]
        videos = []
        audios = []
        for output_seed in _minimax_h3_output_seeds(context["seed"], num_outputs):
            video_latent, audio_latent = self.diffuse(**{**denoise_kwargs, "seed": output_seed})
            video, audio = self.decode(
                video_latent,
                audio_latent,
                height=context["height"],
                width=context["width"],
            )
            videos.append(video)
            audios.append(audio)
        video = torch.cat(videos, dim=0)
        audio = torch.cat(audios, dim=0)
        return DiffusionOutput(
            output=(video, audio),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )

    # ------------------------------------------------------------------
    # Step-wise execution (continuous batching)
    # ------------------------------------------------------------------

    @staticmethod
    def _packed_batch_supported(transformer: MiniMaxH3DiTModel) -> bool:
        """Whether every attention in this DiT honors multi-document cu_seqlens.

        A packed batch is only isolated if *all* of them do: the token refiner
        runs under its own attention role and can resolve to a different backend
        from the DiT blocks. Ring sequence parallelism dispatches through
        ``RingParallelAttention``, whose kernels ignore the packed
        ``cu_seqlens`` metadata regardless of the configured backend; packing
        multiple requests under ring would let attention cross document
        boundaries, so any layer running ring disqualifies the batch.

        The gate probes a per-backend capability rather than a fixed backend
        name: FLASH_ATTN, for example, only isolates arbitrary N-document
        packed cu_seqlens on CUDA/ROCm/MUSA. Its NPU path only accepts a
        ``[real, pad]`` two-document layout and its XPU path ignores
        cu_seqlens outright — either would silently attend across request
        boundaries.
        """
        attentions = [module for module in transformer.modules() if isinstance(module, MiniMaxH3Attention)]
        if not attentions:
            return False
        return all(_attention_isolates_packed_requests(module.attention) for module in attentions)

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Run every request-level stage once and seed the per-request step state."""
        del kwargs
        # Two request-mode features have no place in the shared step contract:
        # a request state carries exactly one latent tensor, and distributed
        # layerwise offload streams the DiT around one whole denoise loop rather
        # than around a single scheduler-driven step.
        num_outputs = _resolve_minimax_h3_num_outputs(state.sampling.num_outputs_per_prompt)
        if num_outputs != 1:
            raise OmniClientError(
                f"MiniMax H3 step execution produces one output per request, got num_outputs_per_prompt={num_outputs}"
            )
        if getattr(self, "_dlo_residency_controller", None) is not None:
            raise ValueError(
                "MiniMax H3 step execution is not compatible with distributed layerwise offload; "
                "the resident-layer window spans a whole denoise loop, so per-step streaming would "
                "reload the DiT every step. Drop --step-execution or --enable-distributed-layerwise-offload."
            )
        # Request-scoped Cache-DiT (quality=high) mutates hook state on the
        # shared transformer rather than on ``StepRequestState``. In step mode
        # two requests can interleave denoise steps, or be co-batched into a
        # single forward, and the second one would then re-enter the DiT with
        # cache buffers shaped for the first. Reject the profile here rather
        # than let it corrupt outputs at runtime; startup-configured Cache-DiT
        # is already blocked in ``DiffusionModelRunner.execute_stepwise``.
        if getattr(state.sampling, "quality", None) == "high":
            raise OmniClientError(
                "MiniMax H3 step execution does not support the high-quality Cache-DiT profile "
                "(quality=high); its hooks live on the shared transformer, so interleaved or "
                "co-batched requests would reuse incompatible cache state. Drop --step-execution "
                "or omit quality=high."
            )
        prompt, multi_modal_data = self._extract_prompt(state.prompt)
        context = self._prepare_request_inputs(
            prompt=prompt,
            multi_modal_data=multi_modal_data,
            sampling=state.sampling,
            text_conditioning=self._extract_text_conditioning(state.prompt),
            prepared_reference_videos=self._extract_prepared_reference_videos(state.prompt),
        )
        inputs = self._build_denoise_inputs(**self._denoise_kwargs(context))

        sigmas_video = inputs["sigmas_video"]
        sigmas_audio = inputs["sigmas_audio"]
        if len(sigmas_video) < 2:
            raise OmniClientError(
                f"MiniMax H3 step execution needs at least one denoise step, got num_inference_steps="
                f"{len(sigmas_video) - 1}"
            )

        branch = inputs["branch"]
        video_rows, audio_rows, cond_anchor, audio_anchor = minimax_h3_prepare_denoise_rows(
            positive=branch,
            initial_video_rows=inputs["video_rows"],
            initial_audio_rows=inputs["audio_rows"],
            keyframe_cond_rows=inputs["cond_anchor"],
            audio_ref_rows=inputs["audio_anchor"],
            device=self.device,
        )

        # The denoise loop consumes sigma pairs, so the schedule carries one more
        # point than there are steps. ``timesteps`` holds the video branch because
        # the shared contract gives a request exactly one timestep sequence; the
        # audio schedule rides along in ``extra``.
        state.timesteps = torch.tensor(
            [1.0 - sigma for sigma in sigmas_video[:-1]],
            dtype=torch.float32,
            device=self.device,
        )
        state.step_index = 0
        # Video rows are the batched tensor the runner slices per request; audio
        # rows have a different width, so they stay request-private.
        state.latents = video_rows
        state.do_true_cfg = False  # H3 checkpoints are CFG-distilled.
        state.extra.update(
            {
                _STEP_BRANCH: branch,
                _STEP_TRANSFORMER: self._transformer_for_task(context["task"]),
                _STEP_AUDIO_ROWS: audio_rows,
                _STEP_COND_ANCHOR: cond_anchor,
                _STEP_AUDIO_ANCHOR: audio_anchor,
                _STEP_SIGMAS_VIDEO: sigmas_video,
                _STEP_SIGMAS_AUDIO: sigmas_audio,
                _STEP_SHAPE: {
                    "height": context["height"],
                    "width": context["width"],
                    "latent_t": context["latent_t"],
                    "latent_h": context["latent_h"],
                    "latent_w": context["latent_w"],
                    "audio_t": context["audio_t"],
                },
            }
        )
        return state

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """Run one denoise forward covering every request in the batch.

        Requests are concatenated into a single packed sequence that keeps one
        attention document each, so the whole batch costs one DiT forward.
        Backends that ignore ``cu_seqlens`` cannot express that isolation, so
        they fall back to one forward per request.
        """
        del kwargs
        batch_states = list(states if states is not None else input_batch.states)

        branches = [state.extra[_STEP_BRANCH] for state in batch_states]
        video_rows = [state.latents for state in batch_states]
        audio_rows = [state.extra[_STEP_AUDIO_ROWS] for state in batch_states]
        schedules = [_minimax_h3_step_schedule(state) for state in batch_states]
        transformers = [state.extra[_STEP_TRANSFORMER] for state in batch_states]
        mixed_transformers = len({id(transformer) for transformer in transformers}) > 1

        # Both execution modes must publish denoise progress for step-gated
        # attention features. Requests can differ in both step index and sigma
        # schedule, so a batch that is not at one single point has nothing to
        # publish and those gates stay dense -- which is their safe default.
        progress = {
            (state.step_index, schedule["sigma_video"], len(state.extra[_STEP_SIGMAS_VIDEO]) - 1)
            for state, schedule in zip(batch_states, schedules)
        }
        minimax_h3_publish_denoise_progress(*(progress.pop() if len(progress) == 1 else (None, None, None)))

        if len(batch_states) > 1 and (mixed_transformers or not self._packed_batch_supported(transformers[0])):
            if mixed_transformers:
                logger.warning_once(
                    "MiniMax H3 step batch contains requests for different task-specific DiTs; "
                    "running %d requests one forward at a time.",
                    len(batch_states),
                )
            elif any(
                getattr(getattr(module, "attention", None), "use_ring", False)
                for module in transformers[0].modules()
                if isinstance(module, MiniMaxH3Attention)
            ):
                logger.warning_once(
                    "MiniMax H3 step batching is disabled when ring attention is active: "
                    "the ring kernels ignore packed cu_seqlens and would attend across request "
                    "boundaries. Running %d requests one forward at a time.",
                    len(batch_states),
                )
            else:
                logger.warning_once(
                    "MiniMax H3 step batching needs every attention on a backend that isolates "
                    "packed multi-document cu_seqlens (see AttentionBackend."
                    "supports_multi_doc_packed_varlen); running %d requests one forward at a time.",
                    len(batch_states),
                )
            video_parts: list[torch.Tensor] = []
            audio_parts: list[torch.Tensor] = []
            for index, branch in enumerate(branches):
                forward_kwargs = branch.forward_kwargs(
                    video_rows=video_rows[index],
                    audio_rows=audio_rows[index],
                    t_video=schedules[index]["t_video"],
                    t_audio=schedules[index]["t_audio"],
                    imgvid_cond_timestep=schedules[index]["imgvid_cond_timestep"],
                    audio_ref_cond_timestep=schedules[index]["audio_ref_cond_timestep"],
                )
                request_video, request_audio = transformers[index](**forward_kwargs)
                video_parts.append(request_video)
                audio_parts.append(request_audio)
            video_velocity = torch.cat(video_parts)
            audio_velocity = torch.cat(audio_parts)
        else:
            forward_kwargs = minimax_h3_batched_forward_kwargs(
                branches=branches,
                video_rows=video_rows,
                audio_rows=audio_rows,
                t_video=[schedule["t_video"] for schedule in schedules],
                t_audio=[schedule["t_audio"] for schedule in schedules],
                imgvid_cond_timesteps=[schedule["imgvid_cond_timestep"] for schedule in schedules],
                audio_ref_cond_timesteps=[schedule["audio_ref_cond_timestep"] for schedule in schedules],
            )
            logger.debug(
                "MiniMax H3 denoise step: %d request(s) packed into %d rows",
                len(batch_states),
                int(forward_kwargs["x"].shape[1]),
            )
            video_velocity, audio_velocity = transformers[0](**forward_kwargs)

        # The shared contract carries one velocity tensor per step, and audio rows
        # are a different width than video rows, so hand the audio branch to
        # step_scheduler() through request-private state.
        audio_parts_by_request = torch.split(audio_velocity, [int(branch.audio_pos.shape[0]) for branch in branches])
        for state, request_audio in zip(batch_states, audio_parts_by_request, strict=True):
            state.extra[_STEP_AUDIO_NOISE_PRED] = request_audio
        return video_velocity

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Apply one Euler-eta0 update to this request's video and audio rows."""
        del kwargs
        # denoise_step() stages the audio half of this step's velocity; popping
        # it keeps a second step_scheduler() call from reusing a stale one.
        audio_noise_pred = state.extra.pop(_STEP_AUDIO_NOISE_PRED)

        branch = state.extra[_STEP_BRANCH]
        schedule = _minimax_h3_step_schedule(state)
        update = branch.update_mask_dev
        audio_update = branch.audio_update_mask_dev
        video_rows = state.latents
        audio_rows = state.extra[_STEP_AUDIO_ROWS]
        cond_anchor = state.extra[_STEP_COND_ANCHOR]
        audio_anchor = state.extra[_STEP_AUDIO_ANCHOR]
        device = video_rows.device

        x0_video = minimax_h3_rf_v_to_x0(
            video_rows[update],
            noise_pred.float()[update],
            torch.tensor(schedule["t_video"], dtype=torch.float32, device=device),
        )
        new_video = minimax_h3_euler_eta0_step(
            video_rows[update],
            x0_video,
            sigma_curr=schedule["sigma_video"],
            sigma_next=schedule["sigma_video_next"],
        )
        video_rows = video_rows.clone()
        video_rows[update] = new_video
        if cond_anchor is not None:
            video_rows[~update] = cond_anchor  # per-step imgvid cond reset

        x0_audio = minimax_h3_rf_v_to_x0(
            audio_rows[audio_update],
            audio_noise_pred.float()[audio_update],
            torch.tensor(schedule["t_audio"], dtype=torch.float32, device=device),
        )
        new_audio = minimax_h3_euler_eta0_step(
            audio_rows[audio_update],
            x0_audio,
            sigma_curr=schedule["sigma_audio"],
            sigma_next=schedule["sigma_audio_next"],
        )
        audio_rows = audio_rows.clone()
        audio_rows[audio_update] = new_audio
        if audio_anchor is not None:
            audio_rows[~audio_update] = audio_anchor  # per-step audio ref reset

        state.latents = video_rows
        state.extra[_STEP_AUDIO_ROWS] = audio_rows
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Unpack the denoised rows and run the joint video/audio VAE decode."""
        del kwargs
        shape = state.extra[_STEP_SHAPE]
        video_latent, audio_latent = self._unpack_denoised_rows(
            state.extra[_STEP_BRANCH],
            state.latents,
            state.extra[_STEP_AUDIO_ROWS],
            latent_t=shape["latent_t"],
            latent_h=shape["latent_h"],
            latent_w=shape["latent_w"],
            audio_t=shape["audio_t"],
        )
        video, audio = self.decode(
            video_latent,
            audio_latent,
            height=shape["height"],
            width=shape["width"],
        )
        return DiffusionOutput(
            output=(video, audio),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )


__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
]
