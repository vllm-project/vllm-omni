# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline for MiniMax H3 FL2VA and Ref2VA partitions."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar

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
    get_dit_group,
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
from vllm_omni.diffusion.offloader import OffloadPlan
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import (
    resolve_component_quant_config as _resolve_component_quant_config,
)
from vllm_omni.quantization.component_config import (
    resolve_encoder_quant_config as _resolve_encoder_quant_config,
)

from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .denoise_loop import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .encoder import MiniMaxH3Qwen3VLEncoder
from .minimax_h3_transformer import MiniMaxH3DiTModel
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .presentation import (
    minimax_h3_multi_image_presentation_ids,
    minimax_h3_multi_image_presentation_token_tags,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
    minimax_h3_text_only_ids,
)
from .quality_policy import MINIMAX_H3_GENERIC_CACHE_KEY, MiniMaxH3QualityPolicy
from .reference_video import (
    load_audio_file,
    load_video_audio,
    load_video_frames,
    prepare_reference_videos,
    sample_reference_video_frames,
    validate_reference_audio_files,
    validate_reference_audio_waveforms,
)
from .time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
    minimax_h3_time_shift_sigmas,
)
from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

logger = init_logger(__name__)

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
MINIMAX_H3_DOWNLOAD_PATTERNS = [
    "FL2VA/**",
    "Ref2VA/model_index.json",
    "Ref2VA/transformer/**",
]
MINIMAX_H3_TASK_DOWNLOAD_PATTERNS = {
    "fl2va": ["FL2VA/**"],
    "ref2va": ["Ref2VA/**"],
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
    task = str(task_type or "auto").lower()
    if task == "auto" and model is not None:
        path = Path(model)
        if path.is_dir() and path.name in {"FL2VA", "Ref2VA"} and (path / "model_index.json").is_file():
            return path.name.lower()
    if task in {"auto", "combined"}:
        return "combined"
    if task in {"t2va", "fl2va"}:
        return "fl2va"
    if task == "ref2va":
        return "ref2va"
    raise ValueError(f"MiniMax-H3 task_type must be one of auto, t2va, fl2va, or ref2va; got {task_type!r}")


def _resolve_minimax_h3_model_root(
    model: str,
    revision: str | None,
    partition: str,
) -> Path:
    path = Path(model)
    if path.is_dir():
        if path.name in {"FL2VA", "Ref2VA"} and (path / "model_index.json").is_file():
            return path.parent
        return path
    allow_patterns = (
        MINIMAX_H3_DOWNLOAD_PATTERNS if partition == "combined" else MINIMAX_H3_TASK_DOWNLOAD_PATTERNS[partition]
    )
    return Path(
        download_weights_from_hf_specific(
            model_name_or_path=model,
            cache_dir=None,
            allow_patterns=allow_patterns,
            revision=revision,
            require_all=True,
        )
    )


def _read_base_schedule(release: Mapping[str, Any]) -> DMD2SigmaSchedule | None:
    """Read a partition's distilled schedule. An absent key means legacy uniform."""
    return DMD2SigmaSchedule.from_metadata(release)


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


def _load_images(value: Any) -> list[Image.Image]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise OmniClientError("MiniMax H3 image input must not be empty")
        return [_load_image(item) for item in value]
    if isinstance(value, (str, os.PathLike)):
        file_size = os.path.getsize(value)
        if file_size > MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES:
            raise OmniClientError("MiniMax H3 reference image exceeds the 30 MiB size limit")
        with Image.open(value) as image:
            image_format = str(image.format or "").lower()
            if image_format and image_format not in MINIMAX_H3_REFERENCE_IMAGE_FORMATS:
                raise OmniClientError(
                    f"MiniMax H3 reference image must use JPG, JPEG, PNG, WEBP, HEIC, or HEIF, got {image.format}"
                )
            return [image.convert("RGB")]
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise OmniClientError(f"image tensor must be [C,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        return [Image.fromarray(array.clip(0, 255).astype(np.uint8)).convert("RGB")]
    raise OmniClientError(f"unsupported MiniMax H3 image input {type(value)!r}")


def _load_audio(value: Any) -> tuple[torch.Tensor, int]:
    if isinstance(value, (list, tuple)) and not (len(value) == 2 and isinstance(value[1], (int, np.integer))):
        audios = _load_audios(value)
        if len(audios) != 1:
            raise OmniClientError(f"MiniMax H3 expected one audio, got {len(audios)}")
        return audios[0]
    if isinstance(value, (str, os.PathLike)):
        return load_audio_file(str(value))
    if isinstance(value, tuple) and len(value) == 2:
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


def _resolve_minimax_h3_aspect_ratio(
    task: str,
    value: Any,
    image: Image.Image | None,
) -> float:
    """Resolve H3's task-specific ratio policy.

    T2VA must name one of the official ratios.  FL2VA always follows the
    first input image, even when a generic client sends ``aspect_ratio``.
    Ref2VA defaults to 16:9; ``adaptive``/``auto`` are retained as aliases
    for that default for compatibility with existing clients.
    """
    if task == "fl2va":
        if image is None:
            raise OmniClientError("fl2va requires an input image to resolve its aspect ratio")
        return float(image.width) / float(image.height)

    if value is None:
        if task == "t2va":
            raise OmniClientError("t2va requires an explicit aspect_ratio")
        return MINIMAX_H3_SUPPORTED_ASPECT_RATIOS["16:9"]

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"adaptive", "auto"}:
            if task == "t2va":
                raise OmniClientError("t2va requires an explicit named aspect_ratio, not adaptive")
            return MINIMAX_H3_SUPPORTED_ASPECT_RATIOS["16:9"]
        if normalized in MINIMAX_H3_SUPPORTED_ASPECT_RATIOS:
            return MINIMAX_H3_SUPPORTED_ASPECT_RATIOS[normalized]
        try:
            numeric_value = float(normalized)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(MINIMAX_H3_SUPPORTED_ASPECT_RATIOS)
            raise OmniClientError(f"MiniMax H3 aspect_ratio must be one of {supported}, got {value!r}") from exc
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric_value = float(value)
    else:
        raise OmniClientError(f"MiniMax H3 aspect_ratio must be a string ratio, got {value!r}")

    if not math.isfinite(numeric_value) or not any(
        math.isclose(numeric_value, ratio, rel_tol=0.0, abs_tol=1e-6)
        for ratio in MINIMAX_H3_SUPPORTED_ASPECT_RATIOS.values()
    ):
        supported = ", ".join(MINIMAX_H3_SUPPORTED_ASPECT_RATIOS)
        raise OmniClientError(f"MiniMax H3 aspect_ratio must be one of {supported}, got {value!r}")
    return numeric_value


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
    group = get_dit_group()
    return group, dist.get_rank(group), dist.get_world_size(group)


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


def _reference_image_shape(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"reference image aspect ratio must be in [0.4, 2.5], got {width}x{height}")
    if min(width, height) < 256 or max(width, height) > 5760:
        raise OmniClientError(f"reference image dimensions must be in [256, 5760] pixels, got {width}x{height}")
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    return (
        _align_multiple(
            width * scale,
            MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
        ),
        _align_multiple(
            height * scale,
            MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
        ),
    )


def _resolve_output_canvas(aspect_ratio: float, short_edge: int) -> tuple[int, int]:
    """Resolve the official H3 ratio/area policy to a 32-pixel canvas."""
    if not math.isfinite(float(aspect_ratio)) or float(aspect_ratio) <= 0:
        raise OmniClientError(f"MiniMax H3 canvas aspect ratio must be positive, got {aspect_ratio!r}")
    if short_edge != MINIMAX_H3_OUTPUT_SHORT_EDGE:
        raise OmniClientError(f"MiniMax H3 target.short_edge must be {MINIMAX_H3_OUTPUT_SHORT_EDGE}, got {short_edge}")
    if aspect_ratio >= 1.0:
        width = float(short_edge) * aspect_ratio
        height = float(short_edge)
    else:
        width = float(short_edge)
        height = float(short_edge) / aspect_ratio
    area = width * height
    if area > MINIMAX_H3_OUTPUT_MAX_PIXELS:
        scale = (MINIMAX_H3_OUTPUT_MAX_PIXELS / area) ** 0.5
        width *= scale
        height *= scale
    return (
        _align_multiple(height, 32),
        _align_multiple(width, 32),
    )


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
        "_encode_video_conditions",
        "_encode_video_audio_conditions",
        "_encode_audio_conditions",
        "diffuse",
        "decode",
    ]
    dummy_run_num_frames: ClassVar[int] = 0
    # Only distilled releases pin a schedule, so the default keeps the legacy
    # uniform path available to partially constructed pipelines.
    _base_schedule_by_partition: ClassVar[Mapping[str, DMD2SigmaSchedule | None]] = {}

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
        self.partition = _minimax_h3_partition_for_task(
            getattr(od_config, "task_type", None),
            str(od_config.model),
        )
        model_root = _resolve_minimax_h3_model_root(
            str(od_config.model),
            od_config.revision,
            self.partition,
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

        _, rank, dit_world = _dit_rank_world()
        self._dit_rank = rank
        text_encoder_tp_size = int(getattr(self.parallel_config, "text_encoder_tp_size", 1))
        if text_encoder_tp_size < 1:
            raise ValueError(f"text_encoder_tp_size must be >= 1, got {text_encoder_tp_size}")
        if text_encoder_tp_size > dit_world:
            raise ValueError(
                f"text_encoder_tp_size must not exceed the DiT group size ({dit_world}), got {text_encoder_tp_size}"
            )
        # The Qwen3-VL text model uses 64 attention heads / 8 KV heads; the
        # encoder shards them across the encoder TP ranks.
        if 64 % text_encoder_tp_size or 8 % text_encoder_tp_size:
            raise ValueError(
                "text_encoder_tp_size must divide both Qwen3-VL "
                f"num_attention_heads (64) and num_key_value_heads (8), "
                f"got {text_encoder_tp_size}"
            )
        self.text_encoder_tp_size = text_encoder_tp_size
        self.text_encoder_group = self._build_text_encoder_group(text_encoder_tp_size)
        load_text_encoder = rank < text_encoder_tp_size
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            os.path.join(model_path, "text_encoder"),
            device=self.device,
            load_model=load_text_encoder,
            encoder_group=self.text_encoder_group,
            quant_config=_resolve_minimax_h3_text_encoder_quant_config(od_config.quantization_config),
        )
        if load_text_encoder:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=str(model_path),
                    subfolder="text_encoder",
                    revision=od_config.revision,
                    prefix="text_encoder.",
                    fall_back_to_pt=False,
                )
            )
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
                    ids = minimax_h3_multi_image_presentation_ids(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=image_token_counts,
                    )
                    tags = minimax_h3_multi_image_presentation_token_tags(
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
        if self.od_config.enable_cpu_offload and not getattr(
            self.od_config, "enable_distributed_layerwise_offload", False
        ):
            # Invoke nn.Module.__call__ so the generic model-level offloader
            # swaps the resident DiT and encoder.
            return self.text_encoder(input_ids, **vision_kwargs)

        if self.od_config.enable_layerwise_offload or getattr(
            self.od_config, "enable_distributed_layerwise_offload", False
        ):
            # Layerwise DiT offload already provides the low-residency encoder
            # phase used by the checkpoint reference.
            self.text_encoder.load_to_device()
            try:
                return self.text_encoder.encode_ids(input_ids, **vision_kwargs)
            finally:
                self.text_encoder.offload_to_cpu()

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

    @contextmanager
    def _component_on_device(self, component: nn.Module):
        staged = self._uses_manual_component_offload()
        if staged:
            component.load_to_device()
        try:
            yield
        finally:
            if staged:
                component.offload_to_cpu()

    def _encode_visual_condition(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        _, rank, _ = _dit_rank_world()
        rows = None
        if rank == 0:
            with self._component_on_device(self.video_vae):
                rows = self.video_vae.encode_image(image)
        return _broadcast_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )

    def _encode_visual_conditions(
        self,
        images: list[Image.Image],
        prepared_videos: list[dict[str, Any]] | None,
        *,
        video_count: int,
    ) -> tuple[torch.Tensor | None, list[tuple[int, int, int]]]:
        rows: list[torch.Tensor] = []
        shapes: list[tuple[int, int, int]] = []
        for image in images:
            rows.append(self._encode_visual_condition(image))
            shapes.append((1, image.height // 16, image.width // 16))
        if video_count:
            video_rows, video_shapes = self._encode_video_conditions(
                prepared_videos,
                count=video_count,
            )
            rows.append(video_rows)
            shapes.extend(video_shapes)
        return (torch.cat(rows) if rows else None), shapes

    def _encode_audio_condition(
        self,
        audio: tuple[torch.Tensor, int],
    ) -> tuple[torch.Tensor, int]:
        _, rank, _ = _dit_rank_world()
        rows = None
        audio_t = 0
        if rank == 0:
            with self._component_on_device(self.audio_vae):
                rows, audio_t = self.audio_vae.encode_waveform(*audio)
        audio_t_tensor = torch.tensor(
            [audio_t],
            dtype=torch.long,
            device=self.device,
        )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(audio_t_tensor, src=0, group=group)
        rows = _broadcast_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )
        return rows, int(audio_t_tensor.item())

    def _encode_audio_conditions(
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

    def _encode_video_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        count: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        with self._component_on_device(self.video_vae):
            return self._encode_video_conditions_resident(prepared_videos, count=count)

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

    def _encode_video_audio_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
    ) -> tuple[torch.Tensor | None, list[int]]:
        with self._component_on_device(self.audio_vae):
            return self._encode_video_audio_conditions_resident(prepared_videos, has_audio=has_audio)

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
        transformer = self._transformer_for_task(task)
        # The static DLO plan keeps leading blocks resident only for the
        # primary ``transformer``. In combined mode ``transformers_ref`` is
        # fully streamed, so a Ref2VA request must not stage the inactive
        # FL2VA transformer resident blocks.
        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):
            with self.progress_bar(total=len(video_sigmas) - 1) as progress:
                video_rows, audio_rows = minimax_h3_denoise_loop(
                    model=transformer,
                    positive=branch,
                    initial_video_rows=initial_video,
                    initial_audio_rows=initial_audio,
                    keyframe_cond_rows=visual_anchor,
                    audio_ref_rows=audio_anchor,
                    sigmas_video=video_sigmas,
                    sigmas_audio=audio_sigmas,
                    device=self.device,
                    imgvid_cond_noise_aug_for_inference=(MINIMAX_H3_IMGVID_COND_TIMESTEP),
                    audio_cond_noise_aug_for_inference=(MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                    on_step_start=lambda step, video_sigma, audio_sigma: self.record_denoise_step(
                        step,
                        normalized_timestep=video_sigma,
                    ),
                    on_step_end=lambda step, video, audio: progress.update(),
                )

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

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise OmniClientError("MiniMax H3 supports one request at a time")
        raw_prompt = request.prompts[0]
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
            multi_modal_data: dict[str, Any] = {}
        else:
            prompt = str(raw_prompt.get("prompt") or "")
            multi_modal_data = raw_prompt.get("multi_modal_data") or {}
        if not prompt:
            raise OmniClientError("MiniMax H3 requires a non-empty prompt")

        sampling = request.sampling_params
        quality = sampling.quality
        logger.debug("MiniMax H3 request quality=%s", quality)
        extra = sampling.extra_args or {}
        task = self._resolve_task(extra.get("task"), multi_modal_data)

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
                prepared_videos = self._prepare_reference_videos(
                    raw_videos,
                    target_frame_count=num_frames,
                    workdir=workdir,
                    start_time_seconds=extra.get("start_time_seconds"),
                )
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
                        group=get_dit_group(),
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

            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                images=prepared_images,
                prepared_videos=prepared_videos,
                condition_labels=condition_labels if task == "ref2va" else None,
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
                embedded_audio_condition, embedded_audio_lengths = self._encode_video_audio_conditions(
                    prepared_videos,
                    has_audio=has_audio,
                )
                external_audio_condition, external_audio_lengths = self._encode_audio_conditions(
                    standalone_audios,
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
        sigma_schedule = self._base_schedule_for_task(task)
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
        videos = []
        audios = []
        for output_seed in _minimax_h3_output_seeds(seed, num_outputs):
            video_latent, audio_latent = self.diffuse(
                task=task,
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                seed=output_seed,
                latent_t=latent_t,
                latent_h=height // 16,
                latent_w=width // 16,
                audio_t=audio_t,
                num_frames=num_frames,
                num_steps=num_steps,
                video_shift=video_shift,
                audio_shift=audio_shift,
                base_schedule=base_schedule,
                visual_condition=visual_condition,
                visual_condition_shape=visual_shape,
                audio_condition=audio_condition,
                ref_audio_t=ref_audio_t,
                ref_blocks=ref_blocks,
                visual_condition_shapes=visual_shapes,
                audio_condition_lengths=audio_lengths,
                keyframe_frame_indices=keyframe_frame_indices,
            )
            video, audio = self.decode(video_latent, audio_latent, height=height, width=width)
            videos.append(video)
            audios.append(audio)
        video = torch.cat(videos, dim=0)
        audio = torch.cat(audios, dim=0)
        return DiffusionOutput(
            output=(video, audio),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )


__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
]
