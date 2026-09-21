# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-scoped LingBot-World v2 causal DMD pipeline."""

from __future__ import annotations

import functools
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T, WanAttentionBlock, WanResample, WanResidualBlock
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.camera import CameraSession
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery, SupportsStepExecution
from vllm_omni.diffusion.models.lingbot_world.actions import (
    LINGBOT_CONTROLLER_TRANSLATION_UNIT,
    LingBotCameraActionFrames,
    LingBotCameraActionScript,
    as_camera_action_script,
    camera_trajectory_from_absolute_pose,
    integrate_lingbot_camera_actions,
    parse_lingbot_camera_action_script,
)
from vllm_omni.diffusion.models.lingbot_world.camera import (
    CameraTrajectory,
    build_plucker_embedding,
    interpolate_camera_trajectory,
    load_camera_trajectory,
    resolve_trusted_action_directory,
)
from vllm_omni.diffusion.models.lingbot_world.dmd_block import ARBlockContext, LingBotDMDBlockRunner
from vllm_omni.diffusion.models.lingbot_world.transformer import (
    CameraModulationCache,
    CausalLingBotWorldTransformer3DModel,
    LingBotAttentionCache,
    LingBotTransformerCache,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config, retrieve_latents
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionChunkMetadata,
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.streaming_decode import (
    StreamingDecodeState,
    WanStreamingDecoder,
)

logger = init_logger(__name__)

# Resident streaming-decoder cache per session, as bytes per output pixel in
# fp32: measured once on the reference checkpoint at 37832 KiB for a 64x64
# output. Every cached tensor is [B, C, <= CACHE_T, H/8, W/8] per causal
# convolution, so the total scales with height * width and never with session
# length; tests/diffusion/ar_diffusion/test_streaming_decode.py pins that
# scaling, and at 832x480 in bf16 this yields the 1801 MiB the decoder was
# measured to hold.
_STREAMING_DECODE_BYTES_PER_PIXEL_FP32 = 37832 * 1024 / (64 * 64)

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmProgressBar

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import (
        ARDiffusionKVState,
    )

LINGBOT_DMD_TIMESTEPS = (1000, 750, 500, 250)
_CAMERA_SPATIAL_FOLD = 8
_MAX_PIXEL_AREA = 480 * 832
_MAX_SOURCE_IMAGE_PIXELS = 4096 * 4096
_MAX_SEQUENCE_LENGTH = 512
# Distinct prompts whose text encodes are kept. One entry is a single padded
# sequence -- a few MiB at 512 tokens -- so a handful covers a session's prompt
# plus the switches it makes, at no meaningful cost next to its KV.
_PROMPT_EMBEDS_CACHE_SIZE = 4
_ACTION_ROOT_ENV = "VLLM_OMNI_LINGBOT_ACTION_ROOT"
_PREPROCESSED_CAMERA_KEY = "_lingbot_camera_trajectory"
_PREPROCESSED_CAMERA_ACTION_SCRIPT_KEY = "_lingbot_camera_action_script"
_SOURCE_IMAGE_ERROR = (
    "Unable to load multi_modal_data.image; expected a decodable image within 4096 * 4096 source pixels."
)


@dataclass(frozen=True)
class _LingBotRequestInputs:
    """Validated model-specific request boundary.

    Pre-processing materializes at most one of ``camera_trajectory`` and
    ``camera_action_script``, chosen from the shape of the incoming control.
    They are not alternative spellings of one input: ``camera_action_script``
    is a *container* whose element is a ``camera_actions``, and the two are
    live at once during step execution.

    - ``camera_trajectory`` is explicit per-frame poses and intrinsics. Offline
      replay loads it from ``extra_args["action_path"]``. Integrating WASD
      actions also produces one, written back onto this field.
    - ``camera_actions`` is WASD key state for *one* three-latent-frame block:
      the unit the camera is embedded in. Pre-processing never fills it; step
      execution derives it per block by slicing ``camera_action_script``.
      ``_prepare_camera`` also reads it as the signal that the trajectory it
      was handed is already latent-aligned.
    - ``camera_action_script`` holds one such action list *per generated chunk*
      for the whole rollout, which only step execution needs, because only it
      runs every block inside a single ``generate()``. ``forward()`` rejects a
      request that carries it.
    """

    prompt: str
    image: PIL.Image.Image | torch.Tensor
    camera_trajectory: CameraTrajectory | None
    camera_actions: LingBotCameraActionFrames | None
    camera_action_script: LingBotCameraActionScript | None
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    # Request mode returns pixels through VideoProcessor in this type. Step execution (realtime sessions)
    # returns every pixel type as ``(F, H, W, 3)`` uint8 frames, the layout the streaming endpoint muxes; only
    # ``"latent"`` differs there. See ``_decode_chunk_to_pixels``.
    output_type: str
    max_sequence_length: int
    flow_shift: float
    generator: torch.Generator


@dataclass
class _LingBotARSessionState:
    """Model-owned state; attention tensors remain runner-owned."""

    next_chunk_index: int = 0
    encoder_cache: list[torch.Tensor | None] | None = None
    pending_encoder_cache: list[torch.Tensor | None] | None = None
    # The condition the encoder settles on once its history has stopped
    # changing; see ``_next_condition_chunk``.
    condition_fixed_point: torch.Tensor | None = None


def _positive_finite_flow_shift(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("flow_shift must be a positive finite number.")
    try:
        flow_shift = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("flow_shift must be a positive finite number.") from exc
    if not math.isfinite(flow_shift) or flow_shift <= 0:
        raise ValueError("flow_shift must be a positive finite number.")
    return flow_shift


def _build_shifted_flow_schedule(
    *,
    flow_shift: float,
) -> tuple[tuple[float, float], ...]:
    """Map checkpoint DMD labels to request-local warped timestep/sigma pairs."""

    # The checkpoint labels index the unshifted [1, ..., 1 / N] training
    # lattice. Apply the request's flow shift before using the value both as
    # the transformer timestep and the sampling sigma.
    base_sigmas = torch.tensor(LINGBOT_DMD_TIMESTEPS, dtype=torch.float64) / 1000
    shifted_numerators = flow_shift * base_sigmas
    shifted_sigmas = shifted_numerators / ((1.0 - base_sigmas) + shifted_numerators)
    warped_timesteps = shifted_sigmas * 1000
    return tuple(
        (float(timestep), float(sigma))
        for timestep, sigma in zip(warped_timesteps.tolist(), shifted_sigmas.tolist(), strict=True)
    )


def _validate_scheduler_config(config: dict[str, Any]) -> None:
    contract = {
        "_class_name": "UniPCMultistepScheduler",
        "num_train_timesteps": 1000,
        "prediction_type": "flow_prediction",
        "predict_x0": True,
        "use_flow_sigmas": True,
        "use_dynamic_shifting": False,
        "use_beta_sigmas": False,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
        "final_sigmas_type": "zero",
        "timestep_spacing": "linspace",
    }
    for name, expected in contract.items():
        if config.get(name) != expected:
            raise ValueError(f"LingBot scheduler config {name} must be {expected!r}, got {config.get(name)!r}.")


def _validate_parallel_config(od_config: OmniDiffusionConfig) -> None:
    if getattr(od_config, "quantization_config", None) is not None:
        raise NotImplementedError("LingBot World v1 does not support quantization.")
    parallel_config = getattr(od_config, "parallel_config", None)
    if parallel_config is None:
        return
    unsupported_sizes = {
        "pipeline_parallel_size": "pipeline parallelism",
        "cfg_parallel_size": "CFG parallelism",
        "vae_patch_parallel_size": "VAE parallelism",
    }
    for field, feature in unsupported_sizes.items():
        size = getattr(parallel_config, field, 1) or 1
        if size > 1:
            raise NotImplementedError(f"LingBot World v1 does not support {feature} ({field}={size}).")
    sequence_parallel_size = getattr(parallel_config, "sequence_parallel_size", 1) or 1
    ulysses_degree = getattr(parallel_config, "ulysses_degree", 1) or 1
    ring_degree = getattr(parallel_config, "ring_degree", 1) or 1
    allgather_degree = getattr(parallel_config, "allgather_degree", 1) or 1
    ulysses_mode = getattr(parallel_config, "ulysses_mode", "strict")
    ulysses_a2a_permute = bool(getattr(parallel_config, "ulysses_a2a_permute", False))
    if (
        sequence_parallel_size != ulysses_degree
        or ring_degree != 1
        or allgather_degree != 1
        or ulysses_mode != "strict"
        or ulysses_a2a_permute
    ):
        raise NotImplementedError(
            "LingBot World sequence parallelism requires pure Ulysses with ulysses_a2a_permute disabled: "
            f"sequence_parallel_size={sequence_parallel_size}, ulysses_degree={ulysses_degree}, "
            f"ring_degree={ring_degree}, allgather_degree={allgather_degree}, ulysses_mode={ulysses_mode!r}, "
            f"ulysses_a2a_permute={ulysses_a2a_permute}."
        )
    if getattr(parallel_config, "use_hsdp", False):
        raise NotImplementedError("LingBot World v1 does not support HSDP.")
    if getattr(parallel_config, "enable_expert_parallel", False):
        raise NotImplementedError("LingBot World v1 does not support expert parallelism.")


def _validate_source_image_size(image: PIL.Image.Image) -> None:
    width, height = image.size
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or width <= 0
        or isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("source image width and height must be positive integers.")
    if width * height > _MAX_SOURCE_IMAGE_PIXELS:
        raise ValueError("source image pixel count must not exceed 4096 * 4096.")


def _decode_source_image(image: PIL.Image.Image) -> PIL.Image.Image:
    _validate_source_image_size(image)
    try:
        return image.convert("RGB")
    except (OSError, SyntaxError, ValueError, PIL.Image.DecompressionBombError):
        raise ValueError(_SOURCE_IMAGE_ERROR) from None


def _source_image_fingerprint(path: str | os.PathLike[str]) -> tuple[str, int, int, int, int, int]:
    """Identify the bytes behind ``path``: the resolved path plus the stat fields that change when it is rewritten."""
    resolved = os.path.realpath(os.fspath(path))
    stat = os.stat(resolved)
    return resolved, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _encoder_caches_identical(
    before: list[torch.Tensor | None] | None, after: list[torch.Tensor | None] | None
) -> bool:
    """Whether two encoder histories hold the same tensors bit for bit, slot by slot."""
    if before is None or after is None or len(before) != len(after):
        return False
    for x, y in zip(before, after, strict=True):
        if x is None or y is None:
            if x is not y:
                return False
        elif x.shape != y.shape or x.dtype != y.dtype or not torch.equal(x, y):
            return False
    return True


def _uint8_frames(video: torch.Tensor) -> np.ndarray:
    """Turn one decoded block ``(1, 3, F, H, W)`` in ``[-1, 1]`` into ``(F, H, W, 3)`` uint8 on the host.

    This is the arithmetic ``VideoProcessor.postprocess_video`` applies for
    ``"np"`` and ``"pil"`` -- denormalise, widen to float32, scale by 255 and
    round -- done on the device before the copy instead of on the host after
    it. The bytes are identical to the PIL path; what changes is that the
    frames cross to the host once, already in the layout and dtype the
    streaming endpoint muxes, instead of as a float32 copy that is then
    converted per frame while the next block waits.
    """
    # Same order of operations as VideoProcessor: denormalise and clamp in
    # the decoder dtype, then widen, scale and round.
    frames = (video[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0)
    frames = frames.float().mul_(255).round_().to(torch.uint8)
    return np.ascontiguousarray(frames.cpu().numpy())


@functools.lru_cache(maxsize=1)
def _decode_source_image_fingerprinted(fingerprint: tuple[str, int, int, int, int, int]) -> PIL.Image.Image:
    """The decoded image behind ``fingerprint`` (resolved path plus the stat fields that change on rewrite).

    One entry: a request-mode caller that replays the same source path decodes it once. The realtime tick
    carries its image inline and never comes through here.
    """
    return _decode_source_image_file(fingerprint[0])


def _load_source_image(path: str | os.PathLike[str]) -> PIL.Image.Image:
    """Decode the source image at ``path``, reusing the last decode while the file is unchanged.

    Every caller gets its own copy, so a request that mutates its image cannot reach the next one. A path
    that cannot be stat'ed goes straight to the uncached decode, which owns the validation error contract.
    """
    try:
        fingerprint = _source_image_fingerprint(path)
    except (OSError, TypeError, ValueError):
        return _decode_source_image_file(path)
    return _decode_source_image_fingerprinted(fingerprint).copy()


def _decode_source_image_file(path: str | os.PathLike[str]) -> PIL.Image.Image:
    try:
        source_image = PIL.Image.open(path)
    except (OSError, SyntaxError, ValueError, PIL.Image.DecompressionBombError):
        raise ValueError(_SOURCE_IMAGE_ERROR) from None
    try:
        return _decode_source_image(source_image)
    finally:
        source_image.close()


def get_lingbot_world_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    """Materialize request-local files once before dispatching GPU workers."""

    model_config = getattr(od_config, "model_config", None) or {}
    configured_action_root = model_config.get("lingbot_action_root") or os.environ.get(_ACTION_ROOT_ENV)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if not isinstance(prompt, dict):
            raise ValueError("LingBot World requires a prompt mapping containing multi_modal_data.image.")
        multi_modal_data = prompt.get("multi_modal_data") or {}
        additional_information = prompt.get("additional_information") or {}
        if not isinstance(multi_modal_data, dict):
            raise ValueError("prompt.multi_modal_data must be a mapping containing image.")
        if not isinstance(additional_information, dict):
            raise ValueError("prompt.additional_information must be a mapping.")
        if "action_path" in additional_information:
            raise ValueError(
                "prompt.additional_information.action_path is not supported; use "
                "sampling_params.extra_args.action_path."
            )

        image = multi_modal_data.get("image")
        if isinstance(image, list):
            raise ValueError("LingBot World requires one image and does not accept an image list.")
        if image is None:
            raise ValueError("LingBot World requires exactly one image in multi_modal_data.image.")
        if isinstance(image, (str, os.PathLike)):
            image = _load_source_image(image)
        elif isinstance(image, PIL.Image.Image):
            image = _decode_source_image(image)
        elif not isinstance(image, torch.Tensor):
            raise ValueError("multi_modal_data.image must be a PIL image, tensor, or file path.")

        extra_args = getattr(request.sampling_params, "extra_args", None) or {}
        if not isinstance(extra_args, dict):
            raise ValueError("sampling_params.extra_args must be a mapping.")
        if extra_args.get("camera_action_script") is not None:
            if extra_args.get("action_path"):
                raise ValueError("camera_action_script cannot be combined with action_path.")
            camera_action_script = parse_lingbot_camera_action_script(
                extra_args.get("camera_action_script"),
                frames_per_chunk=3,
            )
            trajectory = None
        else:
            action_path = extra_args.get("action_path")
            if action_path is None or action_path == "":
                # Stepwise / streaming sessions may omit a request-scoped camera
                # script and instead drive motion via mid-generation camera
                # interaction (idle hold until the first camera event).
                if getattr(od_config, "step_execution", False) or getattr(od_config, "streaming_output", False):
                    trajectory = None
                    camera_action_script = None
                else:
                    raise ValueError("action_path is required in sampling_params.extra_args.action_path.")
            else:
                if not isinstance(action_path, (str, os.PathLike)) or not str(action_path):
                    raise ValueError("action_path is required in sampling_params.extra_args.action_path.")
                if not configured_action_root:
                    raise ValueError(
                        "sampling_params.extra_args.action_path requires a trusted action root configured by "
                        f"model_config.lingbot_action_root or {_ACTION_ROOT_ENV}."
                    )
                action_directory = resolve_trusted_action_directory(
                    action_path,
                    configured_action_root,
                )
                try:
                    trajectory = load_camera_trajectory(action_directory)
                except OSError:
                    raise ValueError(
                        "Unable to load camera trajectory from action_path; expected poses.npy and intrinsics.npy."
                    ) from None
                camera_action_script = None

        updated_prompt = dict(prompt)
        updated_multi_modal_data = dict(multi_modal_data)
        updated_multi_modal_data["image"] = image
        updated_prompt["multi_modal_data"] = updated_multi_modal_data
        request.prompt = updated_prompt
        request.sampling_params.extra_args = {
            **extra_args,
            _PREPROCESSED_CAMERA_KEY: trajectory,
            _PREPROCESSED_CAMERA_ACTION_SCRIPT_KEY: camera_action_script,
        }
        return request

    return pre_process_func


def _fold_camera_embedding(
    camera_embedding: torch.Tensor,
) -> torch.Tensor:
    """Pixel-unshuffle ``[frames, 6, H, W]`` onto the Wan latent grid."""

    folded = F.pixel_unshuffle(camera_embedding, _CAMERA_SPATIAL_FOLD)
    return folded.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def get_lingbot_world_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., Any]:
    del od_config
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params: Any | None = None,
    ) -> Any:
        if isinstance(video, dict) and isinstance(video.get("payload"), dict):
            return video
        if sampling_params is not None:
            output_type = getattr(sampling_params, "output_type", None) or output_type
        if output_type == "latent":
            return video
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {},
        }

    return post_process_func


class LingBotWorldCausalDMDPipeline(
    nn.Module,
    SupportImageInput,
    SupportsComponentDiscovery,
    SupportsStepExecution,
    InteractionMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """LingBot-World v2 I2V generation with a request-local causal cache."""

    supports_step_execution: ClassVar[bool] = True
    # A realtime block is four probes and a commit over one prepared chunk
    # state; nothing between those steps needs the serving scheduler.
    supports_chunk_step_grouping: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    # Generic warmup cannot synthesize the required camera action directory.
    dummy_run_num_frames: ClassVar[int] = 0
    _AR_BRANCH = "main"
    _AR_TEXT_CACHE = "text"

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        del prefix
        _validate_parallel_config(od_config)
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)
        managed_component_placement = bool(
            getattr(od_config, "enable_cpu_offload", False) or getattr(od_config, "enable_layerwise_offload", False)
        )

        # Standard components use from_pretrained; the custom transformer uses the loader.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only)
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        if not managed_component_placement:
            self.text_encoder = self.text_encoder.to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        if not managed_component_placement:
            self.vae = self.vae.to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = CausalLingBotWorldTransformer3DModel.from_config(
            transformer_config,
            quant_config=getattr(od_config, "quantization_config", None),
            prefix="transformer",
        )

        scheduler_config = _load_json(model, "scheduler/scheduler_config.json", local_files_only)
        _validate_scheduler_config(scheduler_config)
        configured_shift = getattr(od_config, "flow_shift", None)
        checkpoint_shift = scheduler_config.get("flow_shift", scheduler_config.get("shift", 5.0))
        scheduler_shift = _positive_finite_flow_shift(
            checkpoint_shift if configured_shift is None else configured_shift
        )
        scheduler_keys = {
            "num_train_timesteps",
            "solver_order",
            "prediction_type",
            "use_dynamic_shifting",
            "thresholding",
            "dynamic_thresholding_ratio",
            "sample_max_value",
            "predict_x0",
            "solver_type",
            "lower_order_final",
            "disable_corrector",
            "timestep_spacing",
            "steps_offset",
            "final_sigmas_type",
        }
        scheduler_kwargs = {name: value for name, value in scheduler_config.items() if name in scheduler_keys}
        scheduler_kwargs["shift"] = scheduler_shift
        self.scheduler = FlowUniPCMultistepScheduler(**scheduler_kwargs)

        self.vae_scale_factor_temporal = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        self.vae_scale_factor_spatial = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        model_config = getattr(od_config, "model_config", None) or {}
        self._ar_height = int(model_config.get("ar_diffusion_height", 480))
        self._ar_width = int(model_config.get("ar_diffusion_width", 832))
        self._ar_diffusion_kv_state: ARDiffusionKVState | None = None
        self._ar_sessions: dict[str, _LingBotARSessionState] = {}
        # Text encodes keyed by what they depend on; see encode_prompt.
        self._prompt_embeds_cache: OrderedDict[tuple[str, int, torch.dtype], torch.Tensor] = OrderedDict()
        # One temporal decoder cache per stepwise session, keyed the way the
        # runner keys AR sessions (session_id == request_id) so both are
        # released together rather than through two independent lifecycles.
        self._streaming_decode_states: dict[str, StreamingDecodeState] = {}
        self._cached_streaming_decoder: WanStreamingDecoder | None = None
        self._streaming_decode_unsupported = False
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=[
                "vae.encode",
                "_prepare_condition_chunk",
                "vae.decode",
                # Streaming decode never reaches vae.decode, so it needs its
                # own stage or a streamed rollout reports no decode time at
                # all -- the one stage this path changes.
                "_streaming_decode_chunk",
                "_generate_block",
                "text_encoder.forward",
                "tokenizer.forward",
            ],
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Describe the fixed worker-local cache geometry for step execution."""
        patch_frames, patch_height, patch_width = self.transformer.config.patch_size
        if patch_frames != 1:
            raise ValueError("LingBot AR-Diffusion requires temporal patch size 1.")
        spatial = self.vae_scale_factor_spatial
        if self._ar_height % (spatial * patch_height) or self._ar_width % (spatial * patch_width):
            raise ValueError("ar_diffusion_height/width must align with VAE and DiT patch sizes.")
        latent_height = self._ar_height // spatial
        latent_width = self._ar_width // spatial
        tokens_per_frame = (latent_height // patch_height) * (latent_width // patch_width)
        total_window_frames = (
            int(self.transformer.config.local_attn_size)
            if int(self.transformer.config.local_attn_size) != -1
            else int(self.transformer.config.sliding_window_num_frames)
        )
        sink_frames = int(self.transformer.config.sink_size)
        recent_window_frames = total_window_frames - sink_frames
        if recent_window_frames <= 0:
            raise ValueError("LingBot AR-Diffusion cache needs a positive recent window after reserving sink frames.")
        condition_latent_frames = int(self.transformer.config.num_frames_per_block)
        condition_channels = self.vae_scale_factor_temporal + int(self.transformer.config.out_channels)
        element_size = torch.empty((), dtype=self.transformer.dtype).element_size()
        condition_bytes_per_session = (
            condition_channels * condition_latent_frames * latent_height * latent_width * element_size
        )
        # Each layer retains scale and shift for the rank-local camera tokens
        # across all forwards of a block, including between scheduler calls.
        camera_tokens = tokens_per_frame * condition_latent_frames
        sp_size = int(self.od_config.parallel_config.ulysses_degree)
        camera_bytes_per_session = (
            2
            * int(self.transformer.config.num_layers)
            * ((camera_tokens + sp_size - 1) // sp_size)
            * int(self.transformer.config.num_attention_heads)
            * int(self.transformer.config.attention_head_dim)
            * element_size
        )
        return ARDiffusionKVCacheSpec(
            num_layers=int(self.transformer.config.num_layers),
            num_kv_heads=int(self.transformer.blocks[0].self_attn.num_sp_heads),
            head_size=int(self.transformer.config.attention_head_dim),
            tokens_per_frame=tokens_per_frame,
            frames_per_block=int(self.transformer.config.num_frames_per_block),
            window_frames=recent_window_frames,
            sink_frames=sink_frames,
            kv_branches=(ARDiffusionKVBranchSpec(self._AR_BRANCH, 0),),
            session_capacity=2,
            cross_attention=(
                ARDiffusionCrossAttentionKVSpec(
                    self._AR_TEXT_CACHE,
                    _MAX_SEQUENCE_LENGTH,
                    # Cross-attention keeps every local head on every rank rather than sharding heads across
                    # the Ulysses group, so the text pool holds that many heads, not the self-attention share.
                    num_kv_heads=int(self.transformer.blocks[0].cross_attn.num_local_heads),
                ),
            ),
            model_owned_state_bytes_per_session=(
                condition_bytes_per_session
                + camera_bytes_per_session
                + 2 * self._condition_encoder_cache_bytes()  # Committed and in-flight histories.
                + self._streaming_decode_bytes_per_session()
            ),
        )

    def _condition_encoder_cache_bytes(self) -> int:
        """Bound one Wan encoder history from its per-convolution input grids."""
        if getattr(self.vae.config, "patch_size", None) is not None:
            raise ValueError("Stateful LingBot conditioning requires an unpatched Wan VAE.")
        encoder = self.vae.encoder
        height, width = self._ar_height, self._ar_width
        elements = CACHE_T * encoder.conv_in.in_channels * height * width
        for layer in encoder.down_blocks:
            if isinstance(layer, WanResidualBlock):
                elements += CACHE_T * (layer.conv1.in_channels + layer.conv2.in_channels) * height * width
            elif isinstance(layer, WanResample) and layer.mode in ("downsample2d", "downsample3d"):
                height, width = height // 2, width // 2
                if layer.mode == "downsample3d":
                    # Temporal downsampling retains one frame after spatial downsampling.
                    elements += layer.time_conv.in_channels * height * width
            elif not isinstance(layer, WanAttentionBlock):
                raise ValueError("Stateful LingBot conditioning requires the Wan encoder cache layout.")
        for layer in encoder.mid_block.resnets:
            elements += CACHE_T * (layer.conv1.in_channels + layer.conv2.in_channels) * height * width
        elements += CACHE_T * encoder.conv_out.in_channels * height * width
        return elements * torch.empty((), dtype=self.vae.dtype).element_size()

    def _streaming_decode_bytes_per_session(self) -> int:
        """Resident streaming-decoder bytes one session can hold, for admission.

        ``model_owned_state_bytes_per_session`` is what the runner subtracts
        from the KV budget before sizing the block pools, and this cache is
        three orders of magnitude larger than the image condition that used to
        be the only entry: leaving it out lets the profiler report a
        configuration as fitting and then OOM partway through the second
        session's rollout.

        Declared for every session rather than only for pixel-output ones: the
        field is documented as a conservative upper bound, it is fixed at load
        time, and whether a request asks for pixels is not known then.
        """
        decoder = self._streaming_decoder()
        if decoder is None:
            return 0
        return decoder.declared_state_bytes(
            height=self._ar_height,
            width=self._ar_width,
            dtype=self.vae.dtype,
        )

    @contextmanager
    def bind_ar_diffusion_state(
        self,
        session_id: str,
        state: ARDiffusionKVState,
    ) -> Iterator[None]:
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("LingBot AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"LingBot bound session mismatch: {state.session_id!r} != {session_id!r}.")
        self._ar_diffusion_kv_state = state
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None

    def _release_condition_encoder_state(self, session_id: str) -> None:
        state = self._ar_sessions.pop(session_id, None)
        if state is not None:
            state.encoder_cache = None
            state.pending_encoder_cache = None
            state.condition_fixed_point = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._release_condition_encoder_state(session_id)
        self._release_streaming_decode_state(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._release_condition_encoder_state(session_id)
        self._release_streaming_decode_state(session_id)

    def _parse_request(self, req: DiffusionRequestBatch) -> _LingBotRequestInputs:
        if req.num_reqs != 1 or len(req.prompts) != 1:
            raise ValueError("LingBot World supports a single prompt request, not request batching.")
        sampling = req.sampling_params
        if int(sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("LingBot World requires num_outputs_per_prompt=1.")
        generator = getattr(sampling, "generator", None)
        if isinstance(generator, list):
            raise ValueError("LingBot World accepts one torch.Generator, not a generator list.")
        if not isinstance(generator, torch.Generator):
            raise ValueError("LingBot World requires the runner-provided torch.Generator.")
        if getattr(sampling, "latents", None) is not None:
            raise ValueError("LingBot World does not support caller-provided latents.")

        prompt_value = req.prompts[0]
        multi_modal_data: dict[str, Any]
        if isinstance(prompt_value, str):
            prompt = prompt_value
            multi_modal_data = {}
        elif isinstance(prompt_value, dict):
            prompt = prompt_value.get("prompt") or ""
            multi_modal_data = prompt_value.get("multi_modal_data") or {}
        else:
            raise ValueError("prompt must be a string or prompt mapping.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must contain non-empty text.")
        if not isinstance(multi_modal_data, dict):
            raise ValueError("prompt.multi_modal_data must be a mapping containing image.")

        image = multi_modal_data.get("image")
        if isinstance(image, list):
            raise ValueError("LingBot World requires one image and does not accept an image list.")
        if image is None:
            raise ValueError("LingBot World requires exactly one image in multi_modal_data.image.")
        if isinstance(image, (str, os.PathLike)):
            raise ValueError("file-path images must be materialized by the LingBot pre-process function.")
        if not isinstance(image, (PIL.Image.Image, torch.Tensor)):
            raise ValueError("multi_modal_data.image must be a PIL image, tensor, or file path.")

        extra_args = getattr(sampling, "extra_args", None) or {}
        if not isinstance(extra_args, dict):
            raise ValueError("sampling_params.extra_args must be a mapping.")
        camera_trajectory = extra_args.get(_PREPROCESSED_CAMERA_KEY)
        camera_action_script = extra_args.get(_PREPROCESSED_CAMERA_ACTION_SCRIPT_KEY)
        if camera_trajectory is not None and not isinstance(camera_trajectory, CameraTrajectory):
            raise ValueError("LingBot camera trajectory must be materialized by the pre-process function.")
        if camera_action_script is not None:
            camera_action_script = as_camera_action_script(camera_action_script)
        camera_sources = int(camera_trajectory is not None) + int(camera_action_script is not None)
        if camera_sources > 1:
            raise ValueError("LingBot pre-processing must materialize at most one camera input.")
        # Zero camera sources are allowed for stepwise camera-interaction sessions;
        # request-mode ``forward`` still requires an explicit trajectory.

        request_flow_shift = (
            extra_args["flow_shift"] if "flow_shift" in extra_args else getattr(self.scheduler.config, "shift", 5.0)
        )
        flow_shift = _positive_finite_flow_shift(request_flow_shift)

        height = getattr(sampling, "height", None)
        width = getattr(sampling, "width", None)
        if isinstance(image, PIL.Image.Image):
            image_width, image_height = image.size
        elif image.ndim == 3 and image.shape[0] == 3:
            image_height, image_width = image.shape[-2:]
        elif image.ndim == 4 and image.shape[:2] == (1, 3):
            image_height, image_width = image.shape[-2:]
        else:
            raise ValueError("tensor image must have shape [3, height, width] or [1, 3, height, width].")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("source image width and height must be positive integers.")
        if image_height * image_width > _MAX_SOURCE_IMAGE_PIXELS:
            raise ValueError("source image pixel count must not exceed 4096 * 4096.")
        if (height is None) != (width is None):
            raise ValueError("height and width must either both be provided or both be omitted.")
        patch_size = tuple(self.transformer.config.patch_size)
        if len(patch_size) != 3 or patch_size[0] != 1:
            raise RuntimeError(
                "transformer.config.patch_size must be a three-dimensional tuple with temporal patch size 1."
            )
        height_divisor = self.vae_scale_factor_spatial * patch_size[1]
        width_divisor = self.vae_scale_factor_spatial * patch_size[2]
        if height is None or width is None:
            aspect_ratio = image_height / image_width
            latent_height = round(
                math.sqrt(_MAX_PIXEL_AREA * aspect_ratio)
                // self.vae_scale_factor_spatial
                // patch_size[1]
                * patch_size[1]
            )
            latent_width = round(
                math.sqrt(_MAX_PIXEL_AREA / aspect_ratio)
                // self.vae_scale_factor_spatial
                // patch_size[2]
                * patch_size[2]
            )
            height = latent_height * self.vae_scale_factor_spatial
            width = latent_width * self.vae_scale_factor_spatial
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be a positive integer, got {height!r}.")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be a positive integer, got {width!r}.")
        if height % height_divisor:
            raise ValueError(f"height must be divisible by {height_divisor}, got {height}.")
        if width % width_divisor:
            raise ValueError(f"width must be divisible by {width_divisor}, got {width}.")
        if height * width > _MAX_PIXEL_AREA:
            raise ValueError("height * width pixel area must not exceed 480 * 832.")

        num_inference_steps = getattr(sampling, "num_inference_steps", None)
        if num_inference_steps is None:
            num_inference_steps = len(LINGBOT_DMD_TIMESTEPS)
        if num_inference_steps != len(LINGBOT_DMD_TIMESTEPS):
            raise ValueError(
                "num_inference_steps must be 4 for LingBot World causal DMD "
                f"timesteps {list(LINGBOT_DMD_TIMESTEPS)}, got {num_inference_steps}."
            )

        num_frames = getattr(sampling, "num_frames", None)
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError(f"num_frames must be a positive integer, got {num_frames!r}.")
        temporal_factor = self.vae_scale_factor_temporal
        if (num_frames - 1) % temporal_factor:
            raise ValueError(
                "num_frames must satisfy the causal Wan VAE geometry "
                f"(num_frames - 1) divisible by {temporal_factor}, got {num_frames}."
            )
        num_latent_frames = (num_frames - 1) // temporal_factor + 1
        block_frames = int(self.transformer.config.num_frames_per_block)
        if num_latent_frames % block_frames:
            raise ValueError(
                "num_frames must map to a whole number of configured three-frame latent blocks; "
                f"got num_frames={num_frames}, latent_frames={num_latent_frames}, block_frames={block_frames}."
            )

        max_sequence_length = getattr(sampling, "max_sequence_length", None)
        if max_sequence_length is None:
            max_sequence_length = _MAX_SEQUENCE_LENGTH
        if isinstance(max_sequence_length, bool) or not isinstance(max_sequence_length, int):
            raise ValueError(f"max_sequence_length must be exactly {_MAX_SEQUENCE_LENGTH}.")
        if max_sequence_length != _MAX_SEQUENCE_LENGTH:
            raise ValueError(f"max_sequence_length must be exactly {_MAX_SEQUENCE_LENGTH}.")

        return _LingBotRequestInputs(
            prompt=prompt.strip(),
            image=image,
            camera_trajectory=camera_trajectory,
            camera_actions=None,
            camera_action_script=camera_action_script,
            height=height,
            width=width,
            num_frames=num_frames,
            num_latent_frames=num_latent_frames,
            output_type=getattr(sampling, "output_type", None) or "np",
            max_sequence_length=max_sequence_length,
            flow_shift=flow_shift,
            generator=generator,
        )

    def _prepare_image_tensor(self, image: PIL.Image.Image | torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            array = np.asarray(image, dtype=np.float32).copy()
            image_tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
        else:
            image_tensor = image.detach()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float32)
            if not torch.isfinite(image_tensor).all():
                raise ValueError("tensor image values must all be finite.")
            minimum = image_tensor.min().item()
            maximum = image_tensor.max().item()
            if minimum >= 0.0 and maximum > 255.0:
                raise ValueError("tensor image values must be in [0, 1], [0, 255], or [-1, 1].")
            if minimum < -1.0 or (minimum < 0.0 and maximum > 1.0):
                raise ValueError("tensor image values must be in [0, 1], [0, 255], or [-1, 1].")
            if maximum > 1.0:
                image_tensor = image_tensor / 255.0
        if image_tensor.min().item() >= 0.0:
            image_tensor = image_tensor * 2.0 - 1.0
        if image_tensor.shape[-2:] != (height, width):
            image_tensor = F.interpolate(
                image_tensor,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        return image_tensor.to(device=self.device, dtype=torch.float32)

    def _vae_latent_stats(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, -1, 1, 1, 1)
        latent_mean = torch.as_tensor(
            self.vae.config.latents_mean,
            device=reference.device,
            dtype=reference.dtype,
        ).view(*shape)
        latent_std = torch.as_tensor(
            self.vae.config.latents_std,
            device=reference.device,
            dtype=reference.dtype,
        ).view(*shape)
        return latent_mean, latent_std

    def _prepare_condition(self, inputs: _LingBotRequestInputs, *, dtype: torch.dtype) -> torch.Tensor:
        """Encode the first frame as ``[mask4, image_latent16]``."""

        image = self._prepare_image_tensor(inputs.image, height=inputs.height, width=inputs.width)
        video_condition = image.new_zeros(1, 3, inputs.num_frames, inputs.height, inputs.width)
        video_condition[:, :, 0] = image
        latent_condition = retrieve_latents(
            self.vae.encode(video_condition.to(dtype=self.vae.dtype)),
            sample_mode="argmax",
        )
        if latent_condition.shape != (
            1,
            self.transformer.config.out_channels,
            inputs.num_latent_frames,
            inputs.height // self.vae_scale_factor_spatial,
            inputs.width // self.vae_scale_factor_spatial,
        ):
            raise RuntimeError(
                f"vae.encode returned an incompatible image latent shape: got {tuple(latent_condition.shape)}."
            )
        return self._condition_from_latents(latent_condition, first_frame=True, dtype=dtype)

    def _condition_from_latents(
        self, latent_condition: torch.Tensor, *, first_frame: bool, dtype: torch.dtype
    ) -> torch.Tensor:
        latent_mean, latent_std = self._vae_latent_stats(latent_condition)
        latent_condition = (latent_condition - latent_mean) / latent_std
        temporal_mask = latent_condition.new_zeros(1, self.vae_scale_factor_temporal, *latent_condition.shape[2:])
        if first_frame:
            temporal_mask[:, :, 0] = 1
        condition = torch.cat((temporal_mask, latent_condition), dim=1).to(dtype=dtype)
        if condition.shape[1] != 20:
            raise RuntimeError(
                "LingBot image condition must contain 4 temporal-mask then 16 image-latent channels, "
                f"got {condition.shape[1]}."
            )
        return condition

    def _next_condition_chunk(
        self,
        inputs: _LingBotRequestInputs,
        *,
        start_frame: int,
        session_state: _LingBotARSessionState,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """The session's condition for the block at ``start_frame``, encoding only while it still changes.

        After a session's opening block the encoder is fed the same all-zero
        pixel frames on every block, so the only thing that can make one
        block's condition differ from the last is the encoder's own history.
        The history is a causal cache with finite temporal reach, and once the
        zero frames have flushed the source image out of it the encoder is at
        a fixed point: identical input and identical state produce the same
        output and the same state on every later block. That fixed point is
        detected, not assumed after some number of blocks: a block whose
        returned history is bit-identical to the history it was given is the
        proof, and from then on the stored condition is returned and the
        encoder is not run. Measured at 480x832 the encode is 178 ms of a
        block and the fixed point arrives by the eleventh; the served
        rollouts this was measured on (40 blocks) are then byte-identical.

        The opening block resets the fixed point, so a session that restarts
        from a new source image encodes again until it settles.
        """
        if start_frame == 0:
            session_state.condition_fixed_point = None
        elif session_state.condition_fixed_point is not None:
            assert session_state.encoder_cache is not None
            # The history is unchanged by construction; the pending copy is
            # what post_decode commits, exactly as an encoded block's would be.
            return session_state.condition_fixed_point.clone(), list(session_state.encoder_cache)
        condition, pending_cache = self._prepare_condition_chunk(
            inputs,
            start_frame=start_frame,
            encoder_cache=session_state.encoder_cache,
            dtype=dtype,
        )
        if start_frame > 0 and _encoder_caches_identical(session_state.encoder_cache, pending_cache):
            session_state.condition_fixed_point = condition.clone()
        return condition, pending_cache

    @torch.no_grad()
    def _prepare_condition_chunk(
        self,
        inputs: _LingBotRequestInputs,
        *,
        start_frame: int,
        encoder_cache: list[torch.Tensor | None] | None,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Continue the causal encoder without modifying its committed history."""
        if getattr(self.vae.config, "patch_size", None) is not None or self.vae_scale_factor_temporal != 4:
            raise ValueError("Stateful LingBot conditioning requires an unpatched Wan VAE with temporal factor 4.")
        if getattr(self.vae, "use_tiling", False) and (
            inputs.height > self.vae.tile_sample_min_height or inputs.width > self.vae.tile_sample_min_width
        ):
            raise ValueError("Stateful LingBot conditioning does not support tiled VAE encoding.")
        if (encoder_cache is None) != (start_frame == 0):
            raise ValueError("LingBot condition encoder history must match the current temporal position.")

        count = int(self.vae._cached_conv_counts["encoder"])
        # Wan replaces cache entries with new tensors; a separate list keeps the
        # committed history intact until this AR block succeeds.
        pending_cache = [None] * count if encoder_cache is None else list(encoder_cache)
        if len(pending_cache) != count:
            raise ValueError("LingBot condition encoder cache has an incompatible number of slots.")
        block_frames = int(self.transformer.config.num_frames_per_block)
        zeros = torch.zeros(
            1,
            3,
            self.vae_scale_factor_temporal,
            inputs.height,
            inputs.width,
            device=self.device,
            dtype=self.vae.dtype,
        )
        image = (
            self._prepare_image_tensor(inputs.image, height=inputs.height, width=inputs.width)
            .unsqueeze(2)
            .to(dtype=self.vae.dtype)
            if start_frame == 0
            else None
        )
        context = getattr(self.vae, "_execution_context", None)
        encoded_frames = []
        with context() if callable(context) else nullcontext():
            for index in range(block_frames):
                # Match Wan's encode loop: one opening pixel frame, then four
                # fresh zero pixel frames per subsequent latent frame.
                pixel_frames = image if start_frame == 0 and index == 0 else zeros
                encoded_frames.append(self.vae.encoder(pixel_frames, feat_cache=pending_cache, feat_idx=[0]))
            # The posterior projection is pointwise in time. GPU bitwise parity
            # with the whole-clip projection still requires the real-weight gate.
            moments = self.vae.quant_conv(torch.cat(encoded_frames, dim=2))
        latent_condition = moments.chunk(2, dim=1)[0]
        expected_shape = (
            1,
            self.transformer.config.out_channels,
            block_frames,
            inputs.height // self.vae_scale_factor_spatial,
            inputs.width // self.vae_scale_factor_spatial,
        )
        if latent_condition.shape != expected_shape:
            raise RuntimeError(
                f"Wan condition encoder returned {tuple(latent_condition.shape)}, expected {expected_shape}."
            )
        condition = self._condition_from_latents(
            latent_condition,
            first_frame=start_frame == 0,
            dtype=dtype,
        )
        return condition, pending_cache

    def _prepare_camera(
        self,
        inputs: _LingBotRequestInputs,
        *,
        dtype: torch.dtype,
        previous: CameraTrajectory | None = None,
        latent_aligned: bool = False,  # iff using realtime interaction inputs
    ) -> tuple[torch.Tensor, CameraTrajectory]:
        """Convert raw camera frames to a latent-aligned ray tensor."""

        trajectory = inputs.camera_trajectory
        if trajectory is None:
            raise RuntimeError("LingBot camera trajectory was not prepared before camera embedding.")
        available_frames = int(trajectory.poses.shape[0])
        if inputs.camera_actions is not None or latent_aligned:
            if available_frames != inputs.num_latent_frames:
                raise ValueError(
                    "camera actions must produce exactly one pose per latent frame; "
                    f"got camera_frames={available_frames}, latent_frames={inputs.num_latent_frames}."
                )
        else:
            if available_frames < inputs.num_frames:
                raise ValueError(
                    "camera trajectory frames must be at least num_frames; "
                    f"got camera_frames={available_frames}, num_frames={inputs.num_frames}."
                )
            trajectory = CameraTrajectory(
                poses=trajectory.poses[: inputs.num_frames],
                intrinsics=trajectory.intrinsics[: inputs.num_frames],
            )
            trajectory = interpolate_camera_trajectory(trajectory, inputs.num_latent_frames)
        embedding_trajectory = trajectory
        drop_anchor = False
        if previous is not None:
            embedding_trajectory = CameraTrajectory(
                poses=torch.cat((previous.poses, trajectory.poses), dim=0),
                intrinsics=torch.cat(
                    (previous.intrinsics, trajectory.intrinsics),
                    dim=0,
                ),
            )
            drop_anchor = True
        elif inputs.camera_actions is not None or latent_aligned:
            # Action/interaction integrators return post-action poses. Prepend the
            # known pre-action state (identity for a new session) so the first
            # pose still contributes a framewise delta.
            identity = torch.eye(
                4,
                device=trajectory.poses.device,
                dtype=trajectory.poses.dtype,
            ).unsqueeze(0)
            embedding_trajectory = CameraTrajectory(
                poses=torch.cat((identity, trajectory.poses), dim=0),
                intrinsics=torch.cat(
                    (trajectory.intrinsics[:1], trajectory.intrinsics),
                    dim=0,
                ),
            )
            drop_anchor = True
        # Realtime interaction cannot see future steps; use the controller translation unit to normalize speed.
        # Full action_path keeps max-norm.
        camera_embedding = build_plucker_embedding(
            embedding_trajectory,
            height=inputs.height,
            width=inputs.width,
            target_height=inputs.height,
            target_width=inputs.width,
            device=self.device,
            dtype=dtype,
            translation_scale=LINGBOT_CONTROLLER_TRANSLATION_UNIT if latent_aligned else None,
        )
        if drop_anchor:
            camera_embedding = camera_embedding[1:]
        tail = CameraTrajectory(
            poses=trajectory.poses[-1:].clone(),
            intrinsics=trajectory.intrinsics[-1:].clone(),
        )
        return _fold_camera_embedding(camera_embedding), tail

    def _ar_text_caches(self, prompt_embeds: torch.Tensor) -> list[LingBotAttentionCache]:
        state = self._ar_diffusion_kv_state
        if state is None:
            raise RuntimeError("LingBot AR text cache requested without a bound state.")
        if not state.is_cross_attention_populated(
            self._AR_BRANCH,
            self._AR_TEXT_CACHE,
        ):
            projected_text = self.transformer.text_embedding(prompt_embeds)

            def layer_kv() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
                for block in self.transformer.blocks:
                    cross_attention = block.cross_attn
                    shape = (cross_attention.num_local_heads, cross_attention.head_dim)
                    key = cross_attention.norm_k(cross_attention.k(projected_text)).unflatten(2, shape)
                    value = cross_attention.v(projected_text).unflatten(2, shape)
                    yield key, value

            state.populate_cross_attention(
                self._AR_BRANCH,
                self._AR_TEXT_CACHE,
                layer_kv(),
            )
        return [
            LingBotAttentionCache(
                key=layer["k"],
                value=layer["v"],
                end=prompt_embeds.shape[1],
                absolute_end=prompt_embeds.shape[1],
                last_start=0,
            )
            for layer in state.get_cross_attention_kv(
                self._AR_BRANCH,
                self._AR_TEXT_CACHE,
            )
        ]

    # ── DMD block helpers ─────────────────────────────────────────────────
    #
    # The DMD math for one AR block lives in ``LingBotDMDBlockRunner`` so that
    # request mode (``_generate_block`` in a loop) and stepwise execution
    # (``denoise_step`` / ``step_scheduler`` / ``post_decode``) share one copy.
    # The pipeline's only jobs here are building the runner and translating
    # the bound AR session into an ``ARBlockContext``.

    @property
    def _dmd_blocks(self) -> LingBotDMDBlockRunner:
        runner = getattr(self, "_dmd_block_runner", None)
        if runner is None:
            runner = LingBotDMDBlockRunner(
                self.transformer,
                device=self.device,
                enforce_eager=bool(self.od_config.enforce_eager),
            )
            self._dmd_block_runner = runner
        return runner

    def _ar_block_context(self, cross_attention: list[LingBotAttentionCache]) -> ARBlockContext:
        """Wrap the bound session for the block runner."""
        state = self._ar_diffusion_kv_state
        if state is None:
            raise RuntimeError("LingBot AR cache requested without a bound state.")
        return ARBlockContext(state=state, cross_attention=cross_attention, branch=self._AR_BRANCH)

    def _generate_block(
        self,
        *,
        condition: torch.Tensor,
        camera: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache,
        start_frame: int,
        schedule: tuple[tuple[float, float], ...],
        generator: torch.Generator,
        progress_bar: TqdmProgressBar[Any],
    ) -> torch.Tensor:
        """Generate one block against request-local KV; step execution uses
        the bound AR session through :meth:`_ar_block_context` instead."""
        return self._dmd_blocks.generate_block(
            condition=condition,
            camera=camera,
            prompt_embeds=prompt_embeds,
            cache=cache,
            ar=None,
            start_frame=start_frame,
            schedule=schedule,
            generator=generator,
            progress_bar=progress_bar,
        )

    def encode_prompt(
        self,
        prompt: str,
        *,
        max_sequence_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode ``prompt`` with UMT5, reusing the result for a prompt already encoded.

        The encode depends only on the whitespace-normalised text, the sequence
        length and the dtype -- the tokenizer and text encoder are fixed once
        loaded -- so those three are the key. That covers every caller the same
        way: a streaming session can repeat its prompt across requests, a
        stepwise request encodes once per request, and a server tends to open
        many sessions with the same scene prompt.

        A hit returns the stored tensor itself. Callers only read it: the
        cross-attention projection and the DMD transformer both work out of
        place, so a shared encode is never changed under another session.
        """
        text = " ".join(prompt.strip().split())
        key = (text, int(max_sequence_length), dtype)
        cached = self._prompt_embeds_cache.get(key)
        if cached is not None:
            self._prompt_embeds_cache.move_to_end(key)
            return cached
        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        prompt_embeds = self.text_encoder(input_ids, attention_mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=dtype)
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(dtype=prompt_embeds.dtype)
        self._prompt_embeds_cache[key] = prompt_embeds
        if len(self._prompt_embeds_cache) > _PROMPT_EMBEDS_CACHE_SIZE:
            self._prompt_embeds_cache.popitem(last=False)
        return prompt_embeds

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        inputs = self._parse_request(req)
        if inputs.camera_action_script is not None:
            # Only step execution walks a whole rollout inside one request, so
            # a per-chunk script here would be silently dropped rather than
            # steering the camera.
            raise ValueError(
                "camera_action_script is read only by LingBot step execution; "
                "request mode takes a camera_trajectory for the request."
            )
        if inputs.camera_trajectory is None:
            raise ValueError("LingBot request mode requires camera_trajectory; omit it only for stepwise sessions.")
        if self._ar_diffusion_kv_state is not None:
            raise ValueError(
                "LingBot request mode does not run under ARDiffusionEngine session "
                "binding; enable step execution for AR-Diffusion serving."
            )
        schedule = _build_shifted_flow_schedule(
            flow_shift=inputs.flow_shift,
        )
        dtype = self.transformer.dtype
        # Phase 1: turn all three user inputs into DiT-ready conditions.
        prompt_embeds = self.encode_prompt(
            inputs.prompt,
            max_sequence_length=inputs.max_sequence_length,
            dtype=dtype,
        )
        condition = self._prepare_condition(inputs, dtype=dtype)
        camera, _ = self._prepare_camera(inputs, dtype=dtype)
        if camera.shape[2:] != condition.shape[2:]:
            raise RuntimeError(
                "folded camera and image condition must share latent frame/height/width geometry; "
                f"got camera={tuple(camera.shape)}, condition={tuple(condition.shape)}."
            )

        block_frames = int(self.transformer.config.num_frames_per_block)
        # Direct/offline oracle keeps request-local contiguous K/V.
        cache: LingBotTransformerCache | None = self.transformer.allocate_cache(
            batch_size=1,
            latent_height=condition.shape[-2],
            latent_width=condition.shape[-1],
            device=self.device,
            dtype=dtype,
        )
        generated_blocks: list[torch.Tensor] = []
        total_steps = (inputs.num_latent_frames // block_frames) * len(LINGBOT_DMD_TIMESTEPS)
        with self.progress_bar(total=total_steps) as progress_bar:
            # Phase 3: generate left-to-right in latent-frame blocks. The
            # monotonically increasing ``start_frame`` is also the temporal
            # position and cache offset used by the Transformer.
            for start_frame in range(0, inputs.num_latent_frames, block_frames):
                stop_frame = start_frame + block_frames
                generated_blocks.append(
                    self._generate_block(
                        condition=condition[:, :, start_frame:stop_frame],
                        camera=camera[:, :, start_frame:stop_frame],
                        prompt_embeds=prompt_embeds,
                        cache=cache,
                        start_frame=start_frame,
                        schedule=schedule,
                        generator=inputs.generator,
                        progress_bar=progress_bar,
                    )
                )
        generated_latents = torch.cat(generated_blocks, dim=2)
        # The direct/offline cache is request-local and no longer needed once
        # all latent blocks have been generated. Release it before VAE decode
        # so the two large allocations do not overlap.
        cache = None

        # Phase 4: either expose model-space latents or invert the checkpoint's
        # latent normalization and decode to pixel-space video.
        if inputs.output_type == "latent":
            output = generated_latents
        else:
            latent_mean, latent_std = self._vae_latent_stats(generated_latents)
            vae_latents = (generated_latents * latent_std + latent_mean).to(dtype=self.vae.dtype)
            output = self.vae.decode(vae_latents, return_dict=False)[0]
            if output.shape[2] != inputs.num_frames:
                raise RuntimeError(
                    "vae.decode returned an incompatible temporal geometry: "
                    f"expected {inputs.num_frames} frames, got {output.shape[2]}."
                )
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _vae_would_tile(self, latents: torch.Tensor) -> bool:
        """Whether ``vae.decode`` would take its tiled path for this shape.

        Tiled decode splits a frame spatially and drives the module's own
        ``feat_cache`` across those tiles, so a per-session cache threaded
        through it is a separate design and is not attempted here. The test
        mirrors ``AutoencoderKLWan._decode``: the flag alone does not tile, the
        latent also has to exceed a tile, so a sub-threshold resolution with
        tiling enabled still streams. A VAE that sets the flag but reports no
        tile size is treated as tiling, because the safe answer when the
        threshold is unknown is the one that keeps ``vae.decode`` in charge.
        """
        if not bool(getattr(self.vae, "use_tiling", False)):
            return False
        compression = int(getattr(self.vae, "spatial_compression_ratio", None) or self.vae_scale_factor_spatial)
        tile_latent_min_height = int(getattr(self.vae, "tile_sample_min_height", None) or 0) // compression
        tile_latent_min_width = int(getattr(self.vae, "tile_sample_min_width", None) or 0) // compression
        return bool(latents.shape[-2] > tile_latent_min_height or latents.shape[-1] > tile_latent_min_width)

    def _streaming_decoder(self) -> WanStreamingDecoder | None:
        """The session-owned streaming decoder, or ``None`` on a VAE that cannot stream.

        Only the decoder's own capability is decided here, and the decision is
        cached including the negative. Whether a particular *shape* would have
        been tiled is a per-call question and lives in :meth:`_vae_would_tile`.
        """
        if self._streaming_decode_unsupported:
            return None
        if self._cached_streaming_decoder is not None:
            return self._cached_streaming_decoder
        try:
            decoder = WanStreamingDecoder(
                self.vae,
                bytes_per_pixel_fp32=_STREAMING_DECODE_BYTES_PER_PIXEL_FP32,
            )
            # Materialize the cache geometry now: a decoder that cannot report
            # it would otherwise fail on the first chunk instead.
            _ = decoder.num_causal_convs
        except TypeError as exc:
            logger.warning(
                "LingBot streaming decode is unavailable (%s); AR blocks will be decoded independently, "
                "which drops each block's opening frame and restarts temporal context at every boundary.",
                exc,
            )
            self._streaming_decode_unsupported = True
            return None
        self._cached_streaming_decoder = decoder
        return decoder

    def _streaming_decode_state(self, decoder: WanStreamingDecoder, session_id: str) -> StreamingDecodeState:
        state = self._streaming_decode_states.get(session_id)
        if state is None:
            state = decoder.new_decode_state(session_id)
            self._streaming_decode_states[session_id] = state
        return state

    def _release_streaming_decode_state(self, session_id: str) -> None:
        """Drop one session's decoder cache. Safe for a session that never decoded.

        The state is released directly rather than through
        ``WanStreamingDecoder.release``, which is the same one-line call: the
        decoder's method exists for a caller holding only the
        ``SupportsStreamingDecode`` protocol, and this pipeline holds the state.
        """
        state = self._streaming_decode_states.pop(session_id, None)
        if state is not None:
            state.release()

    def _decode_chunk_to_pixels(self, latents: torch.Tensor, *, session_id: str | None = None) -> np.ndarray:
        """Decode one AR block so streaming consumers receive pixels, not latents.

        The block comes back as ``(F, H, W, 3)`` uint8 whatever pixel output
        type the request named: the realtime endpoint muxes uint8 frames, so
        ``"pil"`` and ``"np"`` only ever differed in how much work stood
        between the decoder and those bytes. Only ``"latent"`` means
        something else, and the caller keeps latents before reaching here.

        With a ``session_id`` the block is decoded through that session's own
        temporal cache, so chunk *N + 1* continues chunk *N*: the causal decoder
        expands the session's opening latent frame to one raw frame and every
        later one to the full temporal factor, which is the frame timeline an
        offline whole-clip decode of the same latents produces. The cache lives
        on the session rather than on the shared VAE module, so concurrent
        sessions cannot overwrite one another's temporal context.

        Without a ``session_id`` -- or on a VAE that cannot stream -- blocks are
        decoded independently: every block re-expands its own opening frame, so
        the rollout loses frames and continuity at each block boundary.

        The registered post-process hook returns a ``{"payload", "metadata"}``
        envelope untouched, so the pixel conversion it would do for a bare
        tensor has to happen here instead.
        """
        latent_mean, latent_std = self._vae_latent_stats(latents)
        vae_latents = (latents * latent_std + latent_mean).to(dtype=self.vae.dtype)
        decoder = None
        if session_id is not None and not self._vae_would_tile(vae_latents):
            decoder = self._streaming_decoder()
        if decoder is None:
            video = self.vae.decode(vae_latents, return_dict=False)[0]
        else:
            video = self._streaming_decode_chunk(decoder, vae_latents, cast(str, session_id))
        return _uint8_frames(video)

    def _streaming_decode_chunk(
        self, decoder: WanStreamingDecoder, vae_latents: torch.Tensor, session_id: str
    ) -> torch.Tensor:
        """Decode one chunk through ``session_id``'s temporal cache.

        A separate method so the profiler has a stage to wrap: streaming never
        calls ``vae.decode``, and without this a profiled rollout would report
        no decode time for the one stage this path changes.
        """
        state = self._streaming_decode_state(decoder, session_id)
        try:
            return decoder.decode_chunk(vae_latents, state)
        except Exception:
            # A half-advanced cache cannot be resumed: the session's next chunk
            # would continue from a temporal context that was never completed,
            # so drop it here rather than at close.
            self._release_streaming_decode_state(session_id)
            raise

    def _require_bound_ar_state(self) -> None:
        if self._ar_diffusion_kv_state is None:
            raise RuntimeError("LingBot step execution requires AR-Diffusion session binding.")

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        self._require_bound_ar_state()
        req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(
                    prompt=state.prompt,
                    sampling_params=state.sampling,
                    request_id=state.request_id,
                )
            ]
        )
        inputs = self._parse_request(req)
        if (inputs.height, inputs.width) != (self._ar_height, self._ar_width):
            raise ValueError(
                "LingBot AR-Diffusion request resolution must match the "
                f"fixed cache geometry {self._ar_height}x{self._ar_width}."
            )
        block_frames = int(self.transformer.config.num_frames_per_block)
        total_chunks = inputs.num_latent_frames // block_frames
        if inputs.camera_action_script is not None and len(inputs.camera_action_script) != total_chunks:
            raise ValueError(
                "camera_action_script must contain one action list per generated chunk; "
                f"got {len(inputs.camera_action_script)} chunks for {total_chunks} requested chunks."
            )
        dtype = self.transformer.dtype
        prompt_embeds = self.encode_prompt(
            inputs.prompt,
            max_sequence_length=inputs.max_sequence_length,
            dtype=dtype,
        )
        camera_trajectory_cache = None
        camera_embedding_cache = None
        if inputs.camera_trajectory is not None:
            available_frames = int(inputs.camera_trajectory.poses.shape[0])
            if available_frames < inputs.num_frames:
                raise ValueError(
                    "camera trajectory frames must be at least num_frames; "
                    f"got camera_frames={available_frames}, num_frames={inputs.num_frames}."
                )
            truncated = CameraTrajectory(
                poses=inputs.camera_trajectory.poses[: inputs.num_frames],
                intrinsics=inputs.camera_trajectory.intrinsics[: inputs.num_frames],
            )
            camera_trajectory_cache = interpolate_camera_trajectory(truncated, inputs.num_latent_frames)
            # Embed the whole trajectory once, exactly as request mode does, and
            # slice it per block in ``_prepare_next_chunk``. Embedding each chunk
            # on its own would normalize that block's framewise translations by
            # its own largest step, so a slow block would be conditioned as
            # full-speed motion and drift from offline replay of the same path.
            camera_embedding_cache, _ = self._prepare_camera(inputs, dtype=dtype)
        state.prompt_embeds = prompt_embeds
        state.chunk_index = 0
        state.step_index = 0
        state.step_in_chunk = 0
        state.total_chunks = total_chunks
        state.chunk_num_steps = len(LINGBOT_DMD_TIMESTEPS)
        state.extra = {
            "inputs": inputs,
            "schedule": _build_shifted_flow_schedule(flow_shift=inputs.flow_shift),
            "dtype": dtype,
            "block_frames": block_frames,
            "camera_pitch": 0.0,
            "camera_tail": None,
            "camera_action_script": inputs.camera_action_script,
            "camera_trajectory_cache": camera_trajectory_cache,
            "camera_embedding_cache": camera_embedding_cache,
        }
        self._ar_text_caches(prompt_embeds)
        self._ar_sessions[state.request_id] = _LingBotARSessionState()
        return state

    def _prepare_next_chunk(self, state: StepRequestState) -> None:
        extra = state.extra
        inputs: _LingBotRequestInputs = extra["inputs"]
        block_frames = int(extra["block_frames"])
        start_frame = state.chunk_index * block_frames
        stop_frame = start_frame + block_frames
        session_state = self._ar_sessions[state.request_id]
        if session_state.next_chunk_index != state.chunk_index:
            raise ValueError("LingBot condition chunks must be contiguous within the session.")
        condition, pending_encoder_cache = self._next_condition_chunk(
            inputs,
            start_frame=start_frame,
            session_state=session_state,
            dtype=extra["dtype"],
        )
        previous = extra.get("camera_tail")

        if extra.get("camera_action_script") is not None:
            if (
                (camera_session := state.interaction_sessions.get("camera"))
                and isinstance(camera_session, CameraSession)
                and camera_session.has_received_input
            ):
                raise ValueError("Cannot use mid-generation camera interaction together with camera_action_script; ")
            chunk_actions = extra["camera_action_script"][state.chunk_index]
            action_trajectory, camera_pitch = integrate_lingbot_camera_actions(
                chunk_actions,
                width=inputs.width,
                height=inputs.height,
                initial_pose=(previous.poses[-1] if previous is not None else None),
                initial_pitch=float(extra.get("camera_pitch", 0.0)),
            )
            extra["camera_pitch"] = camera_pitch
            media_frames = self._chunk_media_frame_count(state, block_frames=block_frames)
            chunk_inputs = replace(
                inputs,
                camera_trajectory=action_trajectory,
                camera_actions=chunk_actions,
                num_frames=media_frames,
                num_latent_frames=block_frames,
            )
            camera, camera_tail = self._prepare_camera(
                chunk_inputs,
                dtype=extra["dtype"],
                previous=previous,
            )
        elif extra.get("camera_embedding_cache") is not None:
            if (
                (camera_session := state.interaction_sessions.get("camera"))
                and isinstance(camera_session, CameraSession)
                and camera_session.has_received_input
            ):
                raise ValueError("Cannot use mid-generation camera interaction together with camera_embedding_cache; ")
            trajectory = extra["camera_trajectory_cache"]
            camera = extra["camera_embedding_cache"][:, :, start_frame:stop_frame]
            camera_tail = CameraTrajectory(
                poses=trajectory.poses[stop_frame - 1 : stop_frame].clone(),
                intrinsics=trajectory.intrinsics[stop_frame - 1 : stop_frame].clone(),
            )
        elif self._interaction_coordinator is not None and self._interaction_coordinator.has_modality("camera"):
            # Poses come from the latest boundary apply: ``prepare_encode`` for
            # chunk 0, the diffusion runner for later chunks. This method only
            # digests them — it does not apply interactions itself.
            camera_session = state.interaction_sessions.get("camera")
            if not isinstance(camera_session, CameraSession) or camera_session.last_absolute_poses is None:
                raise RuntimeError(
                    "LingBot camera interaction requires apply_interaction_at_chunk_boundary before prepare_next_chunk."
                )
            absolute_poses = camera_session.last_absolute_poses
            media_frames = self._chunk_media_frame_count(state, block_frames=block_frames)
            num_latent_frames = block_frames
            if int(absolute_poses.shape[0]) != num_latent_frames:
                raise ValueError(
                    "camera interaction must produce exactly one pose per latent frame; "
                    f"got {int(absolute_poses.shape[0])}, expected {num_latent_frames}."
                )
            action_trajectory = camera_trajectory_from_absolute_pose(
                absolute_poses,
                width=inputs.width,
                height=inputs.height,
            )
            chunk_inputs = replace(
                inputs,
                camera_trajectory=action_trajectory,
                camera_actions=None,
                num_frames=media_frames,
                num_latent_frames=num_latent_frames,
            )
            camera, camera_tail = self._prepare_camera(
                chunk_inputs,
                dtype=extra["dtype"],
                previous=previous,
                latent_aligned=True,
            )
        else:
            raise RuntimeError("LingBot step execution is missing a camera trajectory or action script.")
        if camera.shape[2:] != condition.shape[2:]:
            raise RuntimeError(
                "folded camera and image condition must share latent frame/height/width geometry; "
                f"got camera={tuple(camera.shape)}, condition={tuple(condition.shape)}."
            )
        generator = getattr(state.sampling, "generator", None)
        if not isinstance(generator, torch.Generator):
            raise ValueError("LingBot World requires the runner-provided torch.Generator.")
        extra["condition"] = condition
        extra["camera"] = camera
        # One camera-modulation cache per block: the four probes and the
        # commit below are separate scheduler steps, so the block runner
        # cannot open the reuse window around them the way generate_block
        # does around its own loop. Holding the cache here makes the window
        # span exactly this block, per request.
        extra["camera_cache"] = CameraModulationCache()
        extra["camera_tail"] = camera_tail
        extra["start_frame"] = start_frame
        extra["ar_cross_attention"] = self._ar_text_caches(cast(torch.Tensor, state.prompt_embeds))
        state.latents = randn_tensor(
            (
                1,
                self.transformer.config.out_channels,
                condition.shape[2],
                condition.shape[3],
                condition.shape[4],
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        state.timesteps = torch.tensor(
            [timestep for timestep, _ in extra["schedule"]],
            device=self.device,
            dtype=torch.float32,
        )
        state.step_in_chunk = 0
        state.step_index = 0
        session_state.pending_encoder_cache = pending_encoder_cache

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("LingBot step execution supports a single request, not a batched request.")
        self._require_bound_ar_state()
        state = states[0]
        extra = state.extra
        latents = state.latents
        if latents is None:
            raise RuntimeError("LingBot step execution requires latents before denoise_step.")
        timestep = state.current_timestep
        if timestep is None:
            raise RuntimeError("LingBot step execution requires a current timestep before denoise_step.")
        schedule = extra["schedule"]
        step_in_chunk = state.step_in_chunk
        return self._dmd_blocks.probe_step(
            current_latents=latents,
            condition=extra["condition"],
            camera=extra["camera"],
            prompt_embeds=cast(torch.Tensor, state.prompt_embeds),
            cache=None,
            ar=self._ar_block_context(extra["ar_cross_attention"]),
            start_frame=int(extra["start_frame"]),
            timestep_value=float(schedule[step_in_chunk][0]),
            step_index=step_in_chunk,
            camera_cache=extra["camera_cache"],
        )

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        extra = state.extra
        latents = state.latents
        if latents is None:
            raise RuntimeError("LingBot step execution requires latents before step_scheduler.")
        generator = getattr(state.sampling, "generator", None)
        if not isinstance(generator, torch.Generator):
            raise ValueError("LingBot World requires the runner-provided torch.Generator.")
        schedule = extra["schedule"]
        step_in_chunk = state.step_in_chunk
        next_sigma = schedule[step_in_chunk + 1][1] if step_in_chunk + 1 < len(schedule) else None
        state.latents = self._dmd_blocks.apply_transition(
            latents,
            noise_pred,
            schedule[step_in_chunk][1],
            next_sigma=next_sigma,
            generator=generator,
        )
        state.step_in_chunk += 1
        state.step_index = state.step_in_chunk

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        self._require_bound_ar_state()
        extra = state.extra
        latents = state.latents
        if latents is None:
            raise RuntimeError("LingBot step execution requires latents before post_decode.")
        session_state = self._ar_sessions[state.request_id]
        if session_state.pending_encoder_cache is None:
            raise RuntimeError("LingBot condition encoder has no prepared chunk to commit.")
        self._dmd_blocks.commit_block_kv(
            latents=latents,
            condition=extra["condition"],
            camera=extra["camera"],
            prompt_embeds=cast(torch.Tensor, state.prompt_embeds),
            cache=None,
            ar=self._ar_block_context(extra["ar_cross_attention"]),
            start_frame=int(extra["start_frame"]),
            camera_cache=extra.pop("camera_cache"),
        )
        completed_chunk_index = state.chunk_index
        inputs: _LingBotRequestInputs = extra["inputs"]
        # "video" is the payload key the output formatter treats as primary, so
        # a streaming client receives frames on ``images`` like any other video
        # pipeline; latent mode keeps the tensor for offline consumers.
        if inputs.output_type == "latent":
            payload: dict[str, Any] = {"latents": latents}
        else:
            payload = {"video": self._decode_chunk_to_pixels(latents, session_id=state.request_id)}
        output = {
            "payload": payload,
            "metadata": {
                "ar_diffusion": ARDiffusionChunkMetadata(
                    session_id=state.request_id,
                    request_id=state.request_id,
                    chunk_index=completed_chunk_index,
                ).to_dict()
            },
        }
        session_state.encoder_cache = session_state.pending_encoder_cache
        session_state.pending_encoder_cache = None
        session_state.next_chunk_index += 1
        state.chunk_index += 1
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            chunk_index=completed_chunk_index,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )

    def _chunk_media_frame_count(self, state: StepRequestState, *, block_frames: int) -> int:
        """Pixel frames this chunk will emit when decoded.

        Independent (non-streaming) decode restarts causal expansion every block,
        so each chunk is ``(block_frames - 1) * temporal + 1`` frames.
        Streaming decode expands only the session's opening latent to one frame and every later latent to the full
        temporal factor, so chunk 0 matches the independent count and later chunks are ``block_frames * temporal``.
        """
        decode_state = self._streaming_decode_states.get(state.request_id)
        if decode_state is not None and decode_state.started:
            return block_frames * self.vae_scale_factor_temporal
        return (block_frames - 1) * self.vae_scale_factor_temporal + 1

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        """Expose this chunk's decoded media extent and latent step count."""
        num_latent_frames = int(state.extra.get("block_frames") or self.transformer.config.num_frames_per_block)
        num_media_frames = self._chunk_media_frame_count(state, block_frames=num_latent_frames)
        fps = state.sampling.fps
        if fps is None or float(fps) <= 0:
            # Media-frame camera controls are not wall-clock paced; a unit fps keeps
            # resolve_event_frame_offset well-defined when the client omits sampling.fps.
            fps = float(num_media_frames)
        return ChunkMediaSpec(
            num_media_frames=num_media_frames,
            fps=fps,
            num_latent_frames=num_latent_frames,
        )

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        """Prepare the next AR block after chunk-boundary interaction apply."""
        if state.request_denoise_completed:
            return
        self._prepare_next_chunk(state)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return cast(set[str], loader.load_weights(weights))
