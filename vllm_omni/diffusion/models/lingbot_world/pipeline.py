# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped LingBot-World v2 causal DMD pipeline."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.lingbot_world.actions import (
    LINGBOT_CAMERA_ACTION_SCHEMA,
    LINGBOT_CAMERA_TRAJECTORY_SCHEMA,
    integrate_lingbot_camera_actions,
    parse_lingbot_camera_action_frames,
)
from vllm_omni.diffusion.models.lingbot_world.camera import (
    CameraTrajectory,
    build_plucker_embedding,
    interpolate_camera_trajectory,
    load_camera_trajectory,
    resolve_trusted_action_directory,
)
from vllm_omni.diffusion.models.lingbot_world.transformer import (
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
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionTickRequest,
)

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmProgressBar

    from vllm_omni.experimental.ar_diffusion.kv_cache.state import (
        ARDiffusionKVState,
    )

LINGBOT_DMD_TIMESTEPS = (1000, 750, 500, 250)
_CAMERA_SPATIAL_FOLD = 8
_MAX_PIXEL_AREA = 480 * 832
_MAX_SOURCE_IMAGE_PIXELS = 4096 * 4096
_MAX_RAW_FRAMES = 117
_MAX_SEQUENCE_LENGTH = 512
_ACTION_ROOT_ENV = "VLLM_OMNI_LINGBOT_ACTION_ROOT"
_PREPROCESSED_CAMERA_KEY = "_lingbot_camera_trajectory"
_PREPROCESSED_CAMERA_ACTIONS_KEY = "_lingbot_camera_actions"
_SOURCE_IMAGE_ERROR = (
    "Unable to load multi_modal_data.image; expected a decodable image within 4096 * 4096 source pixels."
)


@dataclass(frozen=True)
class _LingBotRequestInputs:
    """Validated model-specific request boundary."""

    prompt: str
    image: PIL.Image.Image | torch.Tensor
    camera_trajectory: CameraTrajectory | None
    camera_actions: tuple[tuple[str, ...], ...] | None
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    output_type: str
    max_sequence_length: int
    flow_shift: float
    generator: torch.Generator


@dataclass
class _LingBotARSessionState:
    """Small model-owned state; attention tensors remain runner-owned."""

    next_chunk_index: int = 0
    prompt: str | None = None
    generator_state: torch.Tensor | None = None
    image_condition: torch.Tensor | None = None
    camera_tail: CameraTrajectory | None = None
    camera_pitch: float = 0.0


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
        "sequence_parallel_size": "sequence parallelism",
        "cfg_parallel_size": "CFG parallelism",
        "vae_patch_parallel_size": "VAE parallelism",
    }
    for field, feature in unsupported_sizes.items():
        size = getattr(parallel_config, field, 1) or 1
        if size > 1:
            raise NotImplementedError(f"LingBot World v1 does not support {feature} ({field}={size}).")
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


def _load_source_image(path: str | os.PathLike[str]) -> PIL.Image.Image:
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
        tick = ARDiffusionTickRequest.from_extra_args(extra_args)
        if tick is not None:
            camera_controls = [control for control in tick.controls if control.track == "camera"]
            if len(camera_controls) != 1:
                raise ValueError("LingBot realtime ticks require exactly one camera control.")
            camera_control = camera_controls[0]
            if camera_control.schema == LINGBOT_CAMERA_TRAJECTORY_SCHEMA:
                camera_data = camera_control.data
                try:
                    poses = torch.as_tensor(camera_data["poses"], dtype=torch.float32)
                    intrinsics = torch.as_tensor(
                        camera_data["intrinsics"],
                        dtype=torch.float32,
                    )
                except (KeyError, TypeError, ValueError):
                    raise ValueError(
                        "lingbot.camera_trajectory.v1 requires numeric poses and intrinsics fields."
                    ) from None
                if poses.ndim != 3 or poses.shape[1:] != (4, 4):
                    raise ValueError("camera control poses must have shape [frames, 4, 4].")
                if intrinsics.ndim != 2 or intrinsics.shape[1:] != (4,):
                    raise ValueError("camera control intrinsics must have shape [frames, 4].")
                if poses.shape[0] != intrinsics.shape[0] or poses.shape[0] == 0:
                    raise ValueError(
                        "camera control poses and intrinsics must contain the same positive number of frames."
                    )
                if not torch.isfinite(poses).all() or not torch.isfinite(intrinsics).all():
                    raise ValueError("camera control values must all be finite.")
                trajectory = CameraTrajectory(
                    poses=poses,
                    intrinsics=intrinsics,
                )
                camera_actions = None
            elif camera_control.schema == LINGBOT_CAMERA_ACTION_SCHEMA:
                trajectory = None
                camera_actions = parse_lingbot_camera_action_frames(
                    camera_control.data,
                    expected_frames=3,
                )
            else:
                raise ValueError(
                    "LingBot realtime camera controls require schema "
                    "'lingbot.camera_trajectory.v1' or 'lingbot.camera_actions.v1'."
                )
        else:
            action_path = extra_args.get("action_path")
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
            camera_actions = None

        updated_prompt = dict(prompt)
        updated_multi_modal_data = dict(multi_modal_data)
        updated_multi_modal_data["image"] = image
        updated_prompt["multi_modal_data"] = updated_multi_modal_data
        request.prompt = updated_prompt
        request.sampling_params.extra_args = {
            **extra_args,
            _PREPROCESSED_CAMERA_KEY: trajectory,
            _PREPROCESSED_CAMERA_ACTIONS_KEY: camera_actions,
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
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """LingBot-World v2 I2V generation with a request-local causal cache."""

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
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=[
                "vae.encode",
                "vae.decode",
                "_generate_block",
                "text_encoder.forward",
                "tokenizer.forward",
            ],
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Describe the fixed worker-local cache geometry for realtime ticks."""
        patch_frames, patch_height, patch_width = self.transformer.config.patch_size
        if patch_frames != 1:
            raise ValueError("LingBot AR-Diffusion requires temporal patch size 1.")
        spatial = self.vae_scale_factor_spatial
        if self._ar_height % (spatial * patch_height) or self._ar_width % (spatial * patch_width):
            raise ValueError("ar_diffusion_height/width must align with VAE and DiT patch sizes.")
        latent_height = self._ar_height // spatial
        latent_width = self._ar_width // spatial
        tokens_per_frame = (latent_height // patch_height) * (latent_width // patch_width)
        tp_size = get_tensor_model_parallel_world_size()
        num_local_heads = int(self.transformer.config.num_attention_heads) // tp_size
        total_window_frames = (
            int(self.transformer.config.local_attn_size)
            if int(self.transformer.config.local_attn_size) != -1
            else int(self.transformer.config.sliding_window_num_frames)
        )
        sink_frames = int(self.transformer.config.sink_size)
        recent_window_frames = total_window_frames - sink_frames
        if recent_window_frames <= 0:
            raise ValueError("LingBot AR-Diffusion cache needs a positive recent window after reserving sink frames.")
        horizon_latent_frames = (_MAX_RAW_FRAMES - 1) // self.vae_scale_factor_temporal + 1
        condition_channels = self.vae_scale_factor_temporal + int(self.transformer.config.out_channels)
        condition_bytes_per_session = (
            condition_channels
            * horizon_latent_frames
            * latent_height
            * latent_width
            * torch.empty((), dtype=self.transformer.dtype).element_size()
        )
        return ARDiffusionKVCacheSpec(
            num_layers=int(self.transformer.config.num_layers),
            num_kv_heads=num_local_heads,
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
                ),
            ),
            model_owned_state_bytes_per_session=condition_bytes_per_session,
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

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)

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
        camera_actions = extra_args.get(_PREPROCESSED_CAMERA_ACTIONS_KEY)
        if camera_trajectory is not None and not isinstance(camera_trajectory, CameraTrajectory):
            raise ValueError("LingBot camera trajectory must be materialized by the pre-process function.")
        if camera_actions is not None:
            camera_actions = tuple(tuple(actions) for actions in camera_actions)
        if (camera_trajectory is None) == (camera_actions is None):
            raise ValueError("LingBot pre-processing must materialize exactly one camera input.")

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
        if num_frames > _MAX_RAW_FRAMES:
            raise ValueError(f"num_frames must not exceed {_MAX_RAW_FRAMES}.")
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
            camera_actions=camera_actions,
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
        latent_mean, latent_std = self._vae_latent_stats(latent_condition)
        latent_condition = (latent_condition - latent_mean) / latent_std
        temporal_mask = latent_condition.new_zeros(
            1,
            self.vae_scale_factor_temporal,
            inputs.num_latent_frames,
            latent_condition.shape[-2],
            latent_condition.shape[-1],
        )
        temporal_mask[:, :, 0] = 1
        condition = torch.cat((temporal_mask, latent_condition), dim=1).to(dtype=dtype)
        if condition.shape[1] != 20:
            raise RuntimeError(
                "LingBot image condition must contain 4 temporal-mask then 16 image-latent channels, "
                f"got {condition.shape[1]}."
            )
        return condition

    def _prepare_camera(
        self,
        inputs: _LingBotRequestInputs,
        *,
        dtype: torch.dtype,
        previous: CameraTrajectory | None = None,
    ) -> tuple[torch.Tensor, CameraTrajectory]:
        """Convert raw camera frames to a latent-aligned ray tensor."""

        trajectory = inputs.camera_trajectory
        if trajectory is None:
            raise RuntimeError("LingBot camera trajectory was not prepared before camera embedding.")
        available_frames = int(trajectory.poses.shape[0])
        if inputs.camera_actions is not None:
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
        elif inputs.camera_actions is not None:
            # The action integrator returns post-action poses. Keep the first
            # action visible to framewise-delta conditioning by prepending its
            # known pre-action state (identity for a new realtime session).
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
        camera_embedding = build_plucker_embedding(
            embedding_trajectory,
            height=inputs.height,
            width=inputs.width,
            target_height=inputs.height,
            target_width=inputs.width,
            device=self.device,
            dtype=dtype,
        )
        if drop_anchor:
            camera_embedding = camera_embedding[1:]
        tail = CameraTrajectory(
            poses=trajectory.poses[-1:].clone(),
            intrinsics=trajectory.intrinsics[-1:].clone(),
        )
        return _fold_camera_embedding(camera_embedding), tail

    def _ar_text_caches(
        self,
        prompt_embeds: torch.Tensor,
        *,
        invalidate: bool,
    ) -> list[LingBotAttentionCache]:
        state = self._ar_diffusion_kv_state
        if state is None:
            raise RuntimeError("LingBot AR text cache requested without a bound state.")
        if invalidate:
            state.clear_cross_attention()
        if not state.is_cross_attention_populated(
            self._AR_BRANCH,
            self._AR_TEXT_CACHE,
        ):
            projected_text = self.transformer.text_embedding(prompt_embeds)

            def layer_kv() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
                for block in self.transformer.blocks:
                    cross_attention = block.cross_attn
                    key = cross_attention.norm_k(cross_attention.k(projected_text)).unflatten(
                        2,
                        (
                            cross_attention.num_local_heads,
                            cross_attention.head_dim,
                        ),
                    )
                    value = cross_attention.v(projected_text).unflatten(
                        2,
                        (
                            cross_attention.num_local_heads,
                            cross_attention.head_dim,
                        ),
                    )
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

    def _ar_transformer_cache(
        self,
        *,
        condition: torch.Tensor,
        cross_attention: list[LingBotAttentionCache],
        commit_current: bool,
    ) -> LingBotTransformerCache:
        state = self._ar_diffusion_kv_state
        if state is None:
            raise RuntimeError("LingBot AR cache requested without a bound state.")
        patch_frames, patch_height, patch_width = self.transformer.config.patch_size
        frames = condition.shape[2] // patch_frames
        height = condition.shape[3] // patch_height
        width = condition.shape[4] // patch_width
        seq_len = frames * height * width
        return LingBotTransformerCache(
            self_attention=state.get_kv_caches(
                self._AR_BRANCH,
                seq_len=seq_len,
                commit_current=commit_current,
            ),
            cross_attention=cross_attention,
        )

    def _generate_block(
        self,
        *,
        condition: torch.Tensor,
        camera: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar_cross_attention: list[LingBotAttentionCache] | None,
        start_frame: int,
        schedule: tuple[tuple[float, float], ...],
        generator: torch.Generator,
        progress_bar: TqdmProgressBar[Any],
    ) -> torch.Tensor:
        block_shape = (
            1,
            self.transformer.config.out_channels,
            condition.shape[2],
            condition.shape[3],
            condition.shape[4],
        )
        current_latents = randn_tensor(
            block_shape,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        for step_index, (timestep_value, sigma) in enumerate(schedule):
            if not self.od_config.enforce_eager:
                torch.compiler.cudagraph_mark_step_begin()
            set_forward_context_denoise_step_idx(step_index)
            timestep = torch.full((1,), float(timestep_value), device=self.device, dtype=torch.float32)
            # Checkpoint channel contract:
            # [noise/x_t(16), temporal_mask(4), image_latent(16)] -> 36.
            model_input = torch.cat((current_latents.to(dtype=condition.dtype), condition), dim=1)
            current_cache = (
                self._ar_transformer_cache(
                    condition=condition,
                    cross_attention=cast(
                        list[LingBotAttentionCache],
                        ar_cross_attention,
                    ),
                    commit_current=False,
                )
                if ar_cross_attention is not None
                else cast(LingBotTransformerCache, cache)
            )
            flow_prediction = self.transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                camera_hidden_states=camera,
                cache=current_cache,
                start_frame=start_frame,
                update_cache=False,
            )
            if flow_prediction.shape != current_latents.shape:
                raise RuntimeError(
                    "transformer flow prediction shape must match the 16-channel noise latent, "
                    f"got {tuple(flow_prediction.shape)} and {tuple(current_latents.shape)}."
                )
            # The checkpoint's flow parameterization is inverted by
            # x0 = x_t - sigma * flow. Intermediate steps re-noise that x0
            # estimate at the next sigma; the final step keeps x0 as this
            # block's generated latent.
            x0 = current_latents - sigma * flow_prediction.float()
            if step_index + 1 < len(schedule):
                next_sigma = schedule[step_index + 1][1]
                noise = randn_tensor(
                    current_latents.shape,
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
                current_latents = (1.0 - next_sigma) * x0 + next_sigma * noise
            else:
                current_latents = x0
            progress_bar.update()

        # Commit K/V only for the final clean block, never for noisy probes.
        cache_input = torch.cat((current_latents.to(dtype=condition.dtype), condition), dim=1)
        commit_cache = (
            self._ar_transformer_cache(
                condition=condition,
                cross_attention=cast(
                    list[LingBotAttentionCache],
                    ar_cross_attention,
                ),
                commit_current=True,
            )
            if ar_cross_attention is not None
            else cast(LingBotTransformerCache, cache)
        )
        if not self.od_config.enforce_eager:
            torch.compiler.cudagraph_mark_step_begin()
        self.transformer(
            hidden_states=cache_input,
            timestep=torch.zeros(1, device=self.device, dtype=torch.float32),
            encoder_hidden_states=prompt_embeds,
            camera_hidden_states=camera,
            cache=commit_cache,
            start_frame=start_frame,
            update_cache=True,
        )
        if ar_cross_attention is not None:
            state = self._ar_diffusion_kv_state
            if state is None:
                raise RuntimeError("LingBot AR state disappeared before KV commit.")
            state.commit_paged_context(self._AR_BRANCH)
        return current_latents

    def encode_prompt(
        self,
        prompt: str,
        *,
        max_sequence_length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        text_inputs = self.tokenizer(
            [" ".join(prompt.strip().split())],
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
        return prompt_embeds * attention_mask.unsqueeze(-1).to(dtype=prompt_embeds.dtype)

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        inputs = self._parse_request(req)
        tick = ARDiffusionTickRequest.from_extra_args(req.sampling_params.extra_args)
        if tick is not None and self._ar_diffusion_kv_state is None:
            raise RuntimeError("LingBot typed ticks require ARDiffusionEngine session binding.")
        if tick is None and self._ar_diffusion_kv_state is not None:
            raise ValueError("LingBot ARDiffusionEngine requests must carry ar_diffusion_tick.")
        session_state: _LingBotARSessionState | None = None
        if tick is not None:
            if tick.prompt is not None and " ".join(tick.prompt.split()) != " ".join(inputs.prompt.split()):
                raise ValueError("ar_diffusion_tick.prompt must match the standard request prompt.")
            block_frames = int(self.transformer.config.num_frames_per_block)
            if inputs.num_latent_frames != block_frames:
                raise ValueError("A LingBot AR-Diffusion request must generate exactly one three-latent-frame block.")
            if (inputs.height, inputs.width) != (
                self._ar_height,
                self._ar_width,
            ):
                raise ValueError(
                    "LingBot AR-Diffusion request resolution must match the "
                    "fixed cache geometry "
                    f"{self._ar_height}x{self._ar_width}."
                )
            horizon_latent_frames = (_MAX_RAW_FRAMES - 1) // self.vae_scale_factor_temporal + 1
            max_realtime_ticks = horizon_latent_frames // block_frames
            if tick.chunk_index >= max_realtime_ticks:
                raise ValueError(
                    "LingBot realtime generation currently supports at most "
                    f"{max_realtime_ticks} ticks per generation epoch "
                    f"(chunk_index 0 through {max_realtime_ticks - 1}) because "
                    f"the image-condition horizon is {_MAX_RAW_FRAMES} pixel "
                    "frames; reset or create a session to start a new world."
                )
            session_state = self._ar_sessions.setdefault(
                tick.session_id,
                _LingBotARSessionState(),
            )
            if tick.chunk_index != session_state.next_chunk_index:
                raise ValueError(
                    "LingBot chunk_index must be contiguous within a session: "
                    f"expected {session_state.next_chunk_index}, got "
                    f"{tick.chunk_index}."
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
        if tick is None:
            condition = self._prepare_condition(inputs, dtype=dtype)
        else:
            assert session_state is not None
            if session_state.image_condition is None:
                horizon_latent_frames = (_MAX_RAW_FRAMES - 1) // self.vae_scale_factor_temporal + 1
                session_state.image_condition = self._prepare_condition(
                    replace(
                        inputs,
                        num_frames=_MAX_RAW_FRAMES,
                        num_latent_frames=horizon_latent_frames,
                    ),
                    dtype=dtype,
                )
            block_frames = int(self.transformer.config.num_frames_per_block)
            condition_start = tick.chunk_index * block_frames
            condition_stop = condition_start + block_frames
            if condition_stop > session_state.image_condition.shape[2]:
                raise ValueError("LingBot chunk_index exceeds the configured causal image condition horizon.")
            condition = session_state.image_condition[
                :,
                :,
                condition_start:condition_stop,
            ]
        camera_pitch: float | None = None
        if inputs.camera_actions is not None:
            assert session_state is not None
            action_trajectory, camera_pitch = integrate_lingbot_camera_actions(
                inputs.camera_actions,
                width=inputs.width,
                height=inputs.height,
                initial_pose=(session_state.camera_tail.poses[-1] if session_state.camera_tail is not None else None),
                initial_pitch=session_state.camera_pitch,
            )
            inputs = replace(inputs, camera_trajectory=action_trajectory)
        camera, camera_tail = self._prepare_camera(
            inputs,
            dtype=dtype,
            previous=session_state.camera_tail if session_state is not None else None,
        )
        if camera.shape[2:] != condition.shape[2:]:
            raise RuntimeError(
                "folded camera and image condition must share latent frame/height/width geometry; "
                f"got camera={tuple(camera.shape)}, condition={tuple(condition.shape)}."
            )

        block_frames = int(self.transformer.config.num_frames_per_block)
        cache: LingBotTransformerCache | None = None
        ar_cross_attention: list[LingBotAttentionCache] | None = None
        if tick is None:
            # Direct/offline oracle keeps request-local contiguous K/V.
            cache = self.transformer.allocate_cache(
                batch_size=1,
                latent_height=condition.shape[-2],
                latent_width=condition.shape[-1],
                device=self.device,
                dtype=dtype,
            )
        else:
            assert session_state is not None
            prompt_changed = session_state.prompt is not None and session_state.prompt != inputs.prompt
            if session_state.generator_state is not None:
                inputs.generator.set_state(session_state.generator_state)
            ar_cross_attention = self._ar_text_caches(
                prompt_embeds,
                invalidate=prompt_changed,
            )
        generated_blocks: list[torch.Tensor] = []
        total_steps = (inputs.num_latent_frames // block_frames) * len(LINGBOT_DMD_TIMESTEPS)
        first_start_frame = tick.chunk_index * block_frames if tick is not None else 0
        with self.progress_bar(total=total_steps) as progress_bar:
            # Phase 3: generate left-to-right in latent-frame blocks. The
            # monotonically increasing ``start_frame`` is also the temporal
            # position and cache offset used by the Transformer.
            for local_start_frame in range(
                0,
                inputs.num_latent_frames,
                block_frames,
            ):
                start_frame = first_start_frame + local_start_frame
                stop_frame = local_start_frame + block_frames
                generated_blocks.append(
                    self._generate_block(
                        condition=condition[:, :, local_start_frame:stop_frame],
                        camera=camera[:, :, local_start_frame:stop_frame],
                        prompt_embeds=prompt_embeds,
                        cache=cache,
                        ar_cross_attention=ar_cross_attention,
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
        if tick is not None:
            assert session_state is not None
            session_state.prompt = inputs.prompt
            session_state.generator_state = inputs.generator.get_state()
            session_state.camera_tail = camera_tail
            if camera_pitch is not None:
                session_state.camera_pitch = camera_pitch
            session_state.next_chunk_index += 1

        # Phase 4: either expose model-space latents or invert the checkpoint's
        # latent normalization and decode to pixel-space video.
        if tick is not None:
            if inputs.output_type != "latent":
                raise ValueError(
                    "LingBot realtime ticks currently require output_type='latent'; "
                    "stateful streaming VAE decode is a separate integration step."
                )
            output = {
                "payload": {"latents": generated_latents},
                "metadata": {"ar_diffusion": ARDiffusionChunkMetadata.from_tick(tick).to_dict()},
            }
        elif inputs.output_type == "latent":
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return cast(set[str], loader.load_weights(weights))
