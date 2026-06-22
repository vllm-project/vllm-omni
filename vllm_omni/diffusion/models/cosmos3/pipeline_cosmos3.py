# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 text/image/video-to-video and text-to-image pipeline for vllm-omni.

Single pipeline class supports T2V, I2V, V2V, and T2I; the mode is selected at
runtime by:

* ``prompt["modalities"]`` contains ``"image"``: **T2I** (text-to-image).
* ``prompt["modalities"]`` contains ``"video"`` or is omitted: **T2V**
  (text-to-video).
* ``multi_modal_data['image']`` present on the prompt:  **I2V**
  (handled by :func:`get_cosmos3_pre_process_func`)
* ``multi_modal_data['video']`` present on the prompt with no action mode:
  **V2V** (handled by :func:`get_cosmos3_pre_process_func`)

"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    ReferenceVideoDecodeSpec,
    SupportImageInput,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.entrypoints.openai.video_api_utils import positive_float
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from .action import (
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
    ACTION_MODE_POLICY,
    action_start_frame_offset,
    build_action_condition_mask,
    build_vision_condition_mask,
    find_closest_target_size,
    load_action_tensor,
    normalize_action_mode,
    pad_action_to_dim,
    resolve_domain_id,
)
from .transfer import (
    Cosmos3TransferConfig,
    has_transfer_hints,
    load_or_compute_control_frames,
    media_hw,
    media_to_uint8_cthw,
    normalized_video_to_uint8_cthw,
    pad_temporal_frames,
    resize_center_crop_uint8_cthw,
    resolve_transfer_config,
    transfer_max_frames_from_extra_args,
    uint8_cthw_to_normalized_5d,
)
from .transformer_cosmos3 import Cosmos3VFMTransformer, resolve_sound_gen
from .utils import (
    COSMOS3_DEFAULT_CONDITION_FRAME_INDEXES_VISION,
    COSMOS3_VAE_TEMPORAL_COMPRESSION,
    ROBOLAB_CONCAT_VIEW_DESCRIPTION,
    ROBOLAB_DEFAULT_ACTION_CHUNK_SIZE,
    ROBOLAB_DEFAULT_ACTION_SPACE,
    ROBOLAB_DEFAULT_CONDITIONING_FPS,
    ROBOLAB_DEFAULT_DOMAIN_NAME,
    ROBOLAB_DEFAULT_FLOW_SHIFT,
    ROBOLAB_DEFAULT_GUIDANCE_SCALE,
    ROBOLAB_DEFAULT_IMAGE_HEIGHT,
    ROBOLAB_DEFAULT_IMAGE_WIDTH,
    ROBOLAB_DEFAULT_NUM_INFERENCE_STEPS,
    ROBOLAB_DEFAULT_RAW_ACTION_DIM,
    ROBOLAB_DEFAULT_RESOLUTION,
    ROBOLAB_MIDTRAIN_RAW_ACTION_DIM,
    RoboLabActionPostprocessInputs,
    RoboLabPolicyInputs,
    build_abs_pose_from_components,
    build_robolab_unipc_scheduler,
    condition_pixel_frame_count,
    convert_midtrain_rotation,
    ensure_2d_float_array,
    ensure_gripper_array,
    extract_robolab_image,
    extract_robolab_prompt_image,
    lazy_action_transform_pipeline,
    make_robolab_action_postprocess_inputs,
    next_robolab_seed,
    normalize_condition_frame_indexes_vision,
    normalize_condition_video_keep,
    normalize_robolab_action_space,
    pose_abs_to_rel,
    postprocess_robolab_action,
    resize_rgb_uint8,
)

logger = init_logger(__name__)

COSMOS3_DEFAULT_CONDITION_PIXEL_FRAMES = (
    max(COSMOS3_DEFAULT_CONDITION_FRAME_INDEXES_VISION) * COSMOS3_VAE_TEMPORAL_COMPRESSION + 1
)
COSMOS3_V2V_DEFAULT_FLOW_SHIFT = 10.0
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
COSMOS3_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."
COSMOS3_INVERSE_DURATION_TEMPLATE = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
COSMOS3_INVERSE_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."
COSMOS3_INVERSE_IMAGE_RESOLUTION_TEMPLATE = "This image is not of {height}x{width} resolution."
# NOTE: Intentional typo in "give" instead of "given" to match training setup.
COSMOS3_SYSTEM_PROMPT = "You are a helpful assistant who will generate videos from a give prompt."
COSMOS3_T2I_SYSTEM_PROMPT = "You are a helpful assistant who will generate images from a give prompt."

COSMOS3_T2V_DEFAULT_HEIGHT = 720
COSMOS3_T2V_DEFAULT_WIDTH = 1280
COSMOS3_T2V_DEFAULT_NUM_FRAMES = 189
COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS = 35
COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE = 6.0

COSMOS3_T2I_DEFAULT_HEIGHT = 1024
COSMOS3_T2I_DEFAULT_WIDTH = 1024
COSMOS3_T2I_DEFAULT_NUM_INFERENCE_STEPS = 50
COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE = 7.0
COSMOS3_T2I_DEFAULT_FLOW_SHIFT = 3.0
COSMOS3_T2I_DEFAULT_GUIDANCE_INTERVAL: tuple[float, float] = (400.0, 1000.0)

# Truncation cap on the prompt token count (shared by T2I and T2V).  Prompts
# are tokenized to their natural length (no padding); this only bounds the
# UND pathway / GEN cross-attention cost for pathologically long prompts.
COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH = 4096


# ---------------------------------------------------------------------------
# Post-process function (registered in registry.py)
# ---------------------------------------------------------------------------
def get_cosmos3_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process function for T2V, I2V, V2V, and action inputs.

    For T2V (no image in ``multi_modal_data``), the request is returned
    unchanged after the optional guardrails check.  For I2V (image present),
    the conditioning image is loaded, aspect-resized + center-cropped, and
    stored back on the prompt as ``additional_information.preprocessed_image``.
    For V2V (video present without action mode), source frames are cropped to
    the target size and stored as ``additional_information.preprocessed_video``.
    """
    from .guardrails import check_text_safety, ensure_initialized, is_guardrails_enabled

    video_processor = VideoProcessor(vae_scale_factor=16)
    # Eager-load guardrail models at pipeline build time when the server-level
    # gate is on. Per-request overrides only decide whether the loaded models
    # are *invoked* — they cannot turn checks on without a server-side preload.
    if is_guardrails_enabled(od_config):
        ensure_initialized(od_config)

    def _extra_args(request: OmniDiffusionRequest) -> dict[str, Any]:
        extra = getattr(getattr(request, "sampling_params", None), "extra_args", None)
        return extra if isinstance(extra, dict) else {}

    def _request_action_mode(request: OmniDiffusionRequest) -> str | None:
        return normalize_action_mode(_extra_args(request).get("action_mode"))

    def _set_transfer_size_from_image(request: OmniDiffusionRequest, image: PIL.Image.Image) -> tuple[int, int]:
        extra = _extra_args(request)
        resolution = extra.get("resolution", extra.get("image_size", 720))
        target_w, target_h = find_closest_target_size(image.height, image.width, resolution)
        request.sampling_params.height = target_h
        request.sampling_params.width = target_w
        return int(target_h), int(target_w)

    def _set_action_size_from_image(request: OmniDiffusionRequest, image: PIL.Image.Image) -> tuple[int, int]:
        sp = request.sampling_params
        if sp.height is not None and sp.width is not None:
            return int(sp.height), int(sp.width)

        extra = _extra_args(request)
        resolution = extra.get("resolution", extra.get("image_size", 480))
        target_w, target_h = find_closest_target_size(image.height, image.width, resolution)
        if sp.height is None:
            sp.height = target_h
        if sp.width is None:
            sp.width = target_w
        return int(sp.height), int(sp.width)

    def _pil_to_rgb(value: Any) -> PIL.Image.Image:
        if isinstance(value, str):
            return PIL.Image.open(value).convert("RGB")
        if isinstance(value, PIL.Image.Image):
            return value.convert("RGB")
        if isinstance(value, np.ndarray):
            array = value
            if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                array = np.transpose(array, (1, 2, 0))
            if np.issubdtype(array.dtype, np.floating):
                if array.min() < 0.0 or array.max() > 1.0:
                    array = np.clip(array, -1.0, 1.0) * 0.5 + 0.5
                array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            return PIL.Image.fromarray(array).convert("RGB")
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
                tensor = tensor.permute(1, 2, 0)
            if tensor.is_floating_point():
                if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
                    tensor = tensor.clamp(-1.0, 1.0) * 0.5 + 0.5
                tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            return PIL.Image.fromarray(tensor.numpy()).convert("RGB")
        raise TypeError(
            f"Cosmos3 preprocessing expected PIL image, numpy array, torch tensor, or path, got {type(value)!r}."
        )

    def _resize_and_pad_action_image(image: PIL.Image.Image, target_h: int, target_w: int) -> PIL.Image.Image:
        scale = min(target_w / image.width, target_h / image.height, 1.0)
        resize_w = max(1, int(scale * image.width + 0.5))
        resize_h = max(1, int(scale * image.height + 0.5))
        if (resize_w, resize_h) != image.size:
            image = image.resize((resize_w, resize_h), PIL.Image.Resampling.BICUBIC)

        array = np.asarray(image)
        pad_h = target_h - resize_h
        pad_w = target_w - resize_w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                f"Cosmos3 action image resize exceeded target size: resized={(resize_h, resize_w)}, "
                f"target={(target_h, target_w)}."
            )
        if pad_h == 0 and pad_w == 0:
            return image
        pad_mode = "reflect" if pad_h < resize_h and pad_w < resize_w else "edge"
        padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
        return PIL.Image.fromarray(padded)

    def _preprocess_action_image(image: PIL.Image.Image, target_h: int, target_w: int) -> torch.Tensor:
        image = _resize_and_pad_action_image(image, target_h, target_w)
        return video_processor.preprocess(image, height=target_h, width=target_w)

    def _preprocess_action_video(frames: list[Any], target_h: int, target_w: int) -> torch.Tensor:
        if not frames:
            raise ValueError("Cosmos3 action video input must contain at least one frame.")
        processed = [_preprocess_action_image(_pil_to_rgb(frame), target_h, target_w).squeeze(0) for frame in frames]
        return torch.stack(processed, dim=1).unsqueeze(0).contiguous()

    def _preprocess_condition_image(image: PIL.Image.Image, target_h: int, target_w: int) -> torch.Tensor:
        scale = max(target_w / image.width, target_h / image.height)
        resize_w = int(np.ceil(scale * image.width))
        resize_h = int(np.ceil(scale * image.height))
        image = image.resize((resize_w, resize_h), PIL.Image.Resampling.LANCZOS)
        left = (resize_w - target_w) // 2
        top = (resize_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))
        return video_processor.preprocess(image, height=target_h, width=target_w)

    def _video_payload_value(video: Any, key: str) -> Any:
        if isinstance(video, Mapping):
            return video.get(key)
        return getattr(video, key, None)

    def _video_payload_fps(video: Any) -> float | None:
        for key in ("fps", "frame_rate", "source_fps", "input_fps", "avg_fps", "average_fps"):
            fps = positive_float(_video_payload_value(video, key))
            if fps is not None:
                return fps
        for key in ("metadata", "info"):
            metadata = _video_payload_value(video, key)
            if metadata is None or metadata is video:
                continue
            fps = _video_payload_fps(metadata)
            if fps is not None:
                return fps
        if isinstance(video, Mapping):
            for key in ("frames", "data", "video"):
                nested = video.get(key)
                if nested is None or nested is video:
                    continue
                fps = _video_payload_fps(nested)
                if fps is not None:
                    return fps
        return None

    def _unwrap_video_payload(video: Any) -> Any:
        if isinstance(video, Mapping):
            for key in ("frames", "data", "video"):
                nested = video.get(key)
                if nested is not None:
                    return nested
        return video

    def _video_payload_to_frames(video: Any) -> list[Any]:
        video = _unwrap_video_payload(video)
        if isinstance(video, list):
            return video
        if isinstance(video, torch.Tensor):
            tensor = video.detach().cpu()
            if tensor.ndim == 5:
                if tensor.shape[0] != 1:
                    raise TypeError("Cosmos3 video preprocessing supports only batch size 1.")
                tensor = tensor[0]
            if tensor.ndim == 4 and tensor.shape[0] in (3, 4) and tensor.shape[-1] not in (3, 4):
                return [tensor[:, i] for i in range(tensor.shape[1])]
            if tensor.ndim == 4 and tensor.shape[-1] in (3, 4):
                return [tensor[i] for i in range(tensor.shape[0])]
        if isinstance(video, np.ndarray):
            array = video
            if array.ndim == 5:
                if array.shape[0] != 1:
                    raise TypeError("Cosmos3 video preprocessing supports only batch size 1.")
                array = array[0]
            if array.ndim == 4 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
                return [array[:, i] for i in range(array.shape[1])]
            if array.ndim == 4 and array.shape[-1] in (3, 4):
                return [array[i] for i in range(array.shape[0])]
        raise TypeError("Cosmos3 video input must be a non-empty list of frames or a single video tensor/array.")

    def _select_video_frames(frames: list[Any], max_frames: int, keep: str) -> list[Any]:
        if not frames:
            raise ValueError("Cosmos3 video input must contain at least one frame.")
        if keep == "last":
            return frames[-max_frames:]
        return frames[:max_frames]

    def _preprocess_condition_video(
        frames: list[Any],
        target_h: int,
        target_w: int,
        max_frames: int,
        keep: str,
    ) -> torch.Tensor:
        selected = _select_video_frames(frames, max_frames, keep)
        processed = [
            _preprocess_condition_image(_pil_to_rgb(frame), target_h, target_w).squeeze(0) for frame in selected
        ]
        return torch.stack(processed, dim=1).unsqueeze(0).contiguous()

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        action_mode = _request_action_mode(request)
        if is_guardrails_enabled(od_config, request.sampling_params):
            for prompt in request.prompts:
                text = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
                check_text_safety(text)

        for i, prompt in enumerate(request.prompts):
            if isinstance(prompt, str):
                continue
            multi_modal_data = prompt.get("multi_modal_data", {}) or {}
            raw_image = multi_modal_data.get("image")
            raw_video = multi_modal_data.get("video")
            if raw_image is None and raw_video is None:
                continue
            if raw_image is not None and raw_video is not None and action_mode is None:
                raise ValueError("Cosmos3 non-action generation accepts either image or video input, not both.")

            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            raw_video_frames: list[Any] | None = None
            transfer_input_fps: float | None = None
            if raw_video is not None:
                transfer_input_fps = _video_payload_fps(raw_video)
                raw_video_frames = _video_payload_to_frames(raw_video)
                if not raw_video_frames:
                    raise TypeError("Cosmos3 video input must be a non-empty list of PIL images or image paths.")

            if raw_image is None:
                assert raw_video_frames is not None  # raw_image and raw_video can't both be None here
                image = _pil_to_rgb(raw_video_frames[0])
            else:
                image = _pil_to_rgb(raw_image)
            extra = _extra_args(request)
            transfer_requested = action_mode is None and has_transfer_hints(extra)

            # Auto-calculate H/W from aspect ratio (720p max area)
            if transfer_requested:
                _set_transfer_size_from_image(request, image)
            elif request.sampling_params.height is None or request.sampling_params.width is None:
                if action_mode is not None:
                    _set_action_size_from_image(request, image)
                else:
                    max_area = 720 * 1280
                    aspect_ratio = image.height / image.width
                    mod_value = 16
                    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                    if request.sampling_params.height is None:
                        request.sampling_params.height = height
                    if request.sampling_params.width is None:
                        request.sampling_params.width = width

            target_w = request.sampling_params.width
            target_h = request.sampling_params.height
            if action_mode is not None:
                prompt["additional_information"]["preprocessed_image"] = _preprocess_action_image(
                    image,
                    int(target_h),
                    int(target_w),
                )
            elif raw_video is None:
                prompt["additional_information"]["preprocessed_image"] = _preprocess_condition_image(
                    image,
                    int(target_h),
                    int(target_w),
                )
            else:
                assert raw_video_frames is not None
                if transfer_requested:
                    if transfer_input_fps is not None:
                        prompt["additional_information"]["transfer_input_fps"] = transfer_input_fps
                    transfer_frames = media_to_uint8_cthw(
                        raw_video_frames,
                        height=int(target_h),
                        width=int(target_w),
                        max_frames=transfer_max_frames_from_extra_args(extra),
                    )
                    prompt["additional_information"]["preprocessed_transfer_video"] = uint8_cthw_to_normalized_5d(
                        transfer_frames,
                        dtype=torch.float32,
                    )
                else:
                    condition_frame_indexes_vision = normalize_condition_frame_indexes_vision(
                        extra.get(
                            "condition_frame_indexes_vision",
                            prompt.get("condition_frame_indexes_vision"),
                        )
                    )
                    keep = normalize_condition_video_keep(
                        extra.get("condition_video_keep", prompt.get("condition_video_keep"))
                    )
                    max_frames = condition_pixel_frame_count(condition_frame_indexes_vision)
                    prompt["additional_information"]["preprocessed_video"] = _preprocess_condition_video(
                        raw_video_frames,
                        int(target_h),
                        int(target_w),
                        max_frames,
                        keep,
                    )
                    prompt["additional_information"]["condition_frame_indexes_vision"] = list(
                        condition_frame_indexes_vision
                    )
            if action_mode is not None and raw_video_frames is not None:
                prompt["additional_information"]["preprocessed_video"] = _preprocess_action_video(
                    raw_video_frames,
                    int(target_h),
                    int(target_w),
                )
            request.prompts[i] = prompt

        return request

    return pre_process_func


def get_cosmos3_post_process_func(od_config: OmniDiffusionConfig):
    from .guardrails import check_video_safety, is_guardrails_enabled

    video_processor = VideoProcessor(vae_scale_factor=16)

    def _sampling_param(sampling_params, key: str, default=None):
        extra = getattr(sampling_params, "extra_args", None)
        if isinstance(extra, dict) and extra.get(key) is not None:
            return extra[key]
        value = getattr(sampling_params, key, None)
        return default if value is None else value

    def _resolve_output_fps(sampling_params):
        fps = (
            _sampling_param(sampling_params, "resolved_frame_rate")
            or _sampling_param(sampling_params, "frame_rate")
            or _sampling_param(sampling_params, "fps")
            or 24.0
        )
        try:
            fps_value = float(fps)
        except (TypeError, ValueError):
            fps_value = 24.0
        if fps_value <= 0:
            fps_value = 24.0
        return int(fps_value) if fps_value.is_integer() else fps_value

    def post_process_func(
        output: torch.Tensor | dict[str, torch.Tensor] | tuple,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return output

        audio = None
        audio_sample_rate = None
        if isinstance(output, dict):
            if "image" in output and "video" in output:
                raise ValueError("Cosmos3 output cannot contain both image and video payloads.")
            if "image" in output:
                video = output["image"]
            elif "video" in output:
                video = output["video"]
            else:
                raise ValueError("Cosmos3 postprocess expected an 'image' or 'video' output payload.")
            audio = output.get("audio")
            audio_sample_rate = output.get("audio_sample_rate")
        elif isinstance(output, tuple):
            if len(output) == 3:
                video, audio, audio_sample_rate = output
            elif len(output) == 2:
                video, audio = output
            else:
                raise ValueError(
                    "Cosmos3 postprocess expects output tensor, output dict, or (video, audio[, sample_rate]) tuple."
                )
        else:
            video = output

        if isinstance(output, dict) and "image" in output:
            if audio is not None:
                raise ValueError("Cosmos3 text-to-image postprocess does not support audio output.")
            if video.ndim != 5 or video.shape[2] != 1:
                raise ValueError(
                    "Cosmos3 text-to-image postprocess expects decoded output "
                    f"with shape [B, C, 1, H, W], got {tuple(video.shape)}."
                )
            image = video.squeeze(2)  # [B, 3, H, W]
            if is_guardrails_enabled(od_config, sampling_params):
                # check_video_safety expects a 5D tensor; re-add T axis.
                checked = check_video_safety(image.unsqueeze(2))
                image = checked.squeeze(2)
            return video_processor.postprocess(image, output_type="pil")
        if is_guardrails_enabled(od_config, sampling_params):
            video = check_video_safety(video)
        processed_video = video_processor.postprocess_video(video, output_type=output_type)
        if audio is None:
            return processed_video
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu()
        result = {
            "video": processed_video,
            "audio": audio,
            "fps": _resolve_output_fps(sampling_params),
        }
        if audio_sample_rate is not None:
            result["audio_sample_rate"] = int(audio_sample_rate)
        return result

    return post_process_func


def get_cosmos3_action_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def action_post_process_func(action: Any, custom_output: dict[str, Any] | None = None, sampling_params=None):
        del sampling_params
        inputs = custom_output.get("robolab_action_postprocess") if isinstance(custom_output, dict) else None
        if isinstance(inputs, RoboLabActionPostprocessInputs):
            processed_action = postprocess_robolab_action(action, inputs)
            custom_output.pop("robolab_action_postprocess", None)
            return processed_action
        return action

    return action_post_process_func


def get_cosmos3_ir_op_priority_func(od_config: OmniDiffusionConfig):
    del od_config

    def ir_op_priority_func(ir_op_priority, vllm_config=None):
        del vllm_config
        from vllm.config.kernel import IrOpPriorityConfig

        priority_kwargs = {field.name: list(getattr(ir_op_priority, field.name)) for field in fields(ir_op_priority)}
        priority_kwargs["rms_norm"] = ["native"]
        priority_kwargs["fused_add_rms_norm"] = ["native"]
        return IrOpPriorityConfig(**priority_kwargs)

    return ir_op_priority_func


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class Cosmos3OmniDiffusersPipeline(
    nn.Module, CFGParallelMixin, SupportImageInput, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    """Cosmos3 text/image/video-to-video / text-to-image pipeline.

    Architecture: Mixture-of-Transformers with Qwen3-VL backbone.
    - Understanding pathway: causal self-attention on text (runs once, K/V cached)
    - Generation pathway: cross-attention on noisy visual latents (runs each step)

    Supports T2V, I2V, V2V, and T2I from the same class.  Mode is selected at
    runtime:

    * **T2I** when ``prompt["modalities"]`` contains ``"image"``.  Latent
      T-dim is forced to 1, T2I-specific scheduler defaults are applied (50 steps,
      flow_shift=3.0, guidance_interval=[400, 1000]), the duration
      template is suppressed, and post-process emits PIL images.
    * **I2V** when the request supplies a preprocessed image via
      ``multi_modal_data['image']`` (handled by
      :func:`get_cosmos3_pre_process_func`) and the requested output modality
      is not image.
      Frame 0 of the initial latent is set to the VAE-encoded conditioning
      image, frame-0 noise predictions are masked to zero, and the clean
      image latent is re-injected at frame 0 after each scheduler step.
    * **V2V** when the request supplies a preprocessed video via
      ``multi_modal_data['video']`` without an action mode. Explicit latent
      frame indexes are kept clean with ``noisy_frame_mask`` and re-injected
      after each scheduler step.
    * **T2V** otherwise (default video generation).
    """

    support_image_input: ClassVar[bool] = True
    color_format: ClassVar[str] = "RGB"

    @classmethod
    def reference_video_decode_spec(
        cls,
        *,
        num_frames: int | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> ReferenceVideoDecodeSpec:
        extra_args = extra_args if isinstance(extra_args, dict) else {}
        if has_transfer_hints(extra_args):
            max_frames = transfer_max_frames_from_extra_args(extra_args)
            if num_frames is not None:
                max_frames = min(max_frames, int(num_frames))
            return ReferenceVideoDecodeSpec(max_frames=max_frames, keep="first")

        action_mode = normalize_action_mode(extra_args.get("action_mode"))
        if action_mode is not None:
            if num_frames is not None:
                return ReferenceVideoDecodeSpec(max_frames=int(num_frames), keep="first")
            action_chunk_size = extra_args.get("action_chunk_size")
            if action_chunk_size is not None:
                try:
                    max_frames = int(action_chunk_size) + 1
                except (TypeError, ValueError):
                    max_frames = None
                if max_frames is not None and max_frames > 0:
                    return ReferenceVideoDecodeSpec(max_frames=max_frames, keep="first")
            return ReferenceVideoDecodeSpec(max_frames=None, keep="first")

        condition_indexes = normalize_condition_frame_indexes_vision(extra_args.get("condition_frame_indexes_vision"))
        max_frames = condition_pixel_frame_count(condition_indexes)
        if num_frames is not None:
            max_frames = min(max_frames, int(num_frames))
        keep = normalize_condition_video_keep(extra_args.get("condition_video_keep"))
        return ReferenceVideoDecodeSpec(max_frames=max_frames, keep=keep)

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if od_config.enable_cpu_offload:
            raise ValueError(
                "Cosmos3 has no separate text encoder, so CPU offloading "
                "(transformer↔encoder swapping) is not supported. "
                "Use --enable-layerwise-offload instead."
            )
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = od_config.dtype

        model_path = od_config.model
        local_files_only = os.path.exists(model_path)

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="text_tokenizer",
            local_files_only=local_files_only,
        )

        # --- VAE ---
        self.vae = DistributedAutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        if not hasattr(self.vae.config, "scale_factor_temporal"):
            raise ValueError(
                "Cosmos3 Diffusers VAE config must define scale_factor_temporal "
                "so transformer mRoPE temporal positions can be computed correctly."
            )
        self.vae_scale_factor_temporal = int(self.vae.config.scale_factor_temporal)
        self.vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 16)

        sound_gen = resolve_sound_gen(od_config)
        sound_dim = None
        sound_latent_fps = None
        self._sound_tokenizer = None
        if sound_gen:
            self._sound_tokenizer = self._get_sound_tokenizer()
            sound_dim = self._sound_tokenizer.latent_ch
            sound_latent_fps = self._sound_tokenizer.latent_fps

        # --- Transformer (weights loaded later via weights_sources) ---
        self.transformer = Cosmos3VFMTransformer(
            od_config=od_config,
            temporal_compression_factor=self.vae_scale_factor_temporal,
            sound_gen=sound_gen,
            sound_dim=sound_dim,
            sound_latent_fps=sound_latent_fps,
        )

        # --- Scheduler ---
        # Load from checkpoint to preserve solver_order, timestep_spacing,
        # beta_schedule, sigma bounds, flow_shift, etc. Only override
        # flow_shift when explicitly requested by the user.
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        init_flow_shift = (
            od_config.flow_shift
            if od_config.flow_shift is not None
            else getattr(self.scheduler.config, "flow_shift", 1.0)
        )
        self.scheduler = UniPCMultistepScheduler.from_config(
            self.scheduler.config,
            flow_shift=init_flow_shift,
            use_karras_sigmas=False,
        )
        self._cpu_scheduler_state()

        # --- Video processor for post-decode ---
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # --- Weight sources for DiffusersPipelineLoader ---
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
                allow_patterns_overrides=["transformer/*.safetensors"],
            ),
        ]

        # Snapshot the loaded scheduler config so we can rebuild the
        # scheduler at request time when a per-request flow_shift override
        # is supplied (T2I uses shift=3.0; T2V/I2V use the engine default).
        self._base_scheduler_config = self.scheduler.config
        self._engine_init_flow_shift = float(getattr(self.scheduler.config, "flow_shift", 1.0) or 1.0)
        self._current_flow_shift = self._engine_init_flow_shift

        self._guidance_scale = None
        self._num_timesteps = None
        self._cosmos3_branch_caches: dict[str, tuple[Any, Any]] | None = None
        self._robolab_transform = None

        # Set True by ``enable_cache_for_cosmos3`` when cache-dit is enabled on
        # this pipeline. Tells the sequential-CFG loop to keep paired
        # cond/uncond forwards so cache-dit's has_separate_cfg step accounting
        # stays in sync.
        self._cache_dit_requires_paired_cfg = False

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # -- Weight loading --------------------------------------------------------

    @staticmethod
    def _remap_ckpt_key(key: str) -> str | None:
        """Remap a Diffusers transformer key to the model parameter namespace.

        Checkpoint keys arrive with a synthetic ``transformer.`` prefix from
        ``weights_sources``.  The source checkpoint itself uses the prefixless
        Diffusers transformer namespace: top-level projections plus Qwen3-VL
        backbone keys.  UND and GEN components share each layer in the source
        and are split into separate module lists here.  Some sources wrap the
        transformer namespace under ``model.``; that wrapper is structural and
        is stripped before applying the Cosmos3 leaf-name remap.

        Returns the remapped name under ``transformer.``, or None to skip.
        """
        k = key
        # Strip the weights_sources prefix
        if k.startswith("transformer."):
            k = k[len("transformer.") :]
        if k.startswith("model."):
            k = k[len("model.") :]

        # Top-level generation components.
        if k.startswith(
            (
                "proj_in.",
                "proj_out.",
                "time_embedder.",
                "audio_proj_in.",
                "audio_proj_out.",
                "action_proj_in.",
                "action_proj_out.",
            )
        ):
            return f"transformer.{k}"
        if k in ("audio_modality_embed", "audio_modality_embed.weight"):
            return "transformer.audio_modality_embed"
        if k in ("action_modality_embed", "action_modality_embed.weight"):
            return "transformer.action_modality_embed"
        if k.startswith("action_pos_embed."):
            return None

        # Skip lm_head
        if k.startswith("lm_head."):
            return None

        # embed_tokens / norm -> language_model.*
        if k.startswith("embed_tokens."):
            return f"transformer.language_model.{k}"
        if k.startswith("norm."):
            return f"transformer.language_model.{k}"

        # norm_moe_gen -> top level
        if k.startswith("norm_moe_gen."):
            return f"transformer.{k}"

        if not k.startswith("layers."):
            return None

        parts = k.split(".", 2)  # ['layers', '{i}', '{rest}']
        if len(parts) != 3:
            return None
        layer_idx = parts[1]
        rest = parts[2]

        und_lp = f"transformer.language_model.layers.{layer_idx}"
        gen_lp = f"transformer.gen_layers.{layer_idx}"

        _LAYER_MAP = {
            # UND attention
            "self_attn.to_q.": f"{und_lp}.self_attn.to_q.",
            "self_attn.to_k.": f"{und_lp}.self_attn.to_k.",
            "self_attn.to_v.": f"{und_lp}.self_attn.to_v.",
            "self_attn.to_out.": f"{und_lp}.self_attn.to_out.",
            "self_attn.norm_q.": f"{und_lp}.self_attn.norm_q.",
            "self_attn.norm_k.": f"{und_lp}.self_attn.norm_k.",
            # GEN attention
            "self_attn.add_q_proj.": f"{gen_lp}.cross_attention.to_q.",
            "self_attn.add_k_proj.": f"{gen_lp}.cross_attention.to_k.",
            "self_attn.add_v_proj.": f"{gen_lp}.cross_attention.to_v.",
            "self_attn.to_add_out.": f"{gen_lp}.cross_attention.to_out.",
            "self_attn.norm_added_q.": f"{gen_lp}.cross_attention.norm_q.",
            "self_attn.norm_added_k.": f"{gen_lp}.cross_attention.norm_k.",
            # Norms
            "input_layernorm.": f"{und_lp}.input_layernorm.",
            "post_attention_layernorm.": f"{und_lp}.post_attention_layernorm.",
            "input_layernorm_moe_gen.": f"{gen_lp}.input_layernorm.",
            "post_attention_layernorm_moe_gen.": f"{gen_lp}.post_attention_layernorm.",
            # UND MLP
            "mlp.gate_proj.": f"{und_lp}.mlp.gate_proj.",
            "mlp.up_proj.": f"{und_lp}.mlp.up_proj.",
            "mlp.down_proj.": f"{und_lp}.mlp.down_proj.",
            # GEN MLP
            "mlp_moe_gen.gate_proj.": f"{gen_lp}.mlp.gate_proj.",
            "mlp_moe_gen.up_proj.": f"{gen_lp}.mlp.up_proj.",
            "mlp_moe_gen.down_proj.": f"{gen_lp}.mlp.down_proj.",
        }

        for pattern, replacement in _LAYER_MAP.items():
            if rest.startswith(pattern):
                suffix = rest[len(pattern) :]
                return replacement + suffix

        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream-remap checkpoint weights and load via AutoWeightsLoader.

        Handles quantization, TP-aware weight_loader, and buffer loading.
        Returns the set of loaded parameter names for strict validation.
        """
        state = self.state_dict()
        allowed = set(state.keys())
        tp_aware = {n for n, p in self.named_parameters() if hasattr(p, "weight_loader")}

        def _remapped_weights() -> Iterable[tuple[str, torch.Tensor]]:
            total = kept = 0
            for name, tensor in weights:
                total += 1
                remapped = self._remap_ckpt_key(name)
                if remapped is not None and (remapped in allowed or remapped in tp_aware):
                    kept += 1
                    yield remapped, tensor
            if _is_rank_zero():
                logger.info(
                    "Cosmos3 weight remap: kept %d/%d tensors",
                    kept,
                    total,
                )

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_remapped_weights())
        self.transformer.post_load_weights()
        self.transformer.eval()
        if getattr(self.transformer, "sound_gen", False):
            sound_markers = ("audio_proj_in.", "audio_proj_out.", "audio_modality_embed")
            missing = [marker.rstrip(".") for marker in sound_markers if not any(marker in name for name in loaded)]
            if missing:
                raise ValueError(
                    "Cosmos3 transformer config enables sound generation, but "
                    f"the checkpoint is missing sound weights for {missing}. "
                    "Use a sound-capable transformer checkpoint."
                )
        if getattr(self.transformer, "action_gen", False):
            action_markers = ("action_proj_in.", "action_proj_out.", "action_modality_embed")
            missing = [marker.rstrip(".") for marker in action_markers if not any(marker in name for name in loaded)]
            if missing:
                raise ValueError(
                    "Cosmos3 transformer config enables action generation, but "
                    f"the checkpoint is missing action weights for {missing}. "
                    "Use an action-capable transformer checkpoint."
                )
        return loaded

    def predict_noise(self, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Override CFGParallelMixin.predict_noise for Cosmos3.

        The transformer returns the raw prediction: video-only as a tensor,
        or a tuple in video, action, sound order for multimodal generation.
        """
        cache_key = kwargs.pop("_cosmos3_cache_key", None)
        if cache_key is None:
            return self.transformer(**kwargs)

        branch_caches = self._cosmos3_branch_caches
        if branch_caches is None:
            return self.transformer(**kwargs)

        cache_key = str(cache_key)
        self.transformer.cached_kv, self.transformer.cached_freqs_gen = branch_caches.get(cache_key, (None, None))
        prediction = self.transformer(**kwargs)
        branch_caches[cache_key] = (self.transformer.cached_kv, self.transformer.cached_freqs_gen)
        return prediction

    def combine_multi_branch_cfg_noise(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        true_cfg_scale: float | dict[str, float],
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(true_cfg_scale, dict) or true_cfg_scale.get("mode") != "cosmos3_transfer":
            return super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize)
        mode = str(true_cfg_scale.get("branch_mode", "control_and_text"))
        guidance_scale = float(true_cfg_scale.get("guidance_scale", 1.0))
        control_guidance = float(true_cfg_scale.get("control_guidance", 1.0))

        if mode == "control_only":
            if len(predictions) != 2:
                raise ValueError(f"Cosmos3 transfer control-only CFG expects 2 branches, got {len(predictions)}.")
            cond_full, cond_no_control = predictions
            if isinstance(cond_full, tuple) or isinstance(cond_no_control, tuple):
                raise ValueError("Cosmos3 transfer control-only CFG expects video-only tensor predictions.")
            cfg_reference = cond_full
            combined = cond_no_control + control_guidance * (cond_full - cond_no_control)
        elif mode == "text_only":
            if len(predictions) != 2:
                raise ValueError(f"Cosmos3 transfer text-only CFG expects 2 branches, got {len(predictions)}.")
            cond_full, uncond_full = predictions
            if isinstance(cond_full, tuple) or isinstance(uncond_full, tuple):
                raise ValueError("Cosmos3 transfer text-only CFG expects video-only tensor predictions.")
            cfg_reference = cond_full
            combined = uncond_full + guidance_scale * (cond_full - uncond_full)
        elif mode == "control_and_text":
            if len(predictions) != 3:
                raise ValueError(f"Cosmos3 transfer control+text CFG expects 3 branches, got {len(predictions)}.")
            cond_full, cond_no_control, uncond_full = predictions
            if isinstance(cond_full, tuple) or isinstance(cond_no_control, tuple) or isinstance(uncond_full, tuple):
                raise ValueError("Cosmos3 transfer control+text CFG expects video-only tensor predictions.")
            cfg_reference = cond_full
            control_cond = cond_no_control + control_guidance * (cond_full - cond_no_control)
            combined = uncond_full + guidance_scale * (control_cond - uncond_full)
        else:
            raise ValueError(f"Unknown Cosmos3 transfer CFG branch_mode={mode!r}.")

        if cfg_normalize:
            combined = self.cfg_normalize_function(cfg_reference, combined)
        return combined

    @staticmethod
    def _cfg_parallel_active() -> bool:
        try:
            return get_classifier_free_guidance_world_size() > 1
        except Exception:
            return False

    def _cache_requires_paired_cfg(self) -> bool:
        """Whether the sequential-CFG denoising loop must keep paired forwards.

        cache-dit wraps the GEN pathway with ``has_separate_cfg=True`` and
        distinguishes the conditional vs unconditional passes purely by the
        parity of its transformer-forward counter.  The T2I ``guidance_interval``
        optimization that skips the uncond pass outside the interval would
        desync that accounting (cond passes get mislabeled as uncond and the
        per-generation step counter drifts).  ``enable_cache_for_cosmos3`` sets
        the marker below when it enables cache-dit on this pipeline; the loop
        then keeps both passes and neutralizes CFG via scale=1.0 instead.

        Returns False when cache-dit is not active, preserving the skip speedup.
        """
        return self._cache_dit_requires_paired_cfg

    @staticmethod
    def _get_sp_param(sp: OmniDiffusionSamplingParams, key: str, default: Any = None) -> Any:
        """Read a runtime control from sampling params.

        Order of precedence:
            1. ``sp.extra_args[key]`` - preferred path; the OpenAI image/video
               endpoints surface custom controls here (see e.g.
               ``serving_video.py`` writing ``extra_args['flow_shift']``).
            2. direct attribute on ``sp`` - backward compat for callers that
               set attributes directly.
            3. ``default``.

        Skipping this helper would cause API-driven overrides like
        ``request.flow_shift`` (forwarded as ``extra_args['flow_shift']``) to
        be silently ignored.
        """
        extra = sp.extra_args or {}
        if extra.get(key) is not None:
            return extra[key]
        val = getattr(sp, key, None)
        if val is not None:
            return val
        return default

    def _get_robolab_transform(self):
        if self._robolab_transform is None:
            action_dim = int(getattr(self.transformer, "action_dim", 64))
            self._robolab_transform = lazy_action_transform_pipeline(action_dim)
        return self._robolab_transform

    def _build_robolab_policy_inputs(
        self,
        sp: OmniDiffusionSamplingParams,
        prompt_data: Any | None = None,
        request_id: str | None = None,
    ) -> RoboLabPolicyInputs | None:
        extra = sp.extra_args if isinstance(sp.extra_args, dict) else {}
        obs = extra.get("robot_obs")
        if obs is None:
            obs = extra.get("observation")
        if obs is None:
            return None
        if not isinstance(obs, dict):
            raise TypeError(f"Cosmos3 RoboLab observation must be a dict, got {type(obs)!r}.")

        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("RoboLab observation must contain string key 'prompt'.")

        def extra_param(key: str, default: Any) -> Any:
            value = extra.get(key)
            return default if value is None else value

        def extra_param_alias(primary_key: str, alias_key: str, default: Any) -> Any:
            value = extra.get(primary_key)
            if value is not None:
                return value
            value = extra.get(alias_key)
            return default if value is None else value

        action_space = normalize_robolab_action_space(extra_param("action_space", ROBOLAB_DEFAULT_ACTION_SPACE))
        action_chunk_size = int(extra_param("action_chunk_size", ROBOLAB_DEFAULT_ACTION_CHUNK_SIZE))
        raw_action_dim_default = (
            ROBOLAB_DEFAULT_RAW_ACTION_DIM if action_space == "joint_pos" else ROBOLAB_MIDTRAIN_RAW_ACTION_DIM
        )
        raw_action_dim = int(extra_param("raw_action_dim", raw_action_dim_default))
        image_h = int(extra_param("image_height", ROBOLAB_DEFAULT_IMAGE_HEIGHT))
        image_w = int(extra_param("image_width", ROBOLAB_DEFAULT_IMAGE_WIDTH))
        history_length = int(extra_param("history_length", 1))
        use_state = self._truthy(extra_param("use_state", True))
        resolution = str(extra_param("resolution", ROBOLAB_DEFAULT_RESOLUTION))
        fps = float(extra_param("conditioning_fps", ROBOLAB_DEFAULT_CONDITIONING_FPS))
        domain_name = str(extra_param("domain_name", ROBOLAB_DEFAULT_DOMAIN_NAME))
        domain_id = resolve_domain_id(domain_name=domain_name, require_explicit=True)

        if use_state and history_length < 1:
            raise ValueError("RoboLab history_length must be >= 1 when use_state is true.")
        if action_chunk_size <= 0:
            raise ValueError(f"RoboLab action_chunk_size must be positive, got {action_chunk_size}.")
        if raw_action_dim <= 0:
            raise ValueError(f"RoboLab raw_action_dim must be positive, got {raw_action_dim}.")

        try:
            image = extract_robolab_image(obs)
        except ValueError as exc:
            image = extract_robolab_prompt_image(prompt_data)
            if image is None:
                raise exc
        if image.shape[:2] != (image_h, image_w):
            image = resize_rgb_uint8(image, (image_h, image_w))

        t_frames = action_chunk_size + 1
        video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)

        use_state_rows = 1 if use_state else 0
        action = torch.zeros((action_chunk_size + use_state_rows, raw_action_dim), dtype=torch.float32)
        history_action = None
        num_history_rows = history_length - use_state_rows
        gripper_position = 1.0 - ensure_gripper_array(obs["observation/gripper_position"])

        if action_space == "joint_pos":
            joint_position = ensure_2d_float_array(obs["observation/joint_position"], "observation/joint_position", 7)
            if use_state:
                action[0] = torch.from_numpy(np.concatenate((joint_position[-1], gripper_position[-1])))
            if num_history_rows > 0:
                if len(joint_position) < num_history_rows + 1:
                    raise ValueError("Not enough joint_position rows for requested history_length.")
                history_np = np.concatenate(
                    (joint_position[-num_history_rows - 1 : -1], gripper_position[-num_history_rows - 1 : -1]),
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()
        else:
            eef_pos = ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            if use_state:
                rot6d = convert_midtrain_rotation(eef_quat[-1], "quat_xyzw", "rot6d")
                action[0] = torch.from_numpy(np.concatenate((eef_pos[-1], rot6d, gripper_position[-1])))
            if num_history_rows > 0:
                if len(eef_pos) < num_history_rows + 1 or len(eef_quat) < num_history_rows + 1:
                    raise ValueError("Not enough eef_pos/eef_quat rows for requested history_length.")
                poses_abs = build_abs_pose_from_components(eef_pos, eef_quat, "quat_xyzw")
                poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
                history_np = np.concatenate(
                    [poses_rel[-num_history_rows:], gripper_position[-num_history_rows:]],
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()

        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            # Cosmos Framework consumes this as an integer conditioning bucket.
            "conditioning_fps": torch.tensor(fps, dtype=torch.long),
            "mode": ACTION_MODE_POLICY,
            "domain_id": torch.tensor(domain_id, dtype=torch.long),
            "viewpoint": "concat_view",
            "additional_view_description": ROBOLAB_CONCAT_VIEW_DESCRIPTION,
        }
        if history_action is not None:
            sample["history_action"] = history_action

        sample = self._get_robolab_transform()(sample, resolution)
        sequence_plan = sample["sequence_plan"]
        video_tensor = sample["video"].float() / 127.5 - 1.0
        raw_action_dim_tensor = sample.get("raw_action_dim")
        if isinstance(raw_action_dim_tensor, torch.Tensor):
            transformed_raw_action_dim = int(raw_action_dim_tensor.item())
        else:
            transformed_raw_action_dim = raw_action_dim

        return RoboLabPolicyInputs(
            prompt=sample["ai_caption"],
            video_tensor=video_tensor.unsqueeze(0),
            action_tensor=sample["action"].float(),
            action_condition_indexes=list(getattr(sequence_plan, "condition_frame_indexes_action", []) or []),
            action_start_frame_offset=int(getattr(sequence_plan, "action_start_frame_offset", 1)),
            raw_action_dim=transformed_raw_action_dim,
            domain_id=domain_id,
            fps=fps,
            height=int(video_tensor.shape[-2]),
            width=int(video_tensor.shape[-1]),
            image_size=sample.get("image_size"),
            num_frames=int(video_tensor.shape[1]),
            num_inference_steps=int(
                extra_param_alias("num_inference_steps", "num_steps", ROBOLAB_DEFAULT_NUM_INFERENCE_STEPS)
            ),
            guidance_scale=float(extra_param_alias("guidance_scale", "guidance", ROBOLAB_DEFAULT_GUIDANCE_SCALE)),
            flow_shift=float(extra_param_alias("flow_shift", "shift", ROBOLAB_DEFAULT_FLOW_SHIFT)),
            seed=next_robolab_seed(extra, obs, request_id),
            history_length=history_length,
            action_space=action_space,
            observation=obs,
        )

    @staticmethod
    def _build_action_condition_mask_from_indexes(
        indexes: list[int],
        action_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(1, action_length, 1, device=device, dtype=dtype)
        for idx in indexes:
            if idx < 0 or idx >= action_length:
                raise ValueError(f"Action condition index {idx} is out of range for action length {action_length}.")
            mask[:, idx, :] = 1.0
        return mask

    def _forward_robolab_policy(
        self,
        sp: OmniDiffusionSamplingParams,
        inputs: RoboLabPolicyInputs,
        pipeline_start: float,
    ) -> DiffusionOutput:
        if not getattr(self.transformer, "action_gen", False):
            raise ValueError(
                "Cosmos3 RoboLab policy serving was requested, but the transformer "
                "was initialized without action modules. Check that the checkpoint "
                "config enables action_gen and includes action weights."
            )

        action_mode = ACTION_MODE_POLICY
        height = inputs.height
        width = inputs.width
        num_frames = inputs.num_frames
        action_chunk_size = int(inputs.action_tensor.shape[0])
        num_inference_steps = inputs.num_inference_steps
        guidance_scale = float(inputs.guidance_scale)
        flow_shift_target = float(inputs.flow_shift)
        domain_id = int(inputs.domain_id)
        frame_rate = self._get_sp_param(sp, "resolved_frame_rate") or self._get_sp_param(sp, "frame_rate") or inputs.fps
        max_sequence_length = (
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH)
            or COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH
        )
        use_system_prompt = bool(self._get_sp_param(sp, "use_system_prompt", False))

        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps

        generator = sp.generator
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(int(inputs.seed))

        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            inputs.prompt,
            "",
            num_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=False,
        )

        action_video_tensor = inputs.video_tensor
        if action_video_tensor.ndim == 4:
            action_video_tensor = action_video_tensor.unsqueeze(0)
        if action_video_tensor.ndim != 5:
            raise ValueError(
                "Cosmos3 RoboLab action video tensor must have shape [1, 3, T, H, W] "
                f"or [3, T, H, W], got {tuple(action_video_tensor.shape)}."
            )
        if action_video_tensor.shape[2] < num_frames:
            pad = action_video_tensor[:, :, -1:].repeat(1, 1, num_frames - action_video_tensor.shape[2], 1, 1)
            action_video_tensor = torch.cat([action_video_tensor, pad], dim=2)
        elif action_video_tensor.shape[2] > num_frames:
            action_video_tensor = action_video_tensor[:, :, :num_frames]

        action_latents, action_velocity_mask, action_condition_latents, raw_action_dim = self._prepare_action_latents(
            mode=action_mode,
            action_chunk_size=action_chunk_size,
            raw_action_dim=int(inputs.raw_action_dim),
            generator=generator,
            sp=sp,
            clean_action=inputs.action_tensor,
            condition_indexes=inputs.action_condition_indexes,
        )
        action_offset = int(inputs.action_start_frame_offset)

        latents, velocity_mask, condition_latents = self._prepare_latents_action_video(
            action_video_tensor,
            action_mode,
            height,
            width,
            num_frames,
            generator,
            image_size=inputs.image_size,
        )
        image_latent = condition_latents[:, :, 0:1]

        video_shape = (latents.shape[2], latents.shape[3], latents.shape[4])
        shared_kwargs = dict(
            video_shape=video_shape,
            fps=frame_rate,
            noisy_frame_mask=velocity_mask,
            action_domain_ids=torch.tensor([domain_id], dtype=torch.long, device=self.device),
            action_noisy_mask=action_velocity_mask,
            action_start_frame_offset=action_offset,
            action_fps=float(self._get_sp_param(sp, "action_fps", frame_rate) or frame_rate),
        )

        scheduler = build_robolab_unipc_scheduler(num_inference_steps, flow_shift_target, self.device)
        _, action_latents = self.diffuse(
            latents=latents,
            timesteps=scheduler.timesteps,
            cond_ids=cond_ids,
            cond_mask=cond_mask,
            uncond_ids=uncond_ids,
            uncond_mask=uncond_mask,
            guidance_scale=guidance_scale,
            shared_kwargs=shared_kwargs,
            action_latents=action_latents,
            action_velocity_mask=action_velocity_mask,
            action_condition_latents=action_condition_latents,
            sound_latents=None,
            velocity_mask=velocity_mask,
            image_latent=image_latent,
            condition_latents=condition_latents,
            guidance_interval=None,
            raw_action_dim=raw_action_dim,
            scheduler=scheduler,
        )

        if _is_rank_zero():
            logger.info("Total pipeline time: %.2fs", time.time() - pipeline_start)

        action = action_latents[:, :, :raw_action_dim].detach().cpu()
        custom_action_output: dict[str, Any] = {
            "action": action,
            "raw_action_dim": raw_action_dim,
            "action_mode": action_mode,
            "domain_id": domain_id,
            "action_only_output": True,
            "robolab_action_postprocess": make_robolab_action_postprocess_inputs(inputs),
        }
        return DiffusionOutput(output={}, custom_output=custom_action_output)

    @staticmethod
    def _truthy(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def _get_prompt_param(cls, prompt_data, key: str, default=None):
        if not isinstance(prompt_data, dict):
            return default
        if prompt_data.get(key) is not None:
            return prompt_data[key]
        additional = prompt_data.get("additional_information")
        if isinstance(additional, dict) and additional.get(key) is not None:
            return additional[key]
        return default

    @classmethod
    def _is_sound_request(cls, prompt_data, sp) -> bool:
        for key in ("generate_sound", "sound_gen"):
            if cls._truthy(cls._get_prompt_param(prompt_data, key, None)):
                return True
            if cls._truthy(cls._get_sp_param(sp, key, None)):
                return True
        return False

    @classmethod
    def _get_action_mode(cls, prompt_data, sp) -> str | None:
        return normalize_action_mode(
            cls._get_sp_param(sp, "action_mode", cls._get_prompt_param(prompt_data, "action_mode", None))
        )

    def _get_sound_tokenizer(self):
        if self._sound_tokenizer is None:
            from .sound_tokenizer import Cosmos3SoundTokenizer

            self._sound_tokenizer = Cosmos3SoundTokenizer.from_config(self.od_config)
        return self._sound_tokenizer

    @staticmethod
    def _is_t2i_request(req: OmniDiffusionRequest) -> bool:
        """Detect text-to-image mode from request-level prompt modalities."""
        if not req.prompts:
            return False
        first_prompt = req.prompts[0]
        modalities = first_prompt.get("modalities", []) if isinstance(first_prompt, dict) else []
        if modalities is None:
            modalities = []
        if isinstance(modalities, str):
            modalities = [modalities]
        if "image" in modalities and "video" in modalities:
            raise ValueError("Cosmos3 prompt modalities cannot request both image and video output.")

        accepted_modalities = ["image", "video", "text", "audio"]
        if any(x not in accepted_modalities for x in modalities):
            raise ValueError(f"Incorrect modality value in {modalities}, expected one of {accepted_modalities}.")
        return "image" in modalities

    def _set_flow_shift(self, target_shift: float) -> None:
        """Set the UniPC ``flow_shift`` to a concrete target value.

        The scheduler is rebuilt from the saved base config if
        the target differs from the current shift.  Tracking
        ``self._current_flow_shift`` explicitly is required because the
        previous mode may have rebuilt the scheduler - we cannot rely on
        ``self.scheduler.config.flow_shift`` reflecting the last requested
        target if a rebuild was skipped via the equality check.
        """
        target = float(target_shift)
        if target == float(self._current_flow_shift):
            return
        self.scheduler = UniPCMultistepScheduler.from_config(
            self._base_scheduler_config, flow_shift=target, use_karras_sigmas=False
        )
        self._cpu_scheduler_state()
        self._current_flow_shift = target

    def _cpu_scheduler_state(self) -> None:
        # We need to move scheduler tensors to CPU, as unipc from diffusers assumes they are on CPU.
        # However, after the creation they are on GPU due to "with target_device:" in diffusers_loader.py
        for name, value in vars(self.scheduler).items():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                setattr(self.scheduler, name, value.cpu())

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # -- Prompt formatting -----------------------------------------------------

    @staticmethod
    def _apply_metadata_templates(
        prompt: str,
        num_frames: int,
        frame_rate: float,
        height: int,
        width: int,
        duration_template: str | None = COSMOS3_DURATION_TEMPLATE,
        resolution_template: str | None = COSMOS3_RESOLUTION_TEMPLATE,
        force_duration_template: bool = False,
    ) -> str:
        """
        Append duration and resolution metadata to a prompt.
        """
        parts: list[str] = []
        head = prompt.rstrip(".").strip()
        if head:
            parts.append(head)
        if duration_template is not None and (num_frames > 1 or force_duration_template):
            duration = num_frames / frame_rate
            parts.append(duration_template.format(duration=duration, fps=frame_rate).rstrip("."))
        if resolution_template is not None:
            parts.append(resolution_template.format(height=height, width=width).rstrip("."))
        if not parts:
            return ""
        return ". ".join(parts) + "."

    # -- Tokenization --------------------------------------------------------

    def _tokenize_prompt(
        self,
        text: str,
        max_sequence_length: int,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a prompt using the Qwen2 chat template.

        Returns (input_ids, attention_mask) as [1, S] tensors on device.
        """
        conversations = []
        if use_system_prompt:
            conversations.append(
                {
                    "role": "system",
                    "content": system_prompt or COSMOS3_SYSTEM_PROMPT,
                }
            )
        conversations.append({"role": "user", "content": text})

        token_ids = self._normalize_token_ids(
            self.tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True)
        )
        original_token_count = len(token_ids)
        if original_token_count > max_sequence_length and _is_rank_zero():
            logger.warning(
                "Cosmos3 prompt token_ids shortened to max_sequence_length: "
                "original_token_count=%d, max_sequence_length=%d, removed_token_count=%d",
                original_token_count,
                max_sequence_length,
                original_token_count - max_sequence_length,
            )
        token_ids = token_ids[:max_sequence_length]
        token_ids.append(self.tokenizer.eos_token_id)  # 151645
        token_ids.append(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))  # 151652
        seq_len = len(token_ids)

        # No right-padding: the prompt is tokenized to its natural length.
        # The UND pathway uses causal self-attention with no padding mask and
        # the GEN cross-attention K/V is trimmed to the real text length, so
        # padding to a fixed length only added dead compute and never changed
        # the output.  ``max_sequence_length`` is kept purely as a truncation
        # cap (above).  The mask is therefore all ones.
        attention_mask = [1] * seq_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    @staticmethod
    def _normalize_token_ids(tokenized_output: object) -> list[int]:
        """Normalize tokenizer outputs into a flat ``list[int]``.

        Different Transformers/tokenizers versions can return ``list[int]``,
        a mapping/BatchEncoding with ``input_ids``, tensors, or
        ``tokenizers.Encoding`` objects from ``apply_chat_template``.
        """
        token_ids = tokenized_output
        while True:
            if isinstance(token_ids, dict) and "input_ids" in token_ids:
                token_ids = token_ids["input_ids"]
            elif hasattr(token_ids, "input_ids"):
                token_ids = token_ids.input_ids
            elif hasattr(token_ids, "ids"):
                token_ids = token_ids.ids
            elif hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            elif isinstance(token_ids, tuple):
                token_ids = list(token_ids)
            elif isinstance(token_ids, list) and len(token_ids) == 1:
                first = token_ids[0]
                if isinstance(first, list | tuple):
                    token_ids = list(first)
                elif hasattr(first, "ids") or hasattr(first, "input_ids"):
                    token_ids = first
                elif hasattr(first, "tolist"):
                    first_list = first.tolist()
                    if isinstance(first_list, list | tuple):
                        token_ids = list(first_list)
                    else:
                        break
                else:
                    break
            else:
                break

        if not isinstance(token_ids, list):
            raise TypeError(
                "Cosmos3 tokenizer must return token IDs as a list-like value; "
                f"got {type(token_ids).__name__}: {token_ids!r}"
            )

        normalized_ids = []
        for idx, token_id in enumerate(token_ids):
            if hasattr(token_id, "item"):
                token_id = token_id.item()
            try:
                normalized_ids.append(int(token_id))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Cosmos3 tokenizer returned a non-integer token at "
                    f"index {idx}: {type(token_id).__name__}: {token_id!r}"
                ) from exc
        return normalized_ids

    # -- Latent preparation --------------------------------------------------

    def _prepare_latents(
        self,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        num_channels_latents = self.transformer.latent_channel_size
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

    def _prepare_sound_latents(
        self,
        target_audio_samples: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, int]:
        sound_tokenizer = self._get_sound_tokenizer()
        hop_size = int(
            getattr(sound_tokenizer, "hop_size", None) or getattr(sound_tokenizer, "temporal_compression_factor")
        )
        latent_frames = max(1, math.ceil(max(1, int(target_audio_samples)) / hop_size))
        sound_dim = int(getattr(sound_tokenizer, "latent_ch", 64))
        transformer_sound_dim = int(getattr(self.transformer, "sound_dim", sound_dim))
        if sound_dim != transformer_sound_dim:
            raise ValueError(
                "Cosmos3 sound tokenizer latent channels do not match transformer "
                f"sound_dim: tokenizer={sound_dim}, transformer={transformer_sound_dim}."
            )
        latents = randn_tensor(
            (1, sound_dim, latent_frames),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        return latents, latent_frames

    def _resolve_sound_target_samples(
        self,
        sp,
        num_frames: int,
        frame_rate: float,
    ) -> tuple[int, float, int]:
        sound_tokenizer = self._get_sound_tokenizer()
        duration = self._get_sp_param(sp, "sound_duration", None)
        if duration is None:
            duration = self._get_sp_param(sp, "audio_duration", None)
        if duration is None:
            duration = num_frames / frame_rate
        duration = max(float(duration), 1.0 / max(float(frame_rate), 1.0))
        sample_rate = int(getattr(sound_tokenizer, "sample_rate", 48000))
        return max(1, int(round(duration * sample_rate))), duration, sample_rate

    # -- VAE decode ----------------------------------------------------------

    def _get_latents_mean_std(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cached = getattr(self, "_latents_mean_std", None)
        if cached is not None:
            latents_mean, latents_std = cached
            if latents_mean.device == device and latents_mean.dtype == dtype:
                return latents_mean, latents_std

        latents_mean = torch.as_tensor(self.vae.config.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.as_tensor(self.vae.config.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        self._latents_mean_std = (latents_mean, latents_std)
        return latents_mean, latents_std

    def _to_vae_device(self, tensor: torch.Tensor, *, pin_cpu: bool = False) -> torch.Tensor:
        if tensor.device == self.device and tensor.dtype == self.vae.dtype:
            return tensor

        non_blocking = False
        if tensor.device.type == "cpu" and self.device.type == "cuda":
            if pin_cpu and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            non_blocking = tensor.is_pinned()

        return tensor.to(device=self.device, dtype=self.vae.dtype, non_blocking=non_blocking)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.vae.dtype)

        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean, latents_std = self._get_latents_mean_std(latents.device, latents.dtype)
            latents = (latents * latents_std) + latents_mean
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latents = latents / scaling_factor

        video = self.vae.decode(latents, return_dict=False)[0]
        return video

    def _decode_sound_latents(
        self,
        sound_latents: torch.Tensor,
        target_audio_samples: int,
    ) -> torch.Tensor:
        sound_tokenizer = self._get_sound_tokenizer()
        audio = sound_tokenizer.decode(sound_latents.to(self.dtype))
        if audio.shape[-1] > target_audio_samples:
            audio = audio[..., :target_audio_samples]
        elif audio.shape[-1] < target_audio_samples:
            audio = torch.nn.functional.pad(audio, (0, target_audio_samples - audio.shape[-1]))
        return audio.detach().cpu()

    # -- Prompt formatting + tokenization (shared by T2V and I2V) ------------

    def _format_and_tokenize_prompts(
        self,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        frame_rate: float,
        height: int,
        width: int,
        max_sequence_length: int,
        sp: OmniDiffusionSamplingParams,
        use_system_prompt: bool = False,
        is_t2i: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Format prompts with metadata templates and tokenize.

        Returns (cond_ids, cond_mask, uncond_ids, uncond_mask).

        For T2I (``is_t2i=True``) the duration template is suppressed (no FPS
        or duration concept for a single image) and the image-flavored
        resolution template is used.
        """
        # Route cosmos3-specific controls through ``_get_sp_param`` so they
        # are picked up from ``extra_args`` (OpenAI endpoint path) as well
        # as from direct attributes.
        use_duration_template = bool(self._get_sp_param(sp, "use_duration_template", False)) and not is_t2i
        dur_tmpl = COSMOS3_DURATION_TEMPLATE if use_duration_template else None
        if bool(self._get_sp_param(sp, "use_resolution_template", False)):
            res_tmpl = COSMOS3_IMAGE_RESOLUTION_TEMPLATE if is_t2i else COSMOS3_RESOLUTION_TEMPLATE
        else:
            res_tmpl = None
        prompt = self._apply_metadata_templates(
            prompt,
            num_frames,
            frame_rate,
            height,
            width,
            duration_template=dur_tmpl,
            resolution_template=res_tmpl,
        )
        if _is_rank_zero():
            logger.info("Final prompt: '%s'", prompt)

        # Negative prompt: inverse templates ("not {duration}...", "not {height}x{width}...").
        # Applied whenever the matching positive template is enabled; an empty
        # negative_prompt yields output that starts with the template, not a dot.
        inv_dur = COSMOS3_INVERSE_DURATION_TEMPLATE if dur_tmpl else None
        if res_tmpl is None:
            inv_res = None
        elif is_t2i:
            inv_res = COSMOS3_INVERSE_IMAGE_RESOLUTION_TEMPLATE
        else:
            inv_res = COSMOS3_INVERSE_RESOLUTION_TEMPLATE
        negative_prompt = self._apply_metadata_templates(
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            duration_template=inv_dur,
            resolution_template=inv_res,
            force_duration_template=True,
        )

        default_sys_prompt = COSMOS3_T2I_SYSTEM_PROMPT if is_t2i else COSMOS3_SYSTEM_PROMPT
        sys_prompt = self._get_sp_param(sp, "system_prompt", default_sys_prompt) or default_sys_prompt
        cond_ids, cond_mask = self._tokenize_prompt(
            prompt, max_sequence_length, use_system_prompt, system_prompt=sys_prompt
        )
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt, system_prompt=sys_prompt
        )
        return cond_ids, cond_mask, uncond_ids, uncond_mask

    # -- I2V latent preparation ---------------------------------------------

    def _normalize_vae_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean, latents_std = self._get_latents_mean_std(latent.device, latent.dtype)
            latent = (latent - latents_mean) / latents_std
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latent = latent * scaling_factor

        return latent

    def _encode_conditioning_image_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """VAE-encode the first I2V conditioning frame.

        I2V only consumes the first conditioning latent frame.  Encoding the
        input image as a one-frame video keeps that latent while avoiding VAE
        work for repeated frames that are later replaced by noise.
        """
        image_tensor = self._to_vae_device(image_tensor, pin_cpu=True)
        video = image_tensor.unsqueeze(2)
        latent = self.vae.encode(video).latent_dist.mode()
        latent = self._normalize_vae_latent(latent)
        return latent[:, :, 0:1, :, :].to(self.dtype)

    def _latent_hw_from_image_size(self, image_size: Any | None) -> tuple[int, int] | None:
        if image_size is None:
            return None
        if isinstance(image_size, torch.Tensor):
            frame_size = image_size.detach().cpu().flatten()
        else:
            frame_size = torch.as_tensor(image_size).flatten()
        if frame_size.numel() < 4:
            return None
        orig_h = int(frame_size[2].item())
        orig_w = int(frame_size[3].item())
        spatial_factor = int(self.vae_scale_factor_spatial)
        return max(orig_h // spatial_factor, 1), max(orig_w // spatial_factor, 1)

    def _crop_latent_to_image_size(self, latent: torch.Tensor, image_size: Any | None) -> torch.Tensor:
        latent_hw = self._latent_hw_from_image_size(image_size)
        if latent_hw is None:
            return latent
        h_latent, w_latent = latent_hw
        return latent[:, :, :, :h_latent, :w_latent].contiguous()

    def _encode_video_tensor(self, video_tensor: torch.Tensor, image_size: Any | None = None) -> torch.Tensor:
        """VAE-encode a preprocessed pixel video [1, 3, T, H, W]."""
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)
        if video_tensor.ndim != 5:
            raise ValueError(f"Cosmos3 video tensor must have shape [1, 3, T, H, W], got {tuple(video_tensor.shape)}.")
        if video_tensor.shape[0] != 1 or video_tensor.shape[1] != 3:
            raise ValueError(f"Cosmos3 video tensor must have shape [1, 3, T, H, W], got {tuple(video_tensor.shape)}.")

        video = self._to_vae_device(video_tensor)
        latent = self.vae.encode(video).latent_dist.mode()
        latent = self._normalize_vae_latent(latent)
        latent = self._crop_latent_to_image_size(latent, image_size)
        return latent.to(self.dtype)

    def _prepare_latents_i2v(
        self,
        image_tensor: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare initial latents with frame 0 conditioned on the input image.

        Returns:
            latents: [1, C, T_lat, H_lat, W_lat] with frame 0 = image, rest = noise
            velocity_mask: [1, 1, T_lat, 1, 1] with frame 0 = 0, rest = 1
            image_latent: [1, C, 1, H_lat, W_lat] clean frame 0 for re-injection
        """
        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        H_lat = height // self.vae_scale_factor_spatial
        W_lat = width // self.vae_scale_factor_spatial

        noise = randn_tensor(
            (1, C, T_lat, H_lat, W_lat),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        image_latent = self._encode_conditioning_image_latent(image_tensor)
        latents = noise
        latents[:, :, 0:1, :, :] = image_latent

        velocity_mask = torch.ones(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        velocity_mask[:, :, 0, :, :] = 0.0
        return latents, velocity_mask, image_latent

    def _prepare_latents_v2v(
        self,
        video_tensor: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
        condition_frame_indexes_vision: Iterable[int] | int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare V2V latents with explicit clean conditioned latent frames."""
        del height, width
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)
        if video_tensor.ndim != 5 or video_tensor.shape[0] != 1 or video_tensor.shape[1] != 3:
            raise ValueError(f"Cosmos3 video tensor must have shape [1, 3, T, H, W], got {tuple(video_tensor.shape)}.")
        if video_tensor.shape[2] < 1:
            raise ValueError("Cosmos3 V2V video tensor must contain at least one frame.")

        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        H_lat = video_tensor.shape[-2] // self.vae_scale_factor_spatial
        W_lat = video_tensor.shape[-1] // self.vae_scale_factor_spatial
        indexes = normalize_condition_frame_indexes_vision(condition_frame_indexes_vision)
        out_of_range = [index for index in indexes if index >= T_lat]
        if out_of_range:
            raise ValueError(
                "Cosmos3 condition_frame_indexes_vision contains indexes outside the latent video: "
                f"indexes={indexes}, latent_frames={T_lat}."
            )

        noise = randn_tensor(
            (1, C, T_lat, H_lat, W_lat),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        condition_pixel_frames = condition_pixel_frame_count(indexes, self.vae_scale_factor_temporal)
        condition_video = video_tensor[:, :, :condition_pixel_frames]
        if condition_video.shape[2] < condition_pixel_frames:
            pad = condition_video[:, :, -1:].repeat(1, 1, condition_pixel_frames - condition_video.shape[2], 1, 1)
            condition_video = torch.cat([condition_video, pad], dim=2)

        cond_prefix_latent = self._encode_video_tensor(condition_video)
        expected_prefix = (1, C, max(indexes) + 1, H_lat, W_lat)
        if (
            cond_prefix_latent.shape[0] != expected_prefix[0]
            or cond_prefix_latent.shape[1] != expected_prefix[1]
            or cond_prefix_latent.shape[2] < expected_prefix[2]
            or cond_prefix_latent.shape[3:] != expected_prefix[3:]
        ):
            raise ValueError(
                "Cosmos3 V2V condition latent shape mismatch: "
                f"encoded={tuple(cond_prefix_latent.shape)}, expected at least {expected_prefix}."
            )

        condition_mask = torch.zeros(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        condition_latents = torch.zeros_like(noise)
        for index in indexes:
            condition_mask[:, :, index, :, :] = 1.0
            condition_latents[:, :, index : index + 1] = cond_prefix_latent[:, :, index : index + 1]
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, condition_latents

    def _prepare_latents_action_video(
        self,
        video_tensor: torch.Tensor,
        mode: str,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
        image_size: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare video latents for action modes with mode-specific conditioning."""
        del height, width
        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_hw = self._latent_hw_from_image_size(image_size)
        if latent_hw is None:
            H_lat = video_tensor.shape[-2] // self.vae_scale_factor_spatial
            W_lat = video_tensor.shape[-1] // self.vae_scale_factor_spatial
        else:
            H_lat, W_lat = latent_hw

        noise = randn_tensor(
            (1, C, T_lat, H_lat, W_lat),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        cond_latent = self._encode_video_tensor(video_tensor, image_size=image_size)
        if cond_latent.shape[2:] != noise.shape[2:]:
            raise ValueError(
                "Cosmos3 action video latent shape mismatch: "
                f"encoded={tuple(cond_latent.shape)}, expected={tuple(noise.shape)}."
            )
        condition_mask = build_vision_condition_mask(
            mode,
            num_frames,
            self.vae_scale_factor_temporal,
            device=self.device,
            dtype=self.dtype,
        )
        latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, cond_latent

    def _prepare_action_latents(
        self,
        *,
        mode: str,
        action_chunk_size: int,
        raw_action_dim: int | None,
        generator: torch.Generator,
        sp,
        clean_action: torch.Tensor | None = None,
        condition_indexes: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        action_dim = int(getattr(self.transformer, "action_dim", 64))
        if clean_action is not None:
            action = clean_action.detach().to(dtype=torch.float32)
            if action.ndim == 3 and action.shape[0] == 1:
                action = action.squeeze(0)
            if action.ndim != 2:
                raise ValueError(f"Cosmos3 clean action must have shape [T, D], got {tuple(action.shape)}.")
            if action.shape[0] < action_chunk_size:
                pad = action[-1:].repeat(action_chunk_size - action.shape[0], 1)
                action = torch.cat([action, pad], dim=0)
            elif action.shape[0] > action_chunk_size:
                action = action[:action_chunk_size]
            if raw_action_dim is None:
                raw_action_dim = int(action.shape[-1])
            clean_action = pad_action_to_dim(action, action_dim)
        elif mode == ACTION_MODE_FORWARD_DYNAMICS:
            action = load_action_tensor(self._get_sp_param(sp, "action", None))
            if action.shape[0] < action_chunk_size:
                pad = action[-1:].repeat(action_chunk_size - action.shape[0], 1)
                action = torch.cat([action, pad], dim=0)
            elif action.shape[0] > action_chunk_size:
                action = action[:action_chunk_size]
            if raw_action_dim is None:
                raw_action_dim = int(action.shape[-1])
            clean_action = pad_action_to_dim(action, action_dim)
        else:
            if raw_action_dim is None:
                raise ValueError(
                    "Cosmos3 action_mode='policy' and 'inverse_dynamics' require extra_args['raw_action_dim']."
                )
            clean_action = torch.zeros(action_chunk_size, action_dim, dtype=torch.float32)

        raw_action_dim = int(raw_action_dim)
        if raw_action_dim <= 0 or raw_action_dim > action_dim:
            raise ValueError(f"Cosmos3 raw_action_dim must be in [1, {action_dim}], got {raw_action_dim}.")

        clean_action = clean_action.to(device=self.device, dtype=self.dtype).unsqueeze(0)
        if condition_indexes is None:
            condition_mask = build_action_condition_mask(
                mode,
                action_chunk_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            condition_mask = self._build_action_condition_mask_from_indexes(
                condition_indexes,
                action_chunk_size,
                device=self.device,
                dtype=self.dtype,
            )
        noise = randn_tensor(
            (1, action_chunk_size, action_dim),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        noise[:, :, raw_action_dim:] = 0
        clean_action[:, :, raw_action_dim:] = 0
        action_latents = condition_mask * clean_action + (1.0 - condition_mask) * noise
        action_velocity_mask = 1.0 - condition_mask
        return action_latents, action_velocity_mask, clean_action, raw_action_dim

    # -- Denoising loop (shared by T2V and I2V) -----------------------------

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        uncond_ids: torch.Tensor,
        uncond_mask: torch.Tensor,
        guidance_scale: float,
        shared_kwargs: dict,
        *,
        action_latents: torch.Tensor | None = None,
        action_velocity_mask: torch.Tensor | None = None,
        action_condition_latents: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        velocity_mask: torch.Tensor | None = None,
        image_latent: torch.Tensor | None = None,
        condition_latents: torch.Tensor | None = None,
        guidance_interval: tuple[float, float] | None = None,
        raw_action_dim: int | None = None,
        scheduler: Any | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Denoising loop with 3-mode CFG support (parallel, sequential, none).

        Cosmos3's UND pathway is text-dependent, so CFG needs separate K/V
        caches for conditional and unconditional text.

        Two modes:
          1. CFG parallel (multi-GPU): each rank handles one condition via
             predict_noise_maybe_with_cfg; caching is rank-local.
          2. Sequential CFG (single-GPU or cfg_size=1): two separate
             forward passes with explicit cache swapping.  We cannot
             batch B=2 because different text lengths would cause the
             shorter branch to attend to padding in cross-attention.

        I2V conditioning (when both arguments are supplied):
          * ``velocity_mask`` zeros frame-0 noise predictions before stepping.
          * ``image_latent`` is re-injected into frame 0 after each scheduler
            step, since UniPC's predictor-corrector update rescales the
            sample (sigma-dependent), so even zero velocity does not preserve
            frame 0.

        ``guidance_interval`` (T2I) restricts CFG to
        timesteps inside the closed interval ``[lo, hi]``.  The interval is
        compared against the raw scheduler timestep value; works for both
        the [0, 1000] discrete scale and normalized flow-matching scales.
        Outside the interval the cond/uncond delta is zeroed so all ranks
        continue to execute identical control flow (CFG-Parallel safe).
        """
        do_cfg = guidance_scale > 1.0
        cfg_parallel = self._cfg_parallel_active() and do_cfg
        step_scheduler = scheduler if scheduler is not None else self.scheduler
        self.transformer.reset_cache()

        def _cfg_active_at(t: torch.Tensor) -> bool:
            if guidance_interval is None:
                return True
            t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
            lo, hi = guidance_interval
            return lo <= t_scalar <= hi

        # Joint scheduler step over multiple modalities. Safe for flow-matching schedulers
        # because the update is linear per element; revisit this if Cosmos3 adopts a
        # scheduler with cross-element dependencies (e.g. per-modality timestep).
        def _pack_joint(
            video_tensor: torch.Tensor,
            action_tensor: torch.Tensor | None = None,
            sound_tensor: torch.Tensor | None = None,
        ):
            batch = video_tensor.shape[0]
            tensors = [video_tensor]
            if action_tensor is not None:
                tensors.append(action_tensor)
            if sound_tensor is not None:
                tensors.append(sound_tensor)
            flats = [tensor.reshape(batch, -1) for tensor in tensors]
            return torch.cat(flats, dim=1), [tensor.shape for tensor in tensors], [flat.shape[1] for flat in flats]

        def _unpack_joint(
            packed: torch.Tensor,
            shapes: list[torch.Size],
            numels: list[int],
        ) -> tuple[torch.Tensor, ...]:
            outputs = []
            offset = 0
            for shape, numel in zip(shapes, numels, strict=True):
                outputs.append(packed[:, offset : offset + numel].reshape(shape))
                offset += numel
            return tuple(outputs)

        def _split_noise_pred(
            noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            has_action = action_latents is not None
            has_sound = sound_latents is not None
            if not has_action and not has_sound:
                if isinstance(noise_pred, tuple):
                    raise ValueError("Cosmos3 video-only diffusion received tuple predictions.")
                return noise_pred, None, None
            if not isinstance(noise_pred, tuple):
                raise ValueError("Cosmos3 multimodal diffusion expects transformer predictions as a tuple.")
            expected = 1 + int(has_action) + int(has_sound)
            if len(noise_pred) != expected:
                raise ValueError(
                    f"Cosmos3 multimodal diffusion expected {expected} predictions, got {len(noise_pred)}."
                )
            video_pred = noise_pred[0]
            idx = 1
            action_pred = noise_pred[idx] if has_action else None
            if has_action:
                idx += 1
            sound_pred = noise_pred[idx] if has_sound else None
            return video_pred, action_pred, sound_pred

        def _step(
            noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
            t: torch.Tensor,
            latents: torch.Tensor,
            action_latents: torch.Tensor | None,
            sound_latents: torch.Tensor | None,
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            video_pred, action_pred, sound_pred = _split_noise_pred(noise_pred)
            if velocity_mask is not None:
                if image_latent is not None and condition_latents is None:
                    video_pred[:, :, 0:1, :, :] = 0
                else:
                    video_pred = video_pred * velocity_mask
            if action_pred is not None and action_velocity_mask is not None:
                action_pred = action_pred * action_velocity_mask
                if raw_action_dim is not None and 0 < raw_action_dim < action_pred.shape[-1]:
                    action_pred[..., raw_action_dim:] = 0
            if action_latents is None and sound_latents is None:
                latents = step_scheduler.step(video_pred, t, latents, return_dict=False)[0]
            else:
                packed_noise, shapes, numels = _pack_joint(video_pred, action_pred, sound_pred)
                packed_latents, _, _ = _pack_joint(latents, action_latents, sound_latents)
                packed_next = step_scheduler.step(packed_noise, t, packed_latents, return_dict=False)[0]
                unpacked = _unpack_joint(packed_next, shapes, numels)
                latents = unpacked[0]
                idx = 1
                if action_latents is not None:
                    action_latents = unpacked[idx]
                    idx += 1
                if sound_latents is not None:
                    sound_latents = unpacked[idx]
            if condition_latents is not None and velocity_mask is not None:
                latents = velocity_mask * latents + (1.0 - velocity_mask) * condition_latents
            elif image_latent is not None:
                latents[:, :, 0:1, :, :] = image_latent
            if action_latents is not None and action_condition_latents is not None and action_velocity_mask is not None:
                action_latents = (
                    action_velocity_mask * action_latents + (1.0 - action_velocity_mask) * action_condition_latents
                )
            outputs = [latents]
            if action_latents is not None:
                outputs.append(action_latents)
            if sound_latents is not None:
                outputs.append(sound_latents)
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        def _assign_step_out(step_out: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
            nonlocal latents, action_latents, sound_latents
            if action_latents is None and sound_latents is None:
                assert isinstance(step_out, torch.Tensor)
                latents = step_out
                return
            if not isinstance(step_out, tuple):
                raise ValueError("Cosmos3 multimodal diffusion step returned a non-tuple result.")
            latents = step_out[0]
            idx = 1
            if action_latents is not None:
                action_latents = step_out[idx]
                idx += 1
            if sound_latents is not None:
                sound_latents = step_out[idx]

        if cfg_parallel:
            for t in self.progress_bar(timesteps):
                timestep = t.unsqueeze(0)
                # Out-of-interval steps run with effective scale 1.0 so the
                # combined output equals the cond branch (uncond is dropped).
                # All ranks still execute both branches; no CFG-Parallel
                # divergence.
                step_scale = guidance_scale if _cfg_active_at(t) else 1.0
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=True,
                    true_cfg_scale=step_scale,
                    positive_kwargs=dict(
                        hidden_states=latents,
                        timestep=timestep,
                        text_ids=cond_ids,
                        text_mask=cond_mask,
                        action_latents=action_latents,
                        sound_latents=sound_latents,
                        **shared_kwargs,
                    ),
                    negative_kwargs=dict(
                        hidden_states=latents,
                        timestep=timestep,
                        text_ids=uncond_ids,
                        text_mask=uncond_mask,
                        action_latents=action_latents,
                        sound_latents=sound_latents,
                        **shared_kwargs,
                    ),
                    cfg_normalize=False,
                )
                _assign_step_out(_step(noise_pred, t, latents, action_latents, sound_latents))

        elif do_cfg:
            cond_cache: tuple = (None, None)
            uncond_cache: tuple = (None, None)

            keep_uncond_for_cache = self._cache_requires_paired_cfg()

            for t in self.progress_bar(timesteps):
                timestep = t.unsqueeze(0)
                cfg_active = _cfg_active_at(t)

                self.transformer.cached_kv, self.transformer.cached_freqs_gen = cond_cache
                noise_cond = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    action_latents=action_latents,
                    sound_latents=sound_latents,
                    **shared_kwargs,
                )
                if cond_cache[0] is None:
                    cond_cache = (self.transformer.cached_kv, self.transformer.cached_freqs_gen)

                if cfg_active or keep_uncond_for_cache:
                    self.transformer.cached_kv, self.transformer.cached_freqs_gen = uncond_cache
                    noise_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=timestep,
                        text_ids=uncond_ids,
                        text_mask=uncond_mask,
                        action_latents=action_latents,
                        sound_latents=sound_latents,
                        **shared_kwargs,
                    )
                    if uncond_cache[0] is None:
                        uncond_cache = (self.transformer.cached_kv, self.transformer.cached_freqs_gen)
                    # Outside the interval, scale=1.0 makes the combined result
                    # equal to noise_cond; the uncond pass is computed only to
                    # preserve cache-dit's cond/uncond parity.
                    step_scale = guidance_scale if cfg_active else 1.0
                    noise_pred = self.combine_cfg_noise(noise_cond, noise_uncond, step_scale, cfg_normalize=False)
                else:
                    noise_pred = noise_cond

                _assign_step_out(_step(noise_pred, t, latents, action_latents, sound_latents))

        else:
            for t in self.progress_bar(timesteps):
                timestep = t.unsqueeze(0)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    action_latents=action_latents,
                    sound_latents=sound_latents,
                    **shared_kwargs,
                )
                _assign_step_out(_step(noise_pred, t, latents, action_latents, sound_latents))

        outputs = [latents]
        if action_latents is not None:
            outputs.append(action_latents)
        if sound_latents is not None:
            outputs.append(sound_latents)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    @staticmethod
    def _get_transfer_num_chunks(
        total_frames: int,
        frames_per_chunk: int,
        conditional_frames: int,
    ) -> tuple[int, int]:
        if frames_per_chunk <= 0:
            raise ValueError("Cosmos3 transfer frames_per_chunk must be positive.")
        if total_frames <= frames_per_chunk:
            return 1, frames_per_chunk
        stride = frames_per_chunk - conditional_frames
        if stride <= 0:
            raise ValueError("Cosmos3 transfer num_conditional_frames must be smaller than num_video_frames_per_chunk.")
        remaining = total_frames - frames_per_chunk
        extra_chunks = remaining // stride + (1 if remaining % stride else 0)
        return 1 + extra_chunks, stride

    def _prepare_transfer_latents(
        self,
        target_video: torch.Tensor,
        current_conditional_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_latents = self._encode_video_tensor(target_video)
        noise = randn_tensor(
            condition_latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros(
            1,
            1,
            condition_latents.shape[2],
            1,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        if current_conditional_frames > 0:
            latent_frames = (current_conditional_frames - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[:, :, :latent_frames] = 1.0
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, condition_mask * condition_latents

    def _transfer_bucket_size(
        self,
        sp: OmniDiffusionSamplingParams,
        source_hw: tuple[int, int] | None,
    ) -> tuple[int, int]:
        resolution = self._get_sp_param(sp, "resolution", self._get_sp_param(sp, "image_size", 720))
        source_h, source_w = source_hw or (COSMOS3_T2V_DEFAULT_HEIGHT, COSMOS3_T2V_DEFAULT_WIDTH)
        target_w, target_h = find_closest_target_size(int(source_h), int(source_w), resolution)
        return int(target_h), int(target_w)

    @staticmethod
    def _first_transfer_control_hw(transfer_config: Cosmos3TransferConfig) -> tuple[int, int] | None:
        for hint in transfer_config.ordered_hints:
            if hint.control is not None:
                detected = media_hw(hint.control)
                if detected is not None:
                    return detected
            if hint.control_path is not None:
                detected = media_hw(hint.control_path)
                if detected is not None:
                    return detected
        return None

    def diffuse_transfer(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        uncond_ids: torch.Tensor,
        uncond_mask: torch.Tensor,
        guidance_scale: float,
        control_guidance: float,
        control_guidance_interval: tuple[float, float] | None,
        control_latents: list[torch.Tensor],
        shared_kwargs: dict[str, Any],
        *,
        velocity_mask: torch.Tensor,
        condition_latents: torch.Tensor,
        guidance_interval: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        def _active_at(t: torch.Tensor, interval: tuple[float, float] | None) -> bool:
            if interval is None:
                return True
            t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
            lo, hi = interval
            return lo <= t_scalar <= hi

        self.transformer.reset_cache()
        self._cosmos3_branch_caches = {}
        try:
            for t in self.progress_bar(timesteps):
                timestep = t.unsqueeze(0)
                step_guidance = guidance_scale if _active_at(t, guidance_interval) else 1.0
                step_control = control_guidance if _active_at(t, control_guidance_interval) else 1.0
                needs_text_cfg = step_guidance > 1.0
                needs_control_cfg = step_control != 1.0

                cond_full_kwargs = dict(
                    hidden_states=latents,
                    timestep=timestep,
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    control_latents=control_latents,
                    _cosmos3_cache_key="transfer_cond_full",
                    **shared_kwargs,
                )
                if needs_control_cfg and needs_text_cfg:
                    branches_kwargs = [
                        cond_full_kwargs,
                        dict(
                            hidden_states=latents,
                            timestep=timestep,
                            text_ids=cond_ids,
                            text_mask=cond_mask,
                            control_latents=None,
                            _cosmos3_cache_key="transfer_cond_no_control",
                            **shared_kwargs,
                        ),
                        dict(
                            hidden_states=latents,
                            timestep=timestep,
                            text_ids=uncond_ids,
                            text_mask=uncond_mask,
                            control_latents=control_latents,
                            _cosmos3_cache_key="transfer_uncond_full",
                            **shared_kwargs,
                        ),
                    ]
                    noise_pred = self.predict_noise_with_multi_branch_cfg(
                        do_true_cfg=True,
                        true_cfg_scale={
                            "mode": "cosmos3_transfer",
                            "branch_mode": "control_and_text",
                            "guidance_scale": step_guidance,
                            "control_guidance": step_control,
                        },
                        branches_kwargs=branches_kwargs,
                        cfg_normalize=False,
                    )
                elif needs_control_cfg:
                    branches_kwargs = [
                        cond_full_kwargs,
                        dict(
                            hidden_states=latents,
                            timestep=timestep,
                            text_ids=cond_ids,
                            text_mask=cond_mask,
                            control_latents=None,
                            _cosmos3_cache_key="transfer_cond_no_control",
                            **shared_kwargs,
                        ),
                    ]
                    noise_pred = self.predict_noise_with_multi_branch_cfg(
                        do_true_cfg=True,
                        true_cfg_scale={
                            "mode": "cosmos3_transfer",
                            "branch_mode": "control_only",
                            "control_guidance": step_control,
                        },
                        branches_kwargs=branches_kwargs,
                        cfg_normalize=False,
                    )
                elif needs_text_cfg:
                    branches_kwargs = [
                        cond_full_kwargs,
                        dict(
                            hidden_states=latents,
                            timestep=timestep,
                            text_ids=uncond_ids,
                            text_mask=uncond_mask,
                            control_latents=control_latents,
                            _cosmos3_cache_key="transfer_uncond_full",
                            **shared_kwargs,
                        ),
                    ]
                    noise_pred = self.predict_noise_with_multi_branch_cfg(
                        do_true_cfg=True,
                        true_cfg_scale={
                            "mode": "cosmos3_transfer",
                            "branch_mode": "text_only",
                            "guidance_scale": step_guidance,
                        },
                        branches_kwargs=branches_kwargs,
                        cfg_normalize=False,
                    )
                else:
                    noise_pred = self.predict_noise(**cond_full_kwargs)
                if isinstance(noise_pred, tuple):
                    raise ValueError("Cosmos3 transfer diffusion expects video-only tensor predictions.")
                noise_pred = noise_pred * velocity_mask
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents = velocity_mask * latents + (1.0 - velocity_mask) * condition_latents
        finally:
            self._cosmos3_branch_caches = None
            self.transformer.reset_cache()
        return latents

    def _forward_transfer(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        sp: OmniDiffusionSamplingParams,
        transfer_config: Cosmos3TransferConfig,
        transfer_video_tensor: torch.Tensor | None,
        transfer_input_fps: float | None,
    ) -> DiffusionOutput:
        input_frames = None
        if transfer_video_tensor is not None:
            input_frames = normalized_video_to_uint8_cthw(transfer_video_tensor)
            source_hw = (int(input_frames.shape[-2]), int(input_frames.shape[-1]))
        else:
            source_hw = self._first_transfer_control_hw(transfer_config)
        height, width = self._transfer_bucket_size(sp, source_hw)

        if input_frames is not None:
            if tuple(input_frames.shape[-2:]) != (height, width):
                input_frames = resize_center_crop_uint8_cthw(input_frames, height, width)
            input_frames = input_frames[:, : transfer_config.max_frames]

        per_hint_frames: dict[str, torch.Tensor] = {}
        for hint in transfer_config.ordered_hints:
            frames = load_or_compute_control_frames(
                hint,
                height=height,
                width=width,
                max_frames=transfer_config.max_frames,
                input_frames=input_frames,
            )
            if frames.shape[1] < 1:
                raise ValueError(f"Cosmos3 transfer hint '{hint.key}' produced no frames.")
            per_hint_frames[hint.key] = frames
        if not per_hint_frames:
            raise ValueError("Cosmos3 transfer requires at least one control hint.")

        total_frames = next(iter(per_hint_frames.values())).shape[1]
        if transfer_config.num_frames is not None:
            total_frames = min(total_frames, int(transfer_config.num_frames))
        total_frames = max(1, total_frames)
        per_hint_frames = {key: pad_temporal_frames(frames, total_frames) for key, frames in per_hint_frames.items()}
        if input_frames is not None:
            input_frames = pad_temporal_frames(input_frames, total_frames)

        temporal_compression = self.vae_scale_factor_temporal
        chunk_frames = 1 if total_frames == 1 else transfer_config.num_video_frames_per_chunk
        chunk_frames = math.ceil((chunk_frames - 1) / temporal_compression) * temporal_compression + 1
        num_chunks, stride = self._get_transfer_num_chunks(
            total_frames,
            chunk_frames,
            transfer_config.num_conditional_frames,
        )
        padded_frames = max(total_frames, chunk_frames)
        per_hint_frames = {key: pad_temporal_frames(frames, padded_frames) for key, frames in per_hint_frames.items()}
        if input_frames is not None:
            input_frames = pad_temporal_frames(input_frames, padded_frames)

        configured_frame_rate = positive_float(transfer_config.fps)
        input_frame_rate = positive_float(transfer_input_fps)
        sampling_frame_rate = (
            positive_float(self._get_sp_param(sp, "resolved_frame_rate"))
            or positive_float(self._get_sp_param(sp, "frame_rate"))
            or positive_float(self._get_sp_param(sp, "fps"))
        )
        is_wsm_only = len(transfer_config.hints) == 1 and "wsm" in transfer_config.hints
        if is_wsm_only:
            frame_rate = configured_frame_rate or input_frame_rate or sampling_frame_rate or 24.0
        else:
            frame_rate = input_frame_rate or configured_frame_rate or sampling_frame_rate or 24.0
        num_inference_steps = sp.num_inference_steps or COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = (
            float(transfer_config.guidance_scale)
            if transfer_config.guidance_scale is not None
            else float(sp.guidance_scale or COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE)
        )
        flow_shift_target = float(
            transfer_config.flow_shift
            if transfer_config.flow_shift is not None
            else self._get_sp_param(sp, "flow_shift", COSMOS3_V2V_DEFAULT_FLOW_SHIFT)
        )
        max_sequence_length = (
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH)
            or COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH
        )
        use_system_prompt = bool(self._get_sp_param(sp, "use_system_prompt", False))

        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps
        self._set_flow_shift(flow_shift_target)

        generator = sp.generator
        if generator is None:
            seed = sp.seed if sp.seed is not None else 42
            generator = torch.Generator(device=self.device).manual_seed(seed)

        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            prompt,
            negative_prompt,
            chunk_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=False,
        )

        output_chunks: list[torch.Tensor] = []
        control_chunks_per_hint: dict[str, list[torch.Tensor]] = {key: [] for key in per_hint_frames}
        previous_output: torch.Tensor | None = None

        for chunk_id in range(num_chunks):
            start_frame = chunk_id * stride
            end_frame = min(start_frame + chunk_frames, total_frames)
            control_norms = {
                key: uint8_cthw_to_normalized_5d(
                    pad_temporal_frames(frames[:, start_frame:end_frame], chunk_frames),
                    dtype=self.dtype,
                )
                for key, frames in per_hint_frames.items()
            }
            target_norm = torch.zeros_like(next(iter(control_norms.values())))
            current_conditional_frames = 0

            if chunk_id == 0 and transfer_config.num_first_chunk_conditional_frames > 0:
                if input_frames is None:
                    raise ValueError("Cosmos3 transfer num_first_chunk_conditional_frames > 0 requires a video input.")
                current_conditional_frames = min(
                    transfer_config.num_first_chunk_conditional_frames,
                    input_frames.shape[1],
                    chunk_frames,
                )
                if current_conditional_frames > 0:
                    input_cond = uint8_cthw_to_normalized_5d(
                        input_frames[:, :current_conditional_frames],
                        dtype=self.dtype,
                    )
                    target_norm[:, :, :current_conditional_frames] = input_cond
                    if current_conditional_frames < chunk_frames:
                        fill = target_norm[:, :, current_conditional_frames - 1 : current_conditional_frames]
                        target_norm[:, :, current_conditional_frames:] = fill.expand(
                            -1,
                            -1,
                            chunk_frames - current_conditional_frames,
                            -1,
                            -1,
                        )
            elif chunk_id > 0 and previous_output is not None:
                current_conditional_frames = min(
                    transfer_config.num_conditional_frames,
                    previous_output.shape[2],
                    chunk_frames,
                )
                if current_conditional_frames > 0:
                    target_norm[:, :, :current_conditional_frames] = previous_output[
                        :, :, -current_conditional_frames:
                    ].to(target_norm)
                    if current_conditional_frames < chunk_frames:
                        fill = target_norm[:, :, current_conditional_frames - 1 : current_conditional_frames]
                        target_norm[:, :, current_conditional_frames:] = fill.expand(
                            -1,
                            -1,
                            chunk_frames - current_conditional_frames,
                            -1,
                            -1,
                        )

            control_latents = [self._encode_video_tensor(video) for video in control_norms.values()]
            latents, velocity_mask, condition_latents = self._prepare_transfer_latents(
                target_norm,
                current_conditional_frames,
                generator,
            )
            video_shape = (latents.shape[2], latents.shape[3], latents.shape[4])
            shared_kwargs = dict(
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                transfer_share_vision_temporal_positions=transfer_config.share_vision_temporal_positions,
            )

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            latents = self.diffuse_transfer(
                latents=latents,
                timesteps=self.scheduler.timesteps,
                cond_ids=cond_ids,
                cond_mask=cond_mask,
                uncond_ids=uncond_ids,
                uncond_mask=uncond_mask,
                guidance_scale=guidance_scale,
                control_guidance=transfer_config.control_guidance,
                control_guidance_interval=transfer_config.control_guidance_interval,
                control_latents=control_latents,
                shared_kwargs=shared_kwargs,
                velocity_mask=velocity_mask,
                condition_latents=condition_latents,
            )
            output_video = self._decode_latents(latents).clamp(-1, 1)
            previous_output = output_video

            if chunk_id == 0:
                output_chunks.append(output_video)
                for key, control in control_norms.items():
                    control_chunks_per_hint[key].append(control)
            else:
                output_chunks.append(output_video[:, :, current_conditional_frames:])
                for key, control in control_norms.items():
                    control_chunks_per_hint[key].append(control[:, :, current_conditional_frames:])

        full_output = torch.cat(output_chunks, dim=2)[:, :, :total_frames]
        full_controls = {
            key: torch.cat(chunks, dim=2)[:, :, :total_frames] for key, chunks in control_chunks_per_hint.items()
        }

        if transfer_config.show_control_condition:
            all_controls = torch.cat([full_controls[key] for key in per_hint_frames], dim=-1)
            all_controls = all_controls.to(full_output)
            full_output = torch.cat([all_controls, full_output], dim=-1)
        if transfer_config.show_input and input_frames is not None:
            normalized_input = uint8_cthw_to_normalized_5d(input_frames[:, :total_frames], dtype=torch.float32)
            full_output = torch.cat([normalized_input.to(full_output), full_output], dim=-1)

        return DiffusionOutput(
            output={"video": full_output},
            custom_output={
                "transfer_controls": full_controls,
                "transfer_hints": list(per_hint_frames),
                "fps": frame_rate,
            },
        )

    # -- Forward (main generation entry point) -------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        pipeline_start = time.time()

        # --- Parse request ---
        if len(req.prompts) > 1:
            raise ValueError("Cosmos3OmniDiffusersPipeline currently supports a single prompt per request.")

        sp = req.sampling_params
        prompt_data = req.prompts[0]
        robolab_inputs = self._build_robolab_policy_inputs(sp, prompt_data, getattr(req, "request_id", None))
        if robolab_inputs is not None:
            return self._forward_robolab_policy(sp, robolab_inputs, pipeline_start)

        if isinstance(prompt_data, str):
            prompt = prompt_data
            negative_prompt = None
            image_tensor = None
            video_tensor = None
            transfer_video_tensor = None
            transfer_input_fps = None
        else:
            prompt = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt")
            additional_info = prompt_data.get("additional_information", {}) or {}
            image_tensor = additional_info.get("preprocessed_image")
            video_tensor = additional_info.get("preprocessed_video")
            transfer_video_tensor = additional_info.get("preprocessed_transfer_video")
            transfer_input_fps = positive_float(additional_info.get("transfer_input_fps"))

        is_t2i = self._is_t2i_request(req)
        sound_enabled = self._is_sound_request(prompt_data, sp)
        action_mode = self._get_action_mode(prompt_data, sp)
        action_enabled = action_mode is not None
        transfer_config = resolve_transfer_config(sp, prompt_data)
        action_video_tensor = video_tensor if action_enabled else None
        if transfer_config is not None:
            if is_t2i:
                raise ValueError("Cosmos3 transfer inference is supported only for video outputs.")
            if action_enabled:
                raise ValueError("Cosmos3 transfer inference cannot be combined with action generation.")
            if sound_enabled:
                raise ValueError("Cosmos3 transfer inference cannot be combined with sound generation.")
        if action_enabled and is_t2i:
            raise ValueError("Cosmos3 action generation is supported only for video outputs.")
        if action_enabled and sound_enabled:
            raise ValueError("Cosmos3 action+sound joint generation is not supported in this phase.")
        if action_enabled and not getattr(self.transformer, "action_gen", False):
            raise ValueError(
                "Cosmos3 action generation was requested, but the transformer was "
                "initialized without action modules. Check that the checkpoint config "
                "enables action_gen and includes action weights."
            )
        if sound_enabled and is_t2i:
            raise ValueError(
                "Cosmos3 sound generation is supported only for video outputs in "
                "this phase; text-to-image with sound is unsupported."
            )
        if sound_enabled and not getattr(self.transformer, "sound_gen", False):
            raise ValueError(
                "Cosmos3 sound generation was requested, but the transformer was "
                "initialized without sound modules. Check that the checkpoint config "
                "enables sound_gen or defines sound_dim and includes sound weights."
            )
        if negative_prompt is None:
            negative_prompt = ""
        if transfer_config is not None:
            if image_tensor is not None:
                raise ValueError("Cosmos3 transfer inference accepts video inputs or control_path values, not images.")
            if transfer_video_tensor is None and video_tensor is not None:
                transfer_video_tensor = video_tensor
            return self._forward_transfer(
                prompt=prompt,
                negative_prompt=negative_prompt,
                sp=sp,
                transfer_config=transfer_config,
                transfer_video_tensor=transfer_video_tensor,
                transfer_input_fps=transfer_input_fps,
            )
        if image_tensor is not None and video_tensor is not None and not action_enabled:
            raise ValueError("Cosmos3 non-action generation accepts either image or video input, not both.")
        if video_tensor is not None and is_t2i:
            raise ValueError("Cosmos3 video-to-video generation is supported only for video outputs.")
        is_v2v = video_tensor is not None and not is_t2i and not action_enabled

        # T2I and T2V share the same model + forward path; only defaults
        # differ:
        #   T2I: 1024x1024, 50 steps, shift=3.0, guidance_interval=[400, 1000]
        #   T2V: 720x1280,  35 steps, shift=engine-init, no interval
        if is_t2i:
            height = sp.height or COSMOS3_T2I_DEFAULT_HEIGHT
            width = sp.width or COSMOS3_T2I_DEFAULT_WIDTH
            num_frames = 1
            num_inference_steps = sp.num_inference_steps or COSMOS3_T2I_DEFAULT_NUM_INFERENCE_STEPS
            guidance_scale = sp.guidance_scale if sp.guidance_scale else COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE
            default_flow_shift = COSMOS3_T2I_DEFAULT_FLOW_SHIFT
            default_guidance_interval: tuple[float, float] | None = COSMOS3_T2I_DEFAULT_GUIDANCE_INTERVAL
            batch_size = max(1, int(sp.num_outputs_per_prompt or 1))
        else:
            height = sp.height or COSMOS3_T2V_DEFAULT_HEIGHT
            width = sp.width or COSMOS3_T2V_DEFAULT_WIDTH
            num_frames = sp.num_frames or COSMOS3_T2V_DEFAULT_NUM_FRAMES
            num_inference_steps = sp.num_inference_steps or COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS
            guidance_scale = sp.guidance_scale if sp.guidance_scale else COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE
            # Fall back to the engine-init shift, NOT None: passing None
            # to ``_set_flow_shift`` would leak a prior T2I rebuild
            # (shift=3.0) into a subsequent video request.
            default_flow_shift = COSMOS3_V2V_DEFAULT_FLOW_SHIFT if is_v2v else self._engine_init_flow_shift
            default_guidance_interval = None
            batch_size = 1  # Existing video pipeline assumes B=1.

        if action_enabled:
            action_chunk_param = self._get_sp_param(sp, "action_chunk_size", None)
            if action_chunk_param is not None:
                action_chunk_size = int(action_chunk_param)
                if sp.num_frames is None:
                    num_frames = action_chunk_size + 1
            elif sp.num_frames is None:
                action_chunk_size = 16
                num_frames = action_chunk_size + 1
            else:
                action_chunk_size = int(num_frames) - 1
            if action_chunk_size <= 0:
                raise ValueError(f"Cosmos3 action_chunk_size must be positive, got {action_chunk_size}.")
            if num_frames not in (action_chunk_size, action_chunk_size + 1):
                raise ValueError(
                    "Cosmos3 action requests require num_frames to equal action_chunk_size "
                    f"or action_chunk_size + 1; got num_frames={num_frames}, action_chunk_size={action_chunk_size}."
                )
            num_inference_steps = sp.num_inference_steps or 30
            guidance_scale = sp.guidance_scale if sp.guidance_scale is not None else 1.0
            default_flow_shift = 5.0

        domain_id = None
        if action_enabled:
            domain_id = resolve_domain_id(
                domain_id=self._get_sp_param(sp, "domain_id", None),
                domain_name=self._get_sp_param(sp, "domain_name", None),
                require_explicit=True,
            )

        # Runtime controls: prefer ``extra_args`` (OpenAI endpoints write
        # there) over direct attrs.
        flow_shift_target = float(self._get_sp_param(sp, "flow_shift", default_flow_shift))
        guidance_interval = self._get_sp_param(sp, "guidance_interval", default_guidance_interval)

        frame_rate = self._get_sp_param(sp, "resolved_frame_rate") or self._get_sp_param(sp, "frame_rate") or 24.0
        max_sequence_length = (
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH)
            or COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH
        )
        use_system_prompt = bool(self._get_sp_param(sp, "use_system_prompt", not action_enabled))

        if action_enabled and action_video_tensor is None:
            extra_action_video = self._get_sp_param(sp, "action_video", None)
            if isinstance(extra_action_video, torch.Tensor):
                action_video_tensor = extra_action_video
        if action_enabled and isinstance(action_video_tensor, torch.Tensor):
            if action_video_tensor.ndim == 4:
                action_video_tensor = action_video_tensor.unsqueeze(0)
            if action_video_tensor.ndim != 5:
                raise ValueError(
                    "Cosmos3 extra_args['action_video'] must have shape [1, 3, T, H, W] "
                    f"or [3, T, H, W], got {tuple(action_video_tensor.shape)}."
                )
            if sp.height is None:
                height = int(action_video_tensor.shape[-2])
            if sp.width is None:
                width = int(action_video_tensor.shape[-1])

        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps

        # Always resolve to a concrete target shift for this request, then
        # update the shared Diffusers scheduler.
        self._set_flow_shift(flow_shift_target)

        generator = sp.generator
        if generator is None:
            seed = sp.seed if sp.seed is not None else 42
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # --- Format prompts & tokenize (B=1; reused across loop iterations
        # for T2I num_outputs_per_prompt > 1) ---
        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            prompt,
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=is_t2i,
        )

        # --- Prepare latents (T2I, T2V, or I2V) ---
        # T2I shares _prepare_latents with T2V; the math collapses cleanly
        # at num_frames=1 ((1-1)//4 + 1 = 1 latent frame).  For T2I with
        # ``num_outputs_per_prompt > 1`` we loop the diffusion below;
        # batching B=N together would require expanding text K/V (UND
        # pathway is text-only and cached) and is left as a future
        # optimization.
        action_latents = None
        action_velocity_mask = None
        action_condition_latents = None
        raw_action_dim = None
        action_offset = 1
        if action_enabled:
            if action_video_tensor is not None and action_video_tensor.ndim == 4:
                action_video_tensor = action_video_tensor.unsqueeze(0)
            if action_video_tensor is not None and action_video_tensor.ndim != 5:
                raise ValueError(
                    "Cosmos3 action video tensor must have shape [1, 3, T, H, W] "
                    f"or [3, T, H, W], got {tuple(action_video_tensor.shape)}."
                )
            if action_video_tensor is not None and action_video_tensor.shape[2] < num_frames:
                pad = action_video_tensor[:, :, -1:].repeat(1, 1, num_frames - action_video_tensor.shape[2], 1, 1)
                action_video_tensor = torch.cat([action_video_tensor, pad], dim=2)
            elif action_video_tensor is not None and action_video_tensor.shape[2] > num_frames:
                action_video_tensor = action_video_tensor[:, :, :num_frames]

            if action_mode == ACTION_MODE_INVERSE_DYNAMICS and action_video_tensor is None:
                raise ValueError("Cosmos3 inverse_dynamics action mode requires multi_modal_data['video'].")
            if action_mode in {ACTION_MODE_POLICY, ACTION_MODE_FORWARD_DYNAMICS} and image_tensor is None:
                if action_video_tensor is None:
                    raise ValueError(
                        f"Cosmos3 action_mode={action_mode!r} requires multi_modal_data['image'] "
                        "or multi_modal_data['video']."
                    )
                image_tensor = action_video_tensor[:, :, 0]

            raw_action_dim_param = self._get_sp_param(sp, "raw_action_dim", None)
            raw_action_dim = int(raw_action_dim_param) if raw_action_dim_param is not None else None
            clean_action = None
            action_condition_indexes = None
            action_prepared = self._prepare_action_latents(
                mode=action_mode,
                action_chunk_size=action_chunk_size,
                raw_action_dim=raw_action_dim,
                generator=generator,
                sp=sp,
                clean_action=clean_action,
                condition_indexes=action_condition_indexes,
            )
            action_latents, action_velocity_mask, action_condition_latents, raw_action_dim = action_prepared
            action_offset = action_start_frame_offset(action_mode, action_chunk_size, num_frames)

        if action_enabled and action_video_tensor is not None:
            latents, velocity_mask, condition_latents = self._prepare_latents_action_video(
                action_video_tensor,
                action_mode,
                height,
                width,
                num_frames,
                generator,
            )
            image_latent = condition_latents[:, :, 0:1]
        elif is_v2v:
            condition_frame_indexes_vision = normalize_condition_frame_indexes_vision(
                self._get_sp_param(
                    sp,
                    "condition_frame_indexes_vision",
                    self._get_prompt_param(prompt_data, "condition_frame_indexes_vision", None),
                )
            )
            latents, velocity_mask, condition_latents = self._prepare_latents_v2v(
                video_tensor,
                height,
                width,
                num_frames,
                generator,
                condition_frame_indexes_vision,
            )
            image_latent = None
        elif image_tensor is not None and not is_t2i:
            latents, velocity_mask, image_latent = self._prepare_latents_i2v(
                image_tensor,
                height,
                width,
                num_frames,
                generator,
            )
            condition_latents = None
        else:
            latents = self._prepare_latents(height, width, num_frames, generator)
            velocity_mask = None
            image_latent = None
            condition_latents = None

        sound_latents = None
        target_audio_samples = None
        sound_sample_rate = None
        if sound_enabled:
            target_audio_samples, _, sound_sample_rate = self._resolve_sound_target_samples(sp, num_frames, frame_rate)
            sound_latents, _ = self._prepare_sound_latents(target_audio_samples, generator)

        T_latent = latents.shape[2]
        H_latent = latents.shape[3]
        W_latent = latents.shape[4]
        video_shape = (T_latent, H_latent, W_latent)

        # --- Denoising loop ---
        shared_kwargs = dict(video_shape=video_shape, fps=frame_rate)
        if velocity_mask is not None:
            shared_kwargs["noisy_frame_mask"] = velocity_mask
        if action_enabled:
            shared_kwargs.update(
                action_domain_ids=torch.tensor([domain_id], dtype=torch.long, device=self.device),
                action_noisy_mask=action_velocity_mask,
                action_start_frame_offset=action_offset,
                action_fps=float(self._get_sp_param(sp, "action_fps", frame_rate) or frame_rate),
            )

        def _run_diffusion(start_latents):
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            scheduler = self.scheduler
            return self.diffuse(
                latents=start_latents,
                timesteps=scheduler.timesteps,
                cond_ids=cond_ids,
                cond_mask=cond_mask,
                uncond_ids=uncond_ids,
                uncond_mask=uncond_mask,
                guidance_scale=guidance_scale,
                shared_kwargs=shared_kwargs,
                action_latents=action_latents,
                action_velocity_mask=action_velocity_mask,
                action_condition_latents=action_condition_latents,
                sound_latents=sound_latents,
                velocity_mask=velocity_mask,
                image_latent=image_latent,
                condition_latents=condition_latents,
                guidance_interval=guidance_interval,
                raw_action_dim=raw_action_dim,
                scheduler=scheduler,
            )

        if is_t2i and batch_size > 1:
            # Generate N independent images by re-running the full diffusion
            # loop with different noise seeds.  The first sample reuses
            # ``latents`` already drawn from ``generator``; subsequent
            # samples draw fresh noise from the same generator (state
            # advances per call), giving distinct outputs from a single
            # user-provided seed.  Batched B=N would be more efficient but
            # requires expanding cached UND text K/V to match.
            samples = [_run_diffusion(latents)]
            for _ in range(batch_size - 1):
                next_latents = self._prepare_latents(height, width, num_frames, generator)
                samples.append(_run_diffusion(next_latents))
            latents = torch.cat(samples, dim=0)
        else:
            diffusion_output = _run_diffusion(latents)
            if action_enabled and sound_enabled:
                latents, action_latents, sound_latents = diffusion_output
            elif action_enabled:
                latents, action_latents = diffusion_output
            elif sound_enabled:
                latents, sound_latents = diffusion_output
            else:
                latents = diffusion_output

        # --- Decode ---
        if _is_rank_zero():
            logger.info("Decoding video...")
        decode_start = time.time()
        video = self._decode_latents(latents)
        if _is_rank_zero():
            logger.info("Video decoded in %.2fs", time.time() - decode_start)
            if not sound_enabled:
                logger.info("Total pipeline time: %.2fs", time.time() - pipeline_start)

        if sound_enabled:
            if sound_latents is None or target_audio_samples is None or sound_sample_rate is None:
                raise ValueError("Cosmos3 sound generation finished without sound latents.")
            if _is_rank_zero():
                logger.info("Decoding sound...")
            sound_decode_start = time.time()
            audio = self._decode_sound_latents(sound_latents, target_audio_samples)
            if _is_rank_zero():
                logger.info("Sound tokenizer decoded in %.2fs", time.time() - sound_decode_start)
                logger.info("Total pipeline time: %.2fs", time.time() - pipeline_start)
            return DiffusionOutput(output={"video": video, "audio": audio, "audio_sample_rate": sound_sample_rate})

        if action_enabled:
            if action_latents is None or raw_action_dim is None or domain_id is None:
                raise ValueError("Cosmos3 action generation finished without action latents.")
            action = action_latents[:, :, :raw_action_dim].detach().cpu()
            custom_action_output: dict[str, Any] = {
                "action": action,
                "raw_action_dim": raw_action_dim,
                "action_mode": action_mode,
                "domain_id": domain_id,
            }
            return DiffusionOutput(output={"video": video}, custom_output=custom_action_output)

        return DiffusionOutput(output={"image": video} if is_t2i else {"video": video})
