# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import io
import logging
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from vllm_omni.diffusion.models.skyreels_v3.aspect_ratio import (
    DEFAULT_SKYREELS_V3_RESOLUTION,
    resolve_bucket_size,
)
from vllm_omni.diffusion.models.skyreels_v3.v2v_t5 import T5EncoderModel
from vllm_omni.diffusion.models.skyreels_v3.v2v_transformer import WanModel
from vllm_omni.diffusion.models.skyreels_v3.v2v_vae import WanVAE
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

DEFAULT_SKYREELS_V3_V2V_FPS = 24
DEFAULT_SKYREELS_V3_V2V_DURATION = 5
DEFAULT_SKYREELS_V3_V2V_STEPS = 8
DEFAULT_SKYREELS_V3_V2V_GUIDANCE = 1.0
DEFAULT_SKYREELS_V3_V2V_SHIFT = 8.0
DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES = 25
DEFAULT_SKYREELS_V3_V2V_LATENT_FRAMES_PER_SECOND = 6
DEFAULT_SKYREELS_V3_V2V_MAX_DURATION = 30
DEFAULT_SKYREELS_V3_V2V_RESOLUTION = "720P"


def _resolve_v2v_model_path(model: str | None) -> str:
    if not model:
        raise ValueError("SkyReels V3 V2V requires `od_config.model` to point to a model path or HF repo.")
    if os.path.isdir(model):
        return model
    from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

    allow_patterns = [
        "*.json",
        "*.pth",
        "transformer/**",
        "google/**",
    ]
    return download_weights_from_hf_specific(model, None, allow_patterns)


def _split_duration(duration: int, chunk_seconds: int = 5) -> list[int]:
    if duration <= 0:
        raise ValueError(f"SkyReels V3 V2V duration must be positive, got {duration}.")
    chunks: list[int] = []
    remaining = duration
    while remaining >= chunk_seconds:
        chunks.append(chunk_seconds)
        remaining -= chunk_seconds
    if remaining > 0:
        chunks.append(remaining)
    return chunks


def _coerce_frames_to_uint8(frames: Any) -> tuple[np.ndarray, float | None]:
    """Normalize a supported video input into ``uint8`` RGB frames ``[T, H, W, 3]``."""

    if isinstance(frames, str):
        return _load_video_from_path_or_url(frames)

    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
        if tensor.ndim == 5:
            if tensor.shape[0] != 1:
                raise ValueError("SkyReels V3 V2V supports a single input video per request.")
            tensor = tensor[0]
        if tensor.ndim != 4:
            raise ValueError(f"Unsupported video tensor shape {tuple(tensor.shape)}.")
        if tensor.shape[-1] in (1, 3, 4):
            pass
        elif tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        else:
            raise ValueError(f"Unsupported video tensor layout {tuple(tensor.shape)}.")
        array = tensor.numpy()
        return _normalize_frame_array(array), None

    if isinstance(frames, np.ndarray):
        return _normalize_frame_array(frames), None

    if isinstance(frames, Sequence) and not isinstance(frames, (bytes, bytearray)):
        if not frames:
            raise ValueError("Input video contains no frames.")
        frame_arrays = []
        for frame in frames:
            if isinstance(frame, PIL.Image.Image):
                frame_arrays.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
            elif isinstance(frame, torch.Tensor):
                frame_tensor = frame.detach().cpu()
                if frame_tensor.ndim == 3 and frame_tensor.shape[0] in (1, 3, 4):
                    frame_tensor = frame_tensor.permute(1, 2, 0)
                frame_arrays.append(_normalize_frame_array(frame_tensor.numpy())[0])
            elif isinstance(frame, np.ndarray):
                frame_arrays.append(_normalize_frame_array(frame)[0])
            else:
                raise TypeError(f"Unsupported video frame type {frame.__class__}.")
        return np.stack(frame_arrays, axis=0), None

    raise TypeError(
        "SkyReels V3 V2V video input must be a path/URL, numpy array, torch tensor, "
        f"or a sequence of frames, got {frames.__class__}."
    )


def _load_video_from_path_or_url(video: str) -> tuple[np.ndarray, float | None]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("SkyReels V3 V2V requires PyAV (`av`) to load video paths or URLs.") from exc

    source: str | io.BytesIO
    if video.startswith("data:video"):
        try:
            _, payload = video.split(",", 1)
            source = io.BytesIO(base64.b64decode(payload))
        except ValueError as exc:
            raise ValueError("Invalid data URL video input.") from exc
    else:
        source = video

    with av.open(source) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not frames:
        raise ValueError("Input video contains no decodable frames.")
    return np.stack(frames, axis=0), fps


def _normalize_frame_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 4 and array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 3, 0))
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError(f"Video frames must have shape [T,H,W,C], got {array.shape}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Video frames must have 1, 3, or 4 channels, got shape {array.shape}.")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    array = array.astype(np.float32)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))
    if min_value >= -1.0 and max_value <= 1.0:
        if min_value < 0.0:
            array = (array + 1.0) * 127.5
        else:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).round().astype(np.uint8))


def _resize_video_uint8(frames: np.ndarray, *, height: int, width: int) -> np.ndarray:
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(tensor.shape[2], height, width))
    return tensor.squeeze(0).permute(1, 2, 3, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def _video_uint8_to_vae_tensor(frames: np.ndarray, device: torch.device | str) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float()
    tensor = tensor / (255.0 / 2.0) - 1.0
    return tensor.to(device)


def _resolve_input_video(prompt: OmniTextPrompt, extra_args: dict[str, Any]) -> Any:
    multi_modal_data = prompt.get("multi_modal_data") or {}
    for source, key in (
        (multi_modal_data, "video"),
        (multi_modal_data, "input_video"),
        (multi_modal_data, "cond_video"),
        (extra_args, "video_path"),
        (extra_args, "input_video"),
    ):
        if key in source and source[key] is not None:
            return source[key]
    else:
        raise ValueError(
            "SkyReels V3 V2V requires an input video. Use `multi_modal_data={'video': ...}` "
            "or pass `video_path` in extra_args / extra_body."
        )


def _normalize_duration(req: DiffusionRequestBatch, extra_args: dict[str, Any], fps: float) -> tuple[int, int | None]:
    requested_frames = req.sampling_params.num_frames
    duration = extra_args.get("duration")
    trim_frames: int | None = None
    if duration is None and requested_frames is not None and requested_frames > 0:
        duration = int(math.ceil(float(requested_frames) / max(float(fps), 1.0)))
        trim_frames = int(requested_frames)
    duration = int(duration or DEFAULT_SKYREELS_V3_V2V_DURATION)
    if duration > DEFAULT_SKYREELS_V3_V2V_MAX_DURATION:
        raise ValueError(
            f"SkyReels V3 single-shot extension supports duration <= "
            f"{DEFAULT_SKYREELS_V3_V2V_MAX_DURATION}s, got {duration}s."
        )
    return duration, trim_frames


def get_skyreels_v3_v2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config

    def post_process_func(
        output: Any,
        output_type: str = "np",
        sampling_params=None,
    ):
        if sampling_params is not None and getattr(sampling_params, "output_type", None):
            output_type = sampling_params.output_type
        fps = DEFAULT_SKYREELS_V3_V2V_FPS
        video = output
        if isinstance(output, tuple) and len(output) == 2:
            video, fps = output
        if output_type == "latent":
            return video

        if isinstance(video, torch.Tensor):
            if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
                video, multiplier = interpolate_video_tensor(
                    video,
                    exp=sampling_params.frame_interpolation_exp,
                    scale=sampling_params.frame_interpolation_scale,
                    model_path=sampling_params.frame_interpolation_model_path,
                )
                fps = fps * multiplier
            video = video.detach().cpu()
            if video.ndim == 5:
                video = video[0]
            if video.ndim == 4 and video.shape[0] in (1, 3, 4):
                video = video.permute(1, 2, 3, 0)
            video = _normalize_frame_array(video.numpy())

        if isinstance(video, np.ndarray):
            video = _normalize_frame_array(video)
            if output_type == "pil":
                video = [PIL.Image.fromarray(frame) for frame in video]
            elif output_type in {"pt", "tensor"}:
                video = torch.from_numpy(video).permute(0, 3, 1, 2)

        return {
            "payload": {"video": video},
            "metadata": {"video": {"fps": fps}},
        }

    return post_process_func


def get_skyreels_v3_v2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        extra_args = request.sampling_params.extra_args or {}
        raw_video = _resolve_input_video(prompt, extra_args)
        frames, source_fps = _coerce_frames_to_uint8(raw_video)
        if frames.shape[0] < 1:
            raise ValueError("SkyReels V3 V2V input video must contain at least one frame.")
        condition_frames = int(extra_args.get("condition_frames", DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES))
        if condition_frames <= 0:
            raise ValueError(f"condition_frames must be positive, got {condition_frames}.")
        if frames.shape[0] < condition_frames:
            raise ValueError(
                f"SkyReels V3 V2V requires at least {condition_frames} input frames, got {frames.shape[0]}."
            )

        if request.sampling_params.height is None or request.sampling_params.width is None:
            resolution = str(extra_args.get("resolution", DEFAULT_SKYREELS_V3_V2V_RESOLUTION))
            height, width = resolve_bucket_size(frames.shape[1], frames.shape[2], resolution)
            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        height = int(request.sampling_params.height)
        width = int(request.sampling_params.width)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")

        prompt["multi_modal_data"] = dict(prompt.get("multi_modal_data") or {})
        prompt["multi_modal_data"]["video"] = _resize_video_uint8(frames, height=height, width=width)
        prompt.setdefault("additional_information", {})["source_video_fps"] = source_fps
        request.prompt = prompt
        return request

    return pre_process_func


class SkyReelsV3V2VPipeline(
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    DenoiseProgressMixin,
    SupportsComponentDiscovery,
):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae.vae"]
    # V2V warmup needs a real input video. Skip generic dummy warmup instead of
    # manufacturing a tiny video that does not match extension constraints.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        del prefix
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.param_dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = _resolve_v2v_model_path(od_config.model)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]

        self.text_encoder = T5EncoderModel(
            checkpoint_path=os.path.join(self.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.model_path, "google", "umt5-xxl"),
            shard_fn=None,
        ).to(self.param_dtype)
        self.vae = WanVAE(vae_pth=os.path.join(self.model_path, "Wan2.1_VAE.pth"))
        self.transformer = WanModel.from_config(os.path.join(self.model_path, "transformer", "config.json")).to(
            self.param_dtype
        )

        self.text_encoder.to(self.device)
        self.vae.to(self.device)
        self.scheduler = FlowUniPCMultistepScheduler()
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.sp_size = 1

        self._guidance_scale: float | None = None
        self._num_timesteps: int | None = None
        self._current_timestep: torch.Tensor | None = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def _predict_noise(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        block_offload: bool = False,
    ) -> torch.Tensor:
        transformer_dtype = self.transformer.patch_embedding.weight.dtype
        transformer_device = self.transformer.patch_embedding.weight.device
        noise_pred = self.transformer(
            latent_model_input.to(device=transformer_device, dtype=transformer_dtype),
            t=timestep.to(device=transformer_device),
            context=context.to(device=transformer_device, dtype=transformer_dtype),
            block_offload=block_offload,
        )
        if isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        if not isinstance(noise_pred, torch.Tensor):
            raise TypeError(f"SkyReels V3 V2V transformer returned {type(noise_pred)!r}, expected Tensor.")
        return noise_pred[0]

    def _diffuse_segment(
        self,
        *,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        condition_latents: torch.Tensor,
        latent_num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        shift: float,
        generator: torch.Generator | list[torch.Generator] | None,
        block_offload: bool = False,
    ) -> torch.Tensor:
        target_shape = (
            self.vae.vae.z_dim,
            latent_num_frames,
            height // self.vae_stride[1],
            width // self.vae_stride[2],
        )
        latents = randn_tensor(target_shape, generator=generator, device=self.device, dtype=torch.float32)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, timestep in enumerate(timesteps):
                self._current_timestep = timestep
                self.record_denoise_step(step_idx, timestep)
                latent_model_input = latents.unsqueeze(0)
                latent_model_input = torch.cat([condition_latents, latent_model_input], dim=2)
                model_timestep = timestep.view(1, 1).repeat(1, latent_model_input.shape[2])
                model_timestep[:, : condition_latents.shape[2]] = 0

                if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                    noise_pred_cond = self._predict_noise(
                        latent_model_input,
                        model_timestep,
                        prompt_embeds,
                        block_offload=block_offload,
                    )
                    noise_pred_uncond = self._predict_noise(
                        latent_model_input,
                        model_timestep,
                        negative_prompt_embeds,
                        block_offload=block_offload,
                    )
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self._predict_noise(
                        latent_model_input,
                        model_timestep,
                        prompt_embeds,
                        block_offload=block_offload,
                    )

                noise_pred = noise_pred[:, -latents.shape[1] :].to(dtype=latents.dtype)
                latents = self.scheduler.step(
                    noise_pred.unsqueeze(0),
                    timestep,
                    latents.unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0)
                pbar.update()
        return latents

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("SkyReels V3 V2V only supports a single prompt per request.")
        first_prompt = req.prompts[0]
        if isinstance(first_prompt, str):
            prompt = first_prompt
            negative_prompt = ""
            prompt_data = OmniTextPrompt(prompt=first_prompt)
        else:
            prompt = first_prompt.get("prompt") or ""
            negative_prompt = first_prompt.get("negative_prompt") or ""
            prompt_data = first_prompt
        if not prompt:
            raise ValueError("Prompt is required for SkyReels V3 V2V generation.")

        extra_args = req.sampling_params.extra_args or {}
        raw_video = _resolve_input_video(prompt_data, extra_args)
        frames, source_fps = _coerce_frames_to_uint8(raw_video)
        additional_info = prompt_data.get("additional_information") or {}
        if source_fps is None and isinstance(additional_info, dict):
            stored_fps = additional_info.get("source_video_fps")
            if isinstance(stored_fps, (int, float)) and not isinstance(stored_fps, bool):
                source_fps = float(stored_fps)

        resolution = str(extra_args.get("resolution", DEFAULT_SKYREELS_V3_V2V_RESOLUTION))
        bucket_height, bucket_width = resolve_bucket_size(frames.shape[1], frames.shape[2], resolution)
        height = int(req.sampling_params.height or bucket_height)
        width = int(req.sampling_params.width or bucket_width)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")
        if frames.shape[1] != height or frames.shape[2] != width:
            frames = _resize_video_uint8(frames, height=height, width=width)

        fps = float(extra_args.get("fps") or source_fps or DEFAULT_SKYREELS_V3_V2V_FPS)
        condition_frames = int(extra_args.get("condition_frames", DEFAULT_SKYREELS_V3_V2V_CONDITION_FRAMES))
        if condition_frames <= 0:
            raise ValueError(f"condition_frames must be positive, got {condition_frames}.")
        if frames.shape[0] < condition_frames:
            raise ValueError(
                f"SkyReels V3 V2V requires at least {condition_frames} input frames, got {frames.shape[0]}."
            )
        prefix_np = frames[-condition_frames:]
        if prefix_np.shape[0] < 1:
            raise ValueError("SkyReels V3 V2V input video must contain at least one frame.")
        prefix_video = _video_uint8_to_vae_tensor(prefix_np, self.device)

        duration, trim_frames = _normalize_duration(req, extra_args, fps)
        num_inference_steps = int(
            extra_args.get(
                "sampling_steps",
                req.sampling_params.num_inference_steps or DEFAULT_SKYREELS_V3_V2V_STEPS,
            )
        )
        guidance_scale = float(
            extra_args.get(
                "cfg_text_scale",
                req.sampling_params.guidance_scale
                if req.sampling_params.guidance_scale_provided
                else DEFAULT_SKYREELS_V3_V2V_GUIDANCE,
            )
        )
        shift = float(extra_args.get("shift", self.od_config.flow_shift or DEFAULT_SKYREELS_V3_V2V_SHIFT))
        self._guidance_scale = guidance_scale

        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        if _is_rank_zero():
            logger.debug(
                "SkyReels V3 V2V request: size=%sx%s duration=%ss fps=%s condition_frames=%s steps=%s cfg=%s",
                width,
                height,
                duration,
                fps,
                condition_frames,
                num_inference_steps,
                guidance_scale,
            )

        prompt_embeds = self.text_encoder.encode(prompt).to(device=self.device, dtype=self.param_dtype)
        negative_prompt_embeds = None
        if guidance_scale > 1.0:
            negative_prompt_embeds = self.text_encoder.encode(negative_prompt).to(
                device=self.device,
                dtype=self.param_dtype,
            )

        output_segments: list[np.ndarray] = []
        padding_frames = 0
        block_offload = bool(extra_args.get("block_offload", False))
        for gen_seconds in _split_duration(duration):
            condition_latents = self.vae.encode(prefix_video).to(device=self.device, dtype=torch.float32)
            latent_num_frames = DEFAULT_SKYREELS_V3_V2V_LATENT_FRAMES_PER_SECOND * gen_seconds
            prefix_shape = condition_latents.shape[2]
            rest_frames = (latent_num_frames + prefix_shape) % 8
            if rest_frames > padding_frames:
                padding_frames = padding_frames + (8 - rest_frames)
                latent_num_frames = latent_num_frames - rest_frames + 8
            else:
                padding_frames = padding_frames - rest_frames
                latent_num_frames = latent_num_frames - rest_frames
            if latent_num_frames <= 0:
                raise ValueError(
                    "SkyReels V3 V2V resolved a non-positive latent frame count; "
                    f"duration={duration}, condition_latents={prefix_shape}."
                )

            latents = self._diffuse_segment(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                condition_latents=condition_latents,
                latent_num_frames=latent_num_frames,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=generator,
                block_offload=block_offload,
            )
            decoded = self.vae.decode(
                torch.cat([condition_latents, latents.unsqueeze(0)], dim=2)
            )
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            segment = (
                (decoded[0].permute(1, 2, 3, 0) * 255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            output_segments.append(segment[condition_frames:])
            prefix_np = segment[-condition_frames:]
            prefix_video = _video_uint8_to_vae_tensor(prefix_np, self.device)

        self._current_timestep = None
        output_video = (
            np.concatenate(output_segments, axis=0)
            if output_segments
            else np.empty((0, height, width, 3), dtype=np.uint8)
        )
        if trim_frames is not None:
            output_video = output_video[:trim_frames]
        if bool(extra_args.get("include_input_video", False)):
            output_video = np.concatenate([frames, output_video], axis=0)

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        return DiffusionOutput(
            output=(output_video, fps),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
