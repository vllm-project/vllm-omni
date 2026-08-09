# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from typing import Any, ClassVar, cast

import PIL.Image
import torch
import torchvision.transforms.functional as TF
from diffusers.utils.torch_utils import randn_tensor
from PIL import ImageOps
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.pipeline_parallel import AsyncLatents, PipelineParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import (
    from_pretrained_with_prefetch,
    prefetch_subfolders,
)
from vllm_omni.diffusion.model_metadata import SKYREELS_V3_R2V_MAX_INPUT_IMAGES
from vllm_omni.diffusion.models.interface import (
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.skyreels_v3.aspect_ratio import (
    DEFAULT_SKYREELS_V3_RESOLUTION,
    resolve_bucket_size,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    build_wan_scheduler,
    create_transformer_from_config,
    load_transformer_config,
    resolve_wan_flow_shift,
    resolve_wan_sample_solver,
    retrieve_latents,
)
from vllm_omni.diffusion.postprocess import interpolate_video_tensor
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)
DEBUG_PERF = False

# Keep this in sync with serving-facing admission metadata.
MAX_SKYREELS_R2V_REF_IMAGES = SKYREELS_V3_R2V_MAX_INPUT_IMAGES
DEFAULT_SKYREELS_R2V_HEIGHT = 544
DEFAULT_SKYREELS_R2V_WIDTH = 960
DEFAULT_SKYREELS_R2V_FRAMES = 105
DEFAULT_SKYREELS_R2V_STEPS = 50
DEFAULT_SKYREELS_R2V_GUIDANCE = 7.5
DEFAULT_SKYREELS_R2V_IMAGE_GUIDANCE = 5.0


def _load_image(image: str | PIL.Image.Image) -> PIL.Image.Image:
    if isinstance(image, str):
        return PIL.Image.open(image).convert("RGB")
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image format {image.__class__}.")


def _normalize_ref_images(raw_images: Any) -> list[PIL.Image.Image]:
    if raw_images is None:
        raise ValueError(
            "SkyReels V3 R2V requires reference images. "
            "Set `multi_modal_data` with key `image`, `ref_imgs`, or `reference_images`."
        )
    if isinstance(raw_images, (str, PIL.Image.Image)):
        raw_images = [raw_images]
    if not isinstance(raw_images, list):
        raise TypeError(
            "SkyReels V3 R2V reference images must be an image path, a PIL image, "
            f"or a list of them, got {raw_images.__class__}."
        )
    if not raw_images:
        raise ValueError("SkyReels V3 R2V requires at least one reference image.")
    if len(raw_images) > MAX_SKYREELS_R2V_REF_IMAGES:
        raise ValueError(
            f"SkyReels V3 R2V supports at most {MAX_SKYREELS_R2V_REF_IMAGES} reference images, got {len(raw_images)}."
        )
    return [_load_image(image) for image in raw_images]


def _infer_target_size(image: PIL.Image.Image, resolution: str | None = None) -> tuple[int, int]:
    return resolve_bucket_size(image.height, image.width, resolution)


def _resize_and_pad_ref_images(
    ref_images: list[PIL.Image.Image],
    *,
    height: int,
    width: int,
) -> list[PIL.Image.Image]:
    resized = []
    target_ratio = width / height
    for image in ref_images:
        image = image.convert("RGB")
        image_ratio = image.width / image.height
        if image_ratio > target_ratio:
            new_width = width
            new_height = int(new_width / image_ratio)
        else:
            new_height = height
            new_width = int(new_height * image_ratio)

        image = image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
        delta_w = width - image.size[0]
        delta_h = height - image.size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - delta_w // 2,
            delta_h - delta_h // 2,
        )
        resized.append(ImageOps.expand(image, padding, fill=(255, 255, 255)))
    return resized


def _resolve_guidance_scales(sampling_params: Any, extra_args: dict[str, Any]) -> tuple[float, float]:
    text_scale = (
        sampling_params.guidance_scale if sampling_params.guidance_scale_provided else DEFAULT_SKYREELS_R2V_GUIDANCE
    )
    if "cfg_text_scale" in extra_args:
        text_scale = extra_args["cfg_text_scale"]

    image_scale = extra_args.get(
        "guidance_scale_img",
        extra_args.get("cfg_img_scale", DEFAULT_SKYREELS_R2V_IMAGE_GUIDANCE),
    )
    return float(text_scale), float(image_scale)


def get_skyreels_v3_r2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
        sampling_params=None,
    ):
        if output_type == "latent":
            return video
        video_metadata = {}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            video_metadata["video_fps_multiplier"] = multiplier
        return {
            "payload": {"video": video_processor.postprocess_video(video, output_type=output_type)},
            "metadata": {"video": video_metadata} if video_metadata else {},
        }

    return post_process_func


def get_skyreels_v3_r2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        multi_modal_data = prompt.get("multi_modal_data") or {}
        raw_images = (
            multi_modal_data.get("image")
            or multi_modal_data.get("ref_imgs")
            or multi_modal_data.get("reference_images")
        )
        ref_images = _normalize_ref_images(raw_images)

        extra_args = request.sampling_params.extra_args or {}
        if request.sampling_params.height is None or request.sampling_params.width is None:
            resolution = str(extra_args.get("resolution", DEFAULT_SKYREELS_V3_RESOLUTION))
            height, width = _infer_target_size(ref_images[0], resolution)
            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        height = cast(int, request.sampling_params.height)
        width = cast(int, request.sampling_params.width)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")

        prompt["multi_modal_data"] = dict(multi_modal_data)
        prompt["multi_modal_data"]["image"] = _resize_and_pad_ref_images(ref_images, height=height, width=width)
        request.prompt = prompt
        return request

    return pre_process_func


class SkyReelsV3R2VPipeline(
    nn.Module,
    SupportImageInput,
    PipelineParallelMixin,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    DenoiseProgressMixin,
    SupportsComponentDiscovery,
):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    dummy_run_num_frames: ClassVar[int] = 5

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        component_subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(model, component_subfolders, local_files_only=local_files_only)

        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(
            transformer_config,
            quant_config=od_config.quantization_config,
        )

        self._sample_solver = "unipc"
        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

        self._guidance_scale = None
        self._guidance_scale_img = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_scale_img(self):
        return self._guidance_scale_img

    @property
    def do_classifier_free_guidance(self):
        return (self._guidance_scale is not None and self._guidance_scale != 1.0) or (
            self._guidance_scale_img is not None and self._guidance_scale_img != 1.0
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def combine_multi_branch_cfg_noise(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        true_cfg_scale: float | dict[str, Any],
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(true_cfg_scale, dict) or true_cfg_scale.get("mode") != "skyreels_v3_r2v":
            return super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize)
        if len(predictions) != 3:
            raise ValueError(f"SkyReels V3 R2V CFG expects 3 branches, got {len(predictions)}.")
        pred, pred_text_uncond, pred_text_image_uncond = predictions
        if isinstance(pred, tuple) or isinstance(pred_text_uncond, tuple) or isinstance(pred_text_image_uncond, tuple):
            raise ValueError("SkyReels V3 R2V CFG expects tensor predictions.")

        text_scale = float(true_cfg_scale["guidance_scale"])
        image_scale = float(true_cfg_scale["guidance_scale_img"])
        combined = pred_text_image_uncond + image_scale * (pred_text_uncond - pred_text_image_uncond)
        combined = combined + text_scale * (pred - pred_text_uncond)
        if cfg_normalize:
            combined = self.cfg_normalize_function(pred, combined)
        return combined

    def diffuse(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        guidance_scale_img: float,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any] | None,
    ) -> torch.Tensor | AsyncLatents:
        if attention_kwargs is None:
            attention_kwargs = {}
        uncondition = torch.zeros_like(condition)
        do_true_cfg = (guidance_scale != 1.0 or guidance_scale_img != 1.0) and negative_prompt_embeds is not None
        with self.progress_bar(total=len(timesteps)) as pbar:
            for step_idx, t in enumerate(timesteps):
                self._current_timestep = t
                self.record_denoise_step(step_idx, t)
                timestep = t.expand(latents.shape[0])

                positive_kwargs = {
                    "hidden_states": torch.cat([latents, condition], dim=2).to(dtype),
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "latent_num_frames": latents.shape[2],
                }
                if do_true_cfg:
                    text_uncond_kwargs = {
                        "hidden_states": torch.cat([latents, condition], dim=2).to(dtype),
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "latent_num_frames": latents.shape[2],
                    }
                    text_image_uncond_kwargs = {
                        "hidden_states": torch.cat([latents, uncondition], dim=2).to(dtype),
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                        "latent_num_frames": latents.shape[2],
                    }
                    noise_pred = self.predict_noise_with_multi_branch_cfg(
                        do_true_cfg=True,
                        true_cfg_scale={
                            "mode": "skyreels_v3_r2v",
                            "guidance_scale": guidance_scale,
                            "guidance_scale_img": guidance_scale_img,
                        },
                        branches_kwargs=[positive_kwargs, text_uncond_kwargs, text_image_uncond_kwargs],
                        cfg_normalize=False,
                    )
                else:
                    noise_pred = self.predict_noise(**positive_kwargs)

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        return latents

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        prompt: str | None = None
        negative_prompt: str | None = None
        if len(req.prompts) > 1:
            raise ValueError("SkyReels V3 R2V only supports a single prompt per request, not a batched request.")
        if len(req.prompts) == 1:
            first_prompt = req.prompts[0]
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")

        if not prompt:
            raise ValueError("Prompt is required for SkyReels V3 R2V generation.")

        multi_modal_data = req.prompts[0].get("multi_modal_data", {}) if not isinstance(req.prompts[0], str) else {}
        raw_images = (
            multi_modal_data.get("image")
            or multi_modal_data.get("ref_imgs")
            or multi_modal_data.get("reference_images")
        )
        ref_images = _normalize_ref_images(raw_images)

        extra_args = req.sampling_params.extra_args or {}
        height = req.sampling_params.height or DEFAULT_SKYREELS_R2V_HEIGHT
        width = req.sampling_params.width or DEFAULT_SKYREELS_R2V_WIDTH
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")
        ref_images = _resize_and_pad_ref_images(ref_images, height=height, width=width)

        num_frames = req.sampling_params.num_frames or DEFAULT_SKYREELS_R2V_FRAMES
        if num_frames <= 1:
            num_frames = DEFAULT_SKYREELS_R2V_FRAMES
        num_steps = req.sampling_params.num_inference_steps or DEFAULT_SKYREELS_R2V_STEPS
        output_type = req.sampling_params.output_type or "np"
        max_sequence_length = req.sampling_params.max_sequence_length or 512
        guidance_scale, guidance_scale_img = _resolve_guidance_scales(req.sampling_params, extra_args)
        self._guidance_scale = guidance_scale
        self._guidance_scale_img = guidance_scale_img

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ref_images=ref_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            guidance_scale_img=guidance_scale_img,
        )

        device = self.device
        dtype = self.transformer.dtype
        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_pipeline_start = time.perf_counter()
            _t_text_enc_start = _t_pipeline_start

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=(guidance_scale != 1.0 or guidance_scale_img != 1.0),
            num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_text_enc_ms = (time.perf_counter() - _t_text_enc_start) * 1000

        sample_solver = resolve_wan_sample_solver(req, default=self._sample_solver)
        flow_shift = resolve_wan_flow_shift(req, self.od_config)
        if sample_solver != self._sample_solver or abs(flow_shift - self._flow_shift) > 1e-6:
            self.scheduler = build_wan_scheduler(sample_solver, flow_shift)
            self._sample_solver = sample_solver
            self._flow_shift = flow_shift

        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        if DEBUG_PERF:
            _t_latent_prep_start = time.perf_counter()
        latents, condition = self.prepare_latents(
            ref_images=ref_images,
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=self.transformer.config.out_channels,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.sampling_params.latents,
        )
        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_latent_prep_ms = (time.perf_counter() - _t_latent_prep_start) * 1000

        if DEBUG_PERF:
            _t_denoise_start = time.perf_counter()
        latents = self.diffuse(
            latents=latents,
            condition=condition,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            guidance_scale_img=guidance_scale_img,
            dtype=dtype,
            attention_kwargs=None,
        )
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_denoise_ms = (time.perf_counter() - _t_denoise_start) * 1000

        if DEBUG_PERF:
            _t_decode_start = time.perf_counter()
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        if DEBUG_PERF:
            current_omni_platform.synchronize()
            _t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000
            _t_pipeline_wall_ms = (time.perf_counter() - _t_pipeline_start) * 1000
            _t_stages_sum = _t_text_enc_ms + _t_latent_prep_ms + _t_denoise_ms + _t_decode_ms
            if _is_rank_zero():
                logger.info(
                    "SkyReels V3 R2V timing: TextEncoding=%.2f ms, LatentPreparation=%.2f ms, "
                    "Denoising=%.2f ms (%d steps), Decoding=%.2f ms, StagesSum=%.2f ms, PipelineWall=%.2f ms",
                    _t_text_enc_ms,
                    _t_latent_prep_ms,
                    _t_denoise_ms,
                    len(timesteps),
                    _t_decode_ms,
                    _t_stages_sum,
                    _t_pipeline_wall_ms,
                )

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def predict_noise(
        self,
        latent_num_frames: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        result = self.transformer(**kwargs)
        if isinstance(result, IntermediateTensors):
            return result
        noise_pred = result[0]
        if latent_num_frames is not None:
            noise_pred = noise_pred[:, :, :latent_num_frames, :, :]
        return noise_pred

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        ref_images: list[PIL.Image.Image],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Received {len(generator)} generators but the effective batch size is {batch_size}.")
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        ref_generator = generator[0] if isinstance(generator, list) else generator
        ref_latents = []
        for ref_image in ref_images:
            ref_tensor = TF.to_tensor(ref_image).sub_(0.5).div_(0.5).to(device=device, dtype=self.vae.dtype)
            ref_latent = self.vae.encode(ref_tensor.unsqueeze(1).unsqueeze(0))
            ref_latent = retrieve_latents(ref_latent, generator=ref_generator)
            ref_latent = (ref_latent - latents_mean) * latents_std
            ref_latents.append(ref_latent.to(dtype=dtype))

        while len(ref_latents) < MAX_SKYREELS_R2V_REF_IMAGES:
            ref_latents.append(
                torch.zeros(
                    1,
                    num_channels_latents,
                    1,
                    latent_height,
                    latent_width,
                    device=device,
                    dtype=dtype,
                )
            )

        condition = torch.cat(ref_latents, dim=2)
        condition = condition.repeat(batch_size, 1, 1, 1, 1)
        return latents, condition

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        ref_images,
        height,
        width,
        guidance_scale,
        guidance_scale_img,
    ):
        if not isinstance(prompt, str):
            raise ValueError(f"`prompt` must be a string, got {type(prompt)}.")
        if negative_prompt is not None and not isinstance(negative_prompt, str):
            raise ValueError(f"`negative_prompt` must be a string, got {type(negative_prompt)}.")
        if not ref_images:
            raise ValueError("SkyReels V3 R2V requires at least one reference image.")
        if len(ref_images) > MAX_SKYREELS_R2V_REF_IMAGES:
            raise ValueError(
                f"SkyReels V3 R2V supports at most {MAX_SKYREELS_R2V_REF_IMAGES} reference images, "
                f"got {len(ref_images)}."
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")
        if guidance_scale < 0 or guidance_scale_img < 0:
            raise ValueError("SkyReels V3 R2V guidance scales must be non-negative.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
