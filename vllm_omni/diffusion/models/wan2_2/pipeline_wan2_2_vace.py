# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
VACE (Video Creation and Editing) Pipeline for WAN models.

VACE is an all-in-one model for video creation and editing. The mode is
determined by which inputs are provided (no explicit mode flag):

- T2V: Text-to-Video (prompt only)
- R2V: Reference-to-Video (prompt + reference_images)
- V2V: Video-to-Video editing (prompt + video)
- MV2V: Masked Video-to-Video / inpainting (prompt + video + mask)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import PIL.Image
import torch
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    retrieve_latents,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    get_wan22_post_process_func as get_wan22_vace_post_process_func,  # noqa: F401
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import WanVACETransformer3DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def create_vace_transformer_from_config(config: dict) -> WanVACETransformer3DModel:
    """Create WanVACETransformer3DModel from config dict."""
    kwargs = {}
    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "freq_dim" in config:
        kwargs["freq_dim"] = config["freq_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "added_kv_proj_dim" in config:
        kwargs["added_kv_proj_dim"] = config["added_kv_proj_dim"]
    if "rope_max_seq_len" in config:
        kwargs["rope_max_seq_len"] = config["rope_max_seq_len"]
    if "pos_embed_seq_len" in config:
        kwargs["pos_embed_seq_len"] = config["pos_embed_seq_len"]
    if "vace_layers" in config:
        kwargs["vace_layers"] = config["vace_layers"]
    if "vace_in_channels" in config:
        kwargs["vace_in_channels"] = config["vace_in_channels"]

    return WanVACETransformer3DModel(**kwargs)


def get_wan22_vace_pre_process_func(od_config: OmniDiffusionConfig):
    """Pre-process function for VACE: handle reference images, source videos, and masks."""
    import numpy as np

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if not multi_modal_data:
                request.prompts[i] = prompt
                continue

            # Handle reference images for R2V
            # "image" is the standard key from online serving (SupportImageInput convention)
            # "reference_images" is the offline API key for backwards compatibility
            ref_images = multi_modal_data.get("image") or multi_modal_data.get("reference_images")
            if ref_images is not None:
                if isinstance(ref_images, str):
                    ref_images = [PIL.Image.open(ref_images).convert("RGB")]
                elif isinstance(ref_images, PIL.Image.Image):
                    ref_images = [ref_images]
                elif isinstance(ref_images, list):
                    ref_images = [
                        PIL.Image.open(img).convert("RGB") if isinstance(img, str) else img for img in ref_images
                    ]

                # Calculate dimensions from first reference image if not provided
                if request.sampling_params.height is None or request.sampling_params.width is None:
                    first_img = ref_images[0]
                    max_area = 480 * 832  # VACE default is 480p
                    aspect_ratio = first_img.height / first_img.width
                    mod_value = 16
                    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                    if request.sampling_params.height is None:
                        request.sampling_params.height = height
                    if request.sampling_params.width is None:
                        request.sampling_params.width = width

                prompt["additional_information"]["reference_images"] = ref_images

            # Handle source video for V2V / MV2V
            source_video = multi_modal_data.get("video")
            if source_video is not None:
                if isinstance(source_video, list) and len(source_video) > 0:
                    if isinstance(source_video[0], str):
                        source_video = [PIL.Image.open(f).convert("RGB") for f in source_video]
                prompt["additional_information"]["source_video"] = source_video

            # Handle mask for MV2V / inpainting
            mask = multi_modal_data.get("mask")
            if mask is not None:
                if isinstance(mask, list) and len(mask) > 0:
                    if isinstance(mask[0], str):
                        mask = [PIL.Image.open(m).convert("L") for m in mask]
                elif isinstance(mask, str):
                    mask = [PIL.Image.open(mask).convert("L")]
                elif isinstance(mask, PIL.Image.Image):
                    mask = [mask]
                prompt["additional_information"]["mask"] = mask

            request.prompts[i] = prompt
        return request

    return pre_process_func


class Wan22VACEPipeline(Wan22Pipeline, SupportImageInput):
    """VACE (Video Creation and Editing) Pipeline for Wan2.1.

    Extends Wan22Pipeline with VACE-specific context creation and weight loading.
    All VACE modes (T2V, R2V, V2V, MV2V) are handled by varying the inputs.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        # VACE defaults to flow_shift=3.0 for 480p (base WAN T2V uses 5.0 for 720p)
        if od_config.flow_shift is None:
            od_config = replace(od_config, flow_shift=3.0)

        super().__init__(od_config=od_config, prefix=prefix)

    def _create_transformer(self, config: dict) -> WanVACETransformer3DModel:
        """Build VACE transformer directly from config dict."""
        return create_vace_transformer_from_config(config)

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        dtype: torch.dtype,
        attention_kwargs: dict[str, object],
        vace_context: torch.Tensor | None,
        vace_context_scale: float,
    ) -> torch.Tensor:
        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t
                latent_model_input = latents.to(dtype)
                timestep = t.expand(latents.shape[0])

                do_true_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "vace_context": vace_context,
                    "vace_context_scale": vace_context_scale,
                    "return_dict": False,
                }
                negative_kwargs = (
                    {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "attention_kwargs": attention_kwargs,
                        "vace_context": vace_context,
                        "vace_context_scale": vace_context_scale,
                        "return_dict": False,
                    }
                    if do_true_cfg
                    else None
                )

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)
                pbar.update()

        return latents

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        video=None,
        mask=None,
        reference_images=None,
    ):
        super().check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # VACE-specific: validate video/mask/reference_images consistency
        if video is not None:
            if mask is not None and len(video) != len(mask):
                raise ValueError(
                    f"Length of `video` ({len(video)}) and `mask` ({len(mask)}) do not match. "
                    "Please make sure that they have the same length."
                )
            if reference_images is not None:
                is_pil_image = isinstance(reference_images, PIL.Image.Image)
                is_list_of_pil_images = isinstance(reference_images, list) and all(
                    isinstance(img, PIL.Image.Image) for img in reference_images
                )
                if not (is_pil_image or is_list_of_pil_images):
                    raise ValueError(
                        "`reference_images` has to be of type `PIL.Image.Image` or `list` of `PIL.Image.Image`, "
                        f"but is {type(reference_images)}"
                    )
        elif mask is not None:
            raise ValueError("`mask` can only be passed if `video` is passed as well.")

    def preprocess_conditions(
        self,
        video: list | torch.Tensor | None,
        mask: list | torch.Tensor | None,
        reference_images: list[PIL.Image.Image] | None,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
        """Preprocess video, mask, and reference images for VACE conditioning.

        - If video is None, create zero tensor (T2V mode)
        - If mask is None, create all-ones tensor (generate everything)
        - Reference images are resized maintaining aspect ratio and center-padded

        Returns:
            (video, mask, reference_images_processed) tensors ready for VAE encoding.
        """
        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        if video is None:
            video = torch.zeros(1, 3, num_frames, height, width, dtype=dtype, device=device)
            image_size = (height, width)
        else:
            base = self.vae_scale_factor_spatial * self.transformer_config.patch_size[1]
            if isinstance(video, list):
                video_height, video_width = video_processor.get_default_height_width(video[0])
                # Downscale if video exceeds target area
                if video_height * video_width > height * width:
                    scale = min(width / video_width, height / video_height)
                    video_height, video_width = int(video_height * scale), int(video_width * scale)
                # Align to base
                video_height = (video_height // base) * base
                video_width = (video_width // base) * base
                video = video_processor.preprocess_video(video, video_height, video_width)
            image_size = (video.shape[-2], video.shape[-1])

        if mask is None:
            mask = torch.ones_like(video)
        else:
            if isinstance(mask, list):
                mask = video_processor.preprocess_video(mask, image_size[0], image_size[1])
            mask = torch.clamp((mask + 1) / 2, min=0, max=1)

        video = video.to(dtype=dtype, device=device)
        mask = mask.to(dtype=dtype, device=device)

        # Preprocess reference images: resize with aspect ratio, center-pad on white canvas
        ref_images_processed: list[list[torch.Tensor]] = []
        if reference_images is not None and len(reference_images) > 0:
            preprocessed = []
            for image in reference_images:
                img_tensor = video_processor.preprocess(image, None, None)
                img_h, img_w = img_tensor.shape[-2:]
                scale = min(image_size[0] / img_h, image_size[1] / img_w)
                new_h, new_w = int(img_h * scale), int(img_w * scale)
                resized = torch.nn.functional.interpolate(
                    img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
                ).squeeze(0)
                canvas = torch.ones(3, *image_size, device=device, dtype=dtype)
                top = (image_size[0] - new_h) // 2
                left = (image_size[1] - new_w) // 2
                canvas[:, top : top + new_h, left : left + new_w] = resized
                preprocessed.append(canvas)
            ref_images_processed = [preprocessed]
        else:
            ref_images_processed = [[]]

        return video, mask, ref_images_processed

    def prepare_video_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: list[list[torch.Tensor]],
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode video and reference images into VACE conditioning latents.

        - Encodes inactive (video * (1-mask)) and reactive (video * mask) regions
        - Reference images are encoded and prepended as extra temporal frames
        """
        vae_dtype = self.vae.dtype

        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=device, dtype=torch.float32).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )

        # Binarize mask
        mask = torch.where(mask > 0.5, 1.0, 0.0).to(dtype=vae_dtype)

        # Encode inactive and reactive regions separately
        video = video.to(dtype=vae_dtype)
        inactive = video * (1 - mask)
        reactive = video * mask

        with torch.no_grad():
            inactive_latent = retrieve_latents(self.vae.encode(inactive), generator, sample_mode="argmax")
            reactive_latent = retrieve_latents(self.vae.encode(reactive), generator, sample_mode="argmax")

        inactive_latent = ((inactive_latent.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive_latent = ((reactive_latent.float() - latents_mean) * latents_std).to(vae_dtype)

        # Concatenate inactive + reactive along channels -> [B, 2*z_dim, T, H, W]
        latents = torch.cat([inactive_latent, reactive_latent], dim=1)

        # Prepend reference image latents along temporal dimension
        latent_list = []
        for latent, ref_batch in zip(latents, reference_images):
            for ref_image in ref_batch:
                ref_image = ref_image.to(dtype=vae_dtype)
                ref_image = ref_image[None, :, None, :, :]  # [1, C, 1, H, W]
                with torch.no_grad():
                    ref_latent = retrieve_latents(self.vae.encode(ref_image), generator, sample_mode="argmax")
                ref_latent = ((ref_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                ref_latent = ref_latent.squeeze(0)  # [z_dim, 1, H, W]
                # Double channels with zeros (inactive=ref, reactive=zeros)
                ref_latent = torch.cat([ref_latent, torch.zeros_like(ref_latent)], dim=0)
                # Prepend along temporal dimension
                latent = torch.cat([ref_latent, latent], dim=1)
            latent_list.append(latent)

        return torch.stack(latent_list)

    def prepare_masks(
        self,
        mask: torch.Tensor,
        reference_images: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        """Encode mask using spatial stride sampling and prepend reference padding.

        - 8x8 spatial stride encoding -> 64 channels
        - Zero-masks prepended for reference image frames
        """
        patch_size = self.transformer_config.patch_size if hasattr(self.transformer_config, "patch_size") else (1, 2, 2)
        if isinstance(self.transformer_config, dict):
            patch_size = self.transformer_config.get("patch_size", (1, 2, 2))
        transformer_patch_size = patch_size[1] if isinstance(patch_size, list | tuple) else 2

        mask_list = []
        for mask_, ref_batch in zip(mask, reference_images):
            num_channels, num_frames, height, width = mask_.shape
            new_num_frames = (num_frames + self.vae_scale_factor_temporal - 1) // self.vae_scale_factor_temporal
            new_height = height // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size
            new_width = width // (self.vae_scale_factor_spatial * transformer_patch_size) * transformer_patch_size

            m = mask_[0, :, :, :]  # [T, H, W]
            m = m.view(num_frames, new_height, self.vae_scale_factor_spatial, new_width, self.vae_scale_factor_spatial)
            m = m.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [64, T, H', W']
            m = torch.nn.functional.interpolate(
                m.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
            ).squeeze(0)

            # Prepend zero-masks for reference image frames
            num_ref = len(ref_batch)
            if num_ref > 0:
                mask_padding = torch.zeros_like(m[:, :num_ref, :, :])
                m = torch.cat([mask_padding, m], dim=1)

            mask_list.append(m)

        return torch.stack(mask_list)

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        vace_context_scale: float | list[float] = 1.0,
        **kwargs,
    ) -> DiffusionOutput:
        """Generate or edit video using VACE.

        The mode is determined by which inputs are provided in the request:
        - T2V: prompt only (no video/mask/reference_images)
        - R2V: prompt + reference_images (in multi_modal_data)
        - V2V: prompt + video (in multi_modal_data)
        - MV2V: prompt + video + mask (in multi_modal_data)

        Args:
            req: Diffusion request containing prompt and optional multi-modal data.
            prompt: Text prompt (overridden by req.prompts if provided).
            negative_prompt: Negative prompt for CFG.
            height: Output video height.
            width: Output video width.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale.
            frame_num: Number of output frames.
            output_type: Output format ("np", "pt", or "latent").
            generator: Random generator for reproducibility.
            prompt_embeds: Pre-computed prompt embeddings.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            attention_kwargs: Additional kwargs for attention layers.
            vace_context_scale: VACE conditioning strength.
        """
        # Get parameters from request or arguments
        if len(req.prompts) > 1:
            raise ValueError(
                "This model only supports a single prompt, not a batched request. "
                "Please pass in a single prompt object or string, or a single-item list."
            )

        reference_images = None
        source_video = None
        source_mask = None

        if len(req.prompts) == 1:
            first_prompt = req.prompts[0]
            if isinstance(first_prompt, str):
                prompt = first_prompt
            else:
                prompt = first_prompt.get("prompt")
                negative_prompt = negative_prompt or first_prompt.get("negative_prompt")
                prompt_embeds = prompt_embeds if prompt_embeds is not None else first_prompt.get("prompt_embeds")
                negative_prompt_embeds = (
                    negative_prompt_embeds
                    if negative_prompt_embeds is not None
                    else first_prompt.get("negative_prompt_embeds")
                )

                additional_info = first_prompt.get("additional_information", {})
                reference_images = additional_info.get("reference_images")
                source_video = additional_info.get("source_video")
                source_mask = additional_info.get("mask")

        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for VACE generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames or frame_num
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        generator = req.sampling_params.generator or generator

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        # Ensure dimensions are compatible with VAE and patch size
        mod_value = self.vae_scale_factor_spatial * 2  # 8 * 2 = 16
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video=source_video,
            mask=source_mask,
            reference_images=reference_images,
        )

        device = self.device
        self._guidance_scale = guidance_scale
        dtype = self.transformer.dtype if self.transformer is not None else torch.bfloat16

        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        # Encode prompts
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("Either prompt or prompt_embeds must be provided.")
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                num_videos_per_prompt=req.sampling_params.num_outputs_per_prompt or 1,
                max_sequence_length=req.sampling_params.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            elif guidance_scale > 1.0:
                _, negative_prompt_embeds = self.encode_prompt(
                    prompt="",
                    negative_prompt=None,
                    do_classifier_free_guidance=True,
                    device=device,
                    dtype=dtype,
                )

        num_reference_images = 0
        if self.transformer.vace_patch_embedding is not None:
            video, mask, ref_images_processed = self.preprocess_conditions(
                video=source_video,
                mask=source_mask,
                reference_images=reference_images,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=dtype,
                device=device,
            )

            conditioning_latents = self.prepare_video_latents(video, mask, ref_images_processed, generator, device)
            mask_encoded = self.prepare_masks(mask, ref_images_processed)

            # Unified VACE context: [video_latents, mask] along channels -> [B, C, T, H, W]
            vace_context = torch.cat([conditioning_latents, mask_encoded], dim=1)

            num_reference_images = len(ref_images_processed[0]) if ref_images_processed else 0
        else:
            vace_context = None

        # Prepare noise latents (extra frames for reference images)
        num_channels_latents = self.transformer_config.in_channels
        noise_num_frames = num_frames + num_reference_images * self.vae_scale_factor_temporal
        latents = self.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=noise_num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.sampling_params.latents,
        )

        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            vace_context=vace_context,
            vace_context_scale=vace_context_scale,
        )

        self._current_timestep = None

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        # Trim reference frames from output before decoding
        # (reference images were prepended as extra temporal frames)
        if output_type != "latent" and num_reference_images > 0:
            latents = latents[:, :, num_reference_images:]

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

        return DiffusionOutput(output=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
