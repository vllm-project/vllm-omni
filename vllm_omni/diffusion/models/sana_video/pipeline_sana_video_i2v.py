# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import warnings
from math import sqrt

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
    DistributedAutoencoderKLLTX2Video,
)
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
    model_parallel_is_initialized,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import (
    ASPECT_RATIO_480_BIN,
    ASPECT_RATIO_720_BIN,
    SanaVideoPipeline,
    get_sana_video_default_resolution,
    get_sana_video_post_process_func,
    resolve_sana_video_sample_size,
    retrieve_timesteps,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest, resolve_video_num_frames
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniTextPrompt

SANA_VIDEO_I2V_COMPLEX_HUMAN_INSTRUCTION = [
    (
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for "
        "video generation. Evaluate the level of detail in the user prompt:"
    ),
    (
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and "
        "temporal relationships to create vivid and dynamic scenes."
    ),
    ("- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating."),
    "Here are examples of how to transform or refine prompts:",
    (
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, "
        "peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding "
        "pots of blooming red flowers."
    ),
    (
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street "
        "lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker "
        "bus smoothly passing by towering glass skyscrapers."
    ),
    (
        "Please generate only the enhanced description for the prompt below and avoid including any additional "
        "commentary or evaluations:"
    ),
    "User Prompt: ",
]


def get_sana_video_i2v_post_process_func(od_config: OmniDiffusionConfig):
    return get_sana_video_post_process_func(od_config)


def get_sana_video_i2v_pre_process_func(od_config: OmniDiffusionConfig):
    default_height, default_width = get_sana_video_default_resolution(resolve_sana_video_sample_size(od_config))

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt, multi_modal_data={})

        multi_modal_data = prompt.get("multi_modal_data") or {}
        raw_image = multi_modal_data.get("image")
        if raw_image is None:
            raise ValueError('SANA-Video I2V requires `"multi_modal_data": {"image": <PIL image or file path>}`.')
        if isinstance(raw_image, str):
            if not os.path.isfile(raw_image):
                raise ValueError(f"SANA-Video I2V image path does not exist: {raw_image}")
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            raise TypeError(f"Unsupported SANA-Video I2V image type: {type(raw_image).__name__}")

        if request.sampling_params.height is None or request.sampling_params.width is None:
            max_area = default_height * default_width
            aspect_ratio = image.height / image.width
            inferred_height = round(sqrt(max_area * aspect_ratio)) // 32 * 32
            inferred_width = round(sqrt(max_area / aspect_ratio)) // 32 * 32
            if request.sampling_params.height is None:
                request.sampling_params.height = inferred_height
            if request.sampling_params.width is None:
                request.sampling_params.width = inferred_width

        image = image.resize(
            (request.sampling_params.width, request.sampling_params.height),
            PIL.Image.Resampling.LANCZOS,
        )
        multi_modal_data["image"] = image
        prompt["multi_modal_data"] = multi_modal_data
        request.prompt = prompt
        return request

    return pre_process_func


def _retrieve_latents(encoder_output: torch.Tensor) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from the SANA-Video VAE encoder output.")


class SanaImageToVideoPipeline(SanaVideoPipeline, SupportImageInput):
    """Native vLLM-Omni SANA-Video 2B image-to-video pipeline.

    The first latent frame is encoded from the input image and remains fixed;
    denoising updates only the subsequent video frames.
    """

    def _prepare_i2v_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"Generator list length {len(generator)} does not match effective batch size {batch_size}."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=dtype)
        else:
            latents = latents.to(device=self.device, dtype=dtype)

        image = image.unsqueeze(2).to(device=self.device, dtype=self.vae.dtype)
        image_latents = _retrieve_latents(self.vae.encode(image))
        image_latents = image_latents.repeat(batch_size, 1, 1, 1, 1)

        if isinstance(self.vae, DistributedAutoencoderKLLTX2Video):
            latents_mean = self.vae.latents_mean
            latents_std = self.vae.latents_std
        elif isinstance(self.vae, DistributedAutoencoderKLWan):
            latents_mean = torch.tensor(self.vae.config.latents_mean)
            latents_std = torch.tensor(self.vae.config.latents_std)
        else:
            raise TypeError(f"Unsupported SANA-Video I2V VAE type: {type(self.vae).__name__}")

        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        image_latents = (image_latents - latents_mean) / latents_std
        latents[:, :, :1] = image_latents.to(dtype)
        return latents

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("SANA-Video I2V currently supports exactly one prompt per request.")
        prompt_obj = req.prompts[0]
        if isinstance(prompt_obj, str):
            raise ValueError("SANA-Video I2V requires an image in `multi_modal_data`.")

        prompt = prompt_obj.get("prompt") or ""
        negative_prompt = prompt_obj.get("negative_prompt") or ""
        image = (prompt_obj.get("multi_modal_data") or {}).get("image")
        if not prompt:
            raise ValueError("Prompt is required for SANA-Video I2V generation.")
        if not isinstance(image, PIL.Image.Image):
            raise TypeError("SANA-Video I2V expects a preprocessed PIL image.")
        sampling = req.sampling_params
        default_height, default_width = get_sana_video_default_resolution(self.transformer.config.sample_size)
        height = sampling.height or default_height
        width = sampling.width or default_width
        frames = resolve_video_num_frames(
            sampling.num_frames,
            default_num_frames=81,
            is_dummy_run=req.is_dummy_run(),
        )
        num_steps = (
            sampling.num_inference_steps
            if sampling.num_inference_steps is not None
            else self.default_num_inference_steps
        )
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 6.0
        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        extra_args = sampling.extra_args or {}
        motion_score = extra_args.get("motion_score")
        if motion_score is not None:
            prompt = f"{prompt} motion score: {int(motion_score)}."

        video = self._generate_i2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            frames=frames,
            num_inference_steps=num_steps,
            timesteps=sampling.timesteps,
            sigmas=sampling.sigmas,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=sampling.num_outputs_per_prompt or 1,
            eta=sampling.eta,
            generator=generator,
            latents=sampling.latents,
            clean_caption=bool(extra_args.get("clean_caption", False)),
            use_resolution_binning=bool(extra_args.get("use_resolution_binning", True)),
            max_sequence_length=sampling.max_sequence_length or 300,
            output_type="latent" if sampling.output_type == "latent" else "raw",
        )
        return DiffusionOutput(output=video)

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        guidance_scale: float,
        conditioning_mask: torch.Tensor,
        extra_step_kwargs: dict,
        dtype: torch.dtype,
        output_slice: int | None,
    ) -> torch.Tensor:
        do_true_cfg = self.do_classifier_free_guidance
        # Single-process runs never initialize the parallel groups.
        cfg_parallel = model_parallel_is_initialized() and get_classifier_free_guidance_world_size() > 1
        if cfg_parallel:
            self.check_cfg_parallel_validity(guidance_scale, negative_prompt_embeds is not None)

        if do_true_cfg and not cfg_parallel:
            # Concatenate neg/pos and the mask for a single batch-2 forward.
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask])

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for timestep_value in timesteps:
                # Conditioning frames keep timestep 0 so they are not denoised.
                timestep = timestep_value.expand(conditioning_mask.shape) * (1 - conditioning_mask)

                if cfg_parallel:
                    latent_model_input = latents.to(dtype)
                    positive_kwargs = {
                        "hidden_states": latent_model_input,
                        "encoder_hidden_states": prompt_embeds.to(dtype),
                        "encoder_attention_mask": prompt_attention_mask,
                        "timestep": timestep,
                    }
                    negative_kwargs = (
                        {
                            "hidden_states": latent_model_input,
                            "encoder_hidden_states": negative_prompt_embeds.to(dtype),
                            "encoder_attention_mask": negative_prompt_attention_mask,
                            "timestep": timestep,
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
                        output_slice=output_slice,
                    ).float()
                else:
                    latent_input = torch.cat([latents, latents]) if do_true_cfg else latents
                    noise_pred = self.transformer(
                        latent_input.to(dtype),
                        encoder_hidden_states=prompt_embeds.to(dtype),
                        encoder_attention_mask=prompt_attention_mask,
                        timestep=timestep,
                        return_dict=False,
                    )[0].float()
                    if do_true_cfg:
                        uncond, cond = noise_pred.chunk(2)
                        noise_pred = uncond + guidance_scale * (cond - uncond)
                    if output_slice is not None:
                        noise_pred = noise_pred[:, :output_slice]

                # Only the non-conditioning frames are denoised; the first frame is carried through.
                denoised = self.scheduler.step(
                    noise_pred[:, :, 1:],
                    timestep_value,
                    latents[:, :, 1:],
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                latents = torch.cat([latents[:, :, :1], denoised], dim=2)
                progress_bar.update()

        return latents

    @torch.no_grad()
    def _generate_i2v(
        self,
        *,
        image: PIL.Image.Image,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        use_resolution_binning: bool,
        max_sequence_length: int,
        timesteps: list[int] | torch.Tensor | None = None,
        sigmas: list[float] | None = None,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        clean_caption: bool = False,
        output_type: str = "raw",
        complex_human_instruction: list[str] = SANA_VIDEO_I2V_COMPLEX_HUMAN_INSTRUCTION,
    ) -> torch.Tensor:
        orig_height, orig_width = height, width
        if use_resolution_binning:
            if self.transformer.config.sample_size == 30:
                aspect_ratio_bin = ASPECT_RATIO_480_BIN
            elif self.transformer.config.sample_size == 22:
                aspect_ratio_bin = ASPECT_RATIO_720_BIN
            else:
                raise ValueError(f"Unsupported SANA-Video sample size: {self.transformer.config.sample_size}")
            height, width = self.video_processor.classify_height_width_bin(
                height,
                width,
                ratios=aspect_ratio_bin,
            )

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        prompt_embeds, prompt_mask, negative_embeds, negative_mask = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            device=self.device,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=self.device,
            timesteps=timesteps,
            sigmas=sigmas,
        )
        image_tensor = self.video_processor.preprocess(image, height=height, width=width).to(
            self.device,
            dtype=torch.float32,
        )
        latent_channels = self.transformer.config.in_channels
        latents = self._prepare_i2v_latents(
            image_tensor,
            batch_size=num_videos_per_prompt,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            num_frames=frames,
            dtype=torch.float32,
            generator=generator,
            latents=latents,
        )

        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        conditioning_mask = latents.new_zeros(
            num_videos_per_prompt,
            1,
            latents.shape[2] // patch_t,
            latents.shape[3] // patch_h,
            latents.shape[4] // patch_w,
        )
        conditioning_mask[:, :, 0] = 1

        self._num_timesteps = len(timesteps)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # learned sigma: transformer predicts 2x latent channels; keep the first half.
        output_slice = latent_channels if self.transformer.config.out_channels // 2 == latent_channels else None
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_attention_mask=negative_mask,
            guidance_scale=guidance_scale,
            conditioning_mask=conditioning_mask,
            extra_step_kwargs=extra_step_kwargs,
            dtype=self.transformer.dtype,
            output_slice=output_slice,
        )

        if output_type == "latent":
            return latents

        latents = latents.to(self.vae.dtype)
        if isinstance(self.vae, DistributedAutoencoderKLLTX2Video):
            latents_mean = self.vae.latents_mean
            latents_std = self.vae.latents_std
            latent_dim = self.vae.config.latent_channels
        else:
            latents_mean = torch.tensor(self.vae.config.latents_mean)
            latents_std = torch.tensor(self.vae.config.latents_std)
            latent_dim = self.vae.config.z_dim
        latents_mean = latents_mean.view(1, latent_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, latent_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std + latents_mean

        try:
            video = self.vae.decode(latents, return_dict=False)[0]
        except torch.OutOfMemoryError as error:
            warnings.warn(
                f"{error}. Enable VAE tiling for SANA-Video I2V high-resolution decoding.",
                stacklevel=2,
            )
            raise
        if use_resolution_binning:
            video = self.video_processor.resize_and_crop_tensor(video, orig_width, orig_height)
        return video
