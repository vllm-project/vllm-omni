# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Task-specific conditioning shared by LTX model versions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from . import ltx2_latents as latent_ops
from .ltx2_denoise import I2VVideoAudioScheduler

if TYPE_CHECKING:
    from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
    from .ltx2_request import LTXRequestInputs


@dataclass
class LTXPromptContext:
    """Connector outputs consumed by an LTX denoise phase."""

    batch_size: int
    connector_prompt_embeds: torch.Tensor
    connector_audio_prompt_embeds: torch.Tensor
    connector_attention_mask: torch.Tensor
    positive_connector_prompt_embeds: torch.Tensor
    positive_connector_audio_prompt_embeds: torch.Tensor
    positive_connector_attention_mask: torch.Tensor
    negative_connector_prompt_embeds: torch.Tensor | None
    negative_connector_audio_prompt_embeds: torch.Tensor | None
    negative_connector_attention_mask: torch.Tensor | None


def _repeat_prompt_tensor_for_outputs(tensor: torch.Tensor, num_outputs: int) -> torch.Tensor:
    if num_outputs == 1:
        return tensor
    return tensor.repeat_interleave(num_outputs, dim=0)


class LTXTextConditioningMixin:
    """Shared Gemma encoding and text connector orchestration."""

    connector_batches_cfg = False

    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del scale_factor
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_inputs = self.tokenizer(
            [text.strip() for text in prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_attention_mask = text_inputs.attention_mask.to(device)
        hidden_states = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states

        prompt_embeds = torch.stack(hidden_states, dim=-1).flatten(2, 3).to(dtype=dtype)
        prompt_embeds = _repeat_prompt_tensor_for_outputs(prompt_embeds, num_videos_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = _repeat_prompt_tensor_for_outputs(
            prompt_attention_mask,
            num_videos_per_prompt,
        )
        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]
        negative_prompt_embeds_provided = negative_prompt_embeds is not None

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )
        elif num_videos_per_prompt > 1:
            prompt_embeds = _repeat_prompt_tensor_for_outputs(prompt_embeds, num_videos_per_prompt)
            prompt_attention_mask = _repeat_prompt_tensor_for_outputs(
                prompt_attention_mask,
                num_videos_per_prompt,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type as `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if isinstance(negative_prompt, list) and batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt` has batch size"
                    f" {batch_size}."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )
        elif do_classifier_free_guidance and negative_prompt_embeds_provided and num_videos_per_prompt > 1:
            negative_prompt_embeds = _repeat_prompt_tensor_for_outputs(
                negative_prompt_embeds,
                num_videos_per_prompt,
            )
            negative_prompt_attention_mask = _repeat_prompt_tensor_for_outputs(
                negative_prompt_attention_mask,
                num_videos_per_prompt,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def _prepare_prompt_context(
        self,
        *,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        num_videos_per_prompt: int,
        max_sequence_length: int,
    ) -> LTXPromptContext:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = (
            self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )
        )
        padding_side = getattr(self.tokenizer, "padding_side", "left")

        if self.do_classifier_free_guidance and self.connector_batches_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        video_context, audio_context, attention_mask = self.connectors(
            prompt_embeds,
            prompt_attention_mask,
            padding_side=padding_side,
        )
        positive_video_context = video_context
        positive_audio_context = audio_context
        positive_attention_mask = attention_mask
        negative_video_context = None
        negative_audio_context = None
        negative_attention_mask = None

        if self.do_classifier_free_guidance and self.connector_batches_cfg:
            split_batch = batch_size * num_videos_per_prompt
            negative_video_context = video_context[:split_batch]
            positive_video_context = video_context[split_batch:]
            negative_audio_context = audio_context[:split_batch]
            positive_audio_context = audio_context[split_batch:]
            negative_attention_mask = attention_mask[:split_batch]
            positive_attention_mask = attention_mask[split_batch:]
        elif self.do_classifier_free_guidance:
            negative_video_context, negative_audio_context, negative_attention_mask = self.connectors(
                negative_prompt_embeds,
                negative_prompt_attention_mask,
                padding_side=padding_side,
            )

        return LTXPromptContext(
            batch_size=batch_size,
            connector_prompt_embeds=video_context,
            connector_audio_prompt_embeds=audio_context,
            connector_attention_mask=attention_mask,
            positive_connector_prompt_embeds=positive_video_context,
            positive_connector_audio_prompt_embeds=positive_audio_context,
            positive_connector_attention_mask=positive_attention_mask,
            negative_connector_prompt_embeds=negative_video_context,
            negative_connector_audio_prompt_embeds=negative_audio_context,
            negative_connector_attention_mask=negative_attention_mask,
        )


class LTXI2VConditioningMixin:
    """First-frame conditioning behavior common to LTX2 and LTX2.3."""

    support_image_input = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_spatial_compression_ratio,
            resample="bilinear",
        )

    def forward(self, req: Any, image: Any | None = None, *args: Any, **kwargs: Any) -> Any:
        """Preserve the public I2V positional API while sharing the T2V runner."""
        return super().forward(req, *args, image=image, **kwargs)

    @staticmethod
    def _resolve_single_prompt_image(raw_image: Any) -> Any:
        if isinstance(raw_image, list):
            if len(raw_image) != 1:
                raise ValueError(
                    "LTX I2V prompt dictionaries support exactly one image per prompt. "
                    "Pass one image per prompt for batched I2V requests."
                )
            return raw_image[0]
        return raw_image

    @staticmethod
    def _resolve_additional_image(additional: dict[str, Any]) -> Any:
        for field_name in ("preprocessed_image", "pixel_values", "image"):
            raw_image = additional.get(field_name)
            if raw_image is not None:
                return raw_image
        return None

    def _resolve_request_image(
        self,
        req: Any,
        image: Any | None,
        request_inputs: LTXRequestInputs,
    ) -> Any | None:
        if image is not None or not req.prompts:
            return image

        raw_images = []
        for prompt_item in req.prompts:
            if isinstance(prompt_item, str):
                raw_image = None
            else:
                multi_modal_data = prompt_item.get("multi_modal_data") or {}
                raw_image = multi_modal_data.get("image")
                if raw_image is None:
                    raw_image = self._resolve_additional_image(prompt_item.get("additional_information") or {})
            raw_image = self._resolve_single_prompt_image(raw_image)
            if isinstance(raw_image, str):
                raw_image = PIL.Image.open(raw_image).convert("RGB")
            raw_images.append(raw_image)

        if any(raw_image is None for raw_image in raw_images) and request_inputs.latents is None:
            raise ValueError("Image is required for LTX I2V generation.")
        if len(raw_images) == 1:
            return raw_images[0]
        return raw_images or image

    def _check_forward_inputs(
        self,
        request_inputs: LTXRequestInputs,
        image: Any | None = None,
    ) -> None:
        if image is None and request_inputs.latents is None:
            raise ValueError("Provide either `image` or `latents`. Cannot leave both undefined.")
        super()._check_forward_inputs(request_inputs, image=image)

    def prepare_latents(
        self,
        image: torch.Tensor | None = None,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_frames, height, width = latent_ops.resolve_video_latent_shape(
            height,
            width,
            num_frames,
            vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
            vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
        )
        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if latents is not None:
            if latents.ndim == 5:
                batch_size, _, num_frames, height, width = latents.shape
                mask_shape = (batch_size, 1, num_frames, height, width)
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0
                latents = latent_ops.normalize_latents(
                    latents,
                    self.vae.latents_mean,
                    self.vae.latents_std,
                    self.vae.config.scaling_factor,
                )
                latents = latent_ops.create_noised_state(
                    latents,
                    noise_scale * (1 - conditioning_mask),
                    generator,
                )
                latents = latent_ops.pack_latents(
                    latents,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
            else:
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

            conditioning_mask = latent_ops.pack_latents(
                conditioning_mask,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            ).squeeze(-1)
            if latents.ndim != 3 or latents.shape[:2] != conditioning_mask.shape:
                raise ValueError(
                    "Provided `latents` tensor has shape"
                    f" {latents.shape}, but the expected shape is {conditioning_mask.shape + (num_channels_latents,)}."
                )
            return latents.to(device=device, dtype=dtype), conditioning_mask

        if image is None:
            raise ValueError("`image` must be provided when `latents` is None.")

        image_batch_size = image.shape[0]
        if image_batch_size == 0:
            raise ValueError("`image` batch is empty.")
        if batch_size % image_batch_size != 0:
            raise ValueError(
                f"`batch_size` ({batch_size}) must be divisible by image batch size ({image_batch_size}) "
                "for image-to-video outputs."
            )
        num_videos_per_prompt = batch_size // image_batch_size

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective"
                    f" batch size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            image_generators = [generator[i * num_videos_per_prompt] for i in range(image_batch_size)]
            init_latents = [
                retrieve_latents(
                    self.vae.encode(image[i].unsqueeze(0).unsqueeze(2)),
                    image_generators[i],
                    "argmax",
                )
                for i in range(image_batch_size)
            ]
        else:
            init_latents = [
                retrieve_latents(
                    self.vae.encode(img.unsqueeze(0).unsqueeze(2)),
                    generator,
                    "argmax",
                )
                for img in image
            ]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        if num_videos_per_prompt > 1:
            init_latents = init_latents.repeat_interleave(num_videos_per_prompt, dim=0)
        init_latents = latent_ops.normalize_latents(
            init_latents,
            self.vae.latents_mean,
            self.vae.latents_std,
        )
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)

        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, 0] = 1.0
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = latent_ops.pack_latents(
            conditioning_mask,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        ).squeeze(-1)
        latents = latent_ops.pack_latents(
            latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        return latents, conditioning_mask

    def _step_video_latents_i2v(
        self,
        noise_pred_video: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        noise_pred_video = latent_ops.unpack_latents(
            noise_pred_video,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents_unpacked = latent_ops.unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        noise_pred_video = noise_pred_video[:, :, 1:]
        noise_latents = latents_unpacked[:, :, 1:]
        pred_latents = self.scheduler.step(
            noise_pred_video,
            timestep,
            noise_latents,
            return_dict=False,
        )[0]
        latents_unpacked = torch.cat([latents_unpacked[:, :, :1], pred_latents], dim=2)
        return latent_ops.pack_latents(
            latents_unpacked,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )

    def _prepare_video_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: LTXPromptContext,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if request_inputs.latents is None:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image.unsqueeze(0)
            elif isinstance(image, list) and image and isinstance(image[0], torch.Tensor):
                image = torch.stack(image, dim=0)
            else:
                image = self.video_processor.preprocess(
                    image,
                    height=request_inputs.height,
                    width=request_inputs.width,
                )
            image = image.to(
                device=device,
                dtype=prompt_context.positive_connector_prompt_embeds.dtype,
            )

        return self.prepare_latents(
            image,
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            self.transformer.config.in_channels,
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            noise_scale,
            torch.float32,
            device,
            request_inputs.generator,
            request_inputs.latents,
        )

    def _make_video_audio_scheduler(
        self,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        return I2VVideoAudioScheduler(
            self,
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
        )

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        denoise_ctx = super()._prepare_denoise_context_for_cfg(forward_ctx, denoise_ctx)
        if denoise_ctx.conditioning_mask is None:
            raise ValueError("LTX I2V denoising requires a conditioning mask.")

        mask_batch = denoise_ctx.conditioning_mask.shape[0]
        model_batch = denoise_ctx.video_coords.shape[0]
        if model_batch % mask_batch != 0:
            raise ValueError(
                "I2V conditioning-mask batch must divide the Transformer input batch, "
                f"but got {mask_batch} and {model_batch}."
            )
        repeats = model_batch // mask_batch
        denoise_ctx.conditioning_mask_for_model = (
            denoise_ctx.conditioning_mask if repeats == 1 else torch.cat([denoise_ctx.conditioning_mask] * repeats)
        )
        return denoise_ctx

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        kwargs = super()._denoise_timestep_kwargs(ts, forward_ctx, denoise_ctx)
        conditioning_mask = (
            denoise_ctx.conditioning_mask if forward_ctx.cfg_parallel_ready else denoise_ctx.conditioning_mask_for_model
        )
        if conditioning_mask is None:
            raise ValueError("LTX I2V denoising requires a conditioning mask.")
        kwargs.update(
            timestep=ts.unsqueeze(-1) * (1 - conditioning_mask),
            audio_timestep=ts,
        )
        return kwargs
