# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.flux2_klein.kv_cache import Flux2KVCache
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
    Flux2KleinPipeline,
    compute_empirical_mu,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_flux2_klein_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import get_flux2_klein_post_process_func as _func

    return _func(od_config)


class Flux2KleinKVPipeline(Flux2KleinPipeline):
    """Flux2 klein pipeline with KV cache optimization for fast image editing.

    On the first denoising step, reference image tokens are included in the forward
    pass and their attention K/V projections are cached. On subsequent steps, the
    cached K/V are reused without recomputing, providing faster inference when
    using reference images.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
        is_distilled: bool = True,
    ):
        super().__init__(od_config=od_config, prefix=prefix, is_distilled=is_distilled)
        self.use_kv_cache = True

    def forward(
        self,
        req: OmniDiffusionRequest,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> DiffusionOutput:
        """Forward with KV cache optimization."""
        if len(req.prompts) > 1:
            logger.warning(
                "This model only supports a single prompt, not a batched request. Taking only the first image for now."
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        is_dummy_warmup = (
            not getattr(req, "request_id", None)
            and prompt == "dummy run"
            and (req.sampling_params.num_inference_steps == 1)
        )

        if (
            raw_image := None
            if isinstance(first_prompt, str)
            else first_prompt.get("multi_modal_data", {}).get("image")
        ) is None:
            pass
        elif isinstance(raw_image, list):
            image = [PIL.Image.open(im) if isinstance(im, str) else cast(PIL.Image.Image, im) for im in raw_image]
        else:
            image = PIL.Image.open(raw_image) if isinstance(raw_image, str) else cast(PIL.Image.Image, raw_image)

        if is_dummy_warmup and image is not None:
            image = None

        # KV path is only needed for image-edit requests with reference images.
        # Fall back to the baseline pipeline for text-to-image requests.
        if image is None:
            return super().forward(
                req,
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        text_encoder_out_layers = req.sampling_params.extra_args.get("text_encoder_out_layers", text_encoder_out_layers)

        req_prompt_embeds = [p.get("prompt_embeds") if not isinstance(p, str) else None for p in req.prompts]
        if any(p is not None for p in req_prompt_embeds):
            prompt_embeds = torch.stack(req_prompt_embeds)  # type: ignore[arg-type]

        req_negative_prompt_embeds = [
            p.get("negative_prompt_embeds") if not isinstance(p, str) else None for p in req.prompts
        ]
        if any(p is not None for p in req_negative_prompt_embeds):
            negative_prompt_embeds = torch.stack(req_negative_prompt_embeds)  # type: ignore[arg-type]

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        negative_text_ids = None
        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        condition_images = None
        if image is not None and not isinstance(image, list):
            image = [image]
        if image is not None:
            condition_images = []
            for img in image:
                self.image_processor.check_image_input(img)
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size
                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        kv_cache_pos: Flux2KVCache | None = None
        kv_cache_neg: Flux2KVCache | None = None

        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if i == 0 and image_latents is not None:
                latent_model_input = torch.cat([image_latents, latents], dim=1).to(self.transformer.dtype)
                latent_image_ids = torch.cat([image_latent_ids, latent_ids], dim=1)
                num_ref_tokens = image_latents.shape[1]
            else:
                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids
                num_ref_tokens = 0

            def _run_transformer(
                encoder_states: torch.Tensor,
                encoder_ids: torch.Tensor,
                *,
                cache: Flux2KVCache | None,
                mode: str | None,
                hidden_states: torch.Tensor,
                image_ids: torch.Tensor,
                mode_num_ref_tokens: int = 0,
                mode_total_nontext_tokens: int | None = None,
            ) -> tuple[torch.Tensor, Flux2KVCache | None]:
                local_kwargs = {
                    "hidden_states": hidden_states,
                    "timestep": timestep / 1000,
                    "img_ids": image_ids,
                    "txt_ids": encoder_ids,
                    "guidance": None,
                    "joint_attention_kwargs": self.attention_kwargs,
                    "return_dict": True,
                    "encoder_hidden_states": encoder_states,
                }
                if mode is not None:
                    local_kwargs.update(
                        {
                            "kv_cache": cache,
                            "kv_cache_mode": mode,
                            "num_ref_tokens": mode_num_ref_tokens,
                            "kv_total_nontext_tokens": mode_total_nontext_tokens,
                        }
                    )
                    if mode == "extract":
                        # Keep reference modulation timestep fixed so cached ref K/V
                        # remains temporally consistent across denoising steps.
                        local_kwargs["ref_fixed_timestep"] = 0.0

                result = self.transformer(**local_kwargs)
                if mode == "extract":
                    output, local_kv_cache = result
                    if output.sample.shape[1] == 0:
                        logger.debug(
                            "flux2_klein_kv empty extract output at step=%s hidden=%s image_ids=%s txt=%s num_ref=%s",
                            i,
                            tuple(hidden_states.shape),
                            tuple(image_ids.shape),
                            tuple(encoder_states.shape),
                            mode_num_ref_tokens,
                        )
                    return output.sample, local_kv_cache
                if result.sample.shape[1] == 0:
                    logger.debug(
                        "flux2_klein_kv empty cached/plain output at step=%s mode=%s hidden=%s image_ids=%s txt=%s",
                        i,
                        mode,
                        tuple(hidden_states.shape),
                        tuple(image_ids.shape),
                        tuple(encoder_states.shape),
                    )
                return result.sample, cache

            if i == 0 and image_latents is not None:
                positive_noise_pred, kv_cache_pos = _run_transformer(
                    prompt_embeds,
                    text_ids,
                    cache=None,
                    mode="extract",
                    hidden_states=latent_model_input,
                    image_ids=latent_image_ids,
                    mode_num_ref_tokens=num_ref_tokens,
                    mode_total_nontext_tokens=latent_model_input.shape[1],
                )
                if self.do_classifier_free_guidance:
                    negative_noise_pred, kv_cache_neg = _run_transformer(
                        negative_prompt_embeds,
                        negative_text_ids,
                        cache=None,
                        mode="extract",
                        hidden_states=latent_model_input,
                        image_ids=latent_image_ids,
                        mode_num_ref_tokens=num_ref_tokens,
                        mode_total_nontext_tokens=latent_model_input.shape[1],
                    )
                    noise_pred = self.combine_cfg_noise(
                        positive_noise_pred,
                        negative_noise_pred,
                        guidance_scale,
                        cfg_normalize=False,
                    )
                    if (
                        positive_noise_pred.shape[1] == 0
                        or negative_noise_pred.shape[1] == 0
                        or noise_pred.shape[1] == 0
                    ):
                        logger.debug(
                            "flux2_klein_kv first-step cfg shapes pos=%s neg=%s "
                            "out=%s latents=%s latent_model_input=%s num_ref=%s",
                            tuple(positive_noise_pred.shape),
                            tuple(negative_noise_pred.shape),
                            tuple(noise_pred.shape),
                            tuple(latents.shape),
                            tuple(latent_model_input.shape),
                            num_ref_tokens,
                        )
                else:
                    noise_pred = positive_noise_pred
            else:
                kv_cache_mode_pos = "cached" if kv_cache_pos is not None else None
                positive_noise_pred, _ = _run_transformer(
                    prompt_embeds,
                    text_ids,
                    cache=kv_cache_pos,
                    mode=kv_cache_mode_pos,
                    hidden_states=latent_model_input,
                    image_ids=latent_image_ids,
                    mode_total_nontext_tokens=latent_model_input.shape[1],
                )
                if self.do_classifier_free_guidance:
                    kv_cache_mode_neg = "cached" if kv_cache_neg is not None else None
                    negative_noise_pred, _ = _run_transformer(
                        negative_prompt_embeds,
                        negative_text_ids,
                        cache=kv_cache_neg,
                        mode=kv_cache_mode_neg,
                        hidden_states=latent_model_input,
                        image_ids=latent_image_ids,
                        mode_total_nontext_tokens=latent_model_input.shape[1],
                    )
                    noise_pred = self.combine_cfg_noise(
                        positive_noise_pred,
                        negative_noise_pred,
                        guidance_scale,
                        cfg_normalize=False,
                    )
                    if (
                        positive_noise_pred.shape[1] == 0
                        or negative_noise_pred.shape[1] == 0
                        or noise_pred.shape[1] == 0
                    ):
                        logger.debug(
                            "flux2_klein_kv later-step cfg shapes step=%s mode_pos=%s "
                            "mode_neg=%s pos=%s neg=%s out=%s latents=%s",
                            i,
                            kv_cache_mode_pos,
                            kv_cache_mode_neg,
                            tuple(positive_noise_pred.shape),
                            tuple(negative_noise_pred.shape),
                            tuple(noise_pred.shape),
                            tuple(latents.shape),
                        )
                else:
                    noise_pred = positive_noise_pred

            if noise_pred.shape[1] == 0:
                logger.debug(
                    "flux2_klein_kv zero noise_pred before scheduler step=%s "
                    "latents=%s image_latents=%s kv_cache_pos=%s kv_cache_neg=%s",
                    i,
                    tuple(latents.shape),
                    None if image_latents is None else tuple(image_latents.shape),
                    kv_cache_pos is not None,
                    kv_cache_neg is not None,
                )

            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, self.do_classifier_free_guidance)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        self._current_timestep = None

        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if latents.dtype != self.vae.dtype:
            latents = latents.to(self.vae.dtype)
        image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)
