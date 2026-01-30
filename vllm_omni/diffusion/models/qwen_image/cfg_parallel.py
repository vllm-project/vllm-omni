# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CFG Parallel Mixin for Qwen Image series
Shared by
- QwenImagePipeline
- QwenImageEditPipeline
- QwenImageEditPlusPipeline
- QwenImageLayeredPipeline
"""

from typing import Any

import torch

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin


class QwenImageCFGParallelMixin(CFGParallelMixin):
    """
    Base Mixin class for Qwen Image pipelines providing shared CFG methods.
    """

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        latents: torch.Tensor,
        img_shapes: torch.Tensor,
        txt_seq_lens: torch.Tensor,
        negative_txt_seq_lens: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        guidance: torch.Tensor,
        true_cfg_scale: float,
        image_latents: torch.Tensor | None = None,
        cfg_normalize: bool = True,
        additional_transformer_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Diffusion loop with optional classifier-free guidance.

        Args:
            prompt_embeds: Positive prompt embeddings
            prompt_embeds_mask: Mask for positive prompt
            negative_prompt_embeds: Negative prompt embeddings
            negative_prompt_embeds_mask: Mask for negative prompt
            latents: Noise latents to denoise
            img_shapes: Image shape information
            txt_seq_lens: Text sequence lengths for positive prompts
            negative_txt_seq_lens: Text sequence lengths for negative prompts
            timesteps: Diffusion timesteps
            do_true_cfg: Whether to apply CFG
            guidance: Guidance scale tensor
            true_cfg_scale: CFG scale factor
            image_latents: Conditional image latents for editing (default: None)
            cfg_normalize: Whether to normalize CFG output (default: True)
            additional_transformer_kwargs: Extra kwargs to pass to transformer (default: None)

        Returns:
            Denoised latents
        """
        self.scheduler.set_begin_index(0)
        self.transformer.do_true_cfg = do_true_cfg
        additional_transformer_kwargs = additional_transformer_kwargs or {}

        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = t

            # Broadcast timestep to match batch size
            timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            # Concatenate image latents with noise latents if available (for editing pipelines)
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)

            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "encoder_hidden_states": prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                **additional_transformer_kwargs,
            }
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "img_shapes": img_shapes,
                    "txt_seq_lens": negative_txt_seq_lens,
                    **additional_transformer_kwargs,
                }
            else:
                negative_kwargs = None

            # For editing pipelines, we need to slice the output to remove condition latents
            output_slice = latents.size(1) if image_latents is not None else None

            # Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
                output_slice,
            )

            # Compute the previous noisy sample x_t -> x_t-1 with automatic CFG sync
            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

        return latents
