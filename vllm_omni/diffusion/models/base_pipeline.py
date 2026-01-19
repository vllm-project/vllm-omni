# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base pipeline class for Qwen Image models with shared CFG functionality.
"""

import torch
from torch import nn

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


class BasePipeline(nn.Module):
    """
    Base class for Diffusion pipelines providing shared CFG methods.

    All pipelines should inherit from this class to reuse
    classifier-free guidance logic.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg,
        true_cfg_scale,
        positive_kwargs,
        negative_kwargs,
        cfg_group=None,
        cfg_rank=None,
        cfg_normalize=True,
        output_slice=None,
    ):
        """
        Predict noise with optional classifier-free guidance.

        Args:
            do_true_cfg: Whether to apply CFG
            true_cfg_scale: CFG scale factor
            positive_kwargs: Kwargs for positive/conditional prediction
            negative_kwargs: Kwargs for negative/unconditional prediction
            cfg_group: Communication group for CFG parallelism
            cfg_rank: Rank in CFG parallel group
            cfg_normalize: Whether to normalize CFG output (default: True)
            output_slice: If set, slice output to [:, :output_slice] for image editing

        Returns:
            Predicted noise tensor
        """
        if do_true_cfg:
            if cfg_group is not None:
                # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
                assert cfg_rank is not None, "cfg_rank must be provided if cfg_group is provided"
                if cfg_rank == 0:
                    local_pred = self.predict_noise(**positive_kwargs)
                else:
                    local_pred = self.predict_noise(**negative_kwargs)

                # Slice output for image editing pipelines (remove condition latents)
                if output_slice is not None:
                    local_pred = local_pred[:, :output_slice]

                gathered = cfg_group.all_gather(local_pred, separate_tensors=True)
                if cfg_rank == 0:
                    noise_pred = gathered[0]
                    neg_noise_pred = gathered[1]
                    noise_pred = self.combine_cfg_noise(noise_pred, neg_noise_pred, true_cfg_scale, cfg_normalize)
                return noise_pred
            else:
                # Sequential CFG: compute both positive and negative
                positive_noise_pred = self.predict_noise(**positive_kwargs)
                negative_noise_pred = self.predict_noise(**negative_kwargs)

                # Slice output for image editing pipelines
                if output_slice is not None:
                    positive_noise_pred = positive_noise_pred[:, :output_slice]
                    negative_noise_pred = negative_noise_pred[:, :output_slice]

                noise_pred = self.combine_cfg_noise(
                    positive_noise_pred, negative_noise_pred, true_cfg_scale, cfg_normalize
                )
                return noise_pred
        else:
            # No CFG: only compute positive/conditional prediction
            pred = self.predict_noise(**positive_kwargs)
            if output_slice is not None:
                pred = pred[:, :output_slice]
            return pred

    def cfg_normalize_function(self, noise_pred, comb_pred):
        """
        Normalize the combined noise prediction.

        Args:
            noise_pred: positive noise prediction
            comb_pred: combined noise prediction after CFG

        Returns:
            Normalized noise prediction tensor
        """
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        noise_pred = comb_pred * (cond_norm / noise_norm)
        return noise_pred

    def combine_cfg_noise(self, noise_pred, neg_noise_pred, true_cfg_scale, cfg_normalize=True):
        """
        Combine conditional and unconditional noise predictions with CFG.

        Args:
            noise_pred: Conditional noise prediction
            neg_noise_pred: Unconditional noise prediction
            true_cfg_scale: CFG scale factor
            cfg_normalize: Whether to normalize the combined prediction (default: True)

        Returns:
            Combined noise prediction tensor
        """
        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

        if cfg_normalize:
            noise_pred = self.cfg_normalize_function(noise_pred, comb_pred)
        else:
            noise_pred = comb_pred

        return noise_pred

    def predict_noise(self, *args, **kwargs):
        """
        Forward pass through transformer to predict noise.

        Subclasses should override this if they need custom behavior,
        but the default implementation calls self.transformer.
        """
        return self.transformer(*args, **kwargs)[0]

    def diffuse(
        self,
        *args,
        **kwargs,
    ):
        """
        Diffusion loop with optional classifier-free guidance.
        """
        raise NotImplementedError("Subclasses must implement diffuse")

    @property
    def interrupt(self):
        """Property to check if diffusion should be interrupted."""
        return getattr(self, "_interrupt", False)


class BaseQwenImagePipeline(BasePipeline):
    """
    Base class for Qwen Image pipelines providing shared CFG methods.
    """

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
        image_latents=None,
        cfg_normalize=True,
        additional_transformer_kwargs=None,
    ):
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

            # Enable CFG-parallel: rank0 computes positive, rank1 computes negative.
            cfg_parallel_ready = do_true_cfg and get_classifier_free_guidance_world_size() > 1
            cfg_group = get_cfg_group() if cfg_parallel_ready else None
            cfg_rank = get_classifier_free_guidance_rank() if cfg_parallel_ready else None

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

            # For editing pipelines, we need to slice the output to remove condition latents
            output_slice = latents.size(1) if image_latents is not None else None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_group,
                cfg_rank,
                cfg_normalize,
                output_slice,
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if cfg_group is not None:
                cfg_group.broadcast(latents, src=0)

        return latents
