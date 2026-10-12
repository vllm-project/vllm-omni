# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CFG Parallel Mixin for Qwen-Image 2.1.

Differences from the 2.0 mixin (``qwen_image/cfg_parallel.py``):
- The transformer owns a KV cache for the text/condition-image prefix, so the
  denoise loop passes a fresh per-generation ``kv_cache`` and names the CFG
  branch (``cache_branch="cond"/"uncond"``) on every forward.
- The joint sequence is ``[condition images, ..., target image]`` — the target
  noise prediction is the *tail* of the output, not the head.
"""

import logging
from typing import Any

import torch

from vllm_omni.diffusion.distributed.cfg_parallel import (
    CFGParallelMixin,
    _get_cfg_world_size_or_one,
    _unwrap,
    _wrap,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

logger = logging.getLogger(__name__)


class QwenImage21CFGParallelMixin(CFGParallelMixin, ProgressBarMixin):
    """Base Mixin class for the Qwen-Image 2.1 pipeline providing shared CFG methods."""

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = False,
        output_slice: int | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # The base implementation slices `[:, :output_slice]` per branch, which
        # matches the 2.0 condition-last layout. 2.1 concatenates condition
        # latents *before* the target, so the target prediction is the tail of
        # the joint output. Each branch must be tail-sliced *before* the CFG
        # combine: positive and negative prompts tokenize to different lengths,
        # so the full joint outputs do not align positionally.
        def tail_slice(pred: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            if output_slice is None:
                return _wrap(pred)
            return tuple(p[:, -output_slice:] for p in _wrap(pred))

        if do_true_cfg:
            if _get_cfg_world_size_or_one() > 1:
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()
                branch_kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
                local_pred = tail_slice(self.predict_noise(**branch_kwargs))
                gathered = [cfg_group.all_gather(p, separate_tensors=True) for p in local_pred]
                positive_noise_pred = tuple(g[0] for g in gathered)
                negative_noise_pred = tuple(g[1] for g in gathered)
            else:
                positive_noise_pred = tail_slice(self.predict_noise(**positive_kwargs))
                negative_noise_pred = tail_slice(self.predict_noise(**negative_kwargs))
            return self.combine_cfg_noise(
                positive_noise_pred,
                negative_noise_pred,
                true_cfg_scale,
                cfg_normalize,
                **({} if kwargs is None else {"kwargs": kwargs}),
            )

        pred = self.predict_noise(**positive_kwargs)
        if output_slice is not None:
            pred = _unwrap(tail_slice(pred))
        return pred

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        img_mask: torch.Tensor,
        negative_img_mask: torch.Tensor | None,
        latents: torch.Tensor,
        image_latents: torch.Tensor | None,
        img_shapes: list,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
        attention_kwargs: dict[str, Any] | None = None,
        additional_transformer_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Diffusion loop with optional classifier-free guidance.

        Args:
            prompt_embeds: Positive prompt embeddings
            prompt_embeds_mask: Mask for positive prompt
            negative_prompt_embeds: Negative prompt embeddings
            negative_prompt_embeds_mask: Mask for negative prompt
            img_mask: Joint-sequence image-slot mask for the positive branch
                (VLM image slots plus one slot per 2x2 group of target tokens)
            negative_img_mask: Same for the negative branch
            latents: Noise latents to denoise (target image only, packed)
            image_latents: Packed condition-image latents, prepended to the
                noise latents (default: None for pure text-to-image)
            img_shapes: Per-sample list of (frame, height, width) latent-token
                shapes, condition images first and the target last
            timesteps: Diffusion timesteps
            do_true_cfg: Whether to apply CFG
            true_cfg_scale: CFG scale factor
            attention_kwargs: Passed through to the transformer (attn_path hint)
            additional_transformer_kwargs: Extra kwargs for the transformer

        Returns:
            Denoised latents
        """
        # Fresh cache for every generation: the pipeline is a singleton shared
        # across requests, so cache state must never leak between them.
        cache_enabled = getattr(self.transformer, "causal_condition", False)
        kv_cache = [{} for _ in range(len(self.transformer.transformer_blocks))] if cache_enabled else None
        self.scheduler.set_begin_index(0)
        self.transformer.do_true_cfg = do_true_cfg
        additional_transformer_kwargs = additional_transformer_kwargs or {}

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = t

                # Broadcast timestep to match batch size
                timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

                # Condition image latents go in front of the noise latents.
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([image_latents, latents], dim=1)

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep / 1000,
                    "encoder_hidden_states_mask": prompt_embeds_mask,
                    "encoder_hidden_states": prompt_embeds,
                    "img_shapes": img_shapes,
                    "img_mask": img_mask,
                    "attention_kwargs": attention_kwargs,
                    "kv_cache": kv_cache,
                    "cache_branch": "cond",
                    **additional_transformer_kwargs,
                }
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep / 1000,
                        "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "img_shapes": img_shapes,
                        "img_mask": negative_img_mask,
                        "attention_kwargs": attention_kwargs,
                        "kv_cache": kv_cache,
                        "cache_branch": "uncond",
                        **additional_transformer_kwargs,
                    }
                else:
                    negative_kwargs = None

                # Prefill (first step) returns the full joint sequence and
                # caches the prefix; decode steps return target tokens only.
                # Slicing the target tail is correct for both.
                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg,
                    true_cfg_scale,
                    positive_kwargs,
                    negative_kwargs,
                    cfg_normalize=False,
                    output_slice=latents.size(1),
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

                pbar.update()

        return latents
