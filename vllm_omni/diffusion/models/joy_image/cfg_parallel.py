# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

import torch

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

logger = logging.getLogger(__name__)


class JoyImageEditCFGParallelMixin(CFGParallelMixin, ProgressBarMixin):
    def cfg_normalize_function(self, noise_pred: torch.Tensor, comb_pred: torch.Tensor) -> torch.Tensor:
        cond_norm = torch.norm(noise_pred, dim=2, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=2, keepdim=True)
        return comb_pred * (cond_norm / noise_norm.clamp_min(1e-6))

    def diffuse(
        self,
        latents: torch.Tensor,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
        cfg_normalize: bool = True,
    ) -> torch.Tensor:
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=len(timesteps)) as pbar:
            for timestep in timesteps:
                if self.interrupt:
                    continue
                self._current_timestep = timestep

                latents[:, : image_latents.shape[1]] = image_latents
                latent_model_input = latents
                timestep_expand = timestep.expand(latents.shape[0]).to(device=latents.device)

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep_expand,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_mask": prompt_embeds_mask,
                    "return_dict": False,
                }
                negative_kwargs = None
                if do_true_cfg:
                    negative_kwargs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep_expand,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                        "return_dict": False,
                    }

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=true_cfg_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=cfg_normalize,
                    output_slice=None,
                )
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, timestep, latents, do_true_cfg)
                latents[:, : image_latents.shape[1]] = image_latents
                pbar.update()
        return latents

    def check_cfg_parallel_validity(self, true_cfg_scale: float) -> bool:
        if get_classifier_free_guidance_world_size() == 1:
            return True
        if true_cfg_scale <= 1:
            logger.warning("CFG parallel is enabled but Joy true_cfg_scale <= 1, so only the positive branch is used.")
            return False
        return True
