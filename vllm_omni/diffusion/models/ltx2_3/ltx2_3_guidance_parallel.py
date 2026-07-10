# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 guidance forward parallel helpers."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)

from .ltx2_3_guidance import _LTX23GuidanceParams


class LTX23GuidanceParallelMixin:
    @staticmethod
    def _combine_x0_space_cfg(
        sample: torch.Tensor,
        positive_noise_pred: torch.Tensor,
        negative_noise_pred: torch.Tensor,
        sigma: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        x0_cond = sample - positive_noise_pred * sigma
        x0_uncond = sample - negative_noise_pred * sigma
        x0_guided = x0_cond + (guidance_scale - 1) * (x0_cond - x0_uncond)
        return (sample - x0_guided) / sigma

    def _setup_cfg_parallel_runtime(self, guidance_params: _LTX23GuidanceParams) -> bool:
        cfg_world_size = get_classifier_free_guidance_world_size()
        if self.do_classifier_free_guidance and cfg_world_size not in (1, 2):
            raise ValueError(
                f"LTX23Pipeline supports CFG parallelism with cfg_parallel_size 1 or 2, but got {cfg_world_size}."
            )
        if cfg_world_size > 1:
            if (
                guidance_params.video_cfg_scale != guidance_params.audio_cfg_scale
                or guidance_params.do_stg
                or guidance_params.do_modality_guidance
                or guidance_params.do_rescale
            ):
                raise ValueError(
                    "LTX23Pipeline cfg-parallel currently supports CFG-only guidance with identical video/audio "
                    "CFG scales. Disable cfg-parallel for STG, modality guidance, rescale, or separate "
                    "video/audio CFG scales."
                )
        return self.do_classifier_free_guidance and cfg_world_size > 1

    def predict_noise(self, **kwargs):
        with self._transformer_cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = self.transformer(**kwargs)
        return noise_pred_video.float(), noise_pred_audio.float()

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale,
        cfg_normalize=False,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        video_sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ):
        if video_latents is None or audio_latents is None or video_sigma is None or audio_sigma is None:
            raise ValueError("LTX23Pipeline applies CFG in x0-space and requires video/audio latents and sigmas.")

        video_pos, audio_pos = positive_noise_pred
        video_neg, audio_neg = negative_noise_pred
        video_combined = self._combine_x0_space_cfg(
            video_latents,
            video_pos,
            video_neg,
            video_sigma,
            true_cfg_scale,
        )
        audio_combined = self._combine_x0_space_cfg(
            audio_latents,
            audio_pos,
            audio_neg,
            audio_sigma,
            true_cfg_scale,
        )
        if cfg_normalize:
            video_combined = self.cfg_normalize_function(video_pos, video_combined)
            audio_combined = self.cfg_normalize_function(audio_pos, audio_combined)
        return video_combined, audio_combined

    def predict_noise_with_parallel_cfg(
        self,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any],
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        *,
        video_latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        video_sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def maybe_slice(pred: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if output_slice is None:
                return pred
            return pred[0][:, :output_slice], pred[1][:, :output_slice]

        cfg_world_size = get_classifier_free_guidance_world_size()
        if cfg_world_size != 2:
            raise ValueError(f"LTX23Pipeline parallel CFG requires cfg_parallel_size 2, but got {cfg_world_size}.")

        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()
        branch_kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
        local_video_pred, local_audio_pred = maybe_slice(self.predict_noise(**branch_kwargs))

        gathered_video = cfg_group.all_gather(local_video_pred, separate_tensors=True)
        gathered_audio = cfg_group.all_gather(local_audio_pred, separate_tensors=True)
        positive_noise_pred = (gathered_video[0], gathered_audio[0])
        negative_noise_pred = (gathered_video[1], gathered_audio[1])

        return self.combine_cfg_noise(
            positive_noise_pred,
            negative_noise_pred,
            true_cfg_scale,
            cfg_normalize,
            video_latents=video_latents,
            audio_latents=audio_latents,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
        )

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (do_true_cfg and get_classifier_free_guidance_world_size() > 1):
            return latents

        latents = tuple(tensor.contiguous() for tensor in latents)
        device = next((tensor.device for tensor in latents if tensor.is_cuda), None)
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
        return latents
