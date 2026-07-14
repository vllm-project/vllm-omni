# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared runtime surface for LTX pipeline variants."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext

import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin

from . import ltx2_latents as latent_ops


class LTXPipelineBase(nn.Module, CFGParallelMixin, ProgressBarMixin, SupportsComponentDiscovery):
    """Common state and primitives shared by LTX2 and LTX2.3 pipelines."""

    _pack_latents = staticmethod(latent_ops.pack_latents)
    _unpack_latents = staticmethod(latent_ops.unpack_latents)
    _normalize_latents = staticmethod(latent_ops.normalize_latents)
    _normalize_audio_latents = staticmethod(latent_ops.normalize_audio_latents)
    _denormalize_latents = staticmethod(latent_ops.denormalize_latents)
    _denormalize_audio_latents = staticmethod(latent_ops.denormalize_audio_latents)
    _create_noised_state = staticmethod(latent_ops.create_noised_state)
    _pack_audio_latents = staticmethod(latent_ops.pack_audio_latents)
    _unpack_audio_latents = staticmethod(latent_ops.unpack_audio_latents)
    _unpad_audio_latents = staticmethod(latent_ops.unpad_audio_latents)
    _get_sp_padded_audio_latent_length = staticmethod(latent_ops.get_sp_padded_audio_latent_length)
    _resolve_video_latent_shape = staticmethod(latent_ops.resolve_video_latent_shape)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def _transformer_cache_context(self, context_name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        if callable(cache_context):
            return cache_context(context_name)
        return nullcontext()

    def predict_noise(self, **kwargs):
        with self._transformer_cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = self.transformer(**kwargs)
        return noise_pred_video.float(), noise_pred_audio.float()

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (do_true_cfg and get_classifier_free_guidance_world_size() > 1):
            return latents

        # CUDA async execution otherwise permits numerical drift to accumulate
        # across CFG-parallel denoise steps.
        latents = tuple(tensor.contiguous() for tensor in latents)
        device = next((tensor.device for tensor in latents if tensor.is_cuda), None)
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
        return latents

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
