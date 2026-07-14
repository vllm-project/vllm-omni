# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import torch

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


class DiffusionV2AtomDefaultsMixin:
    """Default hooks shared by step/disaggregated diffusion pipelines."""

    supports_step_execution: ClassVar[bool] = True

    def init_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        state.extra.pop("decoded_output", None)
        self._guidance_scale = 1.0
        self._attention_kwargs = {}
        self._current_timestep = None
        self._interrupt = False
        return state

    # TODO: build varlen attention metadata here when step batching
    # starts using packed varlen attention.
    def build_step_attention_metadata(
        self,
        input_batch: InputBatch,
    ) -> object | None:
        return None

    # TODO: keep this hook for future varlen/cudagraph batching and
    # disaggregated DiT batch layouts.
    def build_step_batch(
        self,
        states: list[DiffusionRequestState],
        *,
        cached_batch: InputBatch | None = None,
    ) -> InputBatch:
        from vllm_omni.diffusion.worker.input_batch import InputBatch

        return InputBatch.make_batch(states, cached_batch=cached_batch)

    def diffuse(self, state: DiffusionRequestState) -> DiffusionRequestState:
        while not state.denoise_completed:
            if self.interrupt:
                break
            input_batch = self.build_step_batch([state])
            noise_pred = self.denoise_step(input_batch)
            if noise_pred is None:
                if self.interrupt:
                    break
                raise RuntimeError("denoise_step returned None without pipeline interrupt.")
            self.step_scheduler(state, noise_pred)
        return state

    def _new_request_scheduler(self) -> object:
        scheduler = deepcopy(getattr(self, "scheduler"))
        scheduler.set_begin_index(0)
        return scheduler


class DiffusionV2CFGStepMixin(DiffusionV2AtomDefaultsMixin):
    """Default one-step CFG denoise/scheduler hooks.

    Pipelines using this mixin keep model-specific tensor schema in
    ``_build_denoise_kwargs``.
    """

    def denoise_step(
        self,
        input_batch: InputBatch,
    ) -> torch.Tensor | None:
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg
        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=input_batch.latents,
            timestep=t,
            guidance=input_batch.guidance,
            prompt_embeds=input_batch.prompt_embeds,
            prompt_embeds_mask=input_batch.prompt_embeds_mask,
            img_shapes=input_batch.img_shapes,
            txt_seq_lens=input_batch.txt_seq_lens,
            do_true_cfg=input_batch.do_true_cfg,
            negative_prompt_embeds=input_batch.negative_prompt_embeds,
            negative_prompt_embeds_mask=input_batch.negative_prompt_embeds_mask,
            negative_txt_seq_lens=input_batch.negative_txt_seq_lens,
            image_latents=input_batch.image_latents,
        )
        return self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
            output_slice,
        )

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
    ) -> DiffusionRequestState:
        if self.interrupt:
            return state
        t = state.current_timestep
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1
        return state
