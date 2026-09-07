# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One AR block of LingBot World's causal DMD sampler.

The pipeline drives this in two ways -- request mode loops ``generate_block``
over every block, stepwise execution calls ``probe_step`` / ``apply_transition``
one denoise step at a time and ``commit_block_kv`` from ``post_decode`` -- so the
math lives here once and cannot drift between the two modes.

A block is four probes that must not touch KV, then one clean-x0 commit that
does. Whether a call sees paged (session-bound) or request-local KV is decided
by the ``ar`` argument, so this class never reaches back into the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from diffusers.utils.torch_utils import randn_tensor

from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.models.lingbot_world.transformer import (
    LingBotAttentionCache,
    LingBotTransformerCache,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.lingbot_world.transformer import CausalLingBotWorldTransformer3DModel
    from vllm_omni.diffusion.models.progress_bar import TqdmProgressBar
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState


@dataclass(frozen=True)
class ARBlockContext:
    """The bound AR-Diffusion session one block runs against.

    Built by the pipeline from the state the runner bound for this invocation;
    ``None`` in place of a context means request-local contiguous KV.
    """

    state: ARDiffusionKVState
    cross_attention: list[LingBotAttentionCache]
    branch: str


class LingBotDMDBlockRunner:
    """Runs the DMD probes and the KV commit for one latent-frame block."""

    def __init__(
        self,
        transformer: CausalLingBotWorldTransformer3DModel,
        *,
        device: torch.device,
        enforce_eager: bool,
    ) -> None:
        self.transformer = transformer
        self.device = device
        self.enforce_eager = enforce_eager

    # ── cache selection ──────────────────────────────────────────────────

    def _paged_cache(
        self,
        *,
        condition: torch.Tensor,
        ar: ARBlockContext,
        commit_current: bool,
    ) -> LingBotTransformerCache:
        patch_frames, patch_height, patch_width = self.transformer.config.patch_size
        frames = condition.shape[2] // patch_frames
        height = condition.shape[3] // patch_height
        width = condition.shape[4] // patch_width
        return LingBotTransformerCache(
            self_attention=ar.state.get_kv_caches(
                ar.branch,
                seq_len=frames * height * width,
                commit_current=commit_current,
            ),
            cross_attention=ar.cross_attention,
        )

    def _block_cache(
        self,
        *,
        condition: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: ARBlockContext | None,
        commit_current: bool,
    ) -> LingBotTransformerCache:
        """Pick the cache one transformer call sees: paged when a session is
        bound, else the caller's request-local cache."""
        if ar is not None:
            return self._paged_cache(condition=condition, ar=ar, commit_current=commit_current)
        return cast(LingBotTransformerCache, cache)

    # ── one block ────────────────────────────────────────────────────────

    def probe_step(
        self,
        *,
        current_latents: torch.Tensor,
        condition: torch.Tensor,
        camera: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: ARBlockContext | None,
        start_frame: int,
        timestep_value: float,
        step_index: int,
    ) -> torch.Tensor:
        """Predict flow for one denoise step.

        A probe never writes KV: only the clean x0 of a finished block may
        enter the cache, which is ``commit_block_kv``'s job.
        """
        if not self.enforce_eager:
            torch.compiler.cudagraph_mark_step_begin()
        set_forward_context_denoise_step_idx(step_index)
        timestep = torch.full((1,), float(timestep_value), device=self.device, dtype=torch.float32)
        # Checkpoint channel contract:
        # [noise/x_t(16), temporal_mask(4), image_latent(16)] -> 36.
        model_input = torch.cat((current_latents.to(dtype=condition.dtype), condition), dim=1)
        flow_prediction = self.transformer(
            hidden_states=model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            camera_hidden_states=camera,
            cache=self._block_cache(condition=condition, cache=cache, ar=ar, commit_current=False),
            start_frame=start_frame,
            update_cache=False,
        )
        if flow_prediction.shape != current_latents.shape:
            raise RuntimeError(
                "transformer flow prediction shape must match the 16-channel noise latent, "
                f"got {tuple(flow_prediction.shape)} and {tuple(current_latents.shape)}."
            )
        return flow_prediction

    def apply_transition(
        self,
        current_latents: torch.Tensor,
        flow_prediction: torch.Tensor,
        sigma: float,
        *,
        next_sigma: float | None,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Invert the flow to x0, then re-noise at ``next_sigma``.

        ``next_sigma=None`` marks the final step, so where the caller sits in
        its own loop stays out of this function.
        """
        # The checkpoint's flow parameterization is inverted by
        # x0 = x_t - sigma * flow. Intermediate steps re-noise that x0
        # estimate at the next sigma; the final step keeps x0 as this
        # block's generated latent.
        x0 = current_latents - sigma * flow_prediction.float()
        if next_sigma is None:
            return x0
        noise = randn_tensor(current_latents.shape, generator=generator, device=self.device, dtype=torch.float32)
        return (1.0 - next_sigma) * x0 + next_sigma * noise

    def commit_block_kv(
        self,
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        camera: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: ARBlockContext | None,
        start_frame: int,
    ) -> None:
        """Write the finished block's clean x0 into KV and commit its pages.

        The fifth transformer call of a block, deliberately not a denoise step:
        on the stepwise path it belongs to ``post_decode()``.
        """
        # Commit K/V only for the final clean block, never for noisy probes.
        cache_input = torch.cat((latents.to(dtype=condition.dtype), condition), dim=1)
        if not self.enforce_eager:
            torch.compiler.cudagraph_mark_step_begin()
        self.transformer(
            hidden_states=cache_input,
            timestep=torch.zeros(1, device=self.device, dtype=torch.float32),
            encoder_hidden_states=prompt_embeds,
            camera_hidden_states=camera,
            cache=self._block_cache(condition=condition, cache=cache, ar=ar, commit_current=True),
            start_frame=start_frame,
            update_cache=True,
        )
        if ar is not None:
            ar.state.commit_paged_context(ar.branch)

    def generate_block(
        self,
        *,
        condition: torch.Tensor,
        camera: torch.Tensor,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: ARBlockContext | None,
        start_frame: int,
        schedule: tuple[tuple[float, float], ...],
        generator: torch.Generator,
        progress_bar: TqdmProgressBar[Any],
    ) -> torch.Tensor:
        """Request mode: all probes of one block, then its commit."""
        block_shape = (1, self.transformer.config.out_channels, *condition.shape[2:5])
        current_latents = randn_tensor(block_shape, generator=generator, device=self.device, dtype=torch.float32)
        for step_index, (timestep_value, sigma) in enumerate(schedule):
            flow_prediction = self.probe_step(
                current_latents=current_latents,
                condition=condition,
                camera=camera,
                prompt_embeds=prompt_embeds,
                cache=cache,
                ar=ar,
                start_frame=start_frame,
                timestep_value=timestep_value,
                step_index=step_index,
            )
            next_sigma = schedule[step_index + 1][1] if step_index + 1 < len(schedule) else None
            current_latents = self.apply_transition(
                current_latents, flow_prediction, sigma, next_sigma=next_sigma, generator=generator
            )
            progress_bar.update()
        self.commit_block_kv(
            latents=current_latents,
            condition=condition,
            camera=camera,
            prompt_embeds=prompt_embeds,
            cache=cache,
            ar=ar,
            start_frame=start_frame,
        )
        return current_latents
