# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 x0 model adapters for sampler-owned, coupled video/audio integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.attention.backends.abstract import DenoiseProgress
from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.sampler import ModelEvaluation, SamplingState

from .latent_mask import MiniMaxH3LatentEdit
from .scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0

if TYPE_CHECKING:
    from .denoise_loop import MiniMaxH3DenoiseBranch


class MiniMaxH3Model:
    """Translate sampler evaluations into H3 inputs and raw velocity into x0."""

    def __init__(
        self,
        transformer,
        branch: MiniMaxH3DenoiseBranch,
        rows: tuple[torch.Tensor, torch.Tensor],
        edits: tuple[MiniMaxH3LatentEdit | None, MiniMaxH3LatentEdit | None] = (None, None),
        cond_timesteps: tuple[float, float] = (0.999, 1.0),
        model_rows: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        self.transformer, self.branch, self.rows = transformer, branch, rows
        self.edits, self.cond_timesteps = edits, cond_timesteps
        # Fixed condition rows stay in this buffer; each prediction replaces
        # the target rows. Keep it for the entire request, including retries.
        self.model_rows = model_rows if model_rows is not None else tuple(row.clone() for row in rows)

    def rows_for(self, latents):
        """Fill reusable model inputs; valid until this request's next prediction."""
        rows = self.model_rows
        for row, mask, latent in zip(rows, self.masks, latents, strict=True):
            row[mask] = latent
        if self.branch.locked_audio_rows is not None:
            rows[1][self.masks[1]] = self.branch.locked_audio_rows
        return tuple(rows)

    @property
    def masks(self):
        return self.branch.update_mask_dev, self.branch.audio_update_mask_dev

    def prepare(self, evaluation: ModelEvaluation):
        rows = self.rows_for(evaluation.latents)
        times = tuple(1.0 - sigma for sigma in evaluation.sigmas)
        cond = tuple(max(t, anchor) for t, anchor in zip(times, self.cond_timesteps, strict=True))
        prepared = []
        for row, mask, edit, t, c, sigma in zip(
            rows, self.masks, self.edits, times, cond, evaluation.sigmas, strict=True
        ):
            target_timesteps = None
            if edit is not None:
                row[mask], target_timesteps = edit.prepare(row[mask], t, c, sigma=sigma)
            prepared.append((row, target_timesteps))
        return rows, prepared, times, cond

    def to_denoised(self, prepared, velocities):
        rows, model_rows, times, _ = prepared
        return tuple(
            minimax_h3_rf_v_to_x0(row[mask], v.float()[mask], row.new_tensor(t))
            if edit is None
            else edit.x0(model_row[0][mask], v.float()[mask], t)
            for row, model_row, mask, edit, t, v in zip(
                rows, model_rows, self.masks, self.edits, times, velocities, strict=True
            )
        )

    def validate_write_back(self, state: SamplingState):
        """Check both streams before any request's full rows are updated."""
        if self.branch.locked_audio_rows is not None:
            state.latents = (state.latents[0], self.branch.locked_audio_rows)
        for row, mask, latent in zip(self.rows, self.masks, state.latents, strict=True):
            shape = (int(mask.count_nonzero().item()), *row.shape[1:])
            if latent.shape != shape or latent.dtype != row.dtype or latent.device != row.device:
                raise ValueError("Sampled latents must match the target rows' shape, dtype, and device")
            if not bool(torch.isfinite(latent).all().item()):
                raise ValueError("Sampled latents must be finite before writing rows")

    def write_back(self, state: SamplingState):
        """Write checked latents into the existing full video/audio rows."""
        for row, mask, latent in zip(self.rows, self.masks, state.latents, strict=True):
            row[mask] = latent
        return self.rows


class MiniMaxH3ModelBatch:
    """Callable x0 model; pack compatible evaluations without advancing samplers."""

    def __init__(self, models: dict[SamplingState, MiniMaxH3Model], packed_supported: Callable = lambda _: False):
        self.models = models
        self.packed_supported = packed_supported

    def __call__(self, evaluations: Sequence[ModelEvaluation]):
        from .batched_packing import minimax_h3_batched_forward_kwargs
        from .denoise_loop import minimax_h3_publish_denoise_progress

        if not evaluations:
            return []
        check_request_cancellation()
        models = [self.models[e.state] for e in evaluations]
        if len(models) > 1 and (
            len({id(m.transformer) for m in models}) > 1 or not self.packed_supported(models[0].transformer)
        ):
            return [self([e])[0] for e in evaluations]
        prepared = [m.prepare(e) for m, e in zip(models, evaluations, strict=True)]
        progress = {(e.state.step_index, e.sigmas[0], len(e.state.sigmas[0]) - 1) for e in evaluations}
        minimax_h3_publish_denoise_progress(*(progress.pop() if len(progress) == 1 else (None, None, None)))
        context = get_forward_context() if is_forward_context_available() else None
        try:
            if context is not None:
                context.denoise_progress = {
                    role: tuple(
                        DenoiseProgress(length, e.state.step_index, e.sigmas[0], len(e.state.sigmas[0]) - 1)
                        for m, e in zip(models, evaluations, strict=True)
                        for length in (
                            (m.branch.text_len,)
                            if role == "minimax_h3.token_refiner"
                            else (m.branch.used_len, m.branch.seq_len - m.branch.used_len)
                        )
                        if length
                    )
                    for role in ("minimax_h3.token_refiner", "self")
                }
            if len(models) == 1:
                _, rows, times, cond = prepared[0]
                kwargs = models[0].branch.forward_kwargs(
                    video_rows=rows[0][0],
                    audio_rows=rows[1][0],
                    t_video=times[0],
                    t_audio=times[1],
                    imgvid_cond_timestep=cond[0],
                    audio_ref_cond_timestep=cond[1],
                    video_target_timesteps=rows[0][1],
                    audio_target_timesteps=rows[1][1],
                )
            else:
                kwargs = minimax_h3_batched_forward_kwargs(
                    branches=[m.branch for m in models],
                    video_rows=[p[1][0][0] for p in prepared],
                    audio_rows=[p[1][1][0] for p in prepared],
                    t_video=[p[2][0] for p in prepared],
                    t_audio=[p[2][1] for p in prepared],
                    imgvid_cond_timesteps=[p[3][0] for p in prepared],
                    audio_ref_cond_timesteps=[p[3][1] for p in prepared],
                    video_target_timesteps=[p[1][0][1] for p in prepared],
                    audio_target_timesteps=[p[1][1][1] for p in prepared],
                )
            video, audio = models[0].transformer(**kwargs)
            check_request_cancellation()
            video_parts = video.split([m.rows[0].shape[0] for m in models])
            audio_parts = audio.split([m.rows[1].shape[0] for m in models])
            return [
                m.to_denoised(p, (v, a)) for m, p, v, a in zip(models, prepared, video_parts, audio_parts, strict=True)
            ]
        finally:
            if context is not None:
                context.denoise_progress = {}
            minimax_h3_publish_denoise_progress(None, None, None)
