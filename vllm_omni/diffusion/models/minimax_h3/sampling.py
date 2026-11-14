# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 x0 model adapters for sampler-owned, coupled video/audio integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cancellation import check_request_cancellation
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.sampler import ModelEvaluation, SamplingState

from .latent_mask import MiniMaxH3LatentEdit
from .scheduling_minimax_h3_euler_ancestral import minimax_h3_rf_v_to_x0

if TYPE_CHECKING:
    from .denoise_loop import MiniMaxH3DenoiseBranch

logger = init_logger(__name__)


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
        attention_schedule=None,
        attention_sigma_schedule=(),
    ):
        self.transformer, self.branch, self.rows = transformer, branch, rows
        self.attention_schedule = attention_schedule
        self.attention_sigma_schedule = attention_sigma_schedule
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

    def _unpacked_reason(
        self, models: list[MiniMaxH3Model], *, pdd: bool, scheduled: bool, uniform_progress: bool
    ) -> tuple[Callable[..., None], str] | None:
        """Why these requests cannot share one packed forward, or None if they can."""
        if pdd:
            # PDD heads take one request-local step index per forward.
            return logger.debug, "FastH3 Omni PDD executes requests separately with request-local step indices."
        if len({id(m.transformer) for m in models}) > 1:
            return logger.warning_once, "MiniMax H3 step batch contains requests for different task-specific DiTs."
        if any("reference_kv_state" in m.branch.static_kwargs for m in models):
            # Packed kwargs carry no per-request reference-KV state.
            return logger.debug, "MiniMax H3 reference-KV requests run separately."
        if scheduled and (
            not uniform_progress
            or len({m.attention_schedule for m in models}) > 1
            or len({m.attention_sigma_schedule for m in models}) > 1
        ):
            # A packed forward selects one attention profile from the published progress.
            # Sigma windows split like step ranges. The resolved windows are in the step batch key, and
            # the runner's batch_attention_sigma_schedule() rejects differing ones before denoise_step,
            # so differing windows cannot reach one batch; the check mirrors the step-range one.
            return (
                logger.info_once,
                "MiniMax H3 runs a scheduled step batch whose requests differ in progress or schedule "
                "one request per forward, so each request selects attention at its own step.",
            )
        # Checked under the published progress: a scheduled step's selected
        # profile, not the baseline, must isolate requests.
        if self.packed_supported(models[0].transformer):
            return None
        from .minimax_h3_blocks import MiniMaxH3Attention

        modules = getattr(models[0].transformer, "modules", None)
        if callable(modules) and any(
            getattr(getattr(module, "attention", None), "use_ring", False)
            for module in modules()
            if isinstance(module, MiniMaxH3Attention)
        ):
            return (
                logger.warning_once,
                "MiniMax H3 step batching is disabled when ring attention is active: the ring kernels ignore "
                "packed cu_seqlens and would attend across request boundaries.",
            )
        return (
            logger.warning_once,
            "MiniMax H3 step batching needs every attention on a backend that isolates packed multi-document "
            "cu_seqlens (see AttentionBackend.supports_multi_doc_packed_varlen).",
        )

    def __call__(self, evaluations: Sequence[ModelEvaluation]):
        from .batched_packing import minimax_h3_batched_forward_kwargs
        from .denoise_loop import minimax_h3_publish_denoise_progress

        if not evaluations:
            return []
        check_request_cancellation()
        models = [self.models[e.state] for e in evaluations]
        request_progress = [(e.state.step_index, e.sigmas[0], len(e.state.sigmas[0]) - 1) for e in evaluations]
        uniform_progress = len(set(request_progress)) == 1
        # Requests at different points have nothing common to publish, so
        # step-gated attention stays dense, its safe default.
        minimax_h3_publish_denoise_progress(*(request_progress[0] if uniform_progress else (None, None, None)))
        context = get_forward_context() if is_forward_context_available() else None
        try:
            pdd = getattr(getattr(models[0].transformer, "arch", None), "pdd_steps", None) is not None
            if len(models) > 1:
                scheduled = context is not None and bool(
                    getattr(context, "attention_schedule", None) or getattr(context, "attention_sigma_schedule", None)
                )
                reason = self._unpacked_reason(models, pdd=pdd, scheduled=scheduled, uniform_progress=uniform_progress)
                if reason is not None:
                    level, message = reason
                    level(message + " Running %d requests one forward at a time.", len(models))
                    return [self([e])[0] for e in evaluations]
            prepared = [m.prepare(e) for m, e in zip(models, evaluations, strict=True)]
            hyperflow = hasattr(models[0].transformer, "endpoint_time_embedder")
            if hyperflow and len(models) > 1:
                # Pair-unique (t, r) packing is per-request; unpack batch for now.
                return [self([e])[0] for e in evaluations]
            if len(models) == 1:
                _, rows, times, cond = prepared[0]
                if hyperflow:
                    # Official: r_i = 1 - sigma_{i+1} on the same t = 1 - sigma axis.
                    step = evaluations[0].state.step_index
                    sigmas_v = evaluations[0].state.sigmas[0]
                    sigmas_a = evaluations[0].state.sigmas[1] if len(evaluations[0].state.sigmas) > 1 else sigmas_v
                    kwargs = models[0].branch.forward_kwargs_hyperflow(
                        video_rows=rows[0][0],
                        audio_rows=rows[1][0],
                        t_video=times[0],
                        t_audio=times[1],
                        r_video=1.0 - float(sigmas_v[step + 1]),
                        r_audio=1.0 - float(sigmas_a[step + 1]),
                        imgvid_cond_timestep=cond[0],
                        audio_ref_cond_timestep=cond[1],
                        video_target_timesteps=rows[0][1],
                        audio_target_timesteps=rows[1][1],
                    )
                else:
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
            if pdd:
                kwargs["pdd_step"] = evaluations[0].state.step_index
            reference_kv_state = kwargs.get("reference_kv_state")
            if reference_kv_state is not None:
                reference_kv_state.begin_step(evaluations[0].state.step_index, sigma=float(evaluations[0].sigmas[0]))
            video, audio = models[0].transformer(**kwargs)
            if reference_kv_state is not None:
                reference_kv_state.end_step()
            check_request_cancellation()
            video_parts = video.split([m.rows[0].shape[0] for m in models])
            audio_parts = audio.split([m.rows[1].shape[0] for m in models])
            return [
                m.to_denoised(p, (v, a)) for m, p, v, a in zip(models, prepared, video_parts, audio_parts, strict=True)
            ]
        finally:
            minimax_h3_publish_denoise_progress(None, None, None)
