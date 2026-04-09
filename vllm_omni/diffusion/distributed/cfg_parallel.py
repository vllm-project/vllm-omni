# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Base pipeline class for Diffusion models with shared CFG functionality.
"""

from abc import ABCMeta
from typing import Any

import torch

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)


def _wrap(pred: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Normalize prediction to tuple form."""
    return pred if isinstance(pred, tuple) else (pred,)


def _unwrap(pred: tuple[torch.Tensor, ...]) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Unwrap single-element tuple to plain tensor; keep multi-element as tuple."""
    return pred[0] if len(pred) == 1 else pred


def _slice_pred(pred: tuple[torch.Tensor, ...], output_slice: int) -> tuple[torch.Tensor, ...]:
    """Slice each element along dim 1."""
    return tuple(p[:, :output_slice] for p in pred)


class CFGParallelMixin(metaclass=ABCMeta):
    """
    Base Mixin class for Diffusion pipelines providing shared CFG methods.

    All pipelines should inherit from this class to reuse
    classifier-free guidance logic.

    CFG Parallel Architecture:
        When cfg_world_size > 1, each rank computes one branch (positive or
        negative), then all_gather exchanges results. All ranks then compute
        the CFG combine and scheduler step locally — no broadcast needed
        because the operations are deterministic.

    Multi-output models:
        Models that return tuple from predict_noise() (e.g., video + audio)
        should override combine_cfg_noise() to define per-element combine logic,
        and set self.scheduler to a composite scheduler that handles tuples.
    """

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Predict noise with optional classifier-free guidance.

        Args:
            do_true_cfg: Whether to apply CFG
            true_cfg_scale: CFG scale factor
            positive_kwargs: Kwargs for positive/conditional prediction
            negative_kwargs: Kwargs for negative/unconditional prediction
            cfg_normalize: Whether to normalize CFG output (default: True)
            output_slice: If set, slice each output to [:, :output_slice] for image editing

        Returns:
            Predicted noise tensor or tuple of tensors.
            In CFG parallel mode, result is valid on ALL ranks (not just rank 0).

        Note:
            For multi-output models (e.g., video + audio where predict_noise
            returns a tuple), override combine_cfg_noise() for per-element CFG
            logic and set self.scheduler to a composite scheduler.
        """
        if do_true_cfg:
            # Automatically detect CFG parallel configuration
            cfg_parallel_ready = get_classifier_free_guidance_world_size() > 1

            if cfg_parallel_ready:
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()

                # Each rank computes one branch
                kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
                local_pred = _wrap(self.predict_noise(**kwargs))

                if output_slice is not None:
                    local_pred = _slice_pred(local_pred, output_slice)

                # All-gather each element, reconstruct positive/negative tuples
                gathered = [cfg_group.all_gather(p, separate_tensors=True) for p in local_pred]
                positive_noise_pred = tuple(g[0] for g in gathered)
                negative_noise_pred = tuple(g[1] for g in gathered)

                # All ranks compute combine (deterministic, same result)
                return self.combine_cfg_noise(
                    positive_noise_pred,
                    negative_noise_pred,
                    true_cfg_scale,
                    cfg_normalize,
                )
            else:
                # Sequential CFG: compute both positive and negative
                positive_noise_pred = _wrap(self.predict_noise(**positive_kwargs))
                negative_noise_pred = _wrap(self.predict_noise(**negative_kwargs))

                if output_slice is not None:
                    positive_noise_pred = _slice_pred(positive_noise_pred, output_slice)
                    negative_noise_pred = _slice_pred(negative_noise_pred, output_slice)

                return self.combine_cfg_noise(
                    positive_noise_pred,
                    negative_noise_pred,
                    true_cfg_scale,
                    cfg_normalize,
                )
        else:
            # No CFG: only compute positive/conditional prediction
            pred = self.predict_noise(**positive_kwargs)
            if output_slice is not None:
                pred = _unwrap(_slice_pred(_wrap(pred), output_slice))
            return pred

    def cfg_normalize_function(self, noise_pred: torch.Tensor, comb_pred: torch.Tensor) -> torch.Tensor:
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

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Combine conditional and unconditional noise predictions with CFG.

        Accepts both plain tensors (backward-compatible, used by LTX2 etc.)
        and tuples (multi-output models). Default implementation applies the
        standard CFG formula to every element.

        Multi-output models can override this to apply different logic per element.

        Example override for a model returning (video_pred, audio_pred)::

            def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, scale, normalize):
                (video_pos, audio_pos) = positive_noise_pred
                (video_neg, audio_neg) = negative_noise_pred
                video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
                return (video_combined, audio_pos)  # audio: positive only, no CFG

        Args:
            positive_noise_pred: Positive/conditional prediction(s) — Tensor or tuple
            negative_noise_pred: Negative/unconditional prediction(s) — Tensor or tuple
            true_cfg_scale: CFG scale factor
            cfg_normalize: Whether to normalize the combined prediction (default: False)

        Returns:
            Combined noise prediction(s) — same type as inputs
        """
        pos_t = _wrap(positive_noise_pred)
        neg_t = _wrap(negative_noise_pred)

        results = []
        for p, n in zip(pos_t, neg_t):
            comb = n + true_cfg_scale * (p - n)
            if cfg_normalize:
                comb = self.cfg_normalize_function(p, comb)
            results.append(comb)
        return _unwrap(tuple(results))

    def predict_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Forward pass through transformer to predict noise.

        Subclasses should override this if they need custom behavior,
        but the default implementation calls self.transformer.

        Returns:
            Single Tensor for standard models, or tuple of Tensors for
            multi-output models (e.g., video + audio). Multi-output models
            must also override combine_cfg_noise() and set self.scheduler
            to a composite scheduler that handles tuples.
        """
        return self.transformer(*args, **kwargs)[0]

    def diffuse(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Diffusion loop with optional classifier-free guidance.

        Subclasses MUST implement this method to define the complete
        diffusion/denoising loop for their specific model.

        Typical implementation pattern (single output):
        ```python
        def diffuse(self, latents, timesteps, prompt_embeds, negative_embeds, ...):
            for t in timesteps:
                positive_kwargs = {...}
                negative_kwargs = {...}

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=self.guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                )

                latents = self.scheduler_step_maybe_with_cfg(
                    noise_pred, t, latents, do_true_cfg=do_true_cfg
                )

            return latents
        ```

        Multi-output models (e.g., video + audio) should:
        1. Override ``predict_noise()`` to return a tuple
        2. Override ``combine_cfg_noise()`` for per-element CFG logic
        3. Set ``self.scheduler`` to a composite scheduler that handles tuples

        ```python
        def diffuse(self, video_latents, audio_latents, timesteps_video, timesteps_audio, ...):
            for t_v, t_a in zip(timesteps_video, timesteps_audio):
                positive_kwargs = {...}
                negative_kwargs = {...}

                # Returns tuple: (video_pred, audio_pred)
                video_pred, audio_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=self.guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                )

                # self.scheduler = VideoAudioScheduler(video_sched, audio_sched)
                # which accepts and returns tuples
                video_latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                    (video_pred, audio_pred),
                    (t_v, t_a),
                    (video_latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                )

            return video_latents, audio_latents
        ```
        """
        raise NotImplementedError("Subclasses must implement diffuse")

    def scheduler_step(
        self,
        noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        t: torch.Tensor | tuple[torch.Tensor, ...],
        latents: torch.Tensor | tuple[torch.Tensor, ...],
        per_request_scheduler: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Step the scheduler.

        Default implementation passes inputs directly to ``self.scheduler.step()``.
        For multi-output models, set ``self.scheduler`` to a composite scheduler
        that handles tuples (e.g., ``VideoAudioScheduler``).

        Args:
            noise_pred: Predicted noise (Tensor or tuple for multi-output)
            t: Current timestep (Tensor or tuple when schedulers differ per output)
            latents: Current latents (Tensor or tuple for multi-output)
            per_request_scheduler: Optional request-scoped scheduler that
                overrides ``self.scheduler`` for this call. This is
                primarily used by step-wise execution, where each request
                may keep scheduler state in its own runner-managed state
                object. Request-level execution should usually leave this
                as ``None`` and continue using ``self.scheduler``.
            generator: Optional torch Generator for reproducible sampling.
                When using CFG parallel, both ranks should receive generators
                initialized with the same seed so that non-deterministic
                schedulers (e.g., DDPM) produce identical results.

        Returns:
            Updated latents after scheduler step
        """
        sched = per_request_scheduler if per_request_scheduler is not None else getattr(self, "scheduler", None)
        if sched is None:
            raise ValueError("No scheduler is available. Set self.scheduler or pass per_request_scheduler.")
        if not callable(getattr(sched, "step", None)):
            raise TypeError("per_request_scheduler must provide a callable step(...) method.")
        step_kwargs = dict(return_dict=False)
        if generator is not None:
            step_kwargs["generator"] = generator
        return sched.step(noise_pred, t, latents, **step_kwargs)[0]

    def scheduler_step_maybe_with_cfg(
        self,
        noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        t: torch.Tensor | tuple[torch.Tensor, ...],
        latents: torch.Tensor | tuple[torch.Tensor, ...],
        do_true_cfg: bool,
        per_request_scheduler: Any | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Step the scheduler with automatic CFG parallel handling.

        All ranks compute the scheduler step locally — no broadcast needed
        because predict_noise_maybe_with_cfg already ensures all ranks have
        identical noise_pred after all_gather + local combine.

        Args:
            noise_pred: Predicted noise (Tensor or tuple, valid on all ranks)
            t: Current timestep (Tensor or tuple when schedulers differ per output)
            latents: Current latents (Tensor or tuple)
            do_true_cfg: Whether CFG is enabled
            per_request_scheduler: Optional request-scoped scheduler that
                overrides ``self.scheduler`` for this call. This is mainly
                needed by step-wise execution, where scheduler state may be
                stored per request. Request-level execution should normally
                leave this as ``None``.
            generator: Optional torch Generator for reproducible sampling.
                When using CFG parallel, both ranks should receive generators
                initialized with the same seed so that non-deterministic
                schedulers (e.g., DDPM) produce identical results.

        Returns:
            Updated latents (identical across all CFG ranks)
        """
        return self.scheduler_step(
            noise_pred,
            t,
            latents,
            per_request_scheduler=per_request_scheduler,
            generator=generator,
        )
