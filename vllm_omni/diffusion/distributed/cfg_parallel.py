# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Base pipeline class for Diffusion models with shared CFG functionality.
"""

from abc import ABCMeta
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    is_cfg_group_initialized,
)

logger = init_logger(__name__)


def _get_cfg_world_size_or_one() -> int:
    """Return the CFG world size, defaulting only when no group exists.

    Diffusion workers initialize the CFG process group even when its world
    size is one.  Pipelines are also invoked directly by unit tests and some
    offline integrations, though, where no distributed groups exist. Treat
    that explicit case like a one-rank CFG group. Errors from an existing CFG
    coordinator must propagate instead of silently selecting sequential CFG.
    """
    if not is_cfg_group_initialized():
        return 1
    return get_classifier_free_guidance_world_size()


def _wrap(pred: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Normalize prediction to tuple form.

    A list (MixFusion's per-request packed output) passes through unchanged so
    ``combine_cfg_noise`` can combine it request by request.
    """
    return pred if isinstance(pred, (tuple, list)) else (pred,)


def _unwrap(pred: tuple[torch.Tensor, ...]) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Unwrap single-element tuple to plain tensor; keep multi-element as tuple."""
    return pred[0] if len(pred) == 1 else pred


def _slice_pred(pred: tuple[torch.Tensor, ...], output_slice: int) -> tuple[torch.Tensor, ...]:
    """Slice each element along dim 1."""
    return tuple(p[:, :output_slice] for p in pred)


def _dispatch_branches(n_branches: int, n_ranks: int) -> list[list[int]]:
    """
    Round-robin dispatch N branches to M ranks.

    Rule: branch i → rank (i % n_ranks).

    Examples:
        _dispatch_branches(3, 2) -> [[0, 2], [1]]
        _dispatch_branches(3, 3) -> [[0], [1], [2]]
        _dispatch_branches(4, 2) -> [[0, 2], [1, 3]]
        _dispatch_branches(4, 4) -> [[0], [1], [2], [3]]
    """
    assignments: list[list[int]] = [[] for _ in range(n_ranks)]
    for i in range(n_branches):
        assignments[i % n_ranks].append(i)
    return assignments


class _NotBatchableError(Exception):
    """The positive and negative kwargs cannot be merged into one batch."""


def _batch_cfg_value(pos: Any, neg: Any, n_pos: int, n_neg: int) -> Any:
    """Merge one positive/negative argument pair into the batched call's argument.

    Tensors whose dim 0 is the branch batch size (``n_pos`` / ``n_neg``) are
    concatenated and must agree in every other dim, dtype and device. Any
    other value -- unbatched tensors such as Flux's ``[L, 3]`` position ids,
    objects, ``None`` -- must be the very same object in both branches, or an
    equal plain scalar, and is passed through. Dicts, lists and tuples are
    merged element by element. Raises :class:`_NotBatchableError` otherwise.
    """
    if isinstance(pos, torch.Tensor) and isinstance(neg, torch.Tensor):
        if pos.dim() > 0 and neg.dim() > 0 and pos.shape[0] == n_pos and neg.shape[0] == n_neg:
            if pos.shape[1:] != neg.shape[1:] or pos.dtype != neg.dtype or pos.device != neg.device:
                raise _NotBatchableError
            return torch.cat([pos, neg])
    elif isinstance(pos, dict) and isinstance(neg, dict):
        if pos.keys() != neg.keys():
            raise _NotBatchableError
        return {k: _batch_cfg_value(pos[k], neg[k], n_pos, n_neg) for k in pos}
    elif isinstance(pos, (list, tuple)) and isinstance(neg, (list, tuple)):
        if type(pos) is not type(neg) or len(pos) != len(neg):
            raise _NotBatchableError
        return type(pos)(_batch_cfg_value(p, n, n_pos, n_neg) for p, n in zip(pos, neg))
    if pos is neg:
        return pos
    if isinstance(pos, (bool, int, float, str)) and type(pos) is type(neg) and pos == neg:
        return pos
    raise _NotBatchableError


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

    transformer: torch.nn.Module

    # Pipelines whose predict_noise() accepts the positive and negative branch
    # concatenated along the batch dimension set this to True; with
    # enable_cfg_batching on and a single CFG rank, both branches then run as
    # one call. Leave it False when the branches can differ in shape (prompt
    # lengths, masks) or the denoiser keeps per-branch state.
    supports_cfg_batching: bool = False
    # predict_noise kwarg whose dim 0 is the batch size of a branch.
    cfg_batching_batch_kwarg: str = "hidden_states"
    # predict_noise kwargs that are not batched and are equal in both branches
    # but may be distinct objects (e.g. per-branch ``[L, 3]`` text position
    # ids); the positive value is passed to the batched call unchecked.
    cfg_batching_shared_kwargs: frozenset[str] = frozenset()

    def _use_cfg_batching(self) -> bool:
        if not self.supports_cfg_batching:
            return False
        od_config = getattr(self, "od_config", None)
        if od_config is None or not getattr(od_config, "enable_cfg_batching", False):
            return False
        # Non-last pipeline-parallel stages return IntermediateTensors, which
        # cannot be split back into branches.
        parallel_config = getattr(od_config, "parallel_config", None)
        if getattr(parallel_config, "pipeline_parallel_size", 1) > 1:
            return False
        # Step caches (TeaCache, ...) keep one state per CFG branch, keyed on
        # transformer.cfg_branch; one merged call has no single branch.
        return getattr(od_config, "cache_backend", None) in (None, "none")

    def merge_cfg_branch_kwargs(
        self, positive_kwargs: dict[str, Any], negative_kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], int, int] | None:
        """Merge both branches' predict_noise kwargs into one batch.

        Returns the merged kwargs and the positive / negative batch sizes, or
        None to run the branches sequentially. Tensors batched like
        ``cfg_batching_batch_kwarg`` are concatenated on dim 0; kwargs in
        ``cfg_batching_shared_kwargs`` take the positive value; everything else
        must be the same object or an equal scalar in both branches (see
        ``_batch_cfg_value``). Pipelines that need more (e.g. per-branch masks)
        can override this.
        """
        if positive_kwargs.keys() != negative_kwargs.keys():
            return None
        key = self.cfg_batching_batch_kwarg
        pos_batch, neg_batch = positive_kwargs.get(key), negative_kwargs.get(key)
        if not (isinstance(pos_batch, torch.Tensor) and isinstance(neg_batch, torch.Tensor)):
            return None
        if pos_batch.dim() == 0 or neg_batch.dim() == 0:
            return None
        n_pos, n_neg = pos_batch.shape[0], neg_batch.shape[0]
        shared = self.cfg_batching_shared_kwargs
        try:
            merged = {
                k: positive_kwargs[k] if k in shared else _batch_cfg_value(v, negative_kwargs[k], n_pos, n_neg)
                for k, v in positive_kwargs.items()
            }
        except _NotBatchableError:
            return None
        return merged, n_pos, n_neg

    def _predict_noise_batched_cfg(
        self, positive_kwargs: dict[str, Any], negative_kwargs: dict[str, Any]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]] | None:
        """Both branches in one predict_noise call, or None to run them sequentially."""
        merged = self.merge_cfg_branch_kwargs(positive_kwargs, negative_kwargs)
        if merged is None:
            logger.warning_once(
                "CFG batching: %s branch arguments do not merge into one batch; running them sequentially",
                type(self).__name__,
            )
            return None
        batched, n_pos, n_neg = merged
        # Deliberately not stamped with transformer.cfg_branch: the call covers
        # both branches, and per-branch caches are excluded above.
        pred = _wrap(self.predict_noise(**batched))
        if not all(isinstance(p, torch.Tensor) and p.dim() > 0 and p.shape[0] == n_pos + n_neg for p in pred):
            raise RuntimeError(
                f"{type(self).__name__} sets supports_cfg_batching, but predict_noise did not return "
                f"outputs batched as {n_pos} positive + {n_neg} negative rows"
            )
        return tuple(p[:n_pos] for p in pred), tuple(p[n_pos:] for p in pred)

    def _set_transformer_do_true_cfg(self, do_true_cfg: bool) -> None:
        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            transformer.do_true_cfg = do_true_cfg

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        kwargs: dict[str, Any] | None = None,
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
            kwargs: Optional extra context for custom combine implementations.

        Returns:
            Predicted noise tensor or tuple of tensors.
            In CFG parallel mode, result is valid on ALL ranks (not just rank 0).

        Note:
            For multi-output models (e.g., video + audio where predict_noise
            returns a tuple), override combine_cfg_noise() for per-element CFG
            logic and set self.scheduler to a composite scheduler.

            Sets ``self.transformer.do_true_cfg`` before predicting, and stamps
            ``self.transformer.cfg_branch`` ("positive" / "negative") around
            each ``predict_noise`` call. TeaCache keeps one cache state per CFG
            branch and reads the stamp instead of guessing the branch from how
            many forwards it has seen, which goes wrong once non-CFG forwards
            run in between (e.g. under step execution).

            With one CFG rank and ``enable_cfg_batching`` on (default off),
            pipelines that set ``supports_cfg_batching`` run both branches as
            one ``predict_noise`` call on the concatenated batch (see
            ``merge_cfg_branch_kwargs``), unless a cache backend is configured.
        """
        self._set_transformer_do_true_cfg(do_true_cfg)

        if do_true_cfg:
            if negative_kwargs is None:
                raise ValueError("CFG requires negative prediction arguments.")
            # Automatically detect CFG parallel configuration
            cfg_parallel_ready = _get_cfg_world_size_or_one() > 1

            if cfg_parallel_ready:
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()

                # Each rank computes one branch
                if cfg_rank == 0:
                    logger.debug("CFG Parallel: Rank 0 computing positive branch")
                    local_pred = _wrap(self._predict_noise_for_cfg_branch("positive", positive_kwargs))
                else:
                    logger.debug("CFG Parallel: Rank %d computing negative branch", cfg_rank)
                    local_pred = _wrap(self._predict_noise_for_cfg_branch("negative", negative_kwargs))

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
                    **({} if kwargs is None else {"kwargs": kwargs}),
                )
            else:
                batched = None
                if self._use_cfg_batching():
                    batched = self._predict_noise_batched_cfg(positive_kwargs, negative_kwargs)
                if batched is not None:
                    # Batched CFG: both branches in one call.
                    positive_noise_pred, negative_noise_pred = batched
                else:
                    # Sequential CFG: compute both positive and negative
                    positive_noise_pred = _wrap(self._predict_noise_for_cfg_branch("positive", positive_kwargs))
                    negative_noise_pred = _wrap(self._predict_noise_for_cfg_branch("negative", negative_kwargs))

                if output_slice is not None:
                    positive_noise_pred = _slice_pred(positive_noise_pred, output_slice)
                    negative_noise_pred = _slice_pred(negative_noise_pred, output_slice)

                return self.combine_cfg_noise(
                    positive_noise_pred,
                    negative_noise_pred,
                    true_cfg_scale,
                    cfg_normalize,
                    **({} if kwargs is None else {"kwargs": kwargs}),
                )
        else:
            # No CFG: only compute positive/conditional prediction
            pred = self._predict_noise_for_cfg_branch("positive", positive_kwargs)
            if output_slice is not None:
                pred = _unwrap(_slice_pred(_wrap(pred), output_slice))
            return pred

    def _predict_noise_for_cfg_branch(self, branch: str, predict_kwargs: dict[str, Any] | None) -> Any:
        """Call ``predict_noise`` with ``transformer.cfg_branch`` set to ``branch``.

        The previous value is restored afterwards, also on error, so a stale
        stamp never leaks into calls made outside this helper.
        """
        if predict_kwargs is None:
            raise ValueError(f"predict_noise kwargs for the {branch} CFG branch are required when do_true_cfg=True")
        transformer = getattr(self, "transformer", None)
        if transformer is None:
            return self.predict_noise(**predict_kwargs)
        previous = getattr(transformer, "cfg_branch", None)
        transformer.cfg_branch = branch
        try:
            return self.predict_noise(**predict_kwargs)
        finally:
            transformer.cfg_branch = previous

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
        positive_noise_pred: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]:
        """
        Combine conditional and unconditional noise predictions with CFG.

        Accepts plain tensors (backward-compatible, used by LTX2 etc.), tuples
        (multi-output models) and lists (packed batches, one prediction per
        request). Default implementation applies the standard CFG formula to
        every element and returns the inputs' container type: packed batches
        must stay a list because the runner indexes them by request.

        Multi-output models can override this to apply different logic per element.

        Example override for a model returning (video_pred, audio_pred)::

            def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, scale, normalize):
                (video_pos, audio_pos) = positive_noise_pred
                (video_neg, audio_neg) = negative_noise_pred
                video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
                return (video_combined, audio_pos)  # audio: positive only, no CFG

        Args:
            positive_noise_pred: Positive/conditional prediction(s) — Tensor, list or tuple
            negative_noise_pred: Negative/unconditional prediction(s) — Tensor, list or tuple
            true_cfg_scale: CFG scale factor
            cfg_normalize: Whether to normalize the combined prediction (default: False)
            kwargs: Optional extra context for custom combine implementations.

        Returns:
            Combined noise prediction(s) — same type as inputs
        """
        if isinstance(positive_noise_pred, list):
            # Packed MixFusion batch: one prediction per request.
            return [
                self.combine_cfg_noise(p, n, true_cfg_scale, cfg_normalize, kwargs)
                for p, n in zip(positive_noise_pred, negative_noise_pred, strict=True)
            ]
        pos_t = _wrap(positive_noise_pred)
        neg_t = _wrap(negative_noise_pred)

        results = []
        for p, n in zip(pos_t, neg_t):
            comb = n + true_cfg_scale * (p - n)
            if cfg_normalize:
                comb = self.cfg_normalize_function(p, comb)
            results.append(comb)
        return _unwrap(tuple(results))

    # ── N-branch CFG interface (for 3+ branch models) ──

    def predict_noise_with_multi_branch_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float | dict[str, float],
        branches_kwargs: list[dict[str, Any]],
        cfg_normalize: bool = False,
        output_slice: int | None = None,
        combine_fn: Callable[[list[torch.Tensor | tuple[torch.Tensor, ...]]], torch.Tensor | tuple[torch.Tensor, ...]]
        | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Predict noise with N-branch CFG dispatch across M GPUs.

        This is the multi-branch counterpart of predict_noise_maybe_with_cfg().
        Use this for models with 3 or more CFG branches (e.g., OmniGen2, Bagel).
        Existing 2-branch models should continue using
        predict_noise_maybe_with_cfg().

        Args:
            do_true_cfg: Whether to apply CFG.
            true_cfg_scale: CFG scale factor (passed to combine_multi_branch_cfg_noise).
            branches_kwargs: List of N dicts, each containing kwargs for one
                predict_noise() call. branches_kwargs[0] is always the
                positive/conditional branch.
            cfg_normalize: Whether to normalize (passed to combine_multi_branch_cfg_noise).
            output_slice: If set, slice each output to [:, :output_slice].
            combine_fn: Optional request-local combiner, called after branch
                predictions are available on every rank. Defaults to the
                pipeline's combine_multi_branch_cfg_noise implementation.

        Returns:
            Combined noise prediction, identical on all ranks in CFG parallel.
        """
        if do_true_cfg:
            n_branches = len(branches_kwargs)
            cfg_world_size = _get_cfg_world_size_or_one()
            cfg_parallel_ready = cfg_world_size > 1

            if cfg_parallel_ready:
                return self._predict_multi_branch_parallel(
                    branches_kwargs,
                    n_branches,
                    cfg_world_size,
                    true_cfg_scale,
                    cfg_normalize,
                    output_slice,
                    combine_fn,
                )
            else:
                # Sequential: run all N branches on single device
                preds: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
                for kw in branches_kwargs:
                    pred = _wrap(self.predict_noise(**kw))
                    if output_slice is not None:
                        pred = _slice_pred(pred, output_slice)
                    preds.append(_unwrap(pred))
                if combine_fn is not None:
                    return combine_fn(preds)
                return self.combine_multi_branch_cfg_noise(preds, true_cfg_scale, cfg_normalize)
        else:
            # No CFG: only compute positive/conditional prediction
            pred = self.predict_noise(**branches_kwargs[0])
            if output_slice is not None:
                pred = _unwrap(_slice_pred(_wrap(pred), output_slice))
            return pred

    def _predict_multi_branch_parallel(
        self,
        branches_kwargs: list[dict[str, Any]],
        n_branches: int,
        cfg_world_size: int,
        true_cfg_scale: float | dict[str, float],
        cfg_normalize: bool,
        output_slice: int | None,
        combine_fn: Callable[[list[torch.Tensor | tuple[torch.Tensor, ...]]], torch.Tensor | tuple[torch.Tensor, ...]]
        | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Dispatch N branches across M ranks, all_gather, then combine."""
        cfg_group = get_cfg_group()
        cfg_rank = get_classifier_free_guidance_rank()

        if cfg_world_size > n_branches:
            logger.warning_once(
                "cfg_parallel_size=%d > n_branches=%d, %d GPU(s) will be idle for CFG",
                cfg_world_size,
                n_branches,
                cfg_world_size - n_branches,
            )

        # Assign branches to ranks via round-robin
        assignments = _dispatch_branches(n_branches, cfg_world_size)
        my_branch_ids = assignments[cfg_rank]
        max_per_rank = max(len(a) for a in assignments)
        if max_per_rank > 1 and getattr(self, "_cache_dit_one_call_per_cfg_rank", False):
            # Cache-DiT counts every transformer call on a CFG rank as a denoise step, so a rank
            # running two branches would exhaust its cache budget early while the others don't.
            raise ValueError(
                f"Cache-DiT with cfg_parallel_size={cfg_world_size} does not support {n_branches}-branch CFG: "
                f"a rank would run {max_per_rank} branches per step. Set cfg_parallel_size to at least "
                f"{n_branches}, or disable cache_dit."
            )

        # Run assigned branches
        my_preds: list[tuple[torch.Tensor, ...]] = []
        for bid in my_branch_ids:
            pred = _wrap(self.predict_noise(**branches_kwargs[bid]))
            if output_slice is not None:
                pred = _slice_pred(pred, output_slice)
            my_preds.append(pred)

        # Idle ranks (cfg_world_size > n_branches) run a forward pass to get the output shape for all_gather.
        # Output shape cannot be inferred from kwargs — may be tuple, sliced, etc.
        if not my_preds:
            pred = _wrap(self.predict_noise(**branches_kwargs[0]))
            if output_slice is not None:
                pred = _slice_pred(pred, output_slice)
            my_preds.append(pred)

        # Pad to max_per_rank with zeros so all ranks have same size
        ref_pred = my_preds[0]
        while len(my_preds) < max_per_rank:
            my_preds.append(tuple(torch.zeros_like(t) for t in ref_pred))

        # All-gather each output element separately (like predict_noise_maybe_with_cfg)
        # For each slot, gather across ranks; then pick valid results by owner_rank
        # all_slots[slot][elem_idx] = [rank0_tensor, rank1_tensor, ...]
        all_slots: list[list[list[torch.Tensor]]] = []
        for slot in range(max_per_rank):
            slot_results: list[list[torch.Tensor]] = []
            for p in my_preds[slot]:
                gathered = cfg_group.all_gather(p, separate_tensors=True)
                slot_results.append(gathered)
            all_slots.append(slot_results)

        # Reconstruct final_preds in branch order
        final_preds: list[torch.Tensor | tuple[torch.Tensor, ...]] = []
        for bid in range(n_branches):
            owner_rank = bid % cfg_world_size
            slot_idx = bid // cfg_world_size
            elements = tuple(all_slots[slot_idx][elem_idx][owner_rank] for elem_idx in range(len(ref_pred)))
            final_preds.append(_unwrap(elements))

        if combine_fn is not None:
            return combine_fn(final_preds)
        return self.combine_multi_branch_cfg_noise(final_preds, true_cfg_scale, cfg_normalize)

    def combine_multi_branch_cfg_noise(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        true_cfg_scale: float | dict[str, float],
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Combine N branch predictions. Default: standard 2-branch CFG formula.

        Override this method for custom multi-branch combine logic.

        Args:
            predictions: List of N predictions, where predictions[0] is always
                the positive/conditional branch.
            true_cfg_scale: CFG scale factor (float for 2-branch, dict for multi-branch).
            cfg_normalize: Whether to normalize the combined prediction.

        Returns:
            Combined noise prediction.
        """
        positive = _wrap(predictions[0])
        negative = _wrap(predictions[1])

        results = []
        for p, n in zip(positive, negative):
            comb = n + true_cfg_scale * (p - n)
            if cfg_normalize:
                comb = self.cfg_normalize_function(p, comb)
            results.append(comb)
        return _unwrap(tuple(results))

    def predict_noise(self, *args: Any, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, ...] | IntermediateTensors:
        """
        Forward pass through transformer to predict noise.

        Subclasses should override this if they need custom behavior,
        but the default implementation calls self.transformer.

        Returns:
            Single Tensor for standard models, or tuple of Tensors for
            multi-output models (e.g., video + audio). Multi-output models
            must also override combine_cfg_noise() and set self.scheduler
            to a composite scheduler that handles tuples.
            Non-last Pipeline Parallel stages return `IntermediateTensors`
            instead of final noise tensors wrapped in a tuple.
        """
        result = self.transformer(*args, **kwargs)
        # A MixFusion packed forward returns one tensor per request as a list.
        return result if isinstance(result, (IntermediateTensors, list)) else result[0]

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

    def check_cfg_parallel_validity(self, true_cfg_scale: float, has_neg_prompt: bool) -> bool:
        """
        Check if CFG parallel configuration is valid.

        Args:
            true_cfg_scale: The classifier-free guidance scale value
            has_neg_prompt: Whether a negative prompt is provided

        Returns:
            True if CFG parallel configuration is valid, False otherwise
        """
        if _get_cfg_world_size_or_one() == 1:
            return True

        if true_cfg_scale <= 1:
            logger.warning("CFG parallel is NOT working correctly when true_cfg_scale <= 1.")
            return False

        if not has_neg_prompt:
            logger.warning(
                "CFG parallel is NOT working correctly when there is no negative prompt or negative prompt embeddings."
            )
            return False
        return True

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
