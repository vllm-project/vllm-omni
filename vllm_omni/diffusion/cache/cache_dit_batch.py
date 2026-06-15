# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-aware forward for cache-dit ``CachedBlocks`` patterns.

When a ``CachedContextManager`` has ``_batch_contexts`` set (batch mode),
the patched forward runs: Fn on the full batch -> per-request ``can_cache``
decisions -> Mn on the compute-group subset -> per-request ``apply_cache``
-> Bn on the full batch. Cache-dit's own methods are called unmodified;
``_current_context`` is switched between calls so each runs against the
right request.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Kwargs whose leading dim aligns with the batch row dim; they get
# index_select'd to match the compute subset when Mn runs on a subset.
# Leading dim of ``2 * total_rows`` is the CFG layout (positive/negative
# halves concatenated) and is handled in _slice_compute_tensor.
_BATCH_ALIGNED_KWARG_NAMES = frozenset(
    {
        "temb",
        "modulate_index",
        "encoder_hidden_states_mask",
        "hidden_states_mask",
        "timestep",
        "guidance",
        "pooled_projections",
        "attention_mask",
        "encoder_attention_mask",
    }
)


# --- batch-mode activation on a CachedContextManager ---


def set_batch_contexts(
    context_manager: Any,
    contexts_list: list[dict[str, Any]],
    row_counts: list[int],
) -> None:
    """Activate batch mode on a ``CachedContextManager``."""
    offsets = [0]
    for c in row_counts:
        offsets.append(offsets[-1] + c)
    context_manager._batch_contexts = contexts_list
    context_manager._batch_row_counts = row_counts
    context_manager._batch_row_offsets = offsets


def clear_batch_contexts(context_manager: Any) -> None:
    """Deactivate batch mode on a ``CachedContextManager``."""
    context_manager._batch_contexts = None
    context_manager._batch_row_offsets = None
    context_manager._batch_row_counts = None
    context_manager._current_context = None


def _is_batch_mode(context_manager: Any) -> bool:
    return getattr(context_manager, "_batch_contexts", None) is not None


# --- kwargs slicing for the Mn compute subset ---


def _slice_compute_tensor(value: Any, row_index: torch.Tensor, total_rows: int) -> Any:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    index = row_index.to(device=value.device)
    leading = int(value.shape[0])
    if leading == total_rows:
        return torch.index_select(value, 0, index)
    if leading == total_rows * 2:
        # CFG: positive rows [0:N) + negative rows [N:2N) concatenated on dim 0.
        return torch.index_select(value, 0, torch.cat([index, index + total_rows], dim=0))
    return value


def _slice_batch_aligned_kwargs(
    kwargs: dict[str, Any],
    row_index: torch.Tensor,
    total_rows: int,
) -> dict[str, Any]:
    if not kwargs:
        return kwargs
    sliced = dict(kwargs)
    for key in _BATCH_ALIGNED_KWARG_NAMES:
        if key in sliced:
            sliced[key] = _slice_compute_tensor(sliced[key], row_index, total_rows)
    return sliced


# --- encoder sequence-length trim / restore ---


def _request_encoder_seq_len(
    kwargs: dict[str, Any],
    request_index: int,
    row_start: int,
    row_end: int,
    fallback: int,
) -> int:
    mask = kwargs.get("encoder_hidden_states_mask")
    if isinstance(mask, torch.Tensor) and mask.ndim >= 2 and int(mask.shape[0]) >= row_end:
        mask_slice = mask[row_start:row_end]
        if mask_slice.numel() > 0:
            seq_len = int(mask_slice.sum(dim=1, dtype=torch.int64).max().item())
            return max(0, min(seq_len, fallback))

    txt_seq_lens = kwargs.get("txt_seq_lens")
    if isinstance(txt_seq_lens, (list, tuple)) and request_index < len(txt_seq_lens):
        return max(0, min(int(txt_seq_lens[request_index]), fallback))

    return fallback


def _trim_encoder_slice(
    encoder_slice: torch.Tensor | None,
    kwargs: dict[str, Any],
    request_index: int,
    row_start: int,
    row_end: int,
) -> tuple[torch.Tensor | None, int | None]:
    if encoder_slice is None:
        return None, None
    seq_len = _request_encoder_seq_len(kwargs, request_index, row_start, row_end, int(encoder_slice.shape[1]))
    return encoder_slice[:, :seq_len], seq_len


def _restore_encoder_slice(
    encoder_slice: torch.Tensor | None,
    reference_slice: torch.Tensor | None,
    seq_len: int | None,
) -> torch.Tensor | None:
    if encoder_slice is None or reference_slice is None or seq_len is None:
        return encoder_slice
    restored = reference_slice.clone()
    n = min(int(seq_len), int(restored.shape[1]), int(encoder_slice.shape[1]))
    restored[:, :n] = encoder_slice[:, :n]
    return restored


# --- batched forward (shared between Pattern 0/1/2 and Pattern 3/4/5) ---


def _forward_batched(
    self: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    *args: Any,
    is_pattern_345: bool,
    **kwargs: Any,
) -> Any:
    cm = self.context_manager
    batch_contexts: list[dict[str, Any]] = cm._batch_contexts
    row_offsets: list[int] = cm._batch_row_offsets
    row_counts: list[int] = cm._batch_row_counts
    ctx_name: str = self.cache_context
    prefix: str = self.cache_prefix

    cm._current_context = batch_contexts[0][ctx_name]
    self._check_cache_params()

    use_l1 = cm.is_l1_diff_enabled()
    parallelized = self._is_parallelized()
    fn_check_prefix = f"{prefix}_Fn_hidden_states" if use_l1 else f"{prefix}_Fn_residual"
    bn_hs_prefix = f"{prefix}_Bn_residual" if cm.is_cache_residual() else f"{prefix}_Bn_hidden_states"
    bn_enc_prefix = f"{prefix}_Bn_residual" if cm.is_encoder_cache_residual() else f"{prefix}_Bn_hidden_states"

    # Stage 1: Fn blocks on full batch.
    original_hs = hidden_states
    if is_pattern_345:
        hidden_states, encoder_hidden_states = self.call_Fn_blocks(hidden_states, *args, **kwargs)
    else:
        hidden_states, encoder_hidden_states = self.call_Fn_blocks(
            hidden_states, encoder_hidden_states, *args, **kwargs
        )
    fn_residual = self._get_Fn_residual(original_hs, hidden_states)
    fn_hs = hidden_states.clone() if use_l1 else None
    del original_hs

    # Stage 2: per-request can_cache decisions.
    decisions: list[bool] = []
    for i, ctx_map in enumerate(batch_contexts):
        cm._current_context = ctx_map[ctx_name]
        cm.mark_step_begin()
        start = row_offsets[i]
        end = start + row_counts[i]
        check = hidden_states[start:end] if use_l1 else fn_residual[start:end]
        decisions.append(cm.can_cache(check, parallelized=parallelized, prefix=fn_check_prefix))

    compute_idx = [i for i, d in enumerate(decisions) if not d]
    cache_idx = [i for i, d in enumerate(decisions) if d]

    # Stage 3a: Mn on compute subset, then per-request Fn/Bn buffer storage.
    if compute_idx:
        rows: list[int] = []
        for i in compute_idx:
            rows.extend(range(row_offsets[i], row_offsets[i] + row_counts[i]))
        row_index = torch.tensor(rows, dtype=torch.long, device=hidden_states.device)
        total_rows = int(hidden_states.shape[0])
        compute_kwargs = _slice_batch_aligned_kwargs(kwargs, row_index, total_rows)

        compute_hs = torch.index_select(hidden_states, 0, row_index)
        if is_pattern_345:
            mn_hs, mn_enc, mn_hs_residual = self.call_Mn_blocks(compute_hs, *args, **compute_kwargs)
            mn_enc_residual = None
        else:
            compute_enc = (
                torch.index_select(encoder_hidden_states, 0, row_index) if encoder_hidden_states is not None else None
            )
            mn_hs, mn_enc, mn_hs_residual, mn_enc_residual = self.call_Mn_blocks(
                compute_hs, compute_enc, *args, **compute_kwargs
            )

        # Pattern 3/4/5 derives the encoder residual against the pre-Mn encoder,
        # so capture it before we commit Mn outputs below.
        pre_mn_encoder = encoder_hidden_states
        can_store_enc = (mn_enc is not None) if is_pattern_345 else (mn_enc_residual is not None)

        local = 0
        for i in compute_idx:
            cm._current_context = batch_contexts[i][ctx_name]
            nr = row_counts[i]
            bs = row_offsets[i]

            cm.set_Fn_buffer(fn_residual[bs : bs + nr], prefix=f"{prefix}_Fn_residual")
            if use_l1:
                cm.set_Fn_buffer(fn_hs[bs : bs + nr], f"{prefix}_Fn_hidden_states")

            if cm.is_cache_residual():
                cm.set_Bn_buffer(mn_hs_residual[local : local + nr], prefix=f"{prefix}_Bn_residual")
            else:
                cm.set_Bn_buffer(mn_hs[local : local + nr], prefix=f"{prefix}_Bn_hidden_states")

            if can_store_enc:
                req_mn_enc, _ = _trim_encoder_slice(mn_enc[local : local + nr], kwargs, i, bs, bs + nr)
                if cm.is_encoder_cache_residual():
                    if is_pattern_345:
                        old_enc, _ = _trim_encoder_slice(
                            pre_mn_encoder[bs : bs + nr] if pre_mn_encoder is not None else None,
                            kwargs,
                            i,
                            bs,
                            bs + nr,
                        )
                        enc_res = req_mn_enc - old_enc if old_enc is not None else req_mn_enc
                    else:
                        enc_res, _ = _trim_encoder_slice(mn_enc_residual[local : local + nr], kwargs, i, bs, bs + nr)
                    cm.set_Bn_encoder_buffer(enc_res, prefix=f"{prefix}_Bn_residual")
                else:
                    cm.set_Bn_encoder_buffer(req_mn_enc, prefix=f"{prefix}_Bn_hidden_states")

            local += nr

        # Commit Mn outputs back into the full-batch tensors.
        hidden_states = hidden_states.clone()
        hidden_states.index_copy_(0, row_index, mn_hs)
        if mn_enc is not None:
            encoder_hidden_states = (
                torch.zeros_like(hidden_states) if encoder_hidden_states is None else encoder_hidden_states.clone()
            )
            encoder_hidden_states.index_copy_(0, row_index, mn_enc)

    del fn_residual

    # Stage 3b: apply cached residuals for cache hits.
    for i in cache_idx:
        cm._current_context = batch_contexts[i][ctx_name]
        cm.add_cached_step()
        start = row_offsets[i]
        end = start + row_counts[i]
        enc_slice = encoder_hidden_states[start:end] if encoder_hidden_states is not None else None
        trimmed_enc, seq_len = _trim_encoder_slice(enc_slice, kwargs, i, start, end)

        cached_hs, cached_enc = cm.apply_cache(
            hidden_states[start:end],
            trimmed_enc,
            prefix=bn_hs_prefix,
            encoder_prefix=bn_enc_prefix,
        )
        hidden_states[start:end] = cached_hs
        if encoder_hidden_states is not None and cached_enc is not None:
            encoder_hidden_states[start:end] = _restore_encoder_slice(cached_enc, enc_slice, seq_len)

    # Stage 4: Bn blocks on full batch (Pattern 3/4/5 may skip entirely).
    if not is_pattern_345 or cm.Bn_compute_blocks() > 0:
        if is_pattern_345:
            hidden_states, encoder_hidden_states = self.call_Bn_blocks(hidden_states, *args, **kwargs)
        else:
            hidden_states, encoder_hidden_states = self.call_Bn_blocks(
                hidden_states, encoder_hidden_states, *args, **kwargs
            )

    cm._current_context = None
    return self._process_forward_outputs(hidden_states, encoder_hidden_states)


def _forward_batched_base(
    self: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Batch-aware forward for ``CachedBlocks_Pattern_Base`` (Pattern 0/1/2)."""
    return _forward_batched(
        self,
        hidden_states,
        encoder_hidden_states,
        *args,
        is_pattern_345=False,
        **kwargs,
    )


def _forward_batched_345(
    self: Any,
    hidden_states: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Batch-aware forward for ``CachedBlocks_Pattern_3_4_5`` (Pattern 3/4/5)."""
    return _forward_batched(
        self,
        hidden_states,
        None,
        *args,
        is_pattern_345=True,
        **kwargs,
    )


# --- monkey-patch entry point ---


def _install_patch(cls: Any, batched_forward: Any) -> None:
    if getattr(cls, "_batch_patched", False):
        return
    original = cls.forward

    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_batch_mode(self.context_manager):
            return batched_forward(self, *args, **kwargs)
        return original(self, *args, **kwargs)

    cls.forward = forward
    cls._batch_patched = True
    logger.debug("Patched %s.forward for batch mode.", cls.__name__)


def patch_cache_dit_for_batching() -> None:
    """Route batch-mode forwards on cache-dit CachePattern classes.

    Idempotent; silently skipped if cache-dit is not importable.
    """
    try:
        from cache_dit.caching.cache_blocks.pattern_3_4_5 import CachedBlocks_Pattern_3_4_5
        from cache_dit.caching.cache_blocks.pattern_base import CachedBlocks_Pattern_Base
    except ImportError:
        logger.warning("cache-dit CachePattern classes not found; batch patch skipped.")
        return

    _install_patch(CachedBlocks_Pattern_Base, _forward_batched_base)
    _install_patch(CachedBlocks_Pattern_3_4_5, _forward_batched_345)
