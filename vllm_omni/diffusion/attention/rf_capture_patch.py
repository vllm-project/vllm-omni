# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporary: make MindIE-SD's rf_v2 multi-video preprocessing capturable.

Graph-capturing the head-split attention region fails on the ``video_spans``
(multi-video, i.e. Ref2VA) path, and it fails in MindIE-SD's Python rather than
in the operator: with the attention op taken out entirely, the preprocessing
alone still raises ``aclnnNonzeroV2``. Two constructs are responsible, both in
``mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py`` (mindiesd 3.1.0):

* ``_multi_span_permutation`` selects the dense-context rows with
  ``all_indices[~is_video]``. Boolean-mask indexing lowers to ``aclnnNonzeroV2``,
  whose output shape is data dependent, so capture refuses it.
* ``get_multi_span_blockwise_mask`` builds ``torch.tensor(retained, device=...)``
  from a Python list on every call, a host-to-device copy that capture also
  refuses (``copy_between_host_and_device_opapi``, 107030).

Both are replaceable bit-exactly -- the selected rows are a pure function of the
``video_spans`` Python ints, and the second is the same tensor built one step
earlier -- so this module patches them in place rather than waiting on the
upstream fix. It is deliberately conservative: it verifies every symbol it
expects before touching anything, and a mismatch leaves MindIE-SD alone and
disables itself, because a wrong permutation would corrupt output silently.

Applied only when head-overlap graph capture is switched on.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False
_FAILED = False


def ensure_rf_capture_patch() -> bool:
    """Patch MindIE-SD's rf_v2 multi-video preprocessing. Idempotent.

    Returns True once the patch is in place, False when it could not be applied
    (in which case callers keep working -- capture simply stays unavailable for
    the multi-video path).
    """
    global _PATCHED, _FAILED
    if _PATCHED:
        return True
    if _FAILED:
        return False

    try:
        import mindiesd.layers.flash_attn.sparse_flash_attn as sparse_entry
        import mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 as rf_v2
    except Exception as exc:  # noqa: BLE001 - absent MindIE-SD must not break the caller
        _FAILED = True
        logger.warning("rf capture patch skipped: cannot import MindIE-SD sparse attention (%s)", exc)
        return False

    required = ("_multi_span_plan", "_span_rearranged_indices", "get_mask_index", "_multi_span_permutation")
    missing = [name for name in required if not hasattr(rf_v2, name)]
    if missing:
        _FAILED = True
        logger.warning(
            "rf capture patch skipped: this MindIE-SD build has no %s (expected on 3.1.0); "
            "the multi-video path stays uncapturable.",
            ", ".join(missing),
        )
        return False

    try:
        rf_v2._multi_span_permutation = _make_multi_span_permutation(rf_v2)
        patched_mask = _make_span_blockwise_mask(rf_v2)
        rf_v2.get_multi_span_blockwise_mask = patched_mask
        # sparse_attention binds this name with `from ... import`, so the caller's
        # reference has to be replaced too or the patch silently does nothing.
        sparse_entry.get_multi_span_blockwise_mask = patched_mask
    except Exception as exc:  # noqa: BLE001
        _FAILED = True
        logger.warning("rf capture patch failed (%s: %s); leaving MindIE-SD untouched", type(exc).__name__, exc)
        return False

    _PATCHED = True
    logger.info(
        "rf capture patch applied: MindIE-SD's multi-video permutation uses a closed-form dense-row "
        "gather instead of boolean indexing, and caches its retained-block index tensor"
    )
    return True


def _make_multi_span_permutation(rf_v2):
    """Closed form for ``all_indices[~is_video]``: the spans' complement, ascending."""
    plan = rf_v2._multi_span_plan
    span_indices = rf_v2._span_rearranged_indices

    def _multi_span_permutation(spans, sequence_len, pool_size, input_layout, device):
        segments, dense_blocks, first_frame_blocks = plan(spans, sequence_len, pool_size)

        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, _shape, length in spans:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = start + length
        if cursor < sequence_len:
            gaps.append((cursor, sequence_len))
        dense_indices = (
            torch.cat([torch.arange(lo, hi, device=device, dtype=torch.long) for lo, hi in gaps])
            if gaps
            else torch.empty(0, device=device, dtype=torch.long)
        )

        dense_cursor = 0
        parts = []
        for segment in segments:
            if segment[0] == "clip":
                parts.append(span_indices(segment[1], segment[2], input_layout, device))
            else:
                parts.append(dense_indices[dense_cursor : dense_cursor + segment[1]])
                dense_cursor += segment[1]

        permutation = torch.cat(parts)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(sequence_len, device=device, dtype=torch.long)
        return permutation, inverse, dense_blocks, first_frame_blocks

    return _multi_span_permutation


def _make_span_blockwise_mask(rf_v2):
    """Same mask; the retained-block index tensor is built once, not per call."""
    import math

    get_mask_index = rf_v2.get_mask_index
    retained_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def get_multi_span_blockwise_mask(
        qkv_pool, sparsity, scale, dense_blocks, first_frame_blocks, input_layout, return_binary=False
    ):
        query_pool, key_pool, _ = torch.chunk(qkv_pool, 3, dim=0)
        if input_layout == "BSND":
            scores = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool) * scale
        else:
            scores = torch.einsum("bnld,bnsd->bnls", query_pool, key_pool) * scale
        score_matrix = torch.nn.functional.softmax(scores, dim=-1)
        keep_len = max(1, math.ceil(score_matrix.shape[-1] * (1 - sparsity)))
        threshold = torch.topk(score_matrix, k=keep_len, dim=-1).values[..., -1:]
        mask = score_matrix >= threshold
        retained = sorted(dense_blocks | first_frame_blocks)
        if retained:
            cache_key = tuple(retained)
            retained_idx = retained_cache.get(cache_key)
            if retained_idx is None or retained_idx.device != mask.device:
                retained_idx = torch.tensor(retained, device=mask.device, dtype=torch.long)
                retained_cache[cache_key] = retained_idx
            mask[:, :, retained_idx, :] = True
            mask[:, :, :, retained_idx] = True
        if return_binary:
            return mask.to(torch.int8)
        select_idx = get_mask_index(mask)[0].transpose(0, 1)
        select_num_idx = mask[0].transpose(0, 1).sum(dim=-1)
        return select_idx, select_num_idx

    return get_multi_span_blockwise_mask
