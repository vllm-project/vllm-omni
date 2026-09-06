# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8 quantization utilities for diffusion attention tensors.

Provides per-tensor dynamic quantization of Q/K/V tensors to
float8_e4m3fn format. Designed for diffusion models where Q/K/V are
computed fresh each forward pass (no persistent KV cache).

Two entry points dispatch to the MindIE-SD FIA operator:
``fp8_rotate_quant_fa`` for dense batched layouts (BNSD/BSND) and
``fp8_rotate_quant_kv_slice`` for the packed [real, pad] layout with K/V
sliced to the valid prefix so a plain dense BNSD/BSND FIA call (no varlen
feature) suffices. The kv-slice path is the default behavior of
``--diffusion-kv-cache-dtype fp8`` on NPU for packed inputs (the legacy
``MINDIESD_FP8_KV_SLICE`` opt-in env is obsolete and ignored).

``fp8_rotate_quant_kv_slice`` also accepts a chunk plan from
``vllm_omni.diffusion.attention.chunking`` (duck-typed ``ChunkCall``
sequence; no runtime import so this file stays loadable outside the
``vllm_omni`` package): the single wide FIA call becomes several narrower
ones along the query sequence and/or head axes — a power-envelope
mitigation for specific machine types. Quantization happens once up
front; chunk boundaries align to the Q block-quant row block
(``_Q_BLOCK_SIZE``) so per-chunk dequant scales are exact slices of the
full-length scales and chunked results match the single call.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.chunking import ChunkCall

# Hadamard rotation matrix for QuaRot-style preprocessing
# keyed by (device, dtype, head_dim) to avoid matmul dtype mismatch.
_ROT_MATRIXS: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_ROT_MATRIX_LOCK = threading.Lock()

_FP8_KV_LABELS = frozenset({"fp8"})

# Block-quant row-block sizes for the FIA per-block FP8 path (Q: 128 rows,
# K/V: 256 rows).
_Q_BLOCK_SIZE = 128
_KV_BLOCK_SIZE = 256


def is_quantized_kv_cache(kv_cache_dtype: str | None) -> bool:
    """True if config requests FP8-style KV / QKV quantization for the NPU FA path."""
    return kv_cache_dtype in _FP8_KV_LABELS


@lru_cache(maxsize=1)
def _load_quant_ops():
    try:
        import torch_npu
        from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
        from msmodelslim.processor.quarot.common.quarot_utils import QuaRotMode, create_rot
    except ImportError as e:
        raise ImportError(
            "fp8_rotate_quant_fa requires torch_npu, MindIE-SD (mindiesd), and MSModelSlim. "
            "See https://gitcode.com/Ascend/MindIE-SD and https://gitcode.com/Ascend/msmodelslim"
        ) from e
    # The MindIE-SD FIA wrapper is only needed by fp8_rotate_quant_kv_slice;
    # keep it optional so the dense fp8_rotate_quant_fa path (which calls
    # torch_npu directly) works against MindIE-SD builds that do not provide
    # it yet.
    try:
        from mindiesd.layers.flash_attn.fused_infer_attention_score import (
            fused_infer_attention_score_v2,
        )
    except ImportError:
        fused_infer_attention_score_v2 = None
    return torch_npu, fused_infer_attention_score_v2, fa_block_quant_preprocess, QuaRotMode, create_rot


def _get_rot_matrix(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    qua_rot_mode,
    create_rot,
) -> torch.Tensor:
    key = (device, dtype, head_dim)
    with _ROT_MATRIX_LOCK:
        rot = _ROT_MATRIXS.get(key)
        if rot is None:
            rot = create_rot(qua_rot_mode.HADAMARD, head_dim, seed=425500).to(device=device, dtype=dtype)
            _ROT_MATRIXS[key] = rot
    return rot


def fp8_rotate_quant_fa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str = "BNSD",
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run NPU fused attention with dynamic FP8 Q/K/V and optional QuaRot preprocess.

    Args:
        query: Query tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        key: Key tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        value: Value tensor in ``layout`` order (default BNSD: batch, heads, seq, dim).
        layout: ``BNSD`` or ``BSND`` for ``npu_fused_infer_attention_score_v2``.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.

    Returns:
        Attention output in the same layout as inputs.
    """
    torch_npu, _fia_v2, fa_block_quant_preprocess, qua_rot_mode, create_rot = _load_quant_ops()

    out_dtype = query.dtype
    device = query.device

    if layout == "BNSD":
        _, n, s, d = query.shape
    elif layout == "BSND":
        _, s, n, d = query.shape
    else:
        raise ValueError(f"fp8_rotate_quant_fa: unsupported layout {layout!r}, expected BNSD or BSND")

    rot = _get_rot_matrix(device, query.dtype, d, qua_rot_mode, create_rot)
    q_f = torch.matmul(query, rot)
    k_f = torch.matmul(key, rot)

    q, q_scale = fa_block_quant_preprocess(q_f, block_size=128, dst_type=torch_npu.float8_e4m3fn, layout=layout)
    k, k_scale = fa_block_quant_preprocess(k_f, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)
    v, v_scale = fa_block_quant_preprocess(value, block_size=256, dst_type=torch_npu.float8_e4m3fn, layout=layout)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(d)

    out = torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        input_layout=layout,
        num_query_heads=n,
        softmax_scale=scale,
        pre_tokens=2147483647,  # INT32_MAX: no left-context truncation.
        next_tokens=2147483647,  # INT32_MAX: no right-context truncation.
        query_quant_mode=7,  # NPU mode id for block FP8 dequant path.
        key_quant_mode=7,  # Same quant mode as query branch.
        value_quant_mode=7,  # Same quant mode as key/query branches.
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        out_dtype=out_dtype,
    )[0]

    if out.shape[2] != s:
        if layout == "BNSD":
            out = out[:, :, :s, :]
        elif layout == "BSND":
            out = out[:, :s, :, :]

    return out


def fp8_rotate_quant_kv_slice(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_len: int,
    *,
    layout: str = "BSND",
    softmax_scale: float | None = None,
    plan: Sequence[ChunkCall] | None = None,
    chunk_callback=None,
) -> torch.Tensor | None:
    """Run dense NPU fused attention with dynamic FP8 Q/K/V after slicing K/V
    to the valid prefix.

    Alternative to :func:`fp8_rotate_quant_fa` for the packed [real, pad]
    two-document layout: the padding document is a strict suffix, so dropping
    it from K/V is identical to masking it out, and the FIA operator's varlen
    feature (``actual_seq_*`` / ``NTD_TND``) stays off — the call is a plain
    dense ``BNSD``/``BSND`` FIA whose query is longer than its K/V. K/V are
    sliced (zero-copy views) BEFORE rotation and quantization, so padding rows
    are neither attended nor quantized. Query keeps its full length; outputs
    on padding rows are never consumed downstream (same contract as the
    unquantized prefix-K/V-slice path).

    Query padding rows are quantized together with the real rows; per-block
    scales (128-row blocks) can therefore mix both in the boundary block.
    This is exact when the packing pads with zeros (zeros never raise the
    block absmax) — the MiniMax-H3 packing this path is built for.

    With ``plan`` (from
    ``vllm_omni.diffusion.attention.chunking.build_chunk_plan`` with
    ``row_align=_Q_BLOCK_SIZE``) the wide FIA call runs as several narrower
    ones over the same one-shot quantization — a power-envelope mitigation
    for specific machine types, not a general win. Calls are grouped per q
    chunk (the plan's q-chunk-major order); K/V head slices are materialized
    once and reused across q chunks. ``None`` (default) keeps the single
    wide call.

    Args:
        query: Query tensor in ``layout`` order; its seq length may exceed
            ``kv_len``.
        key: Key tensor in ``layout`` order; sliced to ``kv_len`` on the seq
            axis before quantization.
        value: Value tensor in ``layout`` order; sliced to ``kv_len`` on the seq
            axis before quantization.
        kv_len: Valid K/V prefix length (real document length of the packed
            row).
        layout: Caller-facing tensor layout, ``BNSD`` or ``BSND``. The FIA
            operator itself is always fed BNSD (the quant kernel's output
            layout) and its output is transposed back for ``BSND`` callers.
        softmax_scale: If None, uses ``1 / sqrt(head_dim)``.
        plan: Optional duck-typed ``ChunkCall`` sequence scheduling the FIA
            calls. Row boundaries must start on ``_Q_BLOCK_SIZE``-row blocks
            so per-chunk dequant scales are exact slices of the full-length
            scales.
        chunk_callback: Optional ``fn(out_chunk, call)`` invoked with each q
            chunk's head-merged output in the caller-facing layout. When set,
            nothing is reassembled and the return value is None (the caller
            consumes chunks, e.g. interleaving low-power communication
            between compute bursts).

    Returns:
        Attention output in the same layout as the inputs, at the query's
        full sequence length; None when ``chunk_callback`` consumed the
        per-chunk outputs.
    """
    torch_npu, fia_v2, fa_block_quant_preprocess, qua_rot_mode, create_rot = _load_quant_ops()
    if fia_v2 is None:
        raise ImportError(
            "fp8_rotate_quant_kv_slice requires MindIE-SD with "
            "fused_infer_attention_score_v2 (mindiesd.layers.flash_attn.fused_infer_attention_score); "
            "the installed MindIE-SD does not provide it."
        )

    if layout == "BNSD":
        _, num_heads, seq_len, head_dim = query.shape
        kv_seq_dim = 2
        num_kv_heads = key.shape[1]
    elif layout == "BSND":
        _, seq_len, num_heads, head_dim = query.shape
        kv_seq_dim = 1
        num_kv_heads = key.shape[2]
    else:
        raise ValueError(f"fp8_rotate_quant_kv_slice: unsupported layout {layout!r}, expected BNSD or BSND")

    kv_total = key.shape[kv_seq_dim]
    if not isinstance(kv_len, int) or not 0 < kv_len <= kv_total:
        raise ValueError(f"fp8_rotate_quant_kv_slice: kv_len must be an int in (0, {kv_total}], got {kv_len!r}")

    out_dtype = query.dtype
    device = query.device

    rot = _get_rot_matrix(device, query.dtype, head_dim, qua_rot_mode, create_rot)
    q_f = torch.matmul(query, rot)
    # Slice K/V to the valid prefix (zero-copy views) before rotation and
    # quantization: pad rows are neither attended nor quantized.
    key = key.narrow(kv_seq_dim, 0, kv_len)
    value = value.narrow(kv_seq_dim, 0, kv_len)
    k_f = torch.matmul(key, rot)

    # fa_block_quant_preprocess always returns BNSD-logical tensors (BSND
    # inputs are transposed before the quant kernel), so the FIA call is
    # always dispatched with input_layout="BNSD" and the output is transposed
    # back to the caller's layout below.
    q, q_scale = fa_block_quant_preprocess(
        q_f, block_size=_Q_BLOCK_SIZE, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    k, k_scale = fa_block_quant_preprocess(
        k_f, block_size=_KV_BLOCK_SIZE, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )
    v, v_scale = fa_block_quant_preprocess(
        value, block_size=_KV_BLOCK_SIZE, dst_type=torch_npu.float8_e4m3fn, layout=layout
    )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)

    if plan is None:
        out = fia_v2(
            q,
            k,
            v,
            input_layout="BNSD",
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            softmax_scale=scale,
            pre_tokens=2147483647,  # INT32_MAX: no left-context truncation.
            next_tokens=2147483647,  # INT32_MAX: no right-context truncation.
            query_quant_mode=7,  # NPU mode id for block FP8 dequant path.
            key_quant_mode=7,  # Same quant mode as query branch.
            value_quant_mode=7,  # Same quant mode as key/query branches.
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=out_dtype,
        )[0]
        # The op hands back BNSD-logical output, possibly padded on the seq
        # axis: trim to the query length, then transpose back for BSND.
        if out.shape[2] != seq_len:
            out = out[:, :, :seq_len, :]
        if layout == "BSND":
            out = out.transpose(1, 2)
        return out

    # Chunked dispatch. The plan is q-chunk-major: group consecutive calls
    # sharing a row range into one q chunk, run each of its head slices
    # against the materialized K/V head part, merge on the head axis, then
    # concatenate q chunks on the layout's sequence axis. q may be
    # block-padded by the quant kernel beyond seq_len; the plan only
    # schedules real rows, and each call's output still trims the op's own
    # padding.
    chunks: list[tuple[tuple[int, int], list[ChunkCall]]] = []
    for call in plan:
        rows = (call.row0, call.row1)
        if chunks and chunks[-1][0] == rows:
            chunks[-1][1].append(call)
        else:
            chunks.append((rows, [call]))

    # Head chunking: materialize K/V head slices (and their scales) once and
    # reuse them across q chunks — the per-call K/V footprint drops to
    # L2-resident sizes, which is the point. Head chunking is MHA-only (the
    # plan builder collapses GQA), so a slice's kv heads equal its q heads;
    # without head chunking the real GQA counts are used below.
    head_slices = sorted({(call.h0, call.h1) for call in plan})
    if len(head_slices) > 1:
        kv_head_parts = {
            (h0, h1): (
                k[:, h0:h1].contiguous(),
                v[:, h0:h1].contiguous(),
                k_scale[:, h0:h1].contiguous(),
                v_scale[:, h0:h1].contiguous(),
            )
            for h0, h1 in head_slices
        }
        del k, v, k_scale, v_scale

    out_parts = []
    for (row0, row1), calls in chunks:
        real_rows = min(row1, seq_len) - row0
        if real_rows <= 0:
            continue  # defensive: plans built over seq_len never emit these
        if row0 % _Q_BLOCK_SIZE != 0:
            raise ValueError(
                "fp8_rotate_quant_kv_slice: plan row boundaries must start on "
                f"{_Q_BLOCK_SIZE}-row blocks (got row0={row0}); build the plan with "
                "row_align=_Q_BLOCK_SIZE so per-chunk dequant scales are exact "
                "slices of the full-length scales."
            )
        head_parts = []
        for call in calls:
            h0, h1 = call.h0, call.h1
            if len(head_slices) > 1:
                k_c, v_c, ks_c, vs_c = kv_head_parts[(h0, h1)]
                kv_heads_c = h1 - h0
            else:
                k_c, v_c, ks_c, vs_c = k, v, k_scale, v_scale
                kv_heads_c = num_kv_heads
            out_hc = fia_v2(
                q[:, h0:h1, row0:row1, :].contiguous(),
                k_c,
                v_c,
                input_layout="BNSD",
                num_query_heads=h1 - h0,
                num_key_value_heads=kv_heads_c,
                softmax_scale=scale,
                pre_tokens=2147483647,  # INT32_MAX: no left-context truncation.
                next_tokens=2147483647,  # INT32_MAX: no right-context truncation.
                query_quant_mode=7,  # NPU mode id for block FP8 dequant path.
                key_quant_mode=7,  # Same quant mode as query branch.
                value_quant_mode=7,  # Same quant mode as key/query branches.
                # Per-chunk Q scale slice: block-aligned boundaries make this
                # the exact block range of the full-length quantization.
                dequant_scale_query=q_scale[
                    :, h0:h1, row0 // _Q_BLOCK_SIZE : -(-row1 // _Q_BLOCK_SIZE), :
                ].contiguous(),
                dequant_scale_key=ks_c,
                dequant_scale_value=vs_c,
                out_dtype=out_dtype,
            )[0]
            # The op may hand back a padded seq axis; keep this call's real rows.
            if out_hc.shape[2] != real_rows:
                out_hc = out_hc[:, :, :real_rows, :]
            head_parts.append(out_hc)
        # Reassemble heads (BNSD head axis) before the caller layout transpose.
        out_c = torch.cat(head_parts, dim=1) if len(head_parts) > 1 else head_parts[0]
        if layout == "BSND":
            out_c = out_c.transpose(1, 2)
        if chunk_callback is not None:
            # The caller consumes each q chunk (e.g. interleaved reverse
            # communication); nothing is reassembled here.
            chunk_callback(out_c, calls[0])
        else:
            out_parts.append(out_c)
    if chunk_callback is not None:
        return None
    # Chunks were already transposed to the caller layout in the loop; just
    # concatenate along the layout's sequence axis.
    seq_dim = 1 if layout == "BSND" else 2
    out = torch.cat(out_parts, dim=seq_dim) if len(out_parts) > 1 else out_parts[0]
    return out
