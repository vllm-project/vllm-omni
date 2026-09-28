# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MOSS tile64 launch policy; reuse the upstream attention GPU kernels.

The launch wrapper is derived from vLLM's triton_unified_attention at
ced6857afa0ea7b2e3f0846a62e1394e90f15607. Only the KV tile and launch
parameters differ. backbone_attention restricts this path to MOSS BF16.
"""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_unified_attention import (
    _get_tile_size,
    is_batch_invariant,
    kernel_unified_attention,
    reduce_segments,
)
from vllm.v1.kv_cache_interface import KVQuantMode


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,  # noqa: N803 - upstream launch API
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    # R-SWA support: prefix tokens stay globally visible, generated tokens use
    # a fixed sliding window.
    rswa_prefix_lens=None,
    rswa_window: int | None = None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
    # Tensor-descriptor mode: use ``tl.make_tensor_descriptor`` for Q/K/V
    # loads and output stores.  Enables HW 2D block reads on Intel Xe2/Xe3.
    # The non-TD branch is dead-code-eliminated at Triton compile time so
    # disabling this flag costs nothing.
    use_td: bool = False,
    # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window.
    # Default False keeps the original behavior for every other model.
    mm_prefix_clamp_sliding_window: bool = False,
):
    # Resolve causal: bool or per-seq tensor.
    use_per_seq_causal = isinstance(causal, torch.Tensor)
    use_causal = bool(causal) if not use_per_seq_causal else True
    per_seq_causal_ptr = causal if use_per_seq_causal else None

    # Sub-byte packed mode (INT4) needs a bespoke kernel (split-dot +
    # sub-byte unpack); everything else goes through the core kernel below.
    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        assert use_causal and not use_per_seq_causal, "INT4_PER_TOKEN_HEAD only supports causal attention"
        from vllm.v1.attention.ops.int4_per_token_head import (
            unified_attention_int4,
        )

        if sinks is not None:
            assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"
        unified_attention_int4(
            q=q,
            k_cache=k,
            v_cache=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            sinks=sinks,
            alibi_slopes=alibi_slopes,
            use_alibi_sqrt=use_alibi_sqrt,
            qq_bias=qq_bias,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
        )
        return

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}")

    use_rswa = rswa_window is not None and rswa_prefix_lens is not None

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = 4
    launch_num_stages: int | None = 2

    # head_size 256 with many query rows per sequence (e.g. diffusion-gemma
    # bidirectional canvas passes) is prefill-shaped, but the decode-oriented
    # defaults (BLOCK_Q=8, TILE=32, 4 warps) under-tile it. A wider KV tile +
    # more query rows per block + 8 warps is ~2x faster on B200.
    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and current_platform.is_device_capability_family(100)
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(head_size, sliding_window_val, q.element_size(), is_prefill=True)
    TILE_SIZE_DECODE = _get_tile_size(head_size, sliding_window_val, q.element_size(), is_prefill=False)

    # Wider KV tile for the tuned large-head path (see above). Only the 2D
    # path (used when max_seqlen_q > 1) reads TILE_SIZE_PREFILL.
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128

    TILE_SIZE_PREFILL = 64
    TILE_SIZE_DECODE = 64

    # USE_TD requires BLOCK_SIZE % TILE_SIZE == 0 (enforced by a
    # ``tl.static_assert`` in the kernel).  The default prefill tile
    # size (32) is larger than a common ``block_size=16``, so clamp it
    # down when TD is enabled.  Zero overhead when disabled.
    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)

    # Tensor descriptors for Q load / output store require every element
    # of ``block_shape`` to be a power of 2.  ``num_queries_per_kv`` is
    # not always pow2 (e.g. Qwen2-7B: 28 / 4 = 7), so gate the Q/O paths
    # separately from the KV tile loads (whose ``block_shape`` does not
    # include ``num_queries_per_kv``).
    #
    # The Q/O descriptors also encode ``HEAD_SIZE_PADDED`` on the inner
    # axis while the backing buffers (both flat output and per-segment
    # output) are laid out with ``HEAD_SIZE``.  When they differ (e.g.
    # Phi-3's head_size=96 → HEAD_SIZE_PADDED=128) the store would spill
    # padded lanes into neighbouring heads because tensor-descriptor
    # stores don't mask the padded tail.  Fall back to the pointer path
    # for Q/O in that case — KV tile loads are unaffected because their
    # ``shape`` already matches ``block_shape`` on the inner axis.
    head_size_padded = triton.next_power_of_2(head_size)
    _is_pow2_nq = (num_queries_per_kv & (num_queries_per_kv - 1)) == 0
    _is_pow2_hs = head_size == head_size_padded
    use_td_qo = use_td and _is_pow2_nq and _is_pow2_hs

    # ``_load_q_td`` / ``_store_output_td`` flatten ``(num_queries_per_kv,
    # HEAD_SIZE)`` into a single contiguous inner axis.  That's only
    # equivalent to the pointer path when the ``num_queries_per_kv`` heads
    # for this KV group start at ``kv_head_idx * num_queries_per_kv`` and
    # lie exactly HEAD_SIZE apart — i.e. ``query_stride_1 == HEAD_SIZE``
    # and ``output_stride_1 == head_size``.  This is the default vLLM
    # query/output layout; assert it explicitly so we fail fast if a
    # future caller passes a non-contiguous query tensor.
    if use_td_qo:
        assert q.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous query heads "
            f"(q.stride(1) = {q.stride(1)} != head_size = {head_size}); "
            f"set VLLM_TRITON_USE_TD=0 or pad the query layout."
        )
        assert out.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous output heads (out.stride(1) = {out.stride(1)} != head_size = {head_size})."
        )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are passed as ``None`` when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead so Triton can skip
    # materialising those arguments and the associated registers.
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = None
        vs_blk = vs_slot = vs_head = None
        k_scale_ptr = None
        v_scale_ptr = None
    # 3D needs real segm tensors; 2D never touches them.  Pass ``None`` in
    # 2D mode so Triton can skip materialising these pointer arguments.
    segm_output_ptr = softmax_segm_output if use_3d else None
    segm_max_ptr = softmax_segm_max if use_3d else None
    segm_expsum_ptr = softmax_segm_expsum if use_3d else None
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[int, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    launch_kwargs: dict[str, int] = {}
    if launch_num_warps is not None:
        launch_kwargs["num_warps"] = launch_num_warps
    if launch_num_stages is not None:
        launch_kwargs["num_stages"] = launch_num_stages

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=use_per_seq_causal,
        per_seq_causal_ptr=per_seq_causal_ptr,
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        rswa_prefix_lens_ptr=rswa_prefix_lens if use_rswa else seqused_k,
        R_SWA_WINDOW=rswa_window or 0,
        USE_R_SWA=use_rswa,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        Q_IS_FP8=(q.dtype == current_platform.fp8_dtype()),
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        USE_TD=use_td,
        USE_TD_QO=use_td_qo,
        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,
        **launch_kwargs,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
