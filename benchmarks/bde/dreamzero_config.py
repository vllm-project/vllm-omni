# SPDX-License-Identifier: Apache-2.0
"""Derive a parity-faithful ``BDEKVConfig`` from DreamZero geometry.

For KV-on to match the model-local ``[-max_attention_size:]`` window exactly, the
BDE window must equal ``max_attention_size = local_attn_size * frame_seqlen``. See
``BDE_doc/dreamzero_kv_phase1_profiling_accuracy.md`` §1.
"""

from __future__ import annotations

from vllm_omni.experimental.bde.kv_cache import BDEKVConfig


def bde_config_for_dreamzero(
    *,
    num_frame_per_block: int,
    frame_seqlen: int,
    local_attn_size: int,
    gpu_memory_fraction: float = 0.1,
) -> BDEKVConfig:
    """Build the parity-faithful ``BDEKVConfig`` for DreamZero.

    Frame-granular paging (matches ``BDEModelRunner._preallocate_kv_cache``): one
    pool block = one frame, so the resident window equals ``max_attention_size``
    exactly without rounding to whole causal blocks.

    ``chunk_size      = frame_seqlen``
    ``window (tokens) = local_attn_size * frame_seqlen   (== max_attention_size)``
    ``window_chunks   = local_attn_size``

    ``num_frame_per_block`` no longer governs paging (it sets the reset cadence); it
    is kept as an argument for call-site compatibility.
    """
    if num_frame_per_block <= 0 or frame_seqlen <= 0:
        raise ValueError("num_frame_per_block and frame_seqlen must be > 0")
    if local_attn_size <= 0:
        raise ValueError(
            f"local_attn_size must be > 0 for a bounded BDE window (got {local_attn_size}); "
            "DreamZero -1 (full attention) needs an explicit window size for Phase 1"
        )

    chunk_size = frame_seqlen
    window_chunks = local_attn_size
    config = BDEKVConfig(
        enable=True,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        sink_chunks=0,
        reset_at_boundary=False,
        gpu_memory_fraction=gpu_memory_fraction,
    )
    # Invariant: the BDE window in tokens equals DreamZero's max_attention_size.
    assert config.sliding_window == local_attn_size * frame_seqlen
    return config
