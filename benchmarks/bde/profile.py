# SPDX-License-Identifier: Apache-2.0
"""Profiling utilities for BDE DreamZero (plan §3).

``kv_memory_plateau`` is runnable without weights — it drives a ``BDEKVCache`` over
many chunks and records pool occupancy, demonstrating the bounded-memory win (KV
plateaus at the window vs the model-local grow-then-slice). Latency breakdown
(allocate / write / gather / attention) requires the weighted model and is
collected via ``record_function`` hooks during a real rollout — see the plan.
"""

from __future__ import annotations

from pathlib import Path

import torch

from vllm_omni.experimental.bde.kv_cache import BDEKVCache, BDEKVConfig


def kv_memory_plateau(
    config: BDEKVConfig,
    *,
    num_chunks: int,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype = torch.float16,
    max_model_len: int = 1 << 20,
    available_bytes: int = 1 << 32,
    device: torch.device | None = None,
) -> dict:
    """Drive a ``BDEKVCache`` over ``num_chunks`` and record pool occupancy.

    Returns per-chunk free / used / resident block counts and whether usage
    plateaued once the window filled.
    """
    kv = BDEKVCache(
        config,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        block_size=block_size,
        max_model_len=max_model_len,
        available_bytes=available_bytes,
        device=device,
    )
    total = kv.manager.block_pool.get_num_free_blocks()
    adapter = kv.begin_request("profile")
    free, resident = [], []
    for _ in range(num_chunks):
        kv.allocate_chunk(adapter)
        free.append(kv.manager.block_pool.get_num_free_blocks())
        resident.append(len(kv.window_block_ids(adapter)))
        kv.commit_chunk(adapter)
    kv.end_request(adapter)

    used = [total - f for f in free]
    window = config.window_chunks
    plateaued = used[-1] == used[window] if (window is not None and num_chunks > window) else None
    return {
        "num_chunks": num_chunks,
        "total_blocks": total,
        "window_chunks": window,
        "free": free,
        "used_blocks": used,
        "resident_blocks": resident,
        "peak_used": max(used),
        "plateaued": plateaued,
    }


def plot_memory(path, profile: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(profile["used_blocks"], label="pool blocks used (BDE)")
    ax.plot(profile["resident_blocks"], label="resident (window) blocks")
    if profile["window_chunks"] is not None:
        ax.axhline(profile["window_chunks"] + 1, ls="--", c="g", label="window + 1")
    ax.set_xlabel("chunk")
    ax.set_ylabel("blocks")
    ax.set_title("BDE KV pool occupancy (bounded by window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
