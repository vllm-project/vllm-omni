# SPDX-License-Identifier: Apache-2.0
"""Conditioning-neutral chunk, cache-accounting, and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest


def iter_ar_chunk_ranges(start_frame: int, num_frames: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield the training-aligned partition ``[1, C, C, ...]``."""
    if start_frame < 0 or num_frames < 0 or start_frame > num_frames:
        raise ValueError(f"Invalid Cosmos-Dreams frame range [{start_frame}, {num_frames})")
    if chunk_size <= 0:
        raise ValueError(f"Cosmos-Dreams chunk_size must be positive, got {chunk_size}")
    frame = start_frame
    while frame < num_frames:
        if frame == 0:
            chunk_end = 1
        else:
            chunk_end = 1 + ((frame - 1) // chunk_size + 1) * chunk_size
        chunk_end = min(chunk_end, num_frames)
        yield frame, chunk_end
        frame = chunk_end


def iter_clean_commit_frames(
    chunk_start: int,
    chunk_end: int,
    *,
    target_frame: int,
    terminal_request: bool,
) -> Iterator[tuple[int, int]]:
    """Yield ``(local, absolute)`` clean-refresh frames in commit order.

    The globally final frame has no downstream reader and is omitted only for
    a terminal request. Every other frame is refreshed individually so later
    frames in the same denoised chunk see clean, committed history.
    """
    if chunk_start < 0 or chunk_end <= chunk_start or target_frame < chunk_end:
        raise ValueError(
            f"Invalid Cosmos-Dreams clean-commit range: chunk=[{chunk_start}, {chunk_end}), target={target_frame}"
        )
    for local_idx, frame_idx in enumerate(range(chunk_start, chunk_end)):
        if terminal_request and frame_idx == target_frame - 1:
            continue
        yield local_idx, frame_idx


def prompt_token_hash(token_ids: Sequence[int] | torch.Tensor) -> str:
    """Stable SHA-256 over prompt token IDs, independent of tensor dtype."""
    if isinstance(token_ids, torch.Tensor):
        values = [int(value) for value in token_ids.detach().cpu().reshape(-1).tolist()]
    else:
        values = [int(value) for value in token_ids]
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CosmosDreamsKVMemoryEstimate:
    page_bytes: int
    managed_blocks: int
    scratch_blocks: int
    self_attention_bytes: int
    scratch_bytes: int
    cross_attention_bytes: int
    total_bytes: int


def estimate_kv_memory_bytes(
    manifest: CosmosDreamsManifest,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    num_local_kv_branches: int = 1,
    num_logical_kv_branches: int = 1,
    session_capacity: int = 1,
    frames_per_block: int = 1,
    max_scratch_tokens_per_branch: int = 0,
) -> CosmosDreamsKVMemoryEstimate:
    """Estimate the manager floor, scratch reservation, and text pools."""
    positive = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "num_local_kv_branches": num_local_kv_branches,
        "num_logical_kv_branches": num_logical_kv_branches,
        "session_capacity": session_capacity,
        "frames_per_block": frames_per_block,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"Cosmos-Dreams KV estimate {name} must be positive, got {value}")
    if max_scratch_tokens_per_branch < 0:
        raise ValueError("Cosmos-Dreams max_scratch_tokens_per_branch must be non-negative")
    page_bytes = int(2 * manifest.tokens_per_frame * num_kv_heads * head_size * dtype.itemsize * num_layers)
    managed_blocks = num_local_kv_branches * (manifest.sink_frames + manifest.window_frames + frames_per_block) + 2
    scratch_per_branch = frames_per_block + math.ceil(max_scratch_tokens_per_branch / manifest.tokens_per_frame)
    scratch_blocks = num_local_kv_branches * scratch_per_branch
    self_attention_bytes = managed_blocks * page_bytes
    scratch_bytes = scratch_blocks * page_bytes
    cross_attention_bytes = int(
        2
        * session_capacity
        * num_logical_kv_branches
        * manifest.text_cache_max_len
        * num_kv_heads
        * head_size
        * dtype.itemsize
        * num_layers
    )
    return CosmosDreamsKVMemoryEstimate(
        page_bytes=page_bytes,
        managed_blocks=managed_blocks,
        scratch_blocks=scratch_blocks,
        self_attention_bytes=self_attention_bytes,
        scratch_bytes=scratch_bytes,
        cross_attention_bytes=cross_attention_bytes,
        total_bytes=self_attention_bytes + scratch_bytes + cross_attention_bytes,
    )
