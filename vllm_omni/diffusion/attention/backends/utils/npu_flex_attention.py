# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt request-owned block-causal metadata to MindIE-SD NPU FlexAttention.

Omni owns model selection, distributed placement and request lifetime. MindIE
owns mask construction, compilation, local execution and Ring accumulation.
This module never imports a vendor runtime for another device or FA path.
"""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, QueryRange

NPU_FLEX_SEQUENCE_LENGTH = "npu_flex_seq_len"
NPU_FLEX_VALID_TOKENS = "npu_flex_valid_tokens"


class _FlexRequest(NamedTuple):
    """Device-local state shared only by layers of one immutable prefill."""

    signature: tuple[object, ...]
    mask: object
    query_positions: torch.Tensor | None
    key_positions: torch.Tensor | None
    joint_key_positions: torch.Tensor | None


@cache
def load_mindie_flex_runtime() -> ModuleType:
    """Load the required API lazily, distinguishing missing and broken installs.

    Missing packages or entry points produce actionable ImportErrors. An error
    importing a transitive dependency retains its original type and cause.
    Successful API discovery does not compile or execute the kernel.
    """
    try:
        runtime = importlib.import_module("mindiesd")
    except ModuleNotFoundError as exc:
        if exc.name != "mindiesd":
            raise
        raise ImportError(
            "MindIE-SD with the FlexAttention API is required for attn_path='flex'. "
            "Install a compatible MindIE-SD build before starting NPU FlexAttention."
        ) from exc
    required = ("flex_attention", "parallel_flex_attention", "FlexAttentionMask", "make_block_causal_mask_mod")
    missing = [name for name in required if not callable(getattr(runtime, name, None))]
    if missing:
        raise ImportError(
            "Upgrade your MindIE-SD build to use NPU FlexAttention; "
            f"the installed package lacks these APIs: {', '.join(missing)}."
        )
    return runtime


def _can_use_flex(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata | None,
    *,
    ring_group: dist.ProcessGroup | None = None,
) -> bool:
    """Check the model integration contract before loading MindIE-SD.

    This adapter selects BF16 single-sample prefill. The public MindIE API has
    its own broader tensor contract. SP is supported after Omni resharding;
    leftover joint fields on the pure Ulysses path have already been consumed.
    """
    if (
        query.device.type != "npu"
        or metadata is None
        or query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or query.shape[0] != 1
        or key.shape[0] != 1
        or key.shape != value.shape
        or query.shape[-1] != key.shape[-1]
        or min(query.shape) <= 0
        or min(key.shape) <= 0
        or query.shape[2] % key.shape[2] != 0
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or key.device != query.device
        or value.device != query.device
        or metadata.full_attn_spans is None
        or len(metadata.full_attn_spans) != 1
        or metadata.attn_mask is not None
        or metadata.extra.get("kv_cache_dtype") not in (None, "float", "auto")
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return False
    global_len = metadata.extra.get(NPU_FLEX_SEQUENCE_LENGTH, key.shape[1])
    if not isinstance(global_len, int) or global_len <= 0:
        return False
    if ring_group is None and key.shape[1] != global_len:
        return False
    if ring_group is None and metadata.query_ranges is None and query.shape[1] != global_len:
        return False
    if ring_group is not None and metadata.query_ranges is not None:
        return False
    cursor = 0
    for start, end in metadata.full_attn_spans[0]:
        if not (cursor <= start < end <= global_len):
            return False
        cursor = end
    return True


def _query_positions(
    ranges: tuple[QueryRange, ...], q_len: int, device: torch.device
) -> torch.Tensor:
    """Map contiguous local query storage to possibly disjoint global ranges."""
    parts = []
    covered = 0
    for query_range in ranges:
        start, end, global_start = query_range.local_start, query_range.local_end, query_range.global_start
        if start != covered or end <= start or global_start < 0:
            raise ValueError("FlexAttention query_ranges must cover the local query contiguously.")
        parts.append(torch.arange(global_start, global_start + end - start, device=device))
        covered = end
    if covered != q_len:
        raise ValueError("FlexAttention query_ranges must cover the complete local query.")
    return torch.cat(parts)


def try_npu_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata | None,
    softmax_scale: float,
    *,
    ring_group: dist.ProcessGroup | None = None,
) -> torch.Tensor | None:
    """Execute MindIE Flex after Omni's Ulysses/AllGather preparation.

    A Ring group, when provided, owns the remaining KV exchange. Ulysses is
    never passed to MindIE here: Omni has already exchanged Q/K/V and will
    restore output shards. Unsupported model contracts return None; import,
    compilation, communication and execution failures propagate.
    """
    if not _can_use_flex(query, key, value, metadata, ring_group=ring_group):
        return None
    assert metadata is not None and metadata.full_attn_spans is not None
    runtime = load_mindie_flex_runtime()
    spans = tuple(metadata.full_attn_spans[0])
    global_len = metadata.extra.get(NPU_FLEX_SEQUENCE_LENGTH, key.shape[1])
    valid_tokens = metadata.extra.get(NPU_FLEX_VALID_TOKENS)
    rank = dist.get_rank(ring_group) if ring_group is not None else 0
    world_size = dist.get_world_size(ring_group) if ring_group is not None else 1
    joint_key = metadata.joint_key if ring_group is not None else None
    joint_value = metadata.joint_value if ring_group is not None else None
    joint_len = joint_key.shape[1] if joint_key is not None else 0
    joint_q_len = (
        metadata.joint_query.shape[1] if ring_group is not None and metadata.joint_query is not None else 0
    )
    if ring_group is not None and metadata.joint_strategy != "front":
        raise ValueError("Omni block-causal FlexAttention currently requires front joint KV under Ring SP.")
    if ring_group is not None and (
        (joint_key is None) != (joint_value is None)
        or joint_q_len not in (0, joint_len)
        or query.shape[1] != joint_q_len + key.shape[1]
        or global_len != joint_len + world_size * key.shape[1]
    ):
        raise ValueError("NPU FlexAttention Ring Q/KV layout does not match the global block-causal sequence.")
    # The validity tensor and callback captures are immutable through this
    # prefill. Never retain them in a process-wide cache.
    validity_version = (
        None if valid_tokens is None or valid_tokens.is_inference() else valid_tokens._version
    )
    signature = (
        spans, global_len, str(query.device), query.shape, key.shape,
        metadata.query_ranges, id(valid_tokens), validity_version, rank, world_size, joint_len, joint_q_len,
    )
    shared = metadata.extra.setdefault("_qwen_image21_flex_cache", {})
    request = shared.get("request")
    if request is None or request.signature != signature:
        mask_mod = runtime.make_block_causal_mask_mod(
            spans, global_len, device=query.device, valid_tokens=valid_tokens
        )
        mask = runtime.FlexAttentionMask(mask_mod, batch_shared=True, head_shared=True)
        q_positions = k_positions = joint_positions = None
        if ring_group is not None:
            k_positions = torch.arange(
                joint_len + rank * key.shape[1], joint_len + (rank + 1) * key.shape[1], device=query.device
            )
            joint_positions = torch.arange(joint_len, device=query.device)
            q_positions = torch.cat((joint_positions[:joint_q_len], k_positions))
        elif metadata.query_ranges is not None:
            q_positions = _query_positions(metadata.query_ranges, query.shape[1], query.device)
        request = _FlexRequest(signature, mask, q_positions, k_positions, joint_positions)
        shared["request"] = request
    return runtime.parallel_flex_attention(
        query, key, value, mask=request.mask, ulysses_group=None, ring_group=ring_group,
        scale=softmax_scale, layout="BSND", query_positions=request.query_positions,
        key_positions=request.key_positions, joint_key=joint_key, joint_value=joint_value,
        joint_key_positions=request.joint_key_positions, enable_gqa=query.shape[2] != key.shape[2],
    )
