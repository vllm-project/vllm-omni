# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility helpers for KV cache manipulation, TP routing, and merge/slice."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger

from .initialization import KV_RANK_PORT_STRIDE, KV_TRANSFER_PORT_OFFSET

logger = init_logger(__name__)

LayerKV = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


# ------------------------------------------------------------------ #
#  TP Topology
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class KVTPTopology:
    """Immutable descriptor for a KV-transfer parallel mapping.

    Captures sender/receiver parallel sizes and the local rank within
    that parallel dimension.  Works for any divisible parallel dimension
    (TP, SP, Ring Attention).
    """

    source_tp_size: int
    target_tp_size: int
    local_rank: int

    def __post_init__(self) -> None:
        if self.source_tp_size <= 0 or self.target_tp_size <= 0:
            raise ValueError(
                f"Parallel sizes must be positive: "
                f"source_tp_size={self.source_tp_size}, target_tp_size={self.target_tp_size}"
            )
        if self.local_rank < 0:
            raise ValueError(f"local_rank must be non-negative, got {self.local_rank}")

    @property
    def is_heterogeneous(self) -> bool:
        return self.source_tp_size != self.target_tp_size

    @property
    def ratio(self) -> int:
        """Larger parallel size divided by smaller. Always >= 1."""
        return max(self.source_tp_size, self.target_tp_size) // min(self.source_tp_size, self.target_tp_size)


# ------------------------------------------------------------------ #
#  Runtime TP detection
# ------------------------------------------------------------------ #


def get_local_tp_rank() -> int:
    """Return the TP-local rank of this worker process.

    Uses ``get_tensor_model_parallel_rank()`` which returns the rank
    within the TP group only, not the stage-global rank.
    """
    try:
        return get_tensor_model_parallel_rank()
    except Exception:
        logger.debug("TP parallel state not initialized, falling back to LOCAL_RANK env", exc_info=True)
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except (ValueError, TypeError):
        return 0


def get_tp_world_size() -> int:
    """Return the TP world size (tensor-parallel dimension only).

    Uses ``get_tensor_model_parallel_world_size()`` so that
    cfg_parallel, SP, PP etc. are not included in the count.
    """
    try:
        return get_tensor_model_parallel_world_size()
    except Exception:
        logger.debug("TP parallel state not initialized, defaulting world_size=1", exc_info=True)
    return 1


# ------------------------------------------------------------------ #
#  ZMQ port computation
# ------------------------------------------------------------------ #


def kv_zmq_port(base_port: int, from_stage: int, local_rank: int = 0) -> int:
    """Compute the ZMQ port for a KV-transfer connector.

    Each TP rank gets its own port so that TP > 1 deployments do not
    cause ``EADDRINUSE`` when multiple sender workers bind on the same
    host.  The formula is backward-compatible: rank 0 produces the same
    port as the previous ``base + OFFSET + stage`` formula.
    """
    return base_port + KV_TRANSFER_PORT_OFFSET + local_rank * KV_RANK_PORT_STRIDE + from_stage


# ------------------------------------------------------------------ #
#  TP topology validation and rank routing
# ------------------------------------------------------------------ #


def validate_kv_tp_topology(topo: KVTPTopology) -> None:
    """Reject heterogeneous TP mappings that cannot be routed losslessly."""
    larger = max(topo.source_tp_size, topo.target_tp_size)
    smaller = min(topo.source_tp_size, topo.target_tp_size)
    if larger % smaller != 0:
        raise ValueError(
            f"KV TP mapping must be divisible: "
            f"source_tp_size={topo.source_tp_size}, "
            f"target_tp_size={topo.target_tp_size}"
        )


def get_kv_target_ranks(topo: KVTPTopology) -> list[int]:
    """Which remote ranks this local rank sends KV shards to (send side)."""
    validate_kv_tp_topology(topo)
    if topo.source_tp_size == topo.target_tp_size:
        return [topo.local_rank]
    if topo.source_tp_size > topo.target_tp_size:
        return [topo.local_rank // (topo.source_tp_size // topo.target_tp_size)]
    ratio = topo.target_tp_size // topo.source_tp_size
    return [topo.local_rank * ratio + i for i in range(ratio)]


def get_kv_source_ranks(topo: KVTPTopology) -> list[int]:
    """Which remote ranks this local rank receives KV shards from (recv side)."""
    validate_kv_tp_topology(topo)
    if topo.source_tp_size == topo.target_tp_size:
        return [topo.local_rank]
    if topo.source_tp_size > topo.target_tp_size:
        ratio = topo.source_tp_size // topo.target_tp_size
        return [topo.local_rank * ratio + i for i in range(ratio)]
    return [topo.local_rank // (topo.target_tp_size // topo.source_tp_size)]


# ------------------------------------------------------------------ #
#  Rank-aware connector key building
# ------------------------------------------------------------------ #


def get_kv_connector_key(
    req_id: str,
    from_stage: int | str,
    chunk_id: int,
    from_rank: int,
    to_rank: int,
) -> str:
    """Build connector key that includes rank info for KV transfers.

    Format matches PR #2677: ``{req_id}_{from_stage}_{chunk_id}_{from_rank}_{to_rank}``
    """
    return f"{req_id}_{from_stage}_{chunk_id}_{from_rank}_{to_rank}"


def build_rank_aware_send_keys(
    request_id: str,
    from_stage: str,
    to_stage: str,
    topo: KVTPTopology,
    hook: Callable | None = None,
) -> list[str]:
    """Build send-side connector keys, checking injectable hook first."""
    if callable(hook):
        keys = list(hook(request_id, from_stage, to_stage))
        if keys:
            return keys
    if topo.source_tp_size <= 1 and topo.target_tp_size <= 1:
        return [f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}"]
    target_ranks = get_kv_target_ranks(topo)
    return [get_kv_connector_key(request_id, from_stage, 0, topo.local_rank, r) for r in target_ranks]


def build_rank_aware_recv_keys(
    request_id: str,
    from_stage: str,
    to_stage: str,
    topo: KVTPTopology,
    hook: Callable | None = None,
) -> list[tuple[str, int | None]]:
    """Build recv-side connector keys with sender rank info.

    Returns a list of ``(key, from_rank)`` tuples.  ``from_rank`` is
    ``None`` when TP <= 1 (single sender, no per-rank routing needed).
    For TP > 1, ``from_rank`` identifies which sender rank owns the
    key so that the connector can route metadata queries to the
    correct endpoint.
    """
    if callable(hook):
        raw = list(hook(request_id, from_stage, to_stage))
        if raw:
            if isinstance(raw[0], tuple):
                return raw
            # Hook returned plain strings (e.g. OmniConnectorModelRunnerMixin.
            # get_rank_aware_kv_keys). Reconstruct from_rank from topology so
            # Mooncake connector can route metadata queries to the correct
            # sender endpoint in heterogeneous TP.
            # TODO: have the mixin return (key, from_rank) tuples directly
            # to avoid this indirect reconstruction.
            source_ranks = get_kv_source_ranks(topo)
            if len(raw) == len(source_ranks):
                return list(zip(raw, source_ranks))
            return [(k, None) for k in raw]
    if topo.source_tp_size <= 1 and topo.target_tp_size <= 1:
        return [(f"omni_{from_stage}_to_{to_stage}_kv_cache_{request_id}", None)]
    source_ranks = get_kv_source_ranks(topo)
    return [(get_kv_connector_key(request_id, from_stage, 0, r, topo.local_rank), r) for r in source_ranks]


# ------------------------------------------------------------------ #
#  KV tensor head slicing (heterogeneous TP)
# ------------------------------------------------------------------ #


def slice_kv_tensor_heads(
    tensor: torch.Tensor | None,
    offset_in_shard: int,
    num_slices: int,
) -> torch.Tensor | None:
    """Slice one KV tensor along its head dimension (dim 1)."""
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.dim() < 2:
        raise ValueError(f"Expected KV tensor with a head dimension, got shape={tuple(tensor.shape)}")
    if num_slices <= 0:
        raise ValueError(f"num_slices must be > 0, got {num_slices}")
    if not (0 <= offset_in_shard < num_slices):
        raise ValueError(f"offset_in_shard must be in [0, {num_slices}), got {offset_in_shard}")

    heads_in_shard = tensor.shape[1]
    if heads_in_shard % num_slices != 0:
        raise ValueError(
            "KV head count must be divisible for heterogeneous TP slicing: "
            f"heads_in_shard={heads_in_shard}, num_slices={num_slices}"
        )

    heads_per_slice = heads_in_shard // num_slices
    start = offset_in_shard * heads_per_slice
    end = start + heads_per_slice
    return tensor[:, start:end, ...].contiguous()


def slice_layer_blocks(
    layer_blocks: dict[str, Any],
    offset_in_shard: int,
    num_slices: int,
) -> dict[str, list[torch.Tensor | None]]:
    """Slice all KV layers for one logical receiver rank."""
    sliced_blocks: dict[str, list[torch.Tensor | None]] = {}
    for cache_name in ("key_cache", "value_cache"):
        cache_list = layer_blocks.get(cache_name, [])
        sliced_blocks[cache_name] = [
            slice_kv_tensor_heads(tensor, offset_in_shard, num_slices) for tensor in cache_list
        ]
    return sliced_blocks


# ------------------------------------------------------------------ #
#  Multi-rank merge and receiver-side slice
# ------------------------------------------------------------------ #


def merge_received_rank_shards(
    payloads: list[dict[str, Any]],
    merger: Callable | None = None,
) -> dict[str, Any] | None:
    """Merge multiple source-rank KV shards for one target rank.

    When *merger* is provided (injectable hook), it is called directly.
    Otherwise the default merges along the head dimension (dim 1).
    """
    if callable(merger):
        return merger(payloads)
    if not payloads:
        return None
    if len(payloads) == 1:
        return payloads[0]

    base_payload = payloads[0]
    if not isinstance(base_payload, dict) or "layer_blocks" not in base_payload:
        return base_payload

    merged: dict[str, Any] = {
        "request_id": base_payload.get("request_id"),
        "block_ids": list(base_payload.get("block_ids", [])),
        "metadata": dict(base_payload.get("metadata", {})),
    }
    merged_layer_blocks: dict[str, list[torch.Tensor | None]] = {}

    for cache_name in ("key_cache", "value_cache"):
        cache_lists = [payload.get("layer_blocks", {}).get(cache_name, []) for payload in payloads]
        num_layers = max((len(cache_list) for cache_list in cache_lists), default=0)
        merged_cache: list[torch.Tensor | None] = []

        for layer_idx in range(num_layers):
            layer_tensors = [
                cache_list[layer_idx]
                for cache_list in cache_lists
                if layer_idx < len(cache_list) and cache_list[layer_idx] is not None
            ]
            if not layer_tensors:
                merged_cache.append(None)
            elif len(layer_tensors) == 1 or not isinstance(layer_tensors[0], torch.Tensor):
                merged_cache.append(layer_tensors[0])
            else:
                merged_cache.append(torch.cat(layer_tensors, dim=1).contiguous())

        merged_layer_blocks[cache_name] = merged_cache

    merged["layer_blocks"] = merged_layer_blocks
    return merged


def slice_received_rank_shard(
    payload: dict[str, Any] | None,
    topo: KVTPTopology,
    slicer: Callable | None = None,
) -> dict[str, Any] | None:
    """Optionally slice a received payload to extract this rank's portion.

    Used when ``to_tp > from_tp``: the sender sent full heads and each
    receiver rank slices out its own subset.
    """
    if callable(slicer):
        return slicer(payload)
    if not payload or topo.target_tp_size <= topo.source_tp_size or "layer_blocks" not in payload:
        return payload

    metadata = payload.get("metadata", {})
    slice_metadata = metadata.get("tp_head_slice") if isinstance(metadata, dict) else None
    if isinstance(slice_metadata, dict) and slice_metadata.get("applied"):
        tagged_rank = slice_metadata.get("target_rank")
        if tagged_rank is not None and tagged_rank != topo.local_rank:
            logger.warning(
                "Received pre-sliced KV payload for unexpected target rank: expected=%s got=%s",
                topo.local_rank,
                tagged_rank,
            )
        return payload

    ratio = topo.target_tp_size // topo.source_tp_size
    offset_in_sender = topo.local_rank % ratio
    updated_metadata = dict(metadata) if isinstance(metadata, dict) else {}
    updated_metadata["tp_head_slice"] = {
        "applied": True,
        "side": "receiver",
        "target_rank": topo.local_rank,
        "from_tp": topo.source_tp_size,
        "to_tp": topo.target_tp_size,
        "offset_in_shard": offset_in_sender,
        "num_slices": ratio,
    }
    return {
        "request_id": payload.get("request_id"),
        "layer_blocks": slice_layer_blocks(payload["layer_blocks"], offset_in_sender, ratio),
        "block_ids": list(payload.get("block_ids", [])),
        "metadata": updated_metadata,
    }


def normalize_layer_kv(
    layer_kv: LayerKV,
    *,
    req_id: str = "",
    layer_idx: int = -1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Normalize one layer KV cache to a ``(key_blocks, value_blocks)`` tuple.

    In vLLM, different attention backends return paged-attention KV blocks
    with different layouts.  For example:

    * **FlashAttention** (``vllm/v1/attention/backends/flash_attn.py``)
      returns shape ``(2, num_blocks, block_size, num_kv_heads, head_size)``
      – the key/value dimension is at **dim 0**.
    * **FlashInfer** (``vllm/v1/attention/backends/flashinfer.py``)
      returns shape ``(num_blocks, 2, block_size, num_kv_heads, head_size)``
      – the key/value dimension is at **dim 1**.

    This utility handles both layouts (and the tuple case) so that
    downstream code can work with any backend.

    Supported layouts:

    * **Stacked tensor** ``[2, num_blocks, block_size, n_heads, head_dim]`` –
      dim-0 selects key / value.
    * **Stacked tensor** ``[num_blocks, 2, block_size, n_heads, head_dim]`` –
      dim-1 selects key / value.
    * **Tuple** ``(key_tensor, value_tensor)`` – returned as-is after
      validation.

    Args:
        layer_kv: The raw KV cache (tensor or tuple) for the layer.
        req_id: Request ID used only for diagnostic log messages.
        layer_idx: Layer index used only for diagnostic log messages.

    Returns:
        ``(key_blocks, value_blocks)`` if *layer_kv* is valid, ``None``
        otherwise.
    """
    if isinstance(layer_kv, torch.Tensor):
        if layer_kv.ndim >= 3 and layer_kv.shape[0] == 2:
            key_blocks = layer_kv[0]
            value_blocks = layer_kv[1]
        elif layer_kv.ndim >= 3 and layer_kv.shape[1] == 2:
            key_blocks = layer_kv[:, 0]
            value_blocks = layer_kv[:, 1]
        else:
            logger.warning(
                f"Layer {layer_idx} for request {req_id} has invalid stacked KV shape: "
                f"expected [2, ...] or [..., 2, ...] at dim 0/1, got {tuple(layer_kv.shape)}"
            )
            return None
    elif isinstance(layer_kv, tuple):
        if len(layer_kv) != 2:
            logger.warning(f"Layer {layer_idx} for request {req_id} has KV pair length {len(layer_kv)} (expected 2)")
            return None
        key_blocks, value_blocks = layer_kv
        if not isinstance(key_blocks, torch.Tensor) or not isinstance(value_blocks, torch.Tensor):
            logger.warning(f"Layer {layer_idx} for request {req_id} has non-tensor KV pair entries")
            return None
    else:
        logger.warning(f"Layer {layer_idx} for request {req_id} has unsupported KV type {type(layer_kv).__name__}")
        return None

    # ensure key/value blocks are at least 2D for block indexing
    if key_blocks.ndim < 2 or value_blocks.ndim < 2:
        logger.warning(
            f"Layer {layer_idx} for request {req_id} has invalid KV block shape: "
            f"got key={tuple(key_blocks.shape)} value={tuple(value_blocks.shape)}"
        )
        return None

    return key_blocks, value_blocks
