# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Distributed primitives for the native MAGI-2 preview transformer.

This file is adapted from SandAI's Apache-2.0 MAGI-2 preview context- and
expert-parallel primitives.  It has been modified to reuse vLLM-Omni's
already-initialized sequence-parallel and expert-parallel process groups.

MAGI-2 supports two equivalent layouts for its two parallel views:

* Ulysses context parallelism shards tokens and gathers attention heads.
* Multi-head expert parallelism shards the *MoE head* axis, not experts.

With SP-only deployment both views use the SP group.  With TP enabled,
Ulysses keeps using SP while native column/row tensor parallelism and the MoE
heads use TP.  This gives TP4 and TP2SP2 independent weight/head and token
axes while preserving the released equations and checkpoint hierarchy.

These helpers intentionally do not create or own process groups.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Magi2ParallelGroup:
    """The small process-group surface needed by MAGI-2 kernels."""

    group: dist.ProcessGroup | None
    world_size: int
    rank: int
    replicated_sequence: bool = False


def _dist_group_info(group: dist.ProcessGroup | None) -> Magi2ParallelGroup:
    if group is None or not dist.is_available() or not dist.is_initialized():
        return Magi2ParallelGroup(None, 1, 0)
    return Magi2ParallelGroup(
        group=group,
        world_size=dist.get_world_size(group),
        rank=dist.get_rank(group),
    )


def get_magi2_ulysses_group() -> Magi2ParallelGroup:
    """Return vLLM-Omni's Ulysses subgroup, or a rank-local fallback.

    MAGI-2 does not support Ring or AllGather-KV attention.  The caller is
    expected to validate that ``ulysses_degree == sequence_parallel_size``.
    """

    if not dist.is_available() or not dist.is_initialized():
        return Magi2ParallelGroup(None, 1, 0)
    try:
        from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

        coordinator = get_sp_group()
    except (AssertionError, RuntimeError):
        return Magi2ParallelGroup(None, 1, 0)
    return Magi2ParallelGroup(
        group=coordinator.ulysses_group,
        world_size=coordinator.ulysses_world_size,
        rank=coordinator.ulysses_rank,
    )


def get_magi2_ep_group() -> Magi2ParallelGroup:
    """Return the process group used for MAGI's MoE-head parallelism.

    TP is the explicit MoE-head axis when it is larger than one.  Otherwise
    the released SP-only layout overlaps head parallelism with Ulysses.  Both
    groups are initialized and owned by vLLM-Omni; MAGI creates no ad-hoc
    process groups.
    """

    if not dist.is_available() or not dist.is_initialized():
        return Magi2ParallelGroup(None, 1, 0)
    try:
        from vllm.distributed.parallel_state import get_tp_group

        coordinator = get_tp_group()
    except (AssertionError, RuntimeError):
        return get_magi2_ulysses_group()
    if coordinator.world_size <= 1:
        return get_magi2_ulysses_group()
    return Magi2ParallelGroup(
        group=coordinator.device_group,
        world_size=coordinator.world_size,
        rank=coordinator.rank_in_group,
        replicated_sequence=True,
    )


def get_magi2_tp_group() -> Magi2ParallelGroup:
    """Return vLLM's tensor-parallel group without an SP fallback."""

    if not dist.is_available() or not dist.is_initialized():
        return Magi2ParallelGroup(None, 1, 0, replicated_sequence=True)
    try:
        from vllm.distributed.parallel_state import get_tp_group

        coordinator = get_tp_group()
    except (AssertionError, RuntimeError):
        return Magi2ParallelGroup(None, 1, 0, replicated_sequence=True)
    return Magi2ParallelGroup(
        group=coordinator.device_group,
        world_size=coordinator.world_size,
        rank=coordinator.rank_in_group,
        replicated_sequence=True,
    )


def get_magi2_replica_group(data_parallel_size: int) -> Magi2ParallelGroup:
    """Return the complete TP x SP group for one data-parallel replica.

    The pipeline uses this group for conditioning broadcasts and output-rank
    ownership.  With DP=1 the diffusion world is exactly one TP x SP replica.
    With DP>1, MAGI currently requires TP=1, so the existing SP group is the
    complete per-replica group. CFG x DP is deliberately rejected by
    ``_validate_native_topology``; if that combination is enabled in the
    future, this helper must select a group fixed to one DP replica and include
    its CFG ranks rather than returning the SP group alone.
    """

    if not dist.is_available() or not dist.is_initialized():
        return Magi2ParallelGroup(None, 1, 0)
    if data_parallel_size > 1:
        try:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            coordinator = get_sp_group()
        except (AssertionError, RuntimeError):
            return Magi2ParallelGroup(None, 1, 0)
        return Magi2ParallelGroup(
            group=coordinator.device_group,
            world_size=coordinator.world_size,
            rank=coordinator.rank_in_group,
        )
    return _dist_group_info(dist.group.WORLD)


def balanced_split_sizes(length: int, world_size: int) -> list[int]:
    """Split ``length`` contiguously, putting one extra token on early ranks."""

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    base, remainder = divmod(length, world_size)
    return [base + int(rank < remainder) for rank in range(world_size)]


def shard_sequence(
    tensor: torch.Tensor,
    split_sizes: list[int] | None = None,
    group: Magi2ParallelGroup | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Return this Ulysses rank's contiguous sequence slice.

    Inputs are replicated at the pipeline boundary, matching the diffusion
    runner contract, so no collective is needed for dispatch.
    """

    group = group or get_magi2_ulysses_group()
    split_sizes = split_sizes or balanced_split_sizes(tensor.shape[0], group.world_size)
    if len(split_sizes) != group.world_size or sum(split_sizes) != tensor.shape[0]:
        raise ValueError(
            f"split_sizes={split_sizes} do not partition sequence length "
            f"{tensor.shape[0]} across {group.world_size} ranks"
        )
    start = sum(split_sizes[: group.rank])
    return tensor.narrow(0, start, split_sizes[group.rank]).contiguous(), split_sizes


def gather_sequence(
    tensor: torch.Tensor,
    split_sizes: list[int],
    group: Magi2ParallelGroup | None = None,
) -> torch.Tensor:
    """Gather uneven contiguous sequence shards in rank order."""

    group = group or get_magi2_ulysses_group()
    if group.world_size == 1:
        return tensor
    if len(split_sizes) != group.world_size:
        raise ValueError("split_sizes length must equal the Ulysses world size")
    if tensor.shape[0] != split_sizes[group.rank]:
        raise ValueError(f"rank {group.rank} owns {tensor.shape[0]} tokens, expected {split_sizes[group.rank]}")
    output_shape = (sum(split_sizes), *tensor.shape[1:])
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    chunks = list(torch.split(output, split_sizes, dim=0))
    dist.all_gather(chunks, tensor.contiguous(), group=group.group)
    return output


def scatter_seqlen_gather_heads(
    tensor: torch.Tensor,
    split_sizes: list[int],
    group: Magi2ParallelGroup | None = None,
) -> torch.Tensor:
    """Ulysses ``[sum(S_r), H, D] -> [S_rank, world*H, D]`` exchange."""

    group = group or get_magi2_ulysses_group()
    if group.world_size == 1:
        return tensor
    if tensor.ndim != 3 or not tensor.is_contiguous():
        raise ValueError("Ulysses attention input must be contiguous [S,H,D]")
    if len(split_sizes) != group.world_size or sum(split_sizes) != tensor.shape[0]:
        raise ValueError("split_sizes must partition the input sequence")

    local_tokens = split_sizes[group.rank]
    output = torch.empty(
        (group.world_size * local_tokens, tensor.shape[1], tensor.shape[2]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=[local_tokens] * group.world_size,
        input_split_sizes=split_sizes,
        group=group.group,
    )
    output = (
        output.view(group.world_size, local_tokens, tensor.shape[1], tensor.shape[2])
        .permute(1, 0, 2, 3)
        .reshape(local_tokens, group.world_size * tensor.shape[1], tensor.shape[2])
    )
    return output


def scatter_heads_gather_seqlen(
    tensors: Iterable[torch.Tensor],
    split_sizes: list[int],
    group: Magi2ParallelGroup | None = None,
) -> list[torch.Tensor]:
    """Inverse batched Ulysses exchange for Q/K/V.

    Each input is ``[S_rank, world*H_i, D]`` and each output is
    ``[sum(S_r), H_i, D]``.  Fusing Q/K/V into one all-to-all preserves the
    reference communication ordering and avoids three independent collectives.
    """

    group = group or get_magi2_ulysses_group()
    tensors = list(tensors)
    if group.world_size == 1:
        return tensors
    if not tensors:
        return []
    local_tokens = split_sizes[group.rank]
    if len(split_sizes) != group.world_size:
        raise ValueError("split_sizes length must equal the Ulysses world size")
    if any(t.ndim != 3 or t.shape[0] != local_tokens for t in tensors):
        raise ValueError("all Ulysses inputs must be [local_tokens, heads, dim]")

    reshaped: list[torch.Tensor] = []
    local_head_counts: list[int] = []
    head_dim = tensors[0].shape[-1]
    for tensor in tensors:
        if tensor.shape[-1] != head_dim or tensor.shape[1] % group.world_size:
            raise ValueError("attention heads must divide evenly across Ulysses ranks")
        local_heads = tensor.shape[1] // group.world_size
        local_head_counts.append(local_heads)
        reshaped.append(
            tensor.view(local_tokens, group.world_size, local_heads, head_dim)
            .permute(1, 0, 2, 3)
            .reshape(group.world_size * local_tokens, local_heads, head_dim)
        )

    fused = torch.cat(reshaped, dim=1).contiguous()
    output = torch.empty(
        (sum(split_sizes), fused.shape[1], head_dim),
        dtype=fused.dtype,
        device=fused.device,
    )
    dist.all_to_all_single(
        output,
        fused,
        output_split_sizes=split_sizes,
        input_split_sizes=[local_tokens] * group.world_size,
        group=group.group,
    )
    return list(torch.split(output, local_head_counts, dim=1))


def ep_dispatch(
    tensor: torch.Tensor,
    group: Magi2ParallelGroup | None = None,
    sequence_split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Dispatch ``[S,H,D]`` so each rank evaluates a contiguous head shard."""

    group = group or get_magi2_ep_group()
    if group.world_size == 1:
        return tensor
    if tensor.ndim != 3 or tensor.shape[1] % group.world_size:
        raise ValueError(
            f"MoE head count {tensor.shape[1] if tensor.ndim >= 2 else '?'} must divide by EP size {group.world_size}"
        )
    sequence, heads, dim = tensor.shape
    local_heads = heads // group.world_size
    if sequence_split_sizes is None:
        local_size = torch.tensor([sequence], dtype=torch.int64, device=tensor.device)
        gathered_sizes = [torch.empty_like(local_size) for _ in range(group.world_size)]
        dist.all_gather(gathered_sizes, local_size, group=group.group)
        sequence_split_sizes = [int(size.item()) for size in gathered_sizes]
    if len(sequence_split_sizes) != group.world_size or sequence_split_sizes[group.rank] != sequence:
        raise ValueError("EP sequence split sizes do not describe the local tensor")
    send = tensor.contiguous().view(sequence, group.world_size, local_heads, dim).permute(1, 0, 2, 3).contiguous()
    output = torch.empty(
        (sum(sequence_split_sizes), local_heads, dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    row_width = local_heads * dim
    dist.all_to_all_single(
        output.view(-1),
        send.view(-1),
        output_split_sizes=[size * row_width for size in sequence_split_sizes],
        input_split_sizes=[sequence * row_width] * group.world_size,
        group=group.group,
    )
    return output


def ep_undispatch(
    tensor: torch.Tensor,
    group: Magi2ParallelGroup | None = None,
    sequence_split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Undo :func:`ep_dispatch` and restore the complete MoE head axis."""

    group = group or get_magi2_ep_group()
    if group.world_size == 1:
        return tensor
    if tensor.ndim != 3:
        raise ValueError("EP-dispatched tensor must be [global_tokens,local_heads,dim]")
    if sequence_split_sizes is None:
        if tensor.shape[0] % group.world_size:
            raise ValueError("uneven EP sequence requires explicit split sizes")
        sequence_split_sizes = [tensor.shape[0] // group.world_size] * group.world_size
    if len(sequence_split_sizes) != group.world_size or sum(sequence_split_sizes) != tensor.shape[0]:
        raise ValueError("EP sequence split sizes do not partition the global tensor")
    sequence = sequence_split_sizes[group.rank]
    local_heads, dim = tensor.shape[1:]
    send = tensor.contiguous()
    output = torch.empty(
        (group.world_size, sequence, local_heads, dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    row_width = local_heads * dim
    dist.all_to_all_single(
        output.view(-1),
        send.view(-1),
        output_split_sizes=[sequence * row_width] * group.world_size,
        input_split_sizes=[size * row_width for size in sequence_split_sizes],
        group=group.group,
    )
    return output.permute(1, 0, 2, 3).contiguous().view(sequence, group.world_size * local_heads, dim)


class Magi2SequenceDispatcher:
    """Request-scoped dispatcher that enforces one consistent token split."""

    def __init__(self, group: Magi2ParallelGroup | None = None) -> None:
        self.group = group or get_magi2_ulysses_group()
        self.split_sizes: list[int] | None = None

    def dispatch(self, tensor: torch.Tensor) -> torch.Tensor:
        split_sizes = balanced_split_sizes(tensor.shape[0], self.group.world_size)
        if self.split_sizes is not None and split_sizes != self.split_sizes:
            raise ValueError(
                f"all packed inputs must share a sequence length; got {split_sizes}, expected {self.split_sizes}"
            )
        self.split_sizes = split_sizes
        return shard_sequence(tensor, split_sizes, self.group)[0]

    def undispatch(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.split_sizes is None:
            raise RuntimeError("dispatch must be called before undispatch")
        output = gather_sequence(tensor, self.split_sizes, self.group)
        self.split_sizes = None
        return output
