# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layout and transport primitives for H3 vision sequence parallelism.

Padding is removed before attention, never masked
into the model's sequence. Equal-sized A2A avoids variable-split HCCL calls.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class VisionShardPlan:
    segment_lengths: tuple[int, ...]
    rank_token_indices: tuple[tuple[int, ...], ...]
    rank_merger_indices: tuple[tuple[int, ...], ...]


class _EncoderGroup(Protocol):
    world_size: int
    device_group: dist.ProcessGroup


def build_vision_plan(
    lengths: Sequence[int],
    sp_size: int,
    *,
    num_heads: int = 16,
    merge_size: int = 2,
) -> VisionShardPlan:
    """Partition each segment into contiguous whole spatial merger groups.

    Uneven/empty shards are explicit. This does not imply that padded attention
    or smaller projection GEMMs will be bitwise equivalent on a device.
    """
    if sp_size not in (1, 2, 4, 8, 16) or num_heads <= 0 or num_heads % sp_size:
        raise ValueError("SP must divide the positive head count and be one of 1/2/4/8/16")
    if not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError("merge_size must be a positive integer")
    unit = merge_size**2
    lengths = tuple(lengths)
    if any(not isinstance(n, int) or n < 0 or n % unit for n in lengths):
        raise ValueError("segment lengths must contain whole nonnegative merger groups")
    tokens = [[] for _ in range(sp_size)]
    mergers = [[] for _ in range(sp_size)]
    offset = 0
    for length in lengths:
        groups, remainder = divmod(length // unit, sp_size)
        start = offset
        for rank in range(sp_size):
            stop = start + unit * (groups + (rank < remainder))
            tokens[rank].extend(range(start, stop))
            mergers[rank].extend(range(start // unit, stop // unit))
            start = stop
        offset += length
    return VisionShardPlan(lengths, tuple(map(tuple, tokens)), tuple(map(tuple, mergers)))


def pack_heads(local: torch.Tensor, size: int, capacity: int) -> torch.Tensor:
    count, heads, dim = local.shape
    if heads % size or count > capacity:
        raise ValueError("invalid head split or capacity")
    output = local.new_zeros((size, capacity, heads // size, dim))
    output[:, :count] = local.reshape(count, size, heads // size, dim).permute(1, 0, 2, 3)
    return output


def unpack_tokens(received: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return received.flatten(0, 1).index_select(0, permutation)


def pack_tokens(
    full: torch.Tensor,
    plan: VisionShardPlan,
    permutation: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    capacity = max(1, max(map(len, plan.rank_token_indices)))
    shape = (len(plan.rank_token_indices), capacity, *full.shape[1:])
    if full.shape[0] == 0:
        return full.new_zeros(shape)
    packed = full.index_select(0, permutation)
    if padding_mask is not None:
        packed.masked_fill_(padding_mask.reshape(-1, *([1] * (full.ndim - 1))), 0)
    return packed.reshape(shape)


def unpack_heads(received: torch.Tensor, count: int) -> torch.Tensor:
    size, capacity, heads, dim = received.shape
    if count > capacity:
        raise ValueError("local count exceeds capacity")
    return received[:, :count].permute(1, 0, 2, 3).reshape(count, size * heads, dim).contiguous()


class VisionAllToAll:
    """Permutation-backed token/head transport on an explicit process group."""

    def __init__(self, plan: VisionShardPlan, *, group: dist.ProcessGroup | None = None) -> None:
        self.plan = plan
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.size = dist.get_world_size(group=group)
        if self.size != len(plan.rank_token_indices):
            raise ValueError("plan does not match distributed world")
        self.capacity = max(1, max(map(len, plan.rank_token_indices)))
        self.calls = 0
        self.sent_bytes = 0
        # One transport belongs to one vision forward; never retain activations
        # or indices across requests with different segment layouts.
        self._permutation_device: torch.device | None = None
        self._permutation: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None

    def _permutation_for(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if device != self._permutation_device:
            inverse = [0] * sum(self.plan.segment_lengths)
            source = []
            padding = []
            for rank, indices in enumerate(self.plan.rank_token_indices):
                for slot, token in enumerate(indices):
                    inverse[token] = rank * self.capacity + slot
                source.extend(indices)
                source.extend([0] * (self.capacity - len(indices)))
                padding.extend([False] * len(indices) + [True] * (self.capacity - len(indices)))
            self._permutation = (
                torch.tensor(inverse, dtype=torch.long, device=device),
                torch.tensor(source, dtype=torch.long, device=device),
                torch.tensor(padding, dtype=torch.bool, device=device) if any(padding) else None,
            )
            self._permutation_device = device
        assert self._permutation is not None
        return self._permutation

    def exchange(self, packed: torch.Tensor) -> torch.Tensor:
        packed = packed.contiguous()
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=self.group)
        self.calls += 1
        self.sent_bytes += packed.numel() * packed.element_size()
        return received

    def to_heads(self, local: torch.Tensor) -> torch.Tensor:
        inverse, _, _ = self._permutation_for(local.device)
        received = self.exchange(pack_heads(local, self.size, self.capacity))
        return unpack_tokens(received, inverse)

    def to_tokens(self, full: torch.Tensor) -> torch.Tensor:
        _, source, padding = self._permutation_for(full.device)
        packed = pack_tokens(full, self.plan, source, padding_mask=padding)
        return unpack_heads(self.exchange(packed), len(self.plan.rank_token_indices[self.rank]))


def resolve_vision_sp_group(value: str, encoder_group: _EncoderGroup | None) -> dist.ProcessGroup | None:
    """Serve only the validated opt-in SP8/TP8 layout; never silently fall back."""
    if value == "0":
        return None
    if value != "8":
        raise ValueError("VLLM_OMNI_H3_VISION_SP_SIZE must be 0 (disabled) or 8")
    if encoder_group is None or encoder_group.world_size != 8:
        raise ValueError("H3 vision SP8 requires text_encoder_tp_size=8")
    return encoder_group.device_group


def gather_vision_outputs(
    value: tuple[torch.Tensor, list[torch.Tensor]],
    plan: VisionShardPlan,
    *,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Replicate final + DeepStack in original global merger order on every rank."""
    tensors = [value[0], *value[1]]
    capacity = max(1, max(map(len, plan.rank_merger_indices)))
    packed = tensors[0].new_zeros((len(tensors), capacity, tensors[0].shape[-1]))
    for index, tensor in enumerate(tensors):
        packed[index, : tensor.shape[0]] = tensor
    received = [torch.empty_like(packed) for _ in plan.rank_merger_indices]
    dist.all_gather(received, packed, group=group)
    total = sum(map(len, plan.rank_merger_indices))
    full = packed.new_empty((len(tensors), total, packed.shape[-1]))
    for indices, part in zip(plan.rank_merger_indices, received):
        ids = torch.tensor(indices, dtype=torch.long, device=packed.device)
        full.index_copy_(1, ids, part[:, : len(indices)])
    return full[0], list(full[1:].unbind(0))
