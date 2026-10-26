# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layout and transport primitives for H3 vision sequence parallelism.

Padding is removed before attention, never masked
into the model's sequence. Equal-sized A2A avoids variable-split HCCL calls.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class VisionShardPlan:
    segment_lengths: tuple[int, ...]
    rank_token_indices: tuple[tuple[int, ...], ...]
    rank_merger_indices: tuple[tuple[int, ...], ...]
    heads_per_rank: int


def build_vision_plan(lengths, sp_size, *, num_heads=16, merge_size=2):
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
    return VisionShardPlan(lengths, tuple(map(tuple, tokens)), tuple(map(tuple, mergers)), num_heads // sp_size)


def pack_heads(local, size, capacity):
    count, heads, dim = local.shape
    if heads % size or count > capacity:
        raise ValueError("invalid head split or capacity")
    output = local.new_zeros((size, capacity, heads // size, dim))
    output[:, :count] = local.reshape(count, size, heads // size, dim).permute(1, 0, 2, 3)
    return output


def unpack_tokens(received, plan, *, token_indices=None):
    output = received.new_empty((sum(plan.segment_lengths), *received.shape[2:]))
    for rank, indices in enumerate(plan.rank_token_indices):
        ids = (
            torch.tensor(indices, dtype=torch.long, device=received.device)
            if token_indices is None
            else token_indices[rank]
        )
        output.index_copy_(0, ids, received[rank, : len(indices)])
    return output


def pack_tokens(full, plan, *, token_indices=None):
    capacity = max(1, max(map(len, plan.rank_token_indices)))
    output = full.new_zeros((len(plan.rank_token_indices), capacity, *full.shape[1:]))
    for rank, indices in enumerate(plan.rank_token_indices):
        ids = (
            torch.tensor(indices, dtype=torch.long, device=full.device)
            if token_indices is None
            else token_indices[rank]
        )
        output[rank, : len(indices)] = full.index_select(0, ids)
    return output


def unpack_heads(received, count):
    size, capacity, heads, dim = received.shape
    if count > capacity:
        raise ValueError("local count exceeds capacity")
    return received[:, :count].permute(1, 0, 2, 3).reshape(count, size * heads, dim).contiguous()


class VisionAllToAll:
    """Explicit-group transport; callers coordinate validation failures."""

    def __init__(self, plan, *, group=None):
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
        self._token_indices_device: torch.device | None = None
        self._token_indices: tuple[torch.Tensor, ...] = ()

    def _indices_for(self, device):
        if device != self._token_indices_device:
            self._token_indices = tuple(
                torch.tensor(indices, dtype=torch.long, device=device) for indices in self.plan.rank_token_indices
            )
            self._token_indices_device = device
        return self._token_indices

    def exchange(self, packed):
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed.contiguous(), group=self.group)
        self.calls += 1
        self.sent_bytes += packed.numel() * packed.element_size()
        return received

    def to_heads(self, local):
        return unpack_tokens(
            self.exchange(pack_heads(local, self.size, self.capacity)),
            self.plan,
            token_indices=self._indices_for(local.device),
        )

    def to_tokens(self, full):
        packed = pack_tokens(full, self.plan, token_indices=self._indices_for(full.device))
        return unpack_heads(self.exchange(packed), len(self.plan.rank_token_indices[self.rank]))


def resolve_vision_sp_group(value, encoder_group):
    """Serve only the validated opt-in SP8/TP8 layout; never silently fall back."""
    if value == "0":
        return None
    if value != "8":
        raise ValueError("VLLM_OMNI_H3_VISION_SP_SIZE must be 0 (disabled) or 8")
    if encoder_group is None or encoder_group.world_size != 8:
        raise ValueError("H3 vision SP8 requires text_encoder_tp_size=8")
    return encoder_group.device_group


def gather_vision_outputs(value, plan, *, group):
    """Replicate final + DeepStack in original global merger order on every rank."""
    tensors = [value[0], *value[1]]
    capacity = max(1, max(map(len, plan.rank_merger_indices)))
    packed = tensors[0].new_zeros((len(tensors), capacity, tensors[0].shape[-1]))
    packed[:, : tensors[0].shape[0]] = torch.stack(tensors)
    received = [torch.empty_like(packed) for _ in plan.rank_merger_indices]
    dist.all_gather(received, packed, group=group)
    total = sum(map(len, plan.rank_merger_indices))
    full = packed.new_empty((len(tensors), total, packed.shape[-1]))
    for indices, part in zip(plan.rank_merger_indices, received):
        ids = torch.tensor(indices, dtype=torch.long, device=packed.device)
        full.index_copy_(1, ids, part[:, : len(indices)])
    return full[0], list(full[1:].unbind(0))
