# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Page-aligned continuous KV slots and exclusive request leases."""

from dataclasses import dataclass

import torch

PAGE_SIZE = 16


@dataclass(eq=False)
class KVSlot:
    keys: torch.Tensor
    values: torch.Tensor
    block_table: torch.Tensor
    in_use: bool = False
    retired: torch.Event | None = None

    @property
    def capacity(self) -> int:
        return self.keys.shape[1]


class PrefillKVLease:
    """Exclusive ownership of a slot, shared by snapshot and chunk engine."""

    def __init__(self, slot: KVSlot):
        self.slot = slot
        self.returned = False
        slot.in_use = True

    def views(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.returned or not 1 <= length <= self.slot.capacity:
            raise ValueError("Invalid prefill KV lease")
        return self.slot.keys[:, :length], self.slot.values[:, :length]

    def release(self, event: torch.Event | None = None) -> None:
        """Return a slot, optionally deferring reuse until an event completes."""
        if not self.returned:
            self.slot.retired = event
            self.slot.in_use = False
            self.returned = True


def round_token_capacity(length: int, maximum: int) -> int:
    if not 1 <= length <= maximum:
        raise ValueError("Prefill query length is outside the model context")
    return min(1 << (length - 1).bit_length(), maximum)


class KVSlotPool:
    def __init__(
        self,
        *,
        layers: int,
        kv_heads: int,
        head_dim: int,
        maximum: int,
        max_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.layers, self.kv_heads, self.head_dim = layers, kv_heads, head_dim
        self.maximum, self.max_slots = maximum, max_slots
        self.device, self.dtype = device, dtype
        self.slots: list[KVSlot] = []

    def acquire(self, length: int) -> PrefillKVLease | None:
        capacity = round_token_capacity(length, self.maximum)
        capacity = (capacity + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
        free = [slot for slot in self.slots if not slot.in_use and (slot.retired is None or slot.retired.query())]
        suitable = [slot for slot in free if slot.capacity >= capacity]
        if suitable:
            return PrefillKVLease(min(suitable, key=lambda slot: slot.capacity))
        if len(self.slots) >= self.max_slots:
            if not free:
                return None
            # Retire only storage with no outstanding GPU users.
            obsolete = min(free, key=lambda slot: slot.capacity)
            self.slots = [slot for slot in self.slots if slot is not obsolete]
        shape = (self.layers, capacity, self.kv_heads, self.head_dim)
        keys = torch.empty(shape, device=self.device, dtype=self.dtype)
        slot = KVSlot(
            keys,
            torch.empty_like(keys),
            torch.arange(capacity // PAGE_SIZE, dtype=torch.int32, device=self.device)[None],
        )
        self.slots.append(slot)
        return PrefillKVLease(slot)
