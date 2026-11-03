# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded scratch reuse with CUDA completion and downstream-storage leases.

CPU outputs use a NumPy owner attached to torch.from_numpy's storage deleter.
Consequently slices, detach(), NumPy views and serialization aliases retain the
lease, even when the original output wrapper/tensor has already been released.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch


def leased_buffers_enabled():
    return os.environ.get("EASYMAGPIE_CODEC_LEASED_BUFFERS", "0") == "1"


@dataclass
class _Entry:
    tensor: torch.Tensor
    key: tuple
    capacity: int
    generation: int = 0
    state: str = "free"
    event: object = None


class BufferLease:
    def __init__(self, pool, entry, size):
        self._pool = weakref.ref(pool)
        self._entry = entry
        self._generation = entry.generation
        self.tensor = entry.tensor[:size]
        self._event = None
        self._released = False

    def ready_after(self, event):
        self._event = event

    def release(self):
        if self._released:
            return
        self._released = True
        pool = self._pool()
        if pool is not None:
            pool._release(self._entry, self._generation, self._event)
        self._entry = None
        self.tensor = None

    def __del__(self):
        self.release()


class _LeaseArray(np.ndarray):
    def __array_finalize__(self, parent):
        self._buffer_lease = getattr(parent, "_buffer_lease", None)


def storage_owned_host_tensor(lease):
    """Expose CPU storage with lease lifetime tied to *all* derived aliases."""
    if lease.tensor.device.type != "cpu" or lease.tensor.dtype != torch.float32:
        raise ValueError("PCM lease export requires CPU float32 storage")
    array = lease.tensor.numpy().view(_LeaseArray)
    array._buffer_lease = lease
    # PyTorch retains this exact array object until its storage's final alias
    # disappears. This is stronger than a weakref.finalize on one output view.
    return torch.from_numpy(array)


class BufferLeasePool:
    def __init__(self, max_entries=8, allocator=torch.empty):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.allocator = allocator
        self._entries = []
        self._pending_unpooled = []
        self._lock = threading.Lock()
        self._stats = Counter()

    def acquire(self, size, *, dtype, device, pin_memory=False):
        if size < 0:
            raise ValueError("buffer size must be nonnegative")
        device = torch.device(device)
        key = (str(device), dtype, bool(pin_memory))
        capacity = 1 << max(0, (max(size, 1) - 1).bit_length())
        with self._lock:
            self._pending_unpooled = [entry for entry in self._pending_unpooled if not entry.event.query()]
            for entry in self._entries:
                if entry.state == "pending" and entry.event.query():
                    entry.state = "free"
                    entry.event = None
                    self._stats["event_reclaims"] += 1
            suitable = [
                entry
                for entry in self._entries
                if entry.key == key and entry.state == "free" and entry.capacity >= size
            ]
            if suitable:
                entry = min(suitable, key=lambda value: value.capacity)
                self._stats["reuses"] += 1
            else:
                tensor = self.allocator((capacity,), dtype=dtype, device=device, pin_memory=pin_memory)
                entry = _Entry(tensor, key, capacity)
                self._stats["allocations"] += 1
                if len(self._entries) < self.max_entries:
                    self._entries.append(entry)
                else:
                    # Small startup shapes must not permanently occupy every
                    # retained slot. A free entry has no consumer aliases and
                    # its CUDA event has completed (pending slots were polled
                    # above). Replace only such an entry; live/pending entries
                    # remain immutable and use transient allocations instead.
                    free = [item for item in self._entries if item.state == "free"]
                    same_key = [item for item in free if item.key == key]
                    replaceable = same_key or free
                    if replaceable:
                        previous = min(replaceable, key=lambda item: item.capacity)
                        index = next(i for i, item in enumerate(self._entries) if item is previous)
                        self._entries[index] = entry
                        self._stats["free_slot_replacements"] += 1
                        if previous.key == key and previous.capacity < capacity:
                            self._stats["free_slot_grows"] += 1
                    else:
                        self._stats["unpooled_allocations"] += 1
            entry.state = "leased"
            entry.generation += 1
            self._stats["acquires"] += 1
            self._stats["active"] += 1
            self._stats["peak_active"] = max(self._stats["peak_active"], self._stats["active"])
            return BufferLease(self, entry, size)

    def _release(self, entry, generation, event):
        with self._lock:
            if entry.generation != generation or entry.state != "leased":
                raise RuntimeError("stale or duplicate buffer lease release")
            self._stats["active"] -= 1
            self._stats["releases"] += 1
            if event is not None and not event.query():
                entry.event = event
                entry.state = "pending"
                # Unpooled in-flight tensors must stay alive until DMA completes.
                if not any(item is entry for item in self._entries):
                    self._pending_unpooled.append(entry)
            else:
                entry.state = "free"
                entry.event = None

    def status(self):
        with self._lock:
            return {
                **dict(self._stats),
                "retained_entries": len(self._entries),
                "retained_bytes": sum(entry.tensor.numel() * entry.tensor.element_size() for entry in self._entries),
                "states": dict(Counter(entry.state for entry in self._entries)),
                "pending_unpooled": len(self._pending_unpooled),
            }
