# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Host-to-device copies that do not block the host on queued GPU work.

A copy from pageable host memory (``tensor.to("cuda")``, ``torch.tensor(...,
device="cuda")``, ``torch.as_tensor(..., device="cuda")``) makes the CPU wait
until every kernel already queued on the stream has finished. In a per-step
path under async scheduling that removes the CPU/GPU overlap the batch queue
exists for. Staging the data in
pinned memory and copying with ``non_blocking`` keeps the host running ahead;
the caching host allocator keeps the pinned source alive until the copy's
stream event completes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from vllm.utils.platform_utils import is_uva_available


def to_device_nonblocking(tensor: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """``tensor.to(device)`` without a host sync when ``tensor`` is on the CPU and ``device`` is CUDA."""
    device = torch.device(device)
    if tensor.device.type != "cpu" or device.type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def index_to_device(values: Sequence[int], device: torch.device | str, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """A host index list as a device tensor, without a host sync."""
    if torch.device(device).type != "cuda":
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


class DeviceStager:
    """Small per-step host arrays read by the GPU in place, through UVA, with no copy launch.

    The values go into a ring of pinned host slots, and the result is a device
    view of the slot (unified virtual addressing): the kernels that consume it
    read host memory directly. This costs a host ``memcpy`` (~3 us) instead of
    a pinned allocation, a device allocation and a copy launch (~15-50 us).

    A view stays valid until its slot comes around again. The slots form
    groups; leaving a group records an event on the current stream, by which
    time the readers of every result in it are queued there, and a group is
    reused only after its event completes. Readers on other streams can hold a
    result's slot with :meth:`retire`. So a result must be consumed by kernels
    queued before the next call to the same stager (or retired with an event
    that follows its last reader), and must not be kept across steps or used
    inside a CUDA graph capture.
    """

    def __init__(self, dtype: torch.dtype = torch.int64, slots: int = 16, group: int = 4, capacity: int = 1024) -> None:
        assert slots % group == 0 and slots // group >= 2
        self.dtype = dtype
        self._slots = slots
        self._group = group
        self._capacity = [0] * slots
        self._host_np: list[np.ndarray | None] = [None] * slots
        self._view: list[torch.Tensor | None] = [None] * slots
        # (slot, shape) -> device view; shapes repeat while the batch size holds.
        self._shaped: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        self._fence: list[torch.cuda.Event | None] = [None] * (slots // group)
        self._retired: list[list[torch.cuda.Event]] = [[] for _ in range(slots)]
        self._np_dtype = torch.empty((), dtype=dtype).numpy().dtype
        self._init_capacity = capacity
        self._prev = -1

    def _grow(self, slot: int, n: int, device: torch.device) -> None:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        capacity = max(self._init_capacity, 1 << (n - 1).bit_length())
        host = torch.empty(capacity, dtype=self.dtype, pin_memory=True)
        self._host_np[slot] = host.numpy()
        with torch.accelerator.device_index(device.index):
            # The view keeps the pinned tensor alive.
            self._view[slot] = get_accelerator_view_from_cpu_tensor(host)
        self._capacity[slot] = capacity
        for key in [key for key in self._shaped if key[0] == slot]:
            del self._shaped[key]

    def retire(self, event: torch.cuda.Event) -> None:
        """Keep the most recent result's slot until ``event`` completes (for readers on other streams)."""
        if self._prev >= 0:
            self._retired[self._prev].append(event)

    def __call__(self, values: np.ndarray | Sequence[int], device: torch.device | str) -> torch.Tensor:
        arr = np.asarray(values, dtype=self._np_dtype)
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda" or not is_uva_available():
            return to_device_nonblocking(torch.from_numpy(np.ascontiguousarray(arr)), device)
        slot = (self._prev + 1) % self._slots
        if slot % self._group == 0:
            if self._prev >= 0:
                # Leaving a group: the readers of all its results are queued on this stream.
                g = self._prev // self._group
                fence = self._fence[g]
                if fence is None:
                    fence = self._fence[g] = torch.cuda.Event()
                fence.record(torch.cuda.current_stream(device))
            fence = self._fence[slot // self._group]
            if fence is not None and not fence.query():
                fence.synchronize()
        retired = self._retired[slot]
        if retired:
            for event in retired:
                event.synchronize()
            retired.clear()
        n = arr.size
        if n > self._capacity[slot]:
            self._grow(slot, n, device)
        host_np = self._host_np[slot]
        assert host_np is not None
        host_np[:n] = arr.reshape(-1)
        self._prev = slot
        key = (slot, arr.shape)
        out = self._shaped.get(key)
        if out is None:
            view = self._view[slot]
            assert view is not None
            out = self._shaped[key] = view[:n].view(arr.shape)
        return out
