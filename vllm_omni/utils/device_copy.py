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

import torch


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


class HostCopyBatch:
    """Device-to-host copies of one step's outputs behind one host wait.

    ``wait`` waits on an event recorded after the copies, and is free when a
    later sync already covered them. Off CUDA, or without pinned memory,
    :meth:`copy` is the blocking ``tensor.to("cpu")``. A contiguous host tensor
    passes through.
    """

    def __init__(self, pin_memory: bool) -> None:
        self._pin_memory = bool(pin_memory)
        self._event: torch.cuda.Event | None = None

    def copy(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cpu" and not tensor.requires_grad and tensor.is_contiguous():
            return tensor
        tensor = tensor.detach()
        if tensor.device.type != "cuda" or not self._pin_memory:
            return tensor.to("cpu").contiguous()
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
        host.copy_(tensor, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._event = event
        return host

    def wait(self) -> None:
        event = self._event
        if event is None:
            return
        self._event = None
        if not event.query():
            event.synchronize()
