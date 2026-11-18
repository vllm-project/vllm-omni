# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Host-to-device copies that do not block the host on queued CUDA/NPU work.

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
    """Copy CPU tensors to CUDA/NPU without waiting for queued device work."""
    device = torch.device(device)
    if tensor.device.type != "cpu" or device.type not in {"cuda", "npu"}:
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


def index_to_device(values: Sequence[int], device: torch.device | str, dtype: torch.dtype = torch.long) -> torch.Tensor:
    """A host index list as a device tensor, without a host sync."""
    if torch.device(device).type not in {"cuda", "npu"}:
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
