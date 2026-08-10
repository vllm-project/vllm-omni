# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-demand module staging backed by an immutable pinned CPU snapshot."""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torch import nn

from vllm_omni.platforms import current_omni_platform

from .tensor_utils import set_tensor_storage


class PinnedModuleStager:
    """Stage an immutable module without copying device weights back to CPU.

    ``nn.Module.to("cpu")`` performs a device-to-host copy for every parameter
    after each forward. In inference the weights are immutable, so retain one
    pinned CPU master instead. ``load`` materializes fresh device tensors from
    that master; ``offload`` only rebinds the Parameters/Buffers to it and
    releases the device allocations.
    """

    def __init__(
        self,
        module: nn.Module,
        device: torch.device,
        *,
        pin_memory: bool = True,
        copy_stream: Any | None = None,
    ) -> None:
        self.device = device
        self.copy_stream = copy_stream or current_omni_platform.Stream()
        self.loaded = False
        self._entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._device_tensors: list[torch.Tensor] = []

        # parameters()/buffers() remove duplicate objects by default, which
        # preserves tied weights and buffer aliases when their storage moves.
        for target in chain(module.parameters(), module.buffers()):
            master = target.detach()
            if master.device.type != "cpu":
                master = master.to("cpu")
            if pin_memory and not master.is_pinned():
                master = master.pin_memory()
            set_tensor_storage(target, master)
            self._entries.append((target, master))

    def load(self) -> None:
        if self.loaded:
            return

        # Allocate on the compute stream so the caching allocator can reuse
        # memory released by the preceding encoder/DiT stage.
        device_tensors = [torch.empty_like(master, device=self.device) for _, master in self._entries]
        compute_stream = current_omni_platform.current_stream()
        self.copy_stream.wait_stream(compute_stream)
        ready = current_omni_platform.Event()
        with current_omni_platform.stream(self.copy_stream):
            for device_tensor, (_, master) in zip(device_tensors, self._entries):
                device_tensor.copy_(master, non_blocking=master.is_pinned())
            ready.record(self.copy_stream)

        for device_tensor, (target, _) in zip(device_tensors, self._entries):
            set_tensor_storage(target, device_tensor)
        compute_stream.wait_event(ready)
        self._device_tensors = device_tensors
        self.loaded = True

    def offload(self) -> None:
        if not self.loaded:
            return

        # The module has completed on the compute stream. Synchronize once at
        # the stage boundary, then release HBM by rebinding to the unchanged
        # host masters. No device-to-host transfer is necessary.
        current_omni_platform.synchronize()
        for target, master in self._entries:
            set_tensor_storage(target, master)
        self._device_tensors.clear()
        self.loaded = False
        current_omni_platform.empty_cache()


__all__ = ["PinnedModuleStager"]
