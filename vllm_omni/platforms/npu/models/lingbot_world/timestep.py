# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded inference cache of FP64 host-computed timestep frequencies.

The DMD runner already owns the scalar timestep on the host. Round it through
FP32 exactly as its device ``torch.full`` does before applying the unchanged
FP64 sinusoidal expression. No device tensor is read back and no learned
projection is cached.
"""

import math
import struct
from collections import OrderedDict
from collections.abc import Callable

import torch


class HostTimestepFrequencyCache:
    def __init__(
        self,
        embedding: Callable[[int, torch.Tensor], torch.Tensor],
        *,
        max_entries: int = 32,
    ) -> None:
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.embedding = embedding
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        timestep_value: float,
        *,
        dim: int,
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("Host timestep frequency caching requires inference/no_grad")
        if isinstance(timestep_value, bool) or not isinstance(timestep_value, (int, float)):
            raise TypeError("timestep_value must be a host scalar, never a device tensor")
        if not math.isfinite(timestep_value):
            raise ValueError("timestep_value must be finite")
        if any(type(n) is not int or n <= 0 for n in (dim, batch_size, frames)) or dim % 2:
            raise ValueError("dim must be positive/even; batch_size and frames must be positive")
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Host timestep frequencies require BF16, FP16 or FP32")
        # Bits distinguish signed zero and make equal FP32 values share an entry.
        bits = struct.pack("!f", float(timestep_value))
        canonical = struct.unpack("!f", bits)[0]
        if not math.isfinite(canonical):
            raise ValueError("timestep_value must fit finite FP32")
        device = torch.device(device)
        key = (bits, dim, batch_size, frames, device, dtype)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        host = torch.full((batch_size * frames,), canonical, device="cpu", dtype=torch.float32)
        frequency = self.embedding(dim, host).to(device=device, dtype=dtype)
        self._entries[key] = frequency
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return frequency
