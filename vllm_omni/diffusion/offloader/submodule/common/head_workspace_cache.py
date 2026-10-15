# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded receive buffers for sequential inference on one compute stream.

Every collective must be joined on that stream before leaving the lease.
Exceptions retain buffers and prohibit reuse until a synchronized request reset
or teardown.
"""

from collections import OrderedDict
from contextlib import contextmanager

import torch


def _record_buffers(buffers, stream):
    if isinstance(buffers, (list, tuple)):
        for buffer in buffers:
            _record_buffers(buffer, stream)
    else:
        buffers.record_stream(stream)


class HeadWorkspaceCache:
    def __init__(self, capacity=2):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Capacity must be a positive integer")
        self.capacity = capacity
        self._entries = OrderedDict()
        self._stream_key = None
        self._leased = self._poisoned = self._closed = False
        self._validated_boundaries: tuple[torch.Tensor, int, tuple[int, ...]] | None = None

    def validate_boundaries(self, cu_seqlens: torch.Tensor, seq_ends: list[int]) -> None:
        expected = (0, *seq_ends)
        version = None if cu_seqlens.is_inference() else cu_seqlens._version
        cached = self._validated_boundaries
        if cached is not None and cached[0] is cu_seqlens and cached[1:] == (version, expected):
            return
        if tuple(cu_seqlens.tolist()) != expected:
            raise ValueError("H3 TND requires the original single-request real/padding boundaries")
        # Share one device-to-host validation across blocks, but recheck mutations.
        # Inference tensors have no version counter and must always be checked.
        self._validated_boundaries = (cu_seqlens, version, expected) if version is not None else None

    def reset_after_synchronize(self) -> None:
        if self._closed or self._leased:
            raise RuntimeError("Cannot reset a closed or leased workspace")
        self._entries.clear()
        self._stream_key = None
        self._poisoned = False
        self._validated_boundaries = None

    @contextmanager
    def lease(self, key, allocate, *, stream):
        stream_key = (stream.device, stream.npu_stream)
        if self._closed or self._poisoned or self._leased:
            raise RuntimeError("Workspace is closed, poisoned, or already leased")
        if self._stream_key is not None and stream_key != self._stream_key:
            raise RuntimeError("Workspace requires one compute stream")
        if key not in self._entries:
            if len(self._entries) == self.capacity:
                self._entries.popitem(last=False)
            self._entries[key] = allocate()
        self._entries.move_to_end(key)
        self._stream_key, self._leased = stream_key, True
        try:
            yield self._entries[key]
            # Work.wait establishes completion dependencies; record_stream
            # keeps evicted buffers alive for pending reads on that stream.
            _record_buffers(self._entries[key], stream)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._leased = False

    def close_after_synchronize(self):
        self._entries.clear()
        self._validated_boundaries = None
        self._closed = True
