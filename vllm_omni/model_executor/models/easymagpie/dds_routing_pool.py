# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded event-owned pinned uploads for the existing single-owner DDS state.

Each allocation remains unavailable until the event after its GPU index_copy_
has completed. A busy pool or capture uses the original routing construction;
there is no pool wait, overwrite of pending storage or request-time allocation.
"""

from collections import Counter

import torch


class RoutingUploadPool:
    def __init__(self, device, capacity, slots=4):
        self.device = torch.device(device)
        if self.device.type != "cuda" or capacity <= 0 or slots != 4:
            raise ValueError("DDS routing pool requires CUDA, positive capacity and four slots")
        self.capacity = capacity
        self.entries = []
        self.next_slot = 0
        self.counts = Counter()
        with torch.accelerator.device_index(self.device.index):
            for _ in range(slots):
                host = torch.empty(2 * capacity, dtype=torch.long, device="cpu", pin_memory=True)
                self.entries.append(
                    {
                        "host": host,
                        "array": host.numpy(),
                        "device": torch.empty_like(host, device=self.device),
                        "event": torch.cuda.Event(),
                        "busy": False,
                        "recorded": False,
                    }
                )

    def try_upload(self, target, routing):
        n = len(routing)
        if (
            type(target) is not torch.Tensor
            or target.device != self.device
            or target.dtype != torch.long
            or target.ndim != 1
            or not target.is_contiguous()
            or n <= 0
            or n > self.capacity
        ):
            self.counts["unsupported_fallbacks"] += 1
            return False
        with torch.accelerator.device_index(self.device.index):
            if torch.cuda.is_current_stream_capturing():
                self.counts["capture_fallbacks"] += 1
                return False
            chosen = None
            for distance in range(len(self.entries)):
                index = (self.next_slot + distance) % len(self.entries)
                entry = self.entries[index]
                if entry["busy"]:
                    # An unrecorded event after a failed enqueue is not proof
                    # of retirement. Leave that slot unavailable permanently.
                    if not entry["recorded"] or not entry["event"].query():
                        continue
                    self.counts["retired_reuses"] += 1
                chosen = index, entry
                break
            if chosen is None:
                self.counts["busy_fallbacks"] += 1
                return False
            index, entry = chosen
            stream = torch.cuda.current_stream(self.device)
            # Only CPU-owned, fully retired storage is touched here. Preserve
            # dict insertion order, exactly as both original tensor() calls.
            entry["array"][:n] = tuple(routing)
            entry["array"][n : 2 * n] = tuple(routing.values())
            entry["busy"] = True
            entry["recorded"] = False
            try:
                entry["device"][: 2 * n].copy_(entry["host"][: 2 * n], non_blocking=True)
                target.index_copy_(0, entry["device"][:n], entry["device"][n : 2 * n])
            finally:
                entry["event"].record(stream)
                entry["recorded"] = True
            self.next_slot = (index + 1) % len(self.entries)
            self.counts["pooled_uploads"] += 1
            self.counts["rows_uploaded"] += n
            return True

    def status(self):
        # CPU counters only: status does not retire slots or query the device.
        return {
            "enabled": True,
            "capacity": self.capacity,
            "slots": len(self.entries),
            "pinned_bytes": 4 * 2 * self.capacity * 8,
            "device_bytes": 4 * 2 * self.capacity * 8,
            **dict(self.counts),
        }
