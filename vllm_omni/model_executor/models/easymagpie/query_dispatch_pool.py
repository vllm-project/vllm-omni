# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Event-owned query indices, retired after their last model-forward consumer.

The existing runner serializes prepare/forward on one host owner. Every selected
pair is uploaded and consumed on the same CUDA stream. Pure decode owns no pair;
its FULL graph contains no event operations from this helper. Exceptions leave
unretired entries quarantined, and a full pool takes the original construction.
"""

from collections import Counter

import torch


class QueryDispatchPool:
    def __init__(self, device, capacity, slots=4):
        self.device = torch.device(device)
        if self.device.type != "cuda" or type(capacity) is not int or capacity <= 0 or slots != 4:
            raise ValueError("Query dispatch pool requires CUDA, positive integer capacity and four slots")
        self.capacity = capacity
        self.entries = []
        self.current = None
        self.next_slot = 0
        self.counts = Counter()
        with torch.accelerator.device_index(self.device.index):
            for _ in range(slots):
                hosts = tuple(torch.empty(capacity, dtype=torch.long, device="cpu", pin_memory=True) for _ in range(2))
                self.entries.append(
                    {
                        "hosts": hosts,
                        "arrays": tuple(value.numpy() for value in hosts),
                        "devices": tuple(torch.empty_like(value, device=self.device) for value in hosts),
                        "event": torch.cuda.Event(),
                        "busy": False,
                        "recorded": False,
                        "stream": None,
                        "consumer_started": False,
                    }
                )

    def begin_step(self):
        # A prior exception or skipped forward cannot certify consumer completion.
        # Keep that allocation unavailable forever; never wait or overwrite it.
        if self.current is not None:
            self.counts["quarantined_steps"] += 1
            self.current = None

    def try_upload(self, decode, prefill, device):
        if self.current is not None:
            raise RuntimeError("Query pool requires begin_step before a new upload")
        if torch.device(device) != self.device or len(decode) > self.capacity or not 0 < len(prefill) <= self.capacity:
            self.counts["unsupported_fallbacks"] += 1
            return None
        with torch.accelerator.device_index(self.device.index):
            if torch.cuda.is_current_stream_capturing():
                self.counts["capture_fallbacks"] += 1
                return None
            chosen = None
            for distance in range(len(self.entries)):
                index = (self.next_slot + distance) % len(self.entries)
                entry = self.entries[index]
                if entry["busy"]:
                    if not entry["recorded"] or not entry["event"].query():
                        continue
                    self.counts["retired_reuses"] += 1
                chosen = index, entry
                break
            if chosen is None:
                self.counts["busy_fallbacks"] += 1
                return None
            index, entry = chosen
            for array, values in zip(entry["arrays"], (decode, prefill), strict=True):
                array[: len(values)] = values
            entry["busy"] = True
            entry["recorded"] = False
            entry["consumer_started"] = False
            entry["stream"] = torch.cuda.current_stream(self.device)
            self.current = entry
            # Separate device tensors preserve both original zero-offset layouts.
            # The event must NOT be recorded here: later forward code still uses
            # these indices, including the final prefill phoneme prediction.
            result = []
            for host, target, values in zip(entry["hosts"], entry["devices"], (decode, prefill), strict=True):
                target[: len(values)].copy_(host[: len(values)], non_blocking=True)
                result.append(target[: len(values)])
            self.next_slot = (index + 1) % len(self.entries)
            self.counts["pooled_batches"] += 1
            self.counts["tensor_uploads"] += 2
            self.counts["decode_elements"] += len(decode)
            self.counts["prefill_elements"] += len(prefill)
            return tuple(result)

    def before_forward(self):
        if self.current is None:
            return
        with torch.accelerator.device_index(self.device.index):
            if (
                torch.cuda.is_current_stream_capturing()
                or torch.cuda.current_stream(self.device).cuda_stream != self.current["stream"].cuda_stream
            ):
                self.counts["consumer_contract_failures"] += 1
                raise RuntimeError("Pooled query indices require uncaptured forward on their upload stream")
            if self.current["consumer_started"]:
                self.counts["consumer_contract_failures"] += 1
                raise RuntimeError("Pooled query indices require exactly one forward per prepared pair")
            self.current["consumer_started"] = True

    def complete_forward(self):
        entry = self.current
        if entry is None:
            return
        with torch.accelerator.device_index(self.device.index):
            if (
                not entry["consumer_started"]
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.current_stream(self.device).cuda_stream != entry["stream"].cuda_stream
            ):
                self.counts["consumer_contract_failures"] += 1
                raise RuntimeError("Query pool cannot retire outside its bound uncaptured forward")
            entry["event"].record(entry["stream"])
            entry["recorded"] = True
        self.current = None
        self.counts["completed_forwards"] += 1

    def status(self):
        # Observational CPU metadata only; status never queries or retires events.
        return {
            "enabled": True,
            "capacity": self.capacity,
            "slots": len(self.entries),
            "pinned_bytes": 4 * 2 * self.capacity * 8,
            "device_bytes": 4 * 2 * self.capacity * 8,
            "unretired_entries": sum(e["busy"] and not e["recorded"] for e in self.entries),
            "current_owned": self.current is not None,
            "retirement": "after all model-forward index consumers, same uncaptured stream",
            **dict(self.counts),
        }
