# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.distributed as dist


class CellTiming:
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = []
        if enabled:
            self.started = torch.Event(enable_timing=True)
            self.started.record()

    def mark(self, chunk, latent=None):
        if self.enabled:
            event = torch.Event(enable_timing=True)
            event.record()
            self.events.append((chunk, event))

    def durations(self, pp):
        if not self.enabled:
            return {}
        last = pp is None or pp.rank_in_group == pp.world_size - 1
        payload: list[dict[str, float] | None] = [None]
        if last:
            self.events[-1][1].synchronize()
            previous = 0.0
            durations = {}
            for chunk, event in self.events:
                elapsed = self.started.elapsed_time(event) / 1000.0
                durations[f"physis.chunk_{chunk:04d}"] = elapsed - previous
                previous = elapsed
            durations["physis.dit"] = previous
            payload[0] = durations
        if pp is not None and pp.world_size > 1:
            dist.broadcast_object_list(payload, src=pp.ranks[-1], group=pp.cpu_group)
        return payload[0]
