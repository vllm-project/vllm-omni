# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, slot-owned reuse of exact time/AdaLN projections for TP graphs."""

from collections import OrderedDict

import torch


class RealizationTimeConditioning:
    def __init__(self, model, capacity=8):
        self.model = model
        self.capacity = capacity
        self.entries = OrderedDict()
        self.signature = None
        self.buffer = None
        self.views = None
        self.hits = self.misses = 0
        from vllm.distributed import get_tp_group

        self.group = get_tp_group()
        self.agreement = torch.ones(2, device=next(model.parameters()).device, dtype=torch.int32)

    def _signature(self):
        roots = (
            self.model.time_embedder,
            *(b.adaln_proj for b in self.model.blocks),
            self.model.final_layer.adaln_proj,
        )
        # Changes to parameters, dtype/device or LoRA scale invalidate entries.
        # Inference tensors without version tracking cannot safely use this cache.
        tensors = tuple(
            (id(p), p.data_ptr(), p._version, p.shape, p.dtype, p.device) for root in roots for p in root.parameters()
        )
        scales = tuple((id(m), getattr(m, "scale", None)) for root in roots for m in root.modules())
        return (
            tensors,
            scales,
            torch.get_float32_matmul_precision(),
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

    @torch.no_grad()
    def prepare(self, times):
        signature = self._signature()
        # Exact bytes distinguish signed zero and preserve the original M=1/2
        # GEMM geometry. No batching of different timestep tables is performed.
        key = (times.dtype, tuple(times.shape), times.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        unchanged = signature == self.signature
        hit = unchanged and key in self.entries
        if self.group.world_size > 1:
            # Any rank's mutation invalidates every rank's entire cache. Any
            # miss makes all ranks execute the same projection collectives.
            self.agreement.copy_(torch.tensor([unchanged, hit], dtype=torch.int32))
            torch.distributed.all_reduce(
                self.agreement, op=torch.distributed.ReduceOp.MIN, group=self.group.device_group
            )
            unchanged, hit = (bool(x) for x in self.agreement.cpu().tolist())
        if not unchanged:
            self.entries.clear()
        self.signature = signature
        packed = self.entries.get(key) if hit else None
        if packed is None:
            outputs = self.model.time_conditioning(times)
            flat = [x for group in outputs for x in group]
            packed = torch.cat([x.reshape(-1) for x in flat])
            if self.buffer is None:
                self.buffer = torch.empty_like(packed)
                cursor, groups = 0, []
                for group in outputs:
                    views = []
                    for value in group:
                        n = value.numel()
                        views.append(self.buffer[cursor : cursor + n].view(value.shape))
                        cursor += n
                    groups.append(tuple(views))
                self.views = tuple(groups)
            if (
                packed.shape != self.buffer.shape
                or packed.dtype != self.buffer.dtype
                or packed.device != self.buffer.device
            ):
                raise RuntimeError("Captured time conditioning signature changed; recreate graph slot")
            self.entries[key] = packed
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            self.misses += 1
        else:
            self.entries.move_to_end(key)
            self.hits += 1
        # Graphs read stable views. Every replay gets the current table, including
        # switches among all four denoise times and the separate clean t=1 slot.
        self.buffer.copy_(packed)
        return self.views

    @torch.no_grad()
    def prepare_grid(self, tables):
        """Stage an immutable timestep grid with one TP agreement per chunk.

        Stable copies remain graph inputs after cache invalidation. Parameter
        signatures and every grid entry participate in the all-rank agreement.
        """
        signature = self._signature()
        keys = tuple(
            (t.dtype, tuple(t.shape), t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()) for t in tables
        )
        unchanged = signature == self.signature
        hit = unchanged and all(key in self.entries for key in keys)
        if self.group.world_size > 1:
            self.agreement.copy_(torch.tensor([unchanged, hit], dtype=torch.int32))
            torch.distributed.all_reduce(
                self.agreement, op=torch.distributed.ReduceOp.MIN, group=self.group.device_group
            )
            unchanged, hit = (bool(x) for x in self.agreement.cpu().tolist())
        if not unchanged:
            self.entries.clear()
        self.signature = signature
        # Initial prepare creates the original stable buffer/view geometry.
        # All ranks take the same miss branch, including model TP collectives.
        if self.buffer is None:
            self.prepare(tables[0])
        if not hit:
            self.grid_generation = getattr(self, "grid_generation", 0) + 1
            for key, table in zip(keys, tables, strict=True):
                outputs = self.model.time_conditioning(table)
                packed = torch.cat([x.reshape(-1) for group in outputs for x in group])
                if packed.shape != self.buffer.shape or packed.dtype != self.buffer.dtype:
                    raise RuntimeError("Realization timestep grid signature changed; recreate graph slot")
                self.entries[key] = packed
                self.misses += 1
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        else:
            self.hits += len(keys)
        if not hasattr(self, "grid_buffers"):
            self.grid_buffers = tuple(torch.empty_like(self.buffer) for _ in keys)
        if len(self.grid_buffers) != len(keys):
            raise ValueError("Captured Realization timestep count changed")
        for target, key in zip(self.grid_buffers, keys, strict=True):
            target.copy_(self.entries[key])
            self.entries.move_to_end(key)
        return self.grid_buffers
