# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serialized native Realization single-NFE graph slot, with in-graph FP32 sampling.

Graphs capture model compute only, not allocator/refcount/publication. Stable
sequence, time, page-table, retain/slot buffers are updated in place. A slot is
bounded by query geometry and a finite KV capacity, never padded history SDPA.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import torch

from .attention import RealizationPagedAttentionPlan
from .kernels import new_kv_finite
from .paged_cache import RealizationGraphWriteBinding
from .parallel import all_ranks_finite


def clone_sequence(sequence):
    return replace(
        sequence,
        **{
            name: value.clone()
            for name in sequence.__dataclass_fields__
            if isinstance(value := getattr(sequence, name), torch.Tensor)
        },
    )


@contextmanager
def capture_graph(graph):
    """Honor native TP communicator buffer registration during capture."""
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    if hasattr(group, "graph_capture"):
        with group.graph_capture() as context:
            with torch.cuda.graph(graph, stream=context.stream):
                yield
    else:
        with torch.cuda.graph(graph):
            yield


class RealizationGraphSlot:
    def __init__(
        self,
        model,
        pool,
        sequence,
        *,
        capacity,
        commit=False,
        reuse_time_conditioning=False,
        sampling_stage="causal",
        flow_shift=1.0,
        clean_history_chunks=None,
    ):
        if sequence.device.type != "cuda":
            raise ValueError("Realization CUDA graph requires CUDA sequence")
        sequence.validate()
        self.model, self.pool = model, pool
        self.sampling_stage, self.flow_shift = sampling_stage, flow_shift
        if sampling_stage == "dmd" and not commit:
            self.next_time = torch.ones((), device=sequence.device)
            self.step_noise = tuple(torch.zeros_like(x) for x in (sequence.video, sequence.audio))
        self.sequence = clone_sequence(sequence)
        self.plan = RealizationPagedAttentionPlan(
            sequence.groups, capacity=capacity, device=sequence.device, clean_history_chunks=clean_history_chunks
        )
        self.commit = commit
        self.times = torch.ones(1 if commit else 2, device=sequence.device)
        self.dt = torch.zeros((), device=sequence.device)
        self.finite = torch.ones((), device=sequence.device, dtype=torch.bool)
        self.graph = None
        self.output = None
        self.transaction = None
        self.capture_count = 0
        self.replay_count = 0
        self.conditioning = None
        self.time_conditioning = None
        if reuse_time_conditioning:
            from .time_conditioning import RealizationTimeConditioning

            candidate = RealizationTimeConditioning(model)
            try:
                candidate._signature()
                supported = True
            except RuntimeError:
                supported = False
            if all_ranks_finite(supported, sequence.device):
                self.time_conditioning = candidate

    def _compute(self, cache, *, noise=None, next_time=None):
        kwargs = {} if self.conditioning is None else {"conditioning": self.conditioning}
        predictions, _ = self.model(
            self.sequence,
            self.times,
            cache=cache,
            commit=self.commit,
            paged_runtime=(self.plan, self.pool, self.transaction),
            validate=False,
            **kwargs,
        )
        self.finite.copy_(torch.stack([torch.isfinite(x).all() for x in predictions]).all())
        if self.commit:
            # Check only newly written slots, not the full history.
            self.finite.logical_and_(new_kv_finite(self.pool.storage, self.transaction.slots))
        else:
            self.sequence.video.add_(self.dt * predictions[0].float())
            self.sequence.audio.add_(self.dt * predictions[1].float())
            if self.sampling_stage == "dmd":
                from .sampling import renoise

                renoise(
                    (self.sequence.video, self.sequence.audio),
                    self.step_noise if noise is None else noise,
                    self.next_time if next_time is None else next_time,
                )
        # Both paths acknowledge finite output latents. Commit outputs alias
        # inputs, so prediction/clean-KV checks alone do not cover this contract.
        self.finite.logical_and_(torch.isfinite(self.sequence.video).all() & torch.isfinite(self.sequence.audio).all())
        self.output = (self.sequence.video, self.sequence.audio)

    def _copy(self, sequence, times, dt):
        if sequence.groups != self.sequence.groups or sequence.refiner_groups != self.sequence.refiner_groups:
            # Block numbers affect only metadata plans, not tensor compute. Allow
            # semantic changes with identical kind/range geometry.
            for a, b in zip(sequence.groups, self.sequence.groups, strict=True):
                if (a.kind, a.start, a.end) != (b.kind, b.start, b.end):
                    raise ValueError("Realization graph semantic geometry changed")
            for a, b in zip(sequence.refiner_groups, self.sequence.refiner_groups, strict=True):
                if (a.kind, a.start, a.end) != (b.kind, b.start, b.end):
                    raise ValueError("Realization graph refiner geometry changed")
            # Refiner visibility must not depend on a changed sample/block
            # relation: enforce its exact boolean domain matrix.
            from .contracts import visible

            for groups_a, groups_b in ((sequence.refiner_groups, self.sequence.refiner_groups),):
                for ia, ib in zip(groups_a, groups_b, strict=True):
                    for ja, jb in zip(groups_a, groups_b, strict=True):
                        if visible(ia, ja, refiner=True) != visible(ib, jb, refiner=True):
                            raise ValueError("Realization captured refiner visibility changed")
        for name in sequence.__dataclass_fields__:
            src, dst = getattr(sequence, name), getattr(self.sequence, name)
            if isinstance(src, torch.Tensor):
                if src.shape != dst.shape or src.dtype != dst.dtype or src.device != dst.device:
                    raise ValueError("Realization graph input tensor signature changed: " + name)
                dst.copy_(src)
        self.times.copy_(times)
        self.dt.fill_(dt)

    @torch.no_grad()
    def run(self, sequence, times, *, cache, dt=0.0, transaction=None, noise=None, next_time=1.0):
        sequence.validate()
        if not torch.isfinite(times).all() or times.shape != self.times.shape:
            raise ValueError("Invalid Realization graph times")
        if self.commit and (transaction is None or not torch.all(times == 1) or torch.any(sequence.time_ids != 0)):
            raise ValueError("Realization clean graph needs a t=1 page transaction")
        self.plan.update(sequence.groups, cache, self.pool, text_valid_rows=sequence.text_valid_rows)
        if self.sampling_stage == "dmd" and not self.commit:
            self.next_time.fill_(next_time)
            for dst, src in zip(self.step_noise, noise or self.step_noise, strict=True):
                dst.copy_(src)
        if self.commit:
            if self.transaction is None:
                self.transaction = RealizationGraphWriteBinding(transaction.slots.clone(), transaction.retain.clone())
            elif transaction.slots.shape != self.transaction.slots.shape:
                raise ValueError("Realization commit slot geometry changed")
            self.transaction.workspace_offset = (
                self.plan.native_runtime.past_rows if self.plan.native_runtime is not None else None
            )
            self.transaction.slots.copy_(transaction.slots)
            self.transaction.retain.copy_(transaction.retain)
            if transaction is not self.pool._pending or not transaction.pending:
                raise ValueError("Realization graph requires a live reserved transaction")
        if self.time_conditioning is not None:
            self.conditioning = self.time_conditioning.prepare(times)
        if self.graph is None:
            stream = torch.cuda.Stream(device=sequence.device)
            stream.wait_stream(torch.cuda.current_stream(sequence.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._copy(sequence, times, dt)
                    self._compute(cache)
            torch.cuda.current_stream(sequence.device).wait_stream(stream)
            self._copy(sequence, times, dt)
            torch.accelerator.synchronize(sequence.device)
            self.graph = torch.cuda.CUDAGraph()
            with capture_graph(self.graph):
                self._compute(cache)
            self.capture_count += 1
        # Capture executes once. Restore inputs before the requested replay.
        self._copy(sequence, times, dt)
        self.graph.replay()
        self.replay_count += 1
        if transaction is not None:
            transaction.written_layers.update(range(self.pool.storage.shape[0]))
        if not all_ranks_finite(self.finite, sequence.device):
            raise FloatingPointError("Realization graph produced nonfinite velocity/latent/clean KV")
        # Outputs alias graph-owned buffers; consumer must clone before another
        # replay or before queue publication.
        return self.output

    @torch.no_grad()
    def run_chain(self, sequence, *, cache, steps, reuse_static_text=False, start_step=0, text_cache=None, noises=()):
        """Replay the remaining Euler or DMD NFEs with one host completion boundary.

        DMD predicts x0 and re-noises using request-owned fixed random tensors. No clean KV is
        written here. Cancellation fences still guard the later publication.
        """
        if self.commit or steps != 4 or self.time_conditioning is None:
            raise ValueError("Realization chain requires four denoise NFEs and time reuse")
        if not hasattr(self, "chain_times"):
            from .sampling import sampling_grid

            grid, self.chain_dts, self.chain_next = sampling_grid(steps, self.sampling_stage, self.flow_shift)
            self.chain_times = tuple(torch.tensor([1.0, t], device=sequence.device) for t in grid)
            if self.sampling_stage == "dmd":
                self.chain_noises = tuple(tuple(torch.empty_like(x) for x in pair) for pair in noises)
        if self.sampling_stage == "dmd":
            for dst, src in zip(self.chain_noises, noises, strict=True):
                for a, b in zip(dst, src, strict=True):
                    a.copy_(b)
        self.plan.update(sequence.groups, cache, self.pool, text_valid_rows=sequence.text_valid_rows)
        native = self.plan.native_runtime
        reuse_static_text = reuse_static_text and native is not None
        if reuse_static_text:
            native.store_text = True
        if self.graph is None and not hasattr(self, "chain_graphs"):
            # Warm the kernels without capturing an unused single-NFE graph.
            self.conditioning = self.time_conditioning.prepare(self.chain_times[start_step])
            stream = torch.cuda.Stream(device=sequence.device)
            stream.wait_stream(torch.cuda.current_stream(sequence.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._copy(sequence, self.chain_times[start_step], self.chain_dts[start_step])
                    self._compute(cache)
            torch.cuda.current_stream(sequence.device).wait_stream(stream)
        # prepare_encode owns metadata validation; replay changes only media.
        self.plan.update(sequence.groups, cache, self.pool, text_valid_rows=sequence.text_valid_rows)
        tables = self.time_conditioning.prepare_grid(self.chain_times)
        if reuse_static_text and getattr(self, "text_generation", None) != self.time_conditioning.grid_generation:
            # Time slot zero has three modality rows per transformer block.
            # Prove their projection bytes are identical at every sampling time.
            def static_rows(table):
                rows, cursor = [], 0
                for group in self.time_conditioning.views:
                    for view in group:
                        n = view.numel()
                        rows.append(
                            table[cursor : cursor + n].view_as(view)[: 3 if view.shape[0] == 6 else 1].flatten()
                        )
                        cursor += n
                return torch.cat(rows)

            first = static_rows(tables[0])
            stable = all(torch.equal(first, static_rows(table)) for table in tables[1:])
            if not all_ranks_finite(stable, sequence.device):
                raise ValueError("Realization text time conditioning varies across the sampling grid")
            self.text_generation = self.time_conditioning.grid_generation
        self._copy(sequence, self.chain_times[start_step], self.chain_dts[start_step])
        if not hasattr(self, "chain_graphs"):
            self.chain_graphs = {}
            self.chain_finite = torch.ones_like(self.finite)
        key = (start_step, text_cache is not None)
        if text_cache is not None:
            native.text_output.copy_(text_cache)
        if key not in self.chain_graphs:
            torch.accelerator.synchronize(sequence.device)
            graph = torch.cuda.CUDAGraph()
            with capture_graph(graph):
                self.chain_finite.fill_(True)
                for index, (times, table) in enumerate(
                    zip(self.chain_times[start_step:], tables[start_step:], strict=True)
                ):
                    if native is not None:
                        native.reuse_text = reuse_static_text and (index > 0 or text_cache is not None)
                    self.times.copy_(times)
                    self.time_conditioning.buffer.copy_(table)
                    step = index + start_step
                    self.dt.fill_(self.chain_dts[step])
                    self._compute(
                        cache,
                        noise=self.chain_noises[step] if self.sampling_stage == "dmd" and step < 3 else None,
                        next_time=self.chain_next[step],
                    )
                    self.chain_finite.logical_and_(self.finite)
            if native is not None:
                native.reuse_text = False
            self._copy(sequence, self.chain_times[start_step], self.chain_dts[start_step])
            self.chain_graphs[key] = graph
            self.capture_count += 1
        self.chain_graphs[key].replay()
        self.replay_count += steps - start_step
        if not all_ranks_finite(self.chain_finite, sequence.device):
            raise FloatingPointError("Realization chain produced nonfinite velocity/latents")
        return self.output
