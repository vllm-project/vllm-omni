# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Joint clean/next-first graph, optionally followed by three more NFEs.

The first-only mode publishes clean before the remaining denoise segment.
"""

import torch

from .graph import capture_graph, clone_sequence
from .joint_attention import JointAttentionPair
from .joint_forward import joint_forward
from .kernels import new_kv_finite
from .parallel import all_ranks_finite


def copy_sequence(dst, src):
    for name in src.__dataclass_fields__:
        a, b = getattr(dst, name), getattr(src, name)
        if isinstance(a, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("Joint input geometry changed: " + name)
            a.copy_(b)


class RealizationJointCycle:
    def __init__(
        self,
        model,
        pool,
        clean,
        noisy,
        capacity=65536,
        history_blocks=180,
        separate_projections=False,
        sampling_stage="causal",
        flow_shift=1.0,
    ):
        clean.validate()
        noisy.validate()
        self.model, self.pool = model, pool
        self.separate_projections = separate_projections
        self.clean, self.noisy = clone_sequence(clean), clone_sequence(noisy)
        self.pair = JointAttentionPair(pool, capacity, history_blocks)
        self.finite = torch.ones((), device=clean.device, dtype=torch.bool)
        self.clean_times = torch.ones(1, device=clean.device)
        from .sampling import sampling_grid

        grid, self.dts, self.next_times = sampling_grid(4, sampling_stage, flow_shift)
        self.sampling_stage = sampling_stage
        self.times = tuple(torch.tensor([1.0, t], device=clean.device) for t in grid)
        self.noises = (
            tuple(tuple(torch.zeros_like(x) for x in (noisy.video, noisy.audio)) for _ in range(3))
            if sampling_stage == "dmd"
            else ()
        )
        self.ca = model.time_conditioning(self.clean_times)
        self.tables = tuple(model.time_conditioning(t) for t in self.times)
        self.joint_conditioning = tuple(
            tuple(torch.cat((a, b)) for a, b in zip(ca, cb, strict=True))
            for ca, cb in zip(self.ca, self.tables[0], strict=True)
        )
        self.graph = None
        self.first_graph = None
        self.capture_count = 0
        self.replay_count = 0

    def _compute_first(self, cache):
        self.finite.fill_(True)
        a, b = joint_forward(
            self.model,
            self.clean,
            self.noisy,
            self.pair,
            self.ca,
            self.tables[0],
            self.joint_conditioning,
            self.separate_projections,
        )
        self.finite.logical_and_(torch.stack([torch.isfinite(x).all() for x in (*a, *b)]).all())
        self._update(b, 0)
        self.finite.logical_and_(torch.isfinite(self.noisy.video).all() & torch.isfinite(self.noisy.audio).all())
        self.finite.logical_and_(new_kv_finite(self.pool.storage, self.pair.binding.slots))

    def _compute(self, cache):
        self._compute_first(cache)
        native = self.pair.next_plan.native_runtime
        native.reuse_text = True
        for i in range(1, 4):
            pred, _ = self.model(
                self.noisy,
                self.times[i],
                cache=self.pair.private_next,
                commit=False,
                paged_runtime=(self.pair.next_plan, self.pool, None),
                validate=False,
                conditioning=self.tables[i],
            )
            self.finite.logical_and_(torch.stack([torch.isfinite(x).all() for x in pred]).all())
            self._update(pred, i)
        native.reuse_text = False
        self.finite.logical_and_(torch.isfinite(self.noisy.video).all() & torch.isfinite(self.noisy.audio).all())

    def _update(self, pred, index):
        self.noisy.video.add_(pred[0].float() * self.dts[index])
        self.noisy.audio.add_(pred[1].float() * self.dts[index])
        if self.sampling_stage == "dmd" and index < 3:
            from .sampling import renoise

            renoise((self.noisy.video, self.noisy.audio), self.noises[index], self.next_times[index])

    def _restore_stage(self, clean, noisy, cache, tx):
        # Warmup/capture mutate staging. Restore from acknowledged physical KV.
        self.pool._realization_native_attention["cache"] = None
        self.pair.prepare(clean, noisy, cache, tx)
        copy_sequence(self.clean, clean)
        copy_sequence(self.noisy, noisy)

    @torch.no_grad()
    def run(self, clean, noisy, cache, *, first_only=False, noises=()):
        # Prepared metadata is immutable; replay copies media and updates KV.
        for dst, src in zip(self.noises, noises, strict=True):
            for a, b in zip(dst, src, strict=True):
                a.copy_(b)
        tx = self.pool.reserve(cache, clean.groups)
        try:
            self.pair.prepare(clean, noisy, cache, tx)
            graph_name = "first_graph" if first_only else "graph"
            graph = getattr(self, graph_name)
            compute = self._compute_first if first_only else self._compute
            if graph is None:
                self.captured_drop = self.pair.drop
                # Warm up outside capture; restore original historical values each time.
                for _ in range(2):
                    self._restore_stage(clean, noisy, cache, tx)
                    compute(cache)
                self._restore_stage(clean, noisy, cache, tx)
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with capture_graph(graph):
                    compute(cache)
                setattr(self, graph_name, graph)
                self._restore_stage(clean, noisy, cache, tx)
                self.capture_count += 1
            else:
                copy_sequence(self.clean, clean)
                copy_sequence(self.noisy, noisy)
            graph.replay()
            self.replay_count += 1
            if not all_ranks_finite(self.finite, clean.device):
                raise FloatingPointError("Joint clean/noisy/KV finite acknowledgement failed")
            tx.written_layers.update(range(self.pool.storage.shape[0]))
            published = self.pool.publish(tx, acknowledged=True, finite=True)
            first = max(g.block for g in noisy.groups) - self.pair.history_blocks
            published = self.pool.evict(published, first)
            self.pair.stage_published(published)
            return (self.noisy.video.clone(), self.noisy.audio.clone()), published
        except BaseException:
            if tx.pending:
                self.pool.abort(tx)
            if hasattr(self.pool, "_realization_native_attention"):
                self.pool._realization_native_attention["cache"] = None
            raise
