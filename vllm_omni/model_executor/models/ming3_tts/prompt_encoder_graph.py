# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for the Ming3 prompt-audio path.

The AudioVAE encoder and the latent Aggregator run once per reference clip
inside the Stage-0 prefill step. Eager, they launch about a thousand small
kernels and stall every active stream for tens of milliseconds. The frame
encoder and patch aggregator are causal and the Aggregator treats each patch
row independently, so right-padding the input with whole latent groups leaves
the outputs for the real groups unchanged; one graph per padded length covers
any clip up to the largest bucket.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

# Bucket sizes in latent groups (80 ms each): 0.64 s to 30.72 s of audio.
DEFAULT_BUCKET_GROUPS = (8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384)
# Projection rows for the prompts of one prefill step together.
PROJECTION_BUCKET_ROWS = DEFAULT_BUCKET_GROUPS + (512, 768, 1024, 1536, 2048)


class PaddedBucketGraphs:
    """Replay ``fn`` from per-length CUDA graphs over a right-padded input.

    The input is padded along ``in_dim`` to ``bucket * in_unit`` elements; the
    output is cut back to the requested number of groups along ``out_dim``.
    ``fn`` must be causal (or row-independent) along that dimension.
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        template: torch.Tensor,
        *,
        in_dim: int,
        in_unit: int,
        out_dim: int,
        bucket_groups: Sequence[int] = DEFAULT_BUCKET_GROUPS,
    ):
        self.fn = fn
        self.template = template
        self.in_dim = in_dim
        self.in_unit = int(in_unit)
        self.out_dim = out_dim
        self.bucket_groups = sorted(int(groups) for groups in bucket_groups)
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    @torch.inference_mode()
    def capture(self, pool=None) -> None:
        pool = torch.cuda.graph_pool_handle() if pool is None else pool
        # Largest first, so smaller graphs reuse the shared pool's blocks.
        for groups in reversed(self.bucket_groups):
            shape = list(self.template.shape)
            shape[self.in_dim] = groups * self.in_unit
            static_input = self.template.new_zeros(shape)
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                self.fn(static_input)
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                static_output = self.fn(static_input)
            self._graphs[groups] = (graph, static_input, static_output)

    def __call__(self, value: torch.Tensor) -> torch.Tensor | None:
        """Return ``fn(value)`` from a graph, or None when no bucket fits."""
        if value.ndim != self.template.ndim or value.dtype != self.template.dtype:
            return None
        if any(
            size != expected
            for dim, (size, expected) in enumerate(zip(value.shape, self.template.shape))
            if dim != self.in_dim % value.ndim
        ):
            return None
        length = value.shape[self.in_dim]
        if length % self.in_unit:
            return None
        groups = length // self.in_unit
        bucket = next((size for size in self.bucket_groups if size >= groups), None)
        if bucket is None or bucket not in self._graphs:
            return None
        graph, static_input, static_output = self._graphs[bucket]
        static_input.narrow(self.in_dim, 0, length).copy_(value)
        static_input.narrow(self.in_dim, length, static_input.shape[self.in_dim] - length).zero_()
        graph.replay()
        # The caller consumes this view before the next replay.
        return static_output.narrow(self.out_dim, 0, groups)


__all__ = ["DEFAULT_BUCKET_GROUPS", "PROJECTION_BUCKET_ROWS", "PaddedBucketGraphs"]
