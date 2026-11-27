# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FlashInfer backend for packed Flow attention (opt-in).

Full-context segments are whole rows, so they run FlashInfer's ragged prefill
on the packed K/V. Chunk-causal segments read a prefix of their row through
one-frame pages: the packed K/V itself is the paged cache, so nothing is
gathered. Every DiT layer and Euler step of one solve shares one layout, so a
solve plans once on the host and each attention call only runs the kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

# All plans of a device share one float and one int workspace. Solves, graph
# replays and their re-plans run one after another, so they never overlap.
_FLOAT_WORKSPACE_BYTES = 128 << 20
_INT_WORKSPACE_BYTES = 8 << 20
_workspaces: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

# The plan the attention op runs, as the prefix cache's pools are reached:
# Inductor never sees it as a graph input.
_ACTIVE: FlashInferRowPlan | None = None


def _workspace(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if device not in _workspaces:
        _workspaces[device] = (
            torch.empty(_FLOAT_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
            torch.empty(_INT_WORKSPACE_BYTES, dtype=torch.uint8, device=device),
        )
    return _workspaces[device]


def paged_index_capacity(total: int, chunk: int) -> int:
    """Static bound on one-frame KV indices for twin rows of at most ``total``
    frames each, ``2 * total`` frames in all, split into ``chunk``-frame queries.

    A row of length L reads ``chunk * n(n - 1) / 2 + L`` frames over its
    ``n = ceil(L / chunk)`` chunks; that is convex in L, so two full rows
    are the worst case.
    """
    per_row = chunk * (-(-total // chunk)) * (-(-total // chunk) - 1) // 2 + total
    return 2 * per_row


class FlashInferRowPlan:
    """One FlashInfer wrapper and its plan for packed rows.

    ``graph_capacity`` = (segments, kv indices) allocates fixed metadata
    buffers so a CUDA graph captured on this plan replays after any re-plan
    of the same capacity.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        chunked: bool,
        graph_capacity: tuple[int, int] | None = None,
    ) -> None:
        import flashinfer

        self.heads, self.head_dim, self.dtype, self.chunked = heads, head_dim, dtype, chunked
        self.device = device
        float_workspace, int_workspace = _workspace(device)
        graph = graph_capacity is not None
        buffers = {}
        if chunked:
            if graph:
                segments, indices = graph_capacity
                buffers = dict(
                    qo_indptr_buf=torch.zeros(segments + 1, dtype=torch.int32, device=device),
                    paged_kv_indptr_buf=torch.zeros(segments + 1, dtype=torch.int32, device=device),
                    paged_kv_indices_buf=torch.zeros(indices, dtype=torch.int32, device=device),
                    paged_kv_last_page_len_buf=torch.zeros(segments, dtype=torch.int32, device=device),
                )
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                float_workspace, "NHD", use_cuda_graph=graph, **buffers
            )
        else:
            if graph:
                segments, _ = graph_capacity
                buffers = dict(
                    qo_indptr_buf=torch.zeros(segments + 1, dtype=torch.int32, device=device),
                    kv_indptr_buf=torch.zeros(segments + 1, dtype=torch.int32, device=device),
                )
            self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                float_workspace, "NHD", use_cuda_graph=graph, **buffers
            )
        self.wrapper.reset_workspace_buffer(float_workspace, int_workspace)

    def plan(
        self,
        lengths: Sequence[int],
        segment_rows: Sequence[int],
        segment_ends: Sequence[int],
        offsets: Sequence[int],
    ) -> None:
        """Plan segment ``s``: queries ``offsets[s]:offsets[s + 1]`` attend the
        first ``segment_ends[s]`` frames of row ``segment_rows[s]``."""
        qo_indptr = torch.tensor(offsets, dtype=torch.int32)
        # Packed Flow batches already fill the GPU with segments, so split-KV
        # only adds a merge pass. Graph plans still size query tiles for their
        # worst-case layout, so replays may reduce in another order than
        # ordinary solves.
        common = dict(causal=False, q_data_type=self.dtype, disable_split_kv=True)
        if not self.chunked:
            # Full context: each segment's K/V span is its query span.
            self.wrapper.plan(qo_indptr, qo_indptr, self.heads, self.heads, self.head_dim, **common)
            return
        # FlashInfer reads every index tensor as int32.
        ends = torch.tensor(segment_ends, dtype=torch.int32)
        row_starts = F.pad(torch.tensor(lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0))
        kv_indptr = F.pad(ends.cumsum(0, dtype=torch.int32), (1, 0))
        # Segment s reads frames row_start .. row_start + end of the packing.
        shift = row_starts[torch.tensor(segment_rows, dtype=torch.long)] - kv_indptr[:-1]
        indices = torch.arange(int(kv_indptr[-1]), dtype=torch.int32) + shift.repeat_interleave(ends.long())
        if self.device.type == "cuda":
            # FlashInfer copies host indices into a graph's buffer synchronously,
            # which would drain the stream before every re-plan.
            indices = indices.pin_memory().to(self.device, non_blocking=True)
        last_page_len = torch.ones(len(segment_ends), dtype=torch.int32)
        self.wrapper.plan(
            qo_indptr, kv_indptr, indices, last_page_len, self.heads, self.heads, self.head_dim, 1, **common
        )

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.chunked:
            return self.wrapper.run(q, (k[:, None], v[:, None]))
        return self.wrapper.run(q, k, v)


def activate(plan: FlashInferRowPlan) -> None:
    global _ACTIVE
    _ACTIVE = plan


@torch.library.custom_op(
    "vllm_omni_cosyvoice3_perf::flashinfer_row_attention",
    mutates_args=(),
    device_types="cuda",
)
def flashinfer_row_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Packed-row attention on the active FlashInfer plan."""
    return _ACTIVE.run(q, k, v)


@flashinfer_row_attention.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(q)
