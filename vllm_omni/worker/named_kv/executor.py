# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent executor for the named KV branch graph path.

Owns the compiled callable, static execution buffers, and CUDA Graph
capture/replay.  The executor is constructed only when the model-level
``negative_cuda_graph`` switch is enabled and the backend adapter validates
successfully.

The executor shares the Qwen2 model's parameters and compute modules via
the model adapter, but owns its own compiled callable and graph entries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger

from vllm_omni.worker.named_kv.types import NamedKVAppendBatch

if TYPE_CHECKING:
    from vllm_omni.worker.named_kv.runtime import NamedCausalKVBranch

logger = init_logger(__name__)


@dataclass
class ExecutionBuffers:
    """Static per-batch-size execution buffers owned by the executor."""

    embeddings: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    output: torch.Tensor


class NamedKVBranchExecutor:
    """Independent compilation entry, static buffers, and CUDA Graph for the
    negative branch."""

    def __init__(
        self,
        branch: NamedCausalKVBranch,
        model_adapter: Any,
        *,
        max_num_seqs: int,
        max_model_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.branch = branch
        self.model_adapter = model_adapter
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = int(max_model_len)
        self.device = torch.device(device)
        self.dtype = dtype
        self._capacity = self.max_model_len  # fixed upper bound

        # K/V cache views are shared between executor and model adapter.
        self._k_caches = [pair[1] for pair in model_adapter._layer_pairs]
        self._v_caches = [pair[2] for pair in model_adapter._layer_pairs]

        self._buffers: dict[int, ExecutionBuffers] = {}
        self._eager_fn: Callable[..., None] | None = self._build_forward_into()
        self._compiled_fn: Callable[..., None] | None = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_output_refs: dict[int, torch.Tensor] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        step: NamedKVAppendBatch,
        input_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Execute negative Qwen forward with graph replay if available."""
        if self._closed:
            raise RuntimeError("NamedKVBranchExecutor is closed")
        batch_size = len(step.request_ids)
        if not (1 <= batch_size <= self.max_num_seqs):
            raise ValueError(f"Batch size {batch_size} out of range [1, {self.max_num_seqs}]")

        # Never warm up inside a real append transaction: capture writes KV.
        # Uncaptured sizes consume this prepared step via the eager callable.

        bufs = self._ensure_buffers(batch_size)
        self._write_buffers(bufs, batch_size, step, input_embeddings)

        # A workspace, once allocated, is also used by uncaptured sizes.
        if self._compiled_fn is not None:
            self._update_scheduler_metadata(bufs, batch_size)

        if batch_size in self._graphs:
            self._graphs[batch_size].replay()
            return self._graph_output_refs[batch_size][:batch_size].clone()

        # Compiled/eager fallback: use compiled_fn if available, else eager.
        fn = self._compiled_fn if self._compiled_fn is not None else self._eager_fn
        assert fn is not None
        args = self._buffer_args(bufs, batch_size)
        fn(*args)
        return bufs.output[:batch_size].clone()

    def warmup(self, batch_sizes: list[int]) -> None:
        """Compile and capture CUDA Graphs for the specified batch sizes.

        Called from ``warmup_side_graphs`` before serving real requests.
        """
        if self._closed:
            raise RuntimeError("NamedKVBranchExecutor is closed")

        self.branch._ensure_not_entered("warm up")
        if any(not 1 <= b <= self.max_num_seqs for b in batch_sizes):
            raise ValueError("Warmup batch size out of range")
        try:
            # Do not replace workspace pointers already referenced by graphs.
            if self._compiled_fn is None:
                self.model_adapter.init_graph_workspace(self.max_num_seqs)

            # Compile once (not per-B).
            if self._compiled_fn is None and self._eager_fn is not None:
                self._compiled_fn = torch.compile(
                    self._eager_fn,
                    backend="inductor",
                    fullgraph=True,
                    options={"triton.cudagraphs": False},
                )

            for b in batch_sizes:
                if b in self._graphs:
                    continue
                self._capture_owned_batch(b)
        except BaseException:
            # Startup is aborting: release even earlier successful captures.
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close named KV executor after warmup failure")
            raise

    def _capture_owned_batch(self, b: int) -> None:
        """Capture with allocator-owned scratch requests, outside real appends."""
        bufs = self._ensure_buffers(b)
        dummy_ids = [f"__named_kv_warmup_{id(self)}_{b}_{i}" for i in range(b)]
        allocated = []
        try:
            for rid in dummy_ids:
                self.branch.reset(rid)
                allocated.append(rid)
            with self.branch.append_batch(dummy_ids) as step:
                self._write_buffers(bufs, b, step, self._dummy_embeddings(b))
                self._update_scheduler_metadata(bufs, b)
                args = self._buffer_args(bufs, b)
                assert self._compiled_fn is not None
                self._compiled_fn(*args)
                torch.accelerator.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._compiled_fn(*args)
                self._graphs[b] = graph
                self._graph_output_refs[b] = bufs.output
                logger.info("Captured independent named KV graph for batch_size=%d", b)
        except BaseException:
            for rid in allocated:
                try:
                    self.branch.free(rid)
                except Exception:
                    logger.exception("Failed to release warmup request %s", rid)
            raise
        else:
            for rid in allocated:
                self.branch.free(rid)

    def close(self) -> None:
        """Release graphs, static buffers, and pool view references."""
        if self._closed:
            return
        self._graphs.clear()
        self._graph_output_refs.clear()
        self._buffers.clear()
        self._compiled_fn = None
        self._eager_fn = None
        try:
            self.model_adapter.close()
        finally:
            self._k_caches.clear()
            self._v_caches.clear()
            self._closed = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_forward_into(self) -> Callable[..., None]:
        """Build the eager wrapper that writes model output into a buffer."""
        capacity = self._capacity
        model_forward = self.model_adapter.forward

        def forward_into(
            embeddings: torch.Tensor,
            positions: torch.Tensor,
            slot_mapping: torch.Tensor,
            block_table: torch.Tensor,
            query_start_loc: torch.Tensor,
            seq_lens: torch.Tensor,
            output: torch.Tensor,
        ) -> None:
            hidden = model_forward(
                embeddings,
                positions,
                slot_mapping,
                block_table,
                query_start_loc,
                seq_lens,
                max_seq_len=capacity,
            )
            output.copy_(hidden)

        return forward_into

    def _ensure_buffers(self, batch_size: int) -> ExecutionBuffers:
        if batch_size in self._buffers:
            return self._buffers[batch_size]
        max_blocks = self.branch.max_blocks_per_request
        bufs = ExecutionBuffers(
            embeddings=torch.empty(batch_size, self.model_adapter.hidden_size, dtype=self.dtype, device=self.device),
            positions=torch.empty(batch_size, dtype=torch.long, device=self.device),
            slot_mapping=torch.empty(batch_size, dtype=torch.int64, device=self.device),
            block_table=torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=self.device),
            query_start_loc=torch.arange(0, batch_size + 1, dtype=torch.int32, device=self.device),
            seq_lens=torch.empty(batch_size, dtype=torch.int32, device=self.device),
            output=torch.empty(batch_size, self.model_adapter.hidden_size, dtype=self.dtype, device=self.device),
        )
        self._buffers[batch_size] = bufs
        return bufs

    def _write_buffers(
        self,
        bufs: ExecutionBuffers,
        batch_size: int,
        step: NamedKVAppendBatch,
        input_embeddings: list[torch.Tensor],
    ) -> None:
        bufs.embeddings[:batch_size].copy_(torch.cat(input_embeddings, dim=0))
        bufs.positions[:batch_size].copy_(torch.tensor(step.positions, device=self.device))
        bufs.slot_mapping[:batch_size].copy_(torch.tensor(step.slot_values, device=self.device))
        for i, block_ids in enumerate(step.block_ids):
            bufs.block_table[i, : len(block_ids)].copy_(torch.tensor(block_ids, dtype=torch.int32, device=self.device))
        bufs.seq_lens[:batch_size].copy_(torch.tensor(step.seq_lens, device=self.device))

    def _buffer_args(self, bufs: ExecutionBuffers, batch_size: int) -> tuple:
        return (
            bufs.embeddings[:batch_size],
            bufs.positions[:batch_size],
            bufs.slot_mapping[:batch_size],
            bufs.block_table[:batch_size],
            bufs.query_start_loc[: batch_size + 1],
            bufs.seq_lens[:batch_size],
            bufs.output[:batch_size],
        )

    def _update_scheduler_metadata(self, bufs: ExecutionBuffers, batch_size: int) -> None:
        """Compute and store scheduler_metadata before graph replay.

        During CUDA Graph replay, the custom op's Python code does not
        re-execute.  The FA kernel reads scheduler_metadata from the
        pre-allocated workspace tensor.  We must fill it with correct
        values for the current seq_lens BEFORE replaying the graph.
        """
        from vllm.v1.attention.backends.fa_utils import get_scheduler_metadata

        batch_size_int = int(batch_size)
        seq_lens = bufs.seq_lens[:batch_size_int]
        query_start_loc = bufs.query_start_loc[: batch_size_int + 1]

        # Compute scheduler_metadata for each layer.
        # All layers share the same metadata shape (same batch, same seq lens).
        computed = get_scheduler_metadata(
            batch_size=batch_size_int,
            max_seqlen_q=1,
            max_seqlen_k=self._capacity,
            num_heads_q=self.model_adapter._layer_pairs[0][3]["num_heads_q"],
            num_heads_kv=self.model_adapter._layer_pairs[0][3]["num_kv_heads"],
            headdim=self.model_adapter._layer_pairs[0][3]["head_size"],
            cache_seqlens=seq_lens,
            qkv_dtype=self.dtype,
            cu_seqlens_q=query_start_loc,
            page_size=self.model_adapter._block_size,
            causal=True,
            window_size=(-1, -1),
            num_splits=1,
        )
        if computed is None:
            raise RuntimeError("FA3 did not return scheduler metadata")
        # The allocation formula is an upper bound, not the returned ABI size.
        n = computed.numel()
        if computed.ndim != 1 or computed.dtype != torch.int32 or n == 0:
            raise RuntimeError("Unexpected FA3 scheduler metadata shape or dtype")
        sizes = self.model_adapter._scheduler_metadata_sizes
        if batch_size_int in sizes and sizes[batch_size_int] != n:
            raise RuntimeError("FA3 scheduler metadata size changed after initialization")
        for workspace in self.model_adapter._scheduler_metadata:
            if workspace is None or n > workspace.numel() or workspace.device != computed.device:
                raise RuntimeError("FA3 scheduler metadata exceeds or mismatches workspace")
        sizes[batch_size_int] = n
        # Qwen layers have identical attention geometry and scheduling inputs.
        for workspace in self.model_adapter._scheduler_metadata:
            workspace[:n].copy_(computed)
            workspace[n:].zero_()

    def _dummy_embeddings(self, batch_size: int) -> list[torch.Tensor]:
        return [
            torch.zeros(
                1,
                self.model_adapter.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(batch_size)
        ]


__all__ = ["NamedKVBranchExecutor"]
