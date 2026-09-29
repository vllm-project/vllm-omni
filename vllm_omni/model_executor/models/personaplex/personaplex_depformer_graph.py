# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graphs for the duplex depformer step of PersonaPlex Stage 0.

The step (teacher-forcing gather, depformer, frame-state commit) is a few
thousand small kernels. It replays from one graph per padded batch size: vLLM's
cudagraph capture sizes up to ``max_num_seqs``. Padding rows read the neutral
teacher-forcing row and commit to the runtime's scratch row, and every op is row
independent. Without a graph for a bucket (off CUDA, capture failed, or under an
outer capture) the same padded body runs eagerly; only batches above the largest
bucket run eagerly at their own size.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["PersonaPlexDepformerGraphs", "depformer_graph_buckets"]

_GRAPH_WARMUP_ITERS = 2
# Pinned slot-upload buffers in rotation, so the host may run this many steps
# ahead of the device (async scheduling) without a wait.
_HOST_SLOT_BUFFERS = 3


def depformer_graph_buckets(capture_sizes: list[int] | None, max_num_seqs: int) -> list[int]:
    """vLLM's cudagraph capture sizes up to ``max_num_seqs``; without any, powers of two and ``max_num_seqs``."""
    sizes = sorted({int(size) for size in capture_sizes or () if 0 < int(size) <= max_num_seqs})
    return sizes or sorted({max_num_seqs, *(1 << i for i in range(max_num_seqs.bit_length()))})


class PersonaPlexDepformerGraphs:
    """Run the duplex depformer step, replayed from per-bucket CUDA graphs.

    A step is: gather each row's teacher forcing from the runtime's slot buffers,
    run ``depformer`` for ``num_steps`` inner steps, and commit each row's
    effective agent frame and text token back to the slot buffers. The inputs
    live in static buffers sized for the largest bucket; row ``i`` reads slot
    ``read[i]`` and writes slot ``write[i]`` from
    :meth:`PersonaPlexStage0DuplexRuntime.depformer_rows`.
    """

    def __init__(
        self,
        depformer: Any,
        runtime: Any,
        *,
        buckets: list[int],
        num_steps: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.depformer = depformer
        self.runtime = runtime
        self.buckets = sorted(set(int(bucket) for bucket in buckets))
        self.num_steps = int(num_steps)
        self.device = torch.device(device)
        max_rows = self.buckets[-1]
        scratch = runtime.scratch_slot
        self._text = torch.zeros(max_rows, dtype=torch.long, device=self.device)
        self._hidden = torch.zeros(max_rows, 1, hidden_size, dtype=dtype, device=self.device)
        # [read; write] slots of every row; padding rows stay on the scratch row.
        self._slots = torch.full((2, max_rows), scratch, dtype=torch.long, device=self.device)
        self._host_slots = [
            torch.full((2, max_rows), scratch, dtype=torch.long, pin_memory=self._on_cuda)
            for _ in range(_HOST_SLOT_BUFFERS)
        ]
        # Each buffer's last upload; an event never recorded queries as complete.
        self._host_slot_events: list[torch.cuda.Event | None] = (
            [torch.cuda.Event() for _ in range(_HOST_SLOT_BUFFERS)] if self._on_cuda else [None] * _HOST_SLOT_BUFFERS
        )
        self._next_host_slots = 0
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}

    @property
    def _on_cuda(self) -> bool:
        return self.device.type == "cuda"

    @torch.inference_mode()
    def run(self, request_ids: list[str], text_tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Run one post-sample step and return its ``[rows, num_steps]`` codes on the device.

        ``text_tokens`` is ``[rows]`` and ``hidden`` ``[rows, 1, hidden_size]``,
        row ``i`` belonging to ``request_ids[i]``. The codes are the caller's
        own tensor (not the graph's output buffer), and nothing here waits for
        the device.
        """
        rows = len(request_ids)
        if rows == 0:
            return torch.empty((0, self.num_steps), dtype=torch.long, device=self.device)
        read, write = self.runtime.depformer_rows(request_ids)
        padded = next((bucket for bucket in self.buckets if bucket >= rows), None)
        if padded is None:
            slots = torch.tensor([read, write], dtype=torch.long, pin_memory=self._on_cuda)
            slots = slots.to(self.device, non_blocking=True)
            return self._step(
                text_tokens.to(self.device, torch.long),
                hidden.to(self.device, self._hidden.dtype),
                slots[0],
                slots[1],
            )

        # The whole (small, contiguous) buffer goes up, padding rows on the
        # scratch row, once that buffer's previous upload has completed.
        index = self._next_host_slots
        self._next_host_slots = (index + 1) % _HOST_SLOT_BUFFERS
        host, event = self._host_slots[index], self._host_slot_events[index]
        if event is not None and not event.query():
            event.synchronize()
        host[0, :rows] = torch.tensor(read, dtype=torch.long)
        host[1, :rows] = torch.tensor(write, dtype=torch.long)
        host[:, rows:] = self.runtime.scratch_slot
        self._slots.copy_(host, non_blocking=True)
        if event is not None:
            event.record()
        self._text[:rows].copy_(text_tokens.reshape(rows))
        self._hidden[:rows].copy_(hidden.reshape(rows, 1, -1))
        graph = self._graphs.get(padded)
        if graph is not None and not torch.cuda.is_current_stream_capturing():
            graph.replay()
            # The next replay overwrites the output buffer: hand out a copy,
            # made on this stream before that replay can run.
            return self._outputs[padded][:rows].clone()
        return self._step(*self._static_args(padded))[:rows]

    def _static_args(self, rows: int) -> tuple[torch.Tensor, ...]:
        return self._text[:rows], self._hidden[:rows], self._slots[0, :rows], self._slots[1, :rows]

    def _step(
        self,
        text_tokens: torch.Tensor,
        hidden: torch.Tensor,
        read: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        """The captured body: teacher-forcing gather, depformer, frame-state commit."""
        tokens, provided = self.runtime.teacher_forcing_rows(read)
        codes = self.depformer(
            text_tokens,
            hidden,
            audio_tokens=tokens,
            audio_provided=provided,
            num_steps=self.num_steps,
        ).to(torch.long)
        self.runtime.commit_rows(write, text_tokens, codes, tokens, provided)
        return codes

    @torch.inference_mode()
    def capture(self) -> bool:
        """Capture one graph per bucket, largest first, into one private pool.

        Buckets never replay concurrently and each step copies its output to the
        host right after the replay, so the buckets can share a pool; it must not
        be vLLM's, whose Helium output is still read after this step. Warmup and
        capture read and write only the scratch row. Returns whether every
        bucket has a graph.
        """
        if not self._on_cuda:
            return False
        self._slots.fill_(self.runtime.scratch_slot)
        pool = torch.cuda.graph_pool_handle()
        for rows in reversed(self.buckets):
            if rows in self._graphs:
                continue
            args = self._static_args(rows)
            try:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(_GRAPH_WARMUP_ITERS):
                        self._step(*args)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                    output = self._step(*args)
            except Exception:
                logger.warning(
                    "PersonaPlex depformer CUDA graph capture failed at %d rows; those steps run eagerly",
                    rows,
                    exc_info=True,
                )
                continue
            self._graphs[rows] = graph
            self._outputs[rows] = output
        logger.info("Captured PersonaPlex depformer CUDA graphs for batch sizes %s", sorted(self._graphs))
        return len(self._graphs) == len(self.buckets)
