# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture codec-owned work without synchronizing the independent AR thread."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def capture_codec_graph(
    graph: torch.cuda.CUDAGraph,
    pool: tuple[int, int] | None = None,
    *,
    stream: torch.cuda.Stream,
) -> Iterator[None]:
    """Capture on an owner-private stream with thread-local error checking.

    Callers must warm up outside capture, order static inputs/warmup work onto
    the caller stream, and own all captured buffers until the last replay completes.
    A single codec owner serializes captures and replays using its shared pool;
    AR graph construction must be complete before starting that owner.

    Using capture_begin/end directly avoids torch.cuda.graph's device-wide
    synchronize/empty-cache preamble and its shared default capture stream.
    Thread-local mode still rejects unsafe operations on the capturing thread;
    it permits independent AR allocations on other threads. This is not relaxed
    capture and does not relax buffer ownership or cross-stream dependencies.
    """
    # An explicit non-default capture stream also supports the two-stage
    # decoder, whose caller may use CUDA's legacy default stream.
    caller = torch.cuda.current_stream()
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        try:
            yield
        finally:
            graph.capture_end()
    caller.wait_stream(stream)
