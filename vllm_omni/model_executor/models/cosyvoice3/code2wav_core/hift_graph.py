# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded CUDA graph replay for HiFT's deterministic waveform decoder.

F0, sine/noise generation, phase carry, and per-request streaming state remain
outside the graph. Each replay consumes current mel and source tensors and
returns an owned waveform, because multiple requests can reuse one shape.
"""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class _DecodeGraph:
    graph: torch.cuda.CUDAGraph
    mel: torch.Tensor
    source: torch.Tensor
    output: torch.Tensor


class HiFTDecodeGraphs:
    """Capture recurrent exact shapes, with at most 32 resident graphs.

    ``decode`` must have warmed its ISTFT envelope before capture and remain
    deterministic for its mel, source and finalize arguments. Uncommon shapes,
    CPU execution and nested CUDA capture use that same eager implementation.
    """

    def __init__(self, decode: Callable[..., torch.Tensor], max_graphs: int = 32) -> None:
        self.decode = decode
        self.max_graphs = max_graphs
        self.graphs: dict[tuple, _DecodeGraph] = {}
        self.seen: OrderedDict[tuple, int] = OrderedDict()
        self.pool = None

    @torch.inference_mode()
    def run(self, mel: torch.Tensor, source: torch.Tensor, finalize: bool) -> torch.Tensor:
        """Replay a recurring layout without aliasing another request's output."""
        if not mel.is_cuda or not source.is_cuda or torch.cuda.is_current_stream_capturing():
            return self.decode(mel, source, finalize=finalize)
        autocast = torch.is_autocast_enabled("cuda")
        key = (
            mel.shape,
            source.shape,
            mel.device,
            mel.dtype,
            source.dtype,
            finalize,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
            autocast,
            torch.get_autocast_dtype("cuda") if autocast else None,
        )
        state = self.graphs.get(key)
        if state is None:
            if len(self.graphs) >= self.max_graphs:
                return self.decode(mel, source, finalize=finalize)
            count = self.seen.pop(key, 0) + 1
            self.seen[key] = count
            if len(self.seen) > 128:
                self.seen.popitem(last=False)
            if count < 3:
                return self.decode(mel, source, finalize=finalize)
            static_mel, static_source = mel.clone(), source.clone()
            stream = torch.cuda.Stream(device=mel.device)
            stream.wait_stream(torch.cuda.current_stream(mel.device))
            with torch.cuda.stream(stream):
                self.decode(static_mel, static_source, finalize=finalize)
            torch.cuda.current_stream(mel.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                output = self.decode(static_mel, static_source, finalize=finalize)
            if self.pool is None:
                self.pool = graph.pool()
            state = _DecodeGraph(graph, static_mel, static_source, output)
            self.graphs[key] = state
        state.mel.copy_(mel)
        state.source.copy_(source)
        state.graph.replay()
        # A later request may replay this shape before the caller consumes it.
        return state.output.clone()
