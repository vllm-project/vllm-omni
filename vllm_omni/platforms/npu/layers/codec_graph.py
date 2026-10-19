# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup-only NPU graphs for the stateless Qwen3-TTS convolution segment."""

from collections.abc import Callable, Sequence

import torch


class NPUCodecGraphs:
    """One worker owns these scratch buffers; request state stays with the decoder."""

    def __init__(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        weight: torch.Tensor,
        channels: int,
        shapes: Sequence[tuple[int, int]],
    ) -> None:
        self.forward = forward
        self.entries = {}
        self.device = weight.device
        self.dtype = weight.dtype
        self.channels = channels
        shapes = sorted(set(shapes))
        if not shapes or any(b < 1 or t < 1 for b, t in shapes):
            raise ValueError("NPU codec capture shapes must be positive (batch, frames) pairs")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("NPU codec graphs must be prepared before outer graph capture")
        stream = torch.npu.Stream(device=self.device)
        stream.wait_stream(torch.npu.current_stream())
        try:
            for batch, frames in shapes:
                x = torch.zeros(batch, frames, channels, device=self.device, dtype=self.dtype)
                with torch.npu.stream(stream):
                    for _ in range(3):
                        forward(x)
                torch.npu.current_stream().wait_stream(stream)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    y = forward(x)
                self.entries[batch, frames] = (graph, x, y)
        except BaseException:
            self.close()
            raise

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        entry = self.entries.get(tuple(hidden.shape[:2]))
        if (
            entry is None
            or hidden.ndim != 3
            or hidden.shape[-1] != self.channels
            or hidden.device != self.device
            or hidden.dtype != self.dtype
            or torch.is_grad_enabled()
            or torch.npu.is_current_stream_capturing()
        ):
            return self.forward(hidden)
        graph, x, y = entry
        x.copy_(hidden)
        graph.replay()
        # The caller owns the returned waveform: a later request must not
        # overwrite it through our reusable graph output buffer.
        return y.clone()

    def close(self) -> None:
        if self.entries:
            torch.npu.synchronize()
            for graph, _, _ in self.entries.values():
                graph.reset()
            self.entries.clear()
