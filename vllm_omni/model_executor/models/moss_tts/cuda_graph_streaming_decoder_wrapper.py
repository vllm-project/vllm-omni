# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""CUDA graph wrapper for MOSS-TTS Local streaming codec decode."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


@dataclass
class _CapturedStreamingDecodeGraph:
    graph: CUDAGraph
    static_codes: torch.Tensor
    static_lengths: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor


class CUDAGraphStreamingDecoderWrapper:
    """Replay fixed-size streaming ``_decode_frame`` steps with CUDA graphs.

    The wrapped codec must already be inside ``codec.streaming(batch_size)``.
    Graphs are keyed by the streaming step frame count ``T``. The batch width is
    fixed to the codec stream slot count.
    """

    def __init__(
        self,
        codec: nn.Module,
        *,
        batch_size: int,
        num_quantizers: int,
        reset_streaming_state: Callable[[], None] | None = None,
    ) -> None:
        self.codec = codec
        self.batch_size = int(batch_size)
        self.num_quantizers = int(num_quantizers)
        self.reset_streaming_state = reset_streaming_state
        self.graphs: dict[int, _CapturedStreamingDecodeGraph] = {}
        self._pool = None
        self._warmed_up = False

    @property
    def is_ready(self) -> bool:
        return bool(self.graphs)

    @property
    def capture_sizes(self) -> list[int]:
        return sorted(self.graphs)

    @torch.no_grad()
    def warmup(self, device: torch.device, capture_sizes: list[int]) -> None:
        if self._warmed_up:
            return
        self._warmed_up = True
        if device.type != "cuda":
            return
        sizes = sorted({int(size) for size in capture_sizes if int(size) > 0}, reverse=True)
        if not sizes:
            return

        logger.info(
            "MOSS-TTS streaming decoder CUDA graph warmup: batch_size=%d n_vq=%d capture_sizes=%s",
            self.batch_size,
            self.num_quantizers,
            list(reversed(sizes)),
        )
        start_s = time.perf_counter()
        with torch.cuda.device(device):
            for size in sizes:
                try:
                    self._capture(size, device)
                    logger.info("  Captured MOSS-TTS streaming decoder CUDA graph for size=%d", size)
                except Exception:
                    self.graphs.pop(size, None)
                    logger.warning(
                        "  Failed to capture MOSS-TTS streaming decoder CUDA graph for size=%d; "
                        "this size will use eager decode",
                        size,
                        exc_info=True,
                    )
        logger.info(
            "MOSS-TTS streaming decoder CUDA graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(sizes),
            (time.perf_counter() - start_s) * 1000.0,
        )

    @torch.no_grad()
    def _capture(self, size: int, device: torch.device) -> None:
        codes = torch.zeros(
            self.num_quantizers,
            self.batch_size,
            size,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.full((self.batch_size,), size, dtype=torch.long, device=device)
        exec_mask = torch.ones(self.batch_size, dtype=torch.bool, device=device)

        self.codec._set_streaming_exec_mask(exec_mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _ = self.codec._decode_frame(codes, lengths)
        torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize(device)

        if self.reset_streaming_state is not None:
            self.reset_streaming_state()
        self.codec._set_streaming_exec_mask(exec_mask)
        if self._pool is None:
            self._pool = current_platform.get_global_graph_pool()
        graph = CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
            output = self.codec._decode_frame(codes, lengths)

        if output.audio is None or output.audio_lengths is None:
            raise RuntimeError("MOSS-TTS streaming decoder CUDA graph capture produced empty audio.")

        self.graphs[size] = _CapturedStreamingDecodeGraph(
            graph=graph,
            static_codes=codes,
            static_lengths=lengths,
            static_audio=output.audio,
            static_audio_lengths=output.audio_lengths,
        )

    @torch.no_grad()
    def decode(
        self,
        codes_step: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not codes_step.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        n_vq, batch_size, step_t = codes_step.shape
        if int(n_vq) != self.num_quantizers or int(batch_size) != self.batch_size:
            return None
        # The streaming codec advances its internal KV offsets by tensor T, so
        # bucket-padding a smaller step into a larger captured graph would write
        # padded frames into state. Capture exact step sizes instead.
        entry = self.graphs.get(int(step_t))
        if entry is None:
            return None

        self.codec._set_streaming_exec_mask(exec_mask)
        entry.static_codes.copy_(codes_step)
        entry.graph.replay()
        return entry.static_audio, entry.static_audio_lengths


__all__ = ["CUDAGraphStreamingDecoderWrapper"]
