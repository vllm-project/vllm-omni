# Copyright 2026 OpenMOSS and the vLLM-Omni team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""CUDA graphs for compact MOSS-TTS streaming codec batches."""

from __future__ import annotations

import time
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
    static_state_slot_ids: torch.Tensor
    static_valid_rows: torch.Tensor
    static_audio: torch.Tensor
    static_audio_lengths: torch.Tensor


class CUDAGraphStreamingDecoderWrapper:
    """Replay streaming decode graphs keyed by ``(B_bucket, exact_T)``.

    Persistent codec state is addressed through ``state_slot_ids``. Graph
    padding rows use non-leaseable scratch slots, so graph batch width is no
    longer tied to the number of live stream states.
    """

    def __init__(
        self,
        codec: nn.Module,
        *,
        state_capacity: int,
        batch_sizes: list[int],
        frame_sizes: list[int],
        num_quantizers: int,
    ) -> None:
        self.codec = codec
        self.state_capacity = int(state_capacity)
        self.batch_sizes = sorted({int(size) for size in batch_sizes if 0 < int(size) <= state_capacity})
        self.frame_sizes = sorted({int(size) for size in frame_sizes if int(size) > 0})
        self.num_quantizers = int(num_quantizers)
        self.graphs: dict[tuple[int, int], _CapturedStreamingDecodeGraph] = {}
        self._pool = None
        self._warmed_up = False

    @property
    def is_ready(self) -> bool:
        return bool(self.graphs)

    @property
    def capture_sizes(self) -> list[tuple[int, int]]:
        return sorted(self.graphs)

    @property
    def scratch_capacity(self) -> int:
        return max(self.batch_sizes, default=0)

    @torch.no_grad()
    def warmup(self, device: torch.device) -> None:
        if self._warmed_up:
            return
        self._warmed_up = True
        if device.type != "cuda" or not self.batch_sizes or not self.frame_sizes:
            return

        capture_keys = sorted(
            ((batch_size, frame_size) for batch_size in self.batch_sizes for frame_size in self.frame_sizes),
            reverse=True,
        )
        logger.info(
            "MOSS-TTS streaming decoder CUDA graph warmup: n_vq=%d (B,T)=%s",
            self.num_quantizers,
            list(reversed(capture_keys)),
        )
        start_s = time.perf_counter()
        with torch.cuda.device(device):
            for batch_size, frame_size in capture_keys:
                key = (batch_size, frame_size)
                try:
                    self._capture(batch_size, frame_size, device)
                    logger.info("  Captured MOSS-TTS streaming decoder CUDA graph for (B,T)=%s", key)
                except Exception:
                    self.graphs.pop(key, None)
                    logger.warning(
                        "  Failed to capture MOSS-TTS streaming decoder CUDA graph for (B,T)=%s; using eager",
                        key,
                        exc_info=True,
                    )
        logger.info(
            "MOSS-TTS streaming decoder CUDA graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(capture_keys),
            (time.perf_counter() - start_s) * 1000.0,
        )

    @torch.no_grad()
    def _capture(self, batch_size: int, frame_size: int, device: torch.device) -> None:
        codes = torch.zeros(
            self.num_quantizers,
            batch_size,
            frame_size,
            dtype=torch.long,
            device=device,
        )
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        scratch_slots = self.state_capacity + torch.arange(batch_size, dtype=torch.long, device=device)
        valid_rows = torch.zeros(batch_size, dtype=torch.bool, device=device)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _ = self.codec.decode_streaming_batch(codes, lengths, scratch_slots, valid_rows)
        torch.cuda.current_stream().wait_stream(stream)
        torch.accelerator.synchronize(device)
        self.codec.reset_decoder_state_slots(scratch_slots)

        if self._pool is None:
            self._pool = current_platform.get_global_graph_pool()
        graph = CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
            output = self.codec.decode_streaming_batch(codes, lengths, scratch_slots, valid_rows)

        if output.audio is None or output.audio_lengths is None:
            raise RuntimeError("MOSS-TTS streaming decoder CUDA graph capture produced empty audio.")
        self.graphs[(batch_size, frame_size)] = _CapturedStreamingDecodeGraph(
            graph=graph,
            static_codes=codes,
            static_lengths=lengths,
            static_state_slot_ids=scratch_slots,
            static_valid_rows=valid_rows,
            static_audio=output.audio,
            static_audio_lengths=output.audio_lengths,
        )

    def _select_batch_size(self, actual_batch_size: int) -> int | None:
        return next((size for size in self.batch_sizes if size >= actual_batch_size), None)

    def _select_frame_size(self, actual_frame_size: int, allow_padding: bool) -> int | None:
        if actual_frame_size in self.frame_sizes:
            return actual_frame_size
        if not allow_padding:
            return None
        return next((size for size in self.frame_sizes if size >= actual_frame_size), None)

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        state_slot_ids: torch.Tensor,
        *,
        allow_frame_padding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        if not codes.is_cuda or torch.cuda.is_current_stream_capturing():
            return None
        n_vq, actual_batch_size, frame_size = codes.shape
        if int(n_vq) != self.num_quantizers or state_slot_ids.shape != (actual_batch_size,):
            return None
        batch_size = self._select_batch_size(int(actual_batch_size))
        if batch_size is None:
            return None
        graph_frame_size = self._select_frame_size(int(frame_size), allow_frame_padding)
        if graph_frame_size is None:
            return None
        entry = self.graphs.get((batch_size, graph_frame_size))
        if entry is None:
            return None

        entry.static_codes.zero_()
        entry.static_codes[:, :actual_batch_size, :frame_size].copy_(codes, non_blocking=True)
        entry.static_lengths.zero_()
        entry.static_lengths[:actual_batch_size].fill_(int(frame_size))
        entry.static_state_slot_ids.copy_(
            self.state_capacity + torch.arange(batch_size, dtype=torch.long, device=entry.static_state_slot_ids.device)
        )
        entry.static_state_slot_ids[:actual_batch_size].copy_(state_slot_ids, non_blocking=True)
        entry.static_valid_rows.zero_()
        entry.static_valid_rows[:actual_batch_size].fill_(True)
        entry.graph.replay()
        return entry.static_audio, entry.static_audio_lengths, int(actual_batch_size)


__all__ = ["CUDAGraphStreamingDecoderWrapper"]
