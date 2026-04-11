# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes)  # Automatically uses CUDA graph if possible
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device = None

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        """Compute capture sizes from chunking config for high graph hit rate."""
        sizes: set[int] = set()

        # Streaming exact hits
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        # Non-streaming chunked decode: full chunk + last-chunk buckets
        non_stream_max = decode_chunk_size + decode_left_context
        sizes.add(non_stream_max)

        # Power-of-2 buckets covering both streaming IC sizes and non-streaming last-chunk sizes
        for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
            if p2 <= non_stream_max:
                sizes.add(p2)

        return sorted(sizes)

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ):
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )

        logger.info("Starting CUDA Graph warmup for %d sizes: %s", len(self.capture_sizes), self.capture_sizes)

        # Warmup runs to ensure CUDA memory is allocated
        for size in self.capture_sizes:
            dummy = torch.zeros(1, self.num_quantizers, size, dtype=dtype, device=device)
            with torch.no_grad():
                _ = self.decoder(dummy)

        torch.cuda.synchronize(device)

        for size in self.capture_sizes:
            try:
                self._capture(size, device, dtype)
                logger.info("  Captured CUDA Graph for size=%d", size)
            except Exception:
                logger.warning("  Failed to capture graph for size=%d", size, exc_info=True)

        self._warmed_up = True
        logger.info("CUDA Graph warmup complete: %d/%d captured", len(self.graphs), len(self.capture_sizes))

    def _capture(self, size: int, device: torch.device, dtype: torch.dtype):
        static_input = torch.zeros(1, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder(static_input)
        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder(static_input)

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._warmed_up or codes.shape[0] != 1:
            return self.decoder(codes)

        actual_size = codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        if padded_size is None or padded_size not in self.graphs:
            return self.decoder(codes)

        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:, :, :actual_size] = codes
        self.graphs[padded_size].replay()

        actual_out_len = actual_size * self.decoder.total_upsample
        return self.static_outputs[padded_size][..., :actual_out_len].clone()

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self.decode(codes_chunk)

            wavs.append(wav_chunk[..., context_size * total_upsample :])
            start_index = end_index

        return torch.cat(wavs, dim=-1)
