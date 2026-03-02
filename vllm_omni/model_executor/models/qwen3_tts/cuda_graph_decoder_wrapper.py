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

    DEFAULT_CAPTURE_SIZES = [25, 50, 100, 150, 200, 250, 300]

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self.capture_sizes = capture_sizes or self.DEFAULT_CAPTURE_SIZES
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

        self._warmed_up = False
        self._device = None

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self.capture_sizes:
            if actual_size <= size:
                return size
        return None

    def warmup(self, device: torch.device, dtype: torch.dtype = torch.long):
        if device.type != "cuda":
            logger.info("CUDA Graph warmup skipped: device %s is not CUDA", device)
            return

        if not self.enabled:
            logger.info("CUDA Graph is disabled, skipping warmup")
            return

        if self._warmed_up:
            logger.warning("CUDA Graph already warmed up, skipping")
            return

        self._device = device
        self.decoder.eval()

        logger.info("Starting CUDA Graph warmup for %d sizes: %s", len(self.capture_sizes), self.capture_sizes)

        # Warmup runs to ensure CUDA memory is allocated
        for size in self.capture_sizes:
            dummy_codes = torch.zeros(
                1,
                self.num_quantizers,
                size,
                dtype=dtype,
                device=device,
            )
            with torch.no_grad():
                _ = self.decoder(dummy_codes)

        torch.cuda.synchronize(device)

        for size in self.capture_sizes:
            try:
                self._capture_graph_for_size(size, device, dtype)
                logger.info("  Captured CUDA Graph for size=%d", size)
            except Exception:
                logger.warning("  Failed to capture CUDA Graph for size=%d", size, exc_info=True)

        self._warmed_up = True
        logger.info("CUDA Graph warmup complete. Captured %d graphs.", len(self.graphs))

    def _capture_graph_for_size(self, size: int, device: torch.device, dtype: torch.dtype):
        static_input = torch.zeros(
            1,
            self.num_quantizers,
            size,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            _ = self.decoder(static_input)

        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_output = self.decoder(static_input)

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        if codes.shape[0] != 1:
            return self.decoder(codes)

        actual_size = codes.shape[-1]
        padded_size = self._get_padded_size(actual_size)

        if padded_size is None or padded_size not in self.graphs:
            return self.decoder(codes)

        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:, :, :actual_size] = codes

        self.graphs[padded_size].replay()

        output = self.static_outputs[padded_size]
        total_upsample = self.decoder.total_upsample
        actual_output_len = actual_size * total_upsample

        return output[..., :actual_output_len].clone()

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
