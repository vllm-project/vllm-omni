# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Graph wrapper around ``vocoder.decode_mel`` for fixed mel-chunk
sizes. Vocoder's ``[T_mel, 80]`` input is the cleanest fixed-size point
in the streaming pipeline."""

from __future__ import annotations

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# 24 kHz output, 480 samples per mel frame, 80 mel bins (BigVGAN config).
_NUM_MELS = 80


class KimiAudioCudaGraphDecoderWrapper:
    def __init__(
        self,
        vocoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        enabled: bool = True,
    ):
        self.vocoder = vocoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.enabled = enabled

        self.graphs: dict[int, CUDAGraph] = {}
        self.static_inputs: dict[int, torch.Tensor] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}
        self._warmed_up = False
        self._device: torch.device | None = None

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ) -> list[int]:
        sizes: set[int] = set()
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)
        # Cover Kimi's first-chunk (100) and chunk (150) defaults.
        sizes.update({100, 150, 250})
        for p2 in [32, 64, 128, 256, 512]:
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
        dtype: torch.dtype = torch.bfloat16,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
    ) -> None:
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return
        self._device = device
        self.vocoder.eval()

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
            )

        logger.info(
            "KimiAudio CUDA Graph warmup over %d sizes: %s",
            len(self.capture_sizes),
            self.capture_sizes,
        )

        # Eager pass first to allocate workspaces.
        for size in self.capture_sizes:
            dummy = torch.zeros(size, _NUM_MELS, dtype=dtype, device=device)
            with torch.no_grad():
                _ = self.vocoder.decode_mel(dummy)
        torch.accelerator.synchronize(device)

        for size in self.capture_sizes:
            self._capture(size, device, dtype)
            logger.info("  Captured KimiAudio vocoder graph size=%d", size)

        self._warmed_up = True
        logger.info("KimiAudio CUDA Graph warmup complete: %d sizes captured", len(self.graphs))

    def _capture(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        static_input = torch.zeros(size, _NUM_MELS, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.vocoder.decode_mel(static_input)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.vocoder.decode_mel(static_input)

        self.graphs[size] = graph
        self.static_inputs[size] = static_input
        self.static_outputs[size] = static_output

    def decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Drop-in for ``vocoder.decode_mel``. Falls back to eager when the
        input doesn't fit any captured bucket."""
        if not self.enabled or not self._warmed_up or mel.dim() != 2:
            return self.vocoder.decode_mel(mel)
        actual_size = mel.shape[0]
        padded_size = self._get_padded_size(actual_size)
        if padded_size is None or padded_size not in self.graphs:
            return self.vocoder.decode_mel(mel)

        self.static_inputs[padded_size].zero_()
        self.static_inputs[padded_size][:actual_size, :].copy_(mel)
        self.graphs[padded_size].replay()
        # Trim [1, wav_len] to actual length: 480 samples per mel frame.
        actual_out_len = actual_size * 480
        return self.static_outputs[padded_size][..., :actual_out_len].clone()
