# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph wrapper for BigVGAN-style vocoders.

The vocoder forward is a chain of small convolutions (alias-free Snake
activations, residual blocks) that is CPU-launch-bound at streaming chunk
sizes even with the fused alias-free Triton kernel: a few hundred kernel
launches per vocode dominate wall time. Capturing the GPU vocode for fixed
mel-frame buckets and replaying collapses them into one graph launch.

Dynamic mel inputs are right zero-padded to the nearest captured bucket and
the waveform is trimmed back to ``actual_frames * upsample_factor`` after
replay. Exact-size hits replay the same kernels on the same static buffers
as the capture run, so recurring shapes (streaming chunks are a constant
mel width) are bit-stable. Padded hits differ from eager only inside the
conv receptive field of the tail (zero- vs replicate-padding at the
boundary); callers that care validate via their perceptual gates.

Modeled on ``CUDAGraphBigVGANWrapper``
(vllm_omni/model_executor/models/indextts2/bigvgan_cuda_graph.py) and
``CUDAGraphDecoderWrapper``
(vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = ["BigVGANCUDAGraphWrapper"]


class BigVGANCUDAGraphWrapper:
    """Capture a BigVGAN vocode for fixed mel-frame buckets.

    ``capture_fn`` must map ``mel [B, num_mels, T]`` to a GPU waveform
    ``[B, 1, T * upsample_factor]`` with no host syncs inside (pass the
    GPU-only ``vocode`` method, not a ``forward`` that ends in ``.cpu()``).
    The eager fallback routes through the same callable, so graph and eager
    outputs share one implementation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        capture_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_mels: int,
        upsample_factor: int,
        capture_sizes: Sequence[int],
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_fn = capture_fn if capture_fn is not None else model.forward
        self.num_mels = int(num_mels)
        self.upsample_factor = int(upsample_factor)
        self.capture_sizes = sorted({int(size) for size in capture_sizes})
        self.enabled = enabled

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_mel: dict[int, torch.Tensor] = {}
        self.static_wav: dict[int, torch.Tensor] = {}
        self._warmed_up = False
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self.last_call_info: dict | None = None

    # ------------------------------------------------------------------
    # Warmup / capture
    # ------------------------------------------------------------------

    def warmup(self, device: torch.device, dtype: torch.dtype) -> None:
        """Capture every bucket; idempotent. A failing bucket logs one
        warning and is skipped, the rest still replay."""
        if not self.enabled or self._warmed_up or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        if device.index is None:
            # Normalize so later mel.device comparisons (always indexed)
            # do not spuriously mismatch.
            device = torch.device(device.type, torch.accelerator.current_device_index())
        self.model.eval()
        self._precompute_snake_caches()

        logger.info(
            "Starting BigVGAN CUDA graph warmup for mel-frame buckets: %s",
            self.capture_sizes,
        )
        # cuDNN's heuristic algo choice leaves roughly 2x GPU time on the
        # table for the vocoder's small convolutions (measured 13.5 ms vs
        # 5.7 ms per replay at 112 mel frames on an H100). The per-bucket
        # eager warm run in _capture then autotunes each captured shape and
        # the capture freezes the winner. The flag is process-global, so it
        # is restored afterwards; eager callers keep their current algos and
        # the captured graphs are unaffected by the restore.
        prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        # Normal mode so a lazily triggered warmup inside a caller's
        # inference_mode still creates mutable (non-inference) static
        # buffers, which later non-inference replays require.
        try:
            with torch.inference_mode(False), torch.no_grad():
                for size in self.capture_sizes:
                    try:
                        self._capture(size, device, dtype)
                        logger.info("Captured BigVGAN CUDA graph for mel frames=%d", size)
                    except Exception:
                        logger.warning(
                            "Failed to capture BigVGAN CUDA graph for mel frames=%d",
                            size,
                            exc_info=True,
                        )
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark
        self._device = device
        self._dtype = dtype
        self._warmed_up = True

    def _precompute_snake_caches(self) -> None:
        from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta

        for module in self.model.modules():
            if isinstance(module, (Snake, SnakeBeta)):
                module.precompute_exp_cache()

    def _capture(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        mel = torch.zeros(1, self.num_mels, size, device=device, dtype=dtype)
        # Eager warm run so cuDNN autotuning and allocator growth settle
        # outside capture.
        _ = self.capture_fn(mel)
        torch.accelerator.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            self.static_wav[size] = self.capture_fn(mel)
        self.static_mel[size] = mel
        self.graphs[size] = graph

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _padded_size(self, actual_frames: int) -> int | None:
        for size in self.capture_sizes:
            if actual_frames <= size:
                return size
        return None

    def _eager(self, mel: torch.Tensor, reason: str, frames: int) -> torch.Tensor:
        self.last_call_info = {
            "mode": "eager",
            "reason": reason,
            "batch": int(mel.shape[0]) if mel.dim() > 0 else 0,
            "frames": frames,
            "bucket": None,
            "padding": None,
        }
        return self.capture_fn(mel)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocode ``mel`` [B, num_mels, T]; falls back to eager when ineligible."""
        frames = int(mel.shape[-1]) if mel.dim() == 3 else 0

        if (
            not self.enabled
            or mel.dim() != 3
            or mel.shape[0] != 1
            or mel.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            return self._eager(mel, "ineligible", frames)

        if not self._warmed_up:
            self.warmup(mel.device, mel.dtype)
        if not self._warmed_up or mel.device != self._device or mel.dtype != self._dtype:
            return self._eager(mel, "warmup_failed_or_mismatched", frames)

        size = self._padded_size(frames)
        if size is None or size not in self.graphs:
            return self._eager(mel, "no_bucket", frames)

        buf = self.static_mel[size]
        if frames == size:
            buf.copy_(mel)
        else:
            buf.zero_()
            buf[..., :frames].copy_(mel)
        self.graphs[size].replay()
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "batch": 1,
            "frames": frames,
            "bucket": size,
            "padding": size - frames,
        }
        # Clone: the static buffer is overwritten by the next replay.
        return self.static_wav[size][..., : frames * self.upsample_factor].clone()
