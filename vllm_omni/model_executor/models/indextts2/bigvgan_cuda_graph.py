"""CUDA graph wrapper for BigVGAN vocoding in IndexTTS2 Stage 1.

BigVGAN is heavily CPU launch-bound (Snake/AliasFree small-op storm:
~3500 copy_ + ~2000 mul + ~340 cudnn_convolution launches per request,
GPU busy <15%). Capturing the forward for fixed mel-frame buckets and
replaying collapses thousands of kernel launches into one graph launch.

Dynamic mel inputs are right zero-padded to the nearest captured bucket
and the waveform is trimmed back to ``actual_frames * upsample_factor``
after replay. Padding leakage into the kept region is bounded by the
conv receptive field at the tail and is validated via WER/SIM A/B.

Modeled on ``CUDAGraphGLMTTSDiTWrapper``
(vllm_omni/model_executor/models/glm_tts/glm_tts_dit_wrapper.py).
"""

from __future__ import annotations

import math

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Mel-frame buckets at 22050 Hz / hop 256 (~86 fps): 128 ≈ 1.5 s ...
# 2048 ≈ 23.8 s. Longer inputs fall back to the eager path.
DEFAULT_CAPTURE_SIZES = [128, 256, 384, 512, 768, 1024, 1536, 2048]


class CUDAGraphBigVGANWrapper:
    """Capture BigVGAN vocoding for fixed mel-frame buckets."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        enabled: bool = True,
        capture_sizes: list[int] | None = None,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.capture_sizes = sorted(set(capture_sizes or DEFAULT_CAPTURE_SIZES))
        self.num_mels = int(model.h.num_mels)
        self.upsample_factor = int(math.prod(model.h.upsample_rates))

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_mel: dict[int, torch.Tensor] = {}
        self.static_wav: dict[int, torch.Tensor] = {}
        self._warmed_up = False

    def warmup(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self.enabled or self._warmed_up or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        self.model.eval()
        logger.info("Starting BigVGAN CUDA graph warmup for mel-frame buckets: %s", self.capture_sizes)
        for size in self.capture_sizes:
            try:
                self._capture(size, device, dtype)
                logger.info("Captured BigVGAN CUDA graph for mel frames=%d", size)
            except Exception:
                logger.warning("Failed to capture BigVGAN CUDA graph for size=%d", size, exc_info=True)
        self._warmed_up = True

    def _capture(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        mel = torch.zeros(1, self.num_mels, size, device=device, dtype=dtype)
        self.static_mel[size] = mel
        # Eager warm run so cuDNN autotuning happens outside capture.
        with torch.no_grad():
            _ = self.model(mel)
        torch.accelerator.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            self.static_wav[size] = self.model(mel)
        self.graphs[size] = graph

    def _get_padded_size(self, actual_frames: int) -> int | None:
        for size in self.capture_sizes:
            if actual_frames <= size:
                return size
        return None

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocode ``mel`` [1, num_mels, T]; falls back to eager when ineligible."""
        if (
            not self.enabled
            or not self._warmed_up
            or mel.shape[0] != 1
            or mel.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.model(mel)

        actual_frames = int(mel.shape[-1])
        size = self._get_padded_size(actual_frames)
        if size is None or size not in self.graphs:
            return self.model(mel)

        buf = self.static_mel[size]
        buf.zero_()
        buf[..., :actual_frames].copy_(mel.to(buf.dtype))
        self.graphs[size].replay()
        return self.static_wav[size][..., : actual_frames * self.upsample_factor].clone()
