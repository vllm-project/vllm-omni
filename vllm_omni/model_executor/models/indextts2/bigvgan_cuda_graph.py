"""CUDA graph wrapper for BigVGAN vocoding in IndexTTS2 Stage 1.

BigVGAN is heavily CPU launch-bound (Snake/AliasFree small-op storm:
~3500 copy_ + ~2000 mul + ~340 cudnn_convolution launches per request,
GPU busy <15%). Capturing the forward for fixed mel-frame buckets and
replaying collapses thousands of kernel launches into one graph launch.

Dynamic mel inputs are right zero-padded to the nearest captured bucket
and the waveform is trimmed back to ``actual_frames * upsample_factor``
after replay. Padding leakage into the kept region is bounded by the
conv receptive field at the tail (~20-38 mel frames, ~0.2-0.4s audio)
and is validated via WER/SIM A/B.

The compile layer (``compile_shapes``) additionally applies
``torch.compile`` to the BigVGAN forward before CUDA-graph capture,
letting Inductor fuse the Snake/alias-free small-op chain. The
three-tier dispatch at runtime is: compiled-graph > plain-graph > eager.

Modeled on ``CUDAGraphDecoderWrapper``
(vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py)
and ``CUDAGraphGLMTTSDiTWrapper``
(vllm_omni/model_executor/models/glm_tts/glm_tts_dit_wrapper.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# Mel-frame buckets at 22050 Hz / hop 256 (~86 fps).
# Keep buckets modest so that CUDA-graph private pools do not exhaust
# GPU memory when stage0 and stage1 share the same device (~22 GiB).
# Inputs longer than the largest bucket fall back to eager BigVGAN.
DEFAULT_CAPTURE_SIZES = [128, 256, 384, 512]


class CUDAGraphBigVGANWrapper:
    """Capture BigVGAN vocoding for fixed mel-frame buckets.

    Supports an optional torch.compile layer on top of plain CUDA graphs.
    When ``compile_shapes`` is provided, those buckets are compiled with
    ``torch.compile(mode="default", fullgraph=False, dynamic=False)`` and
    then captured into CUDA graphs — giving both Inductor operator fusion
    and zero-launch-overhead replay.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        enabled: bool = True,
        capture_sizes: list[int] | None = None,
        compile_shapes: list[int] | None = None,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.capture_sizes = sorted(set(capture_sizes or DEFAULT_CAPTURE_SIZES))
        self.compile_shapes = sorted(set(compile_shapes or []))
        self.num_mels = int(model.h.num_mels)
        self.upsample_factor = int(math.prod(model.h.upsample_rates))

        # Plain CUDA graph tier
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_mel: dict[int, torch.Tensor] = {}
        self.static_wav: dict[int, torch.Tensor] = {}

        # Compiled CUDA graph tier
        self._compiled_decoder: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._compiled_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._compiled_static_mel: dict[int, torch.Tensor] = {}
        self._compiled_static_wav: dict[int, torch.Tensor] = {}
        self._compiled_shapes: set[int] = set()

        self._warmed_up = False
        self.last_call_info: dict[str, int | str | None] = {}

    def warmup(self, device: torch.device, dtype: torch.dtype) -> None:
        if not self.enabled or self._warmed_up or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        self.model.eval()

        # Precompute Snake/SnakeBeta caches so torch.compile sees stable
        # buffers instead of lazy init inside the traced graph.
        self._precompute_snake_caches()

        logger.info("Starting BigVGAN CUDA graph warmup for mel-frame buckets: %s", self.capture_sizes)
        for size in self.capture_sizes:
            try:
                self._capture(size, device, dtype)
                logger.info("Captured BigVGAN CUDA graph for mel frames=%d", size)
            except Exception:
                logger.warning("Failed to capture BigVGAN CUDA graph for size=%d", size, exc_info=True)

        if self.compile_shapes:
            self._warmup_compile_shapes(device, dtype)

        self._warmed_up = True

    def _precompute_snake_caches(self) -> None:
        from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta

        for module in self.model.modules():
            if isinstance(module, (Snake, SnakeBeta)):
                module.precompute_exp_cache()

    # ------------------------------------------------------------------
    # Plain CUDA graph capture
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Compiled CUDA graph capture (torch.compile + capture)
    # ------------------------------------------------------------------

    def _warmup_compile_shapes(self, device: torch.device, dtype: torch.dtype) -> None:
        logger.info(
            "Starting torch.compile + CUDA Graph warmup for BigVGAN shapes: %s",
            self.compile_shapes,
        )
        try:
            self._compiled_decoder = torch.compile(
                self.model.forward,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
        except Exception:
            logger.warning("Failed to create torch.compile BigVGAN wrapper", exc_info=True)
            self._compiled_decoder = None
            return

        for size in self.compile_shapes:
            try:
                self._capture_compiled(size, device, dtype)
                self._compiled_shapes.add(size)
                logger.info(
                    "  torch.compile + CUDA Graph ready for mel frames=%d",
                    size,
                )
            except Exception:
                logger.warning(
                    "  Failed to capture torch.compile CUDA Graph for size=%d; "
                    "this shape will use plain CUDA graph/eager fallback",
                    size,
                    exc_info=True,
                )
        logger.info(
            "torch.compile + CUDA Graph warmup complete: %d/%d shapes ready",
            len(self._compiled_shapes),
            len(self.compile_shapes),
        )

    def _capture_compiled(self, size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._compiled_decoder is None:
            raise RuntimeError("Compiled decoder is not initialized")

        mel = torch.zeros(1, self.num_mels, size, device=device, dtype=dtype)
        # Warmup iterations let Inductor finish tracing/autotuning before capture.
        with torch.inference_mode():
            for _ in range(5):
                _ = self._compiled_decoder(mel)
        torch.accelerator.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_wav = self._compiled_decoder(mel)

        self._compiled_static_mel[size] = mel
        self._compiled_static_wav[size] = static_wav
        self._compiled_graphs[size] = graph

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _get_padded_size(self, actual_frames: int) -> int | None:
        for size in self.capture_sizes:
            if actual_frames <= size:
                return size
        return None

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Vocode ``mel`` [1, num_mels, T]; falls back to eager when ineligible."""
        actual_frames = int(mel.shape[-1]) if mel.ndim >= 3 else 0

        # Eligibility guard
        if (
            not self.enabled
            or not self._warmed_up
            or mel.shape[0] != 1
            or mel.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            self.last_call_info = {
                "mode": "eager",
                "reason": "ineligible",
                "batch": int(mel.shape[0]) if mel.ndim > 0 else 0,
                "frames": actual_frames,
                "bucket": None,
                "padding": None,
            }
            return self.model(mel)

        # Tier 1: compiled CUDA graph (exact shape match)
        if actual_frames in self._compiled_shapes:
            return self._replay_compiled(actual_frames, actual_frames, mel, "exact_hit")

        # Tier 1b: compiled CUDA graph (padded shape)
        padded = self._get_padded_size(actual_frames)
        if padded is not None and padded in self._compiled_shapes:
            return self._replay_compiled(actual_frames, padded, mel, "padded_hit")

        # Tier 2: plain CUDA graph
        size = padded
        if size is None or size not in self.graphs:
            self.last_call_info = {
                "mode": "eager",
                "reason": "no_bucket",
                "batch": int(mel.shape[0]),
                "frames": actual_frames,
                "bucket": size,
                "padding": None if size is None else int(size - actual_frames),
            }
            return self.model(mel)

        buf = self.static_mel[size]
        if actual_frames == size:
            buf.copy_(mel.to(buf.dtype))
        else:
            buf.zero_()
            buf[..., :actual_frames].copy_(mel.to(buf.dtype))
        self.graphs[size].replay()
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "batch": int(mel.shape[0]),
            "frames": actual_frames,
            "bucket": int(size),
            "padding": int(size - actual_frames),
        }
        return self.static_wav[size][..., : actual_frames * self.upsample_factor].clone()

    def _replay_compiled(
        self,
        actual_frames: int,
        bucket: int,
        mel: torch.Tensor,
        reason: str,
    ) -> torch.Tensor:
        buf = self._compiled_static_mel[bucket]
        if actual_frames == bucket:
            buf.copy_(mel.to(buf.dtype))
        else:
            buf.zero_()
            buf[..., :actual_frames].copy_(mel.to(buf.dtype))
        self._compiled_graphs[bucket].replay()
        self.last_call_info = {
            "mode": "compile_graph",
            "reason": reason,
            "batch": int(mel.shape[0]),
            "frames": actual_frames,
            "bucket": int(bucket),
            "padding": int(bucket - actual_frames),
        }
        return self._compiled_static_wav[bucket][..., : actual_frames * self.upsample_factor].clone()
