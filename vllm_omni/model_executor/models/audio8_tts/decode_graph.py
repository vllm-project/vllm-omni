# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-local CUDA graphs for the Audio8 TTS codec decode.

``ArkttsCodec.decode`` is stateless per call (the streaming contract re-decodes
``[left_context | new_frames]`` windows), so a graph needs no state plumbing:
one static codes buffer and one static waveform per captured frame count.
Only exact frame-count matches replay; every other size (the variable final
chunk, non-default deployments) falls back to eager execution.

Frame counts are not padded up to a bucket: the caller trims the left-context
prefix proportionally to the decoded length, and the codec's length rounding
is only exact in the linear regime, so a padded decode could shift the trim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.audio8_tts.codec import ArkttsCodec

logger = init_logger(__name__)

_GRAPH_WARMUP_ITERS = 3
_GRAPH_STREAMS: dict[int, torch.cuda.Stream] = {}


def _graph_stream() -> torch.cuda.Stream:
    """One capture stream per device so warmup allocations share a segment."""
    device = torch.accelerator.current_device_index()
    stream = _GRAPH_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda.current_blas_handle()  # allocates the stream's cuBLAS workspace
        _GRAPH_STREAMS[device] = stream
    return stream


class ArkttsDecodeGraphRunner:
    """Capture ``codec.decode`` at fixed ``[1, num_codebooks, frames]`` shapes."""

    def __init__(self, codec: ArkttsCodec, *, num_codebooks: int, frames: list[int], device: torch.device) -> None:
        if num_codebooks <= 0:
            raise ValueError(f"ArkttsDecodeGraphRunner needs a positive num_codebooks, got {num_codebooks}.")
        if not frames or any(count <= 0 for count in frames):
            raise ValueError(f"ArkttsDecodeGraphRunner needs positive frame buckets, got {frames}.")
        if device.type != "cuda":
            raise ValueError("ArkttsDecodeGraphRunner requires a CUDA device.")
        self._codec = codec
        self._device = device
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}
        for count in sorted(set(frames)):
            captured = self._capture(count, num_codebooks)
            if captured is not None:
                self._graphs[count] = captured
        if not self._graphs:
            raise RuntimeError("ArkttsDecodeGraphRunner failed to capture every requested frame bucket.")

    @property
    def frames(self) -> tuple[int, ...]:
        """Captured frame counts, ascending."""
        return tuple(sorted(self._graphs))

    def _capture(self, frames: int, num_codebooks: int):
        codec = self._codec
        codes = torch.zeros((1, num_codebooks, frames), dtype=torch.long, device=self._device)
        stream = _graph_stream()
        try:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Warm up: also populates the window transformers' mask/rope
                # caches for this length, so the capture records only decode
                # kernels against stable cache tensors.
                for _ in range(_GRAPH_WARMUP_ITERS):
                    codec.decode(codes)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # A private pool: the codec replays between the stage's other
            # graphs in an order unrelated to their capture order.
            with torch.cuda.graph(
                graph,
                pool=torch.cuda.graph_pool_handle(),
                stream=stream,
                capture_error_mode="thread_local",
            ):
                output = codec.decode(codes)
        except Exception:
            logger.warning("Audio8 TTS codec decode graph capture failed for %d frames", frames, exc_info=True)
            return None
        logger.info("Captured the Audio8 TTS codec decode graph for %d frames", frames)
        return graph, codes, output

    def can_decode(self, codes: torch.Tensor) -> bool:
        """Whether ``codes`` is a ``[1, K, frames]`` batch with a captured shape."""
        return codes.ndim == 3 and codes.shape[0] == 1 and codes.shape[1] > 0 and int(codes.shape[2]) in self._graphs

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Replay the captured decode; returns a fresh ``[1, 1, samples]`` tensor."""
        graph, static_codes, static_output = self._graphs[int(codes.shape[2])]
        if codes.device != self._device or codes.dtype != torch.long:
            codes = codes.to(device=self._device, dtype=torch.long)
        static_codes.copy_(codes)
        graph.replay()
        # Clone: the buffer is reused by the next replay.
        return static_output.clone()
