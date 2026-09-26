# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph replay for the AuK codec decode.

The decoder launches about a thousand small kernels per clip, so it is
bound by launch overhead rather than by the GPU. This wrapper replays the
whole decode as CUDA graphs, in three tiers:

Compiled bucket graphs are built at startup by warmup(): decode is passed
through torch.compile so Inductor fuses the elementwise chains, then one
graph is captured per bucket in compile_shapes. Shorter clips are
right-padded with zeros to their bucket. The padding leaks slightly into
the last frames through the non-causal conv_pre, and the fused kernels
differ from eager in rounding order, so these graphs are close to but not
bit-identical with the eager decode.

Clips longer than tile_frames are decoded in tiles of that many frames
(by default the largest bucket, so every tile replays the same compiled
graph). Neighbouring tiles overlap by the decoder's receptive field, taken
from AuKVAE.decode_context_frames(), and each tile only contributes the
frames that lie outside that halo, so the stitched waveform equals the
whole-clip decode up to floating-point order. Memory and startup cost stay
constant however long the clip; the price is the halo recomputed per tile
(about an eighth of a 512-frame tile). decode_tiles() yields the tiles as
they finish so a caller can stream them.

Plain graphs are captured on demand, one per exact latent length, for
windows no compiled bucket serves (tiling off, or a bucket whose compile
failed). They replay the same kernels as eager and are bit-identical with
it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

from vllm_omni.diffusion.models.auk.auk_vae import AuKVAE, SnakeBeta
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

# Latent-frame buckets for the compiled graphs, at 50 Hz: 2.56, 5.12 and
# 10.24 s. Each bucket costs one Inductor compilation (tens of seconds) at
# startup and one private CUDA graph pool; longer clips are tiled over the
# largest one. Deployments override them with
# ``model_config.auk_vae_compile_shapes`` on the diffusion stage.
DEFAULT_COMPILE_SHAPES: tuple[int, ...] = (128, 256, 512)


def plan_tiles(
    frames: int, tile: int, left: int, right: int, sizes: Sequence[int] = ()
) -> list[tuple[int, int, int, int]]:
    """Split a clip into decode windows: ``(window_start, window_frames, emit_start, emit_end)``.

    Windows are ``tile`` frames wide. A window contributes the frames that
    are at least ``left`` frames after its start and ``right`` frames before
    its end, unless that edge is the clip's own edge, so each emitted frame
    has its full decoder context inside its window. The last window is
    pinned to the clip's end; it shrinks to the smallest of ``sizes`` (the
    compiled buckets) that still holds the remaining frames plus the left
    context, so a clip just past one tile does not pay for a second full
    one. A clip that fits in one tile is a single window of its own length.
    """
    if frames <= tile:
        return [(0, frames, 0, frames)]
    if tile <= left + right:
        raise ValueError(f"tile_frames={tile} must exceed the decoder context {left} + {right}")
    windows: list[tuple[int, int, int, int]] = []
    emit = 0
    while True:
        start = 0 if emit == 0 else emit - left
        if start + tile >= frames:
            width = tile
            for size in sorted(sizes):
                if left + frames - emit <= size <= tile:
                    width = size
                    break
            windows.append((frames - width, width, emit, frames))
            return windows
        end = start + tile - right
        windows.append((start, tile, emit, end))
        emit = end


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    static_latents: torch.Tensor
    static_wav: torch.Tensor


class AuKVAEDecodeGraph:
    """Replay AuKVAE.decode for one clip per call."""

    def __init__(
        self,
        vae: AuKVAE,
        *,
        enabled: bool = True,
        frame_alignment: int = 1,
        max_graphs: int = 32,
        compile_shapes: Sequence[int] = DEFAULT_COMPILE_SHAPES,
        tile_frames: int | None = None,
    ) -> None:
        """``tile_frames``: window for clips longer than it; None = the largest bucket, 0 = no tiling."""
        self.vae = vae
        self.enabled = bool(enabled)
        self.frame_alignment = max(1, int(frame_alignment))
        self.max_graphs = max(1, int(max_graphs))
        self.compile_shapes = sorted({int(size) for size in compile_shapes if int(size) > 0})
        self.context_frames = vae.decode_context_frames()
        if tile_frames is None:
            tile_frames = self.compile_shapes[-1] if self.compile_shapes else 0
            if tile_frames <= sum(self.context_frames):
                logger.warning(
                    "AuK codec decode: largest bucket %d does not exceed the decoder context %s; not tiling",
                    tile_frames,
                    self.context_frames,
                )
                tile_frames = 0
        self.tile_frames = max(0, int(tile_frames))
        if self.tile_frames and self.tile_frames <= sum(self.context_frames):
            raise ValueError(f"tile_frames={self.tile_frames} must exceed the decoder context {self.context_frames}")
        # Plain graphs keyed by latent length, least recently used first.
        self._cache: OrderedDict[int, _GraphEntry] = OrderedDict()
        # Compiled graphs keyed by bucket, filled by warmup().
        self._compiled: dict[int, _GraphEntry] = {}
        self._compiled_decode: Callable[[torch.Tensor], torch.Tensor] | None = None
        # Which path served the last call: "compiled", "graph" or "eager".
        self.last_mode: str | None = None

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode [1, frames, latent_dim] latents into a [1, frames * hop] waveform."""

        # Graphs are captured for one clip at a time. Anything else runs the
        # plain eager decode, whole.
        single_clip = latents.ndim == 3 and latents.shape[0] == 1
        if not self.enabled or not single_clip:
            self.last_mode = "eager"
            return self.vae.decode(latents)

        frames = int(latents.shape[1])
        if not self.tile_frames or frames <= self.tile_frames:
            return self._decode_window(latents).clone()

        hop = self.vae.hop_size
        wav = latents.new_empty((1, frames * hop))
        for emit_start, chunk in self.decode_tiles(latents):
            wav[:, emit_start * hop : emit_start * hop + chunk.shape[1]].copy_(chunk)
        self.last_mode = "tiled"
        return wav

    def decode_tiles(self, latents: torch.Tensor):
        """Yield ``(emit_start_frame, waveform)`` per tile, in order, for a [1, frames, latent_dim] clip.

        The waveforms are views into graph buffers where a graph served the
        tile; copy before the next tile if they must outlive the iteration.
        """
        frames = int(latents.shape[1])
        tile = self.tile_frames or frames
        left, right = self.context_frames
        hop = self.vae.hop_size
        for start, width, emit_start, emit_end in plan_tiles(frames, tile, left, right, self._compiled):
            wav = self._decode_window(latents[:, start : start + width])
            yield emit_start, wav[:, (emit_start - start) * hop : (emit_end - start) * hop]

    def _decode_window(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one window through the best available tier; graph results are views into static buffers."""

        # A call made while an outer graph is being captured, or off the
        # accelerator, runs eager. The capture query is asked last because it
        # needs a CUDA runtime.
        if not latents.is_cuda or torch.cuda.is_current_stream_capturing():
            self.last_mode = "eager"
            return self.vae.decode(latents)

        frames = int(latents.shape[1])
        bucket = self.compiled_bucket(frames)
        if bucket is not None:
            entry = self._compiled[bucket]
            self.last_mode = "compiled"
        else:
            bucket = round_up(frames, self.frame_alignment)
            entry = self._cache.get(bucket)
            if entry is None:
                entry = self._capture(bucket, latents.device, self.vae.decode, warm_iters=2)
                if len(self._cache) >= self.max_graphs:
                    self._cache.popitem(last=False)
                self._cache[bucket] = entry
            else:
                self._cache.move_to_end(bucket)
            self.last_mode = "graph"

        if bucket == frames:
            entry.static_latents.copy_(latents)
        else:
            entry.static_latents.zero_()
            entry.static_latents[:, :frames].copy_(latents)
        entry.graph.replay()
        return entry.static_wav[:, : frames * self.vae.hop_size]

    def compiled_bucket(self, frames: int) -> int | None:
        """The smallest compiled bucket that holds frames latents, or None past the largest."""
        for size in self.compile_shapes:
            if frames <= size and size in self._compiled:
                return size
        return None

    def warmup(self, device: torch.device | str) -> None:
        """Compile the decode and capture every bucket in compile_shapes.

        Meant for service startup, since each bucket costs one Inductor
        compilation. A failure only logs a warning and the bucket is served
        by a plain graph instead.
        """
        device = torch.device(device)
        if not self.enabled or not self.compile_shapes or self._compiled:
            return
        on_accelerator = current_omni_platform.is_cuda_alike() and device.type == current_omni_platform.device_type
        if not on_accelerator or torch.cuda.is_current_stream_capturing():
            return

        # The traced graph must read exp(alpha) from a buffer, not recompute it.
        for module in self.vae.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_exp_cache()

        try:
            self._compiled_decode = torch.compile(self.vae.decode, mode="default", fullgraph=False, dynamic=False)
        except Exception:
            logger.warning("torch.compile of the AuK codec decode failed; using plain CUDA graphs", exc_info=True)
            self._compiled_decode = None
            return

        # Inductor fuses the plain Snake formula with the ops around it.
        for size in self.compile_shapes:
            try:
                self._compiled[size] = self._capture(size, device, self._compiled_decode, warm_iters=5)
                logger.info("Compiled and captured AuK codec decode: latent_frames=%d", size)
            except Exception:
                logger.warning(
                    "Compiled AuK codec decode failed for latent_frames=%d; falling back to plain CUDA graphs",
                    size,
                    exc_info=True,
                )
        logger.info(
            "AuK codec decode compile warmup done: %d/%d buckets", len(self._compiled), len(self.compile_shapes)
        )

    def _capture(
        self,
        bucket: int,
        device: torch.device,
        decode: Callable[[torch.Tensor], torch.Tensor],
        *,
        warm_iters: int,
    ) -> _GraphEntry:
        """Capture one graph of decode on zero latents of bucket frames.

        The warm iterations let cuDNN pick its algorithms and, for the
        compiled decode, Inductor finish tracing and autotuning before
        anything is recorded.
        """
        static_latents = torch.zeros(1, bucket, self.vae.latent_dim, device=device, dtype=torch.float32)
        with torch.inference_mode():
            self.vae.decode(static_latents)
            for _ in range(warm_iters):
                decode(static_latents)
            torch.accelerator.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=current_omni_platform.get_global_graph_pool()):
                static_wav = decode(static_latents)
        logger.info("Captured AuK codec decode CUDA graph: latent_frames=%d", bucket)
        return _GraphEntry(graph=graph, static_latents=static_latents, static_wav=static_wav)


__all__ = ["DEFAULT_COMPILE_SHAPES", "AuKVAEDecodeGraph", "plan_tiles"]
