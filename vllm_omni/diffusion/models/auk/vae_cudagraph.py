# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph replay for the AuK codec decode.

The decoder launches about a thousand small kernels per clip, so it is
bound by launch overhead rather than by the GPU. This wrapper replays the
whole decode as CUDA graphs, in three tiers:

Compiled bucket graphs are built at startup by warmup(): decode is passed
through torch.compile so Inductor fuses the elementwise chains, then one
graph is captured per bucket in compile_shapes. Shorter clips are
right-padded to their bucket with the normalized latent that decodes to
raw zero, which is what the eager decode's own convolution padding sees,
so the padding only reaches the last frames through the upsamplers'
lookahead. The fused kernels differ from eager in rounding order, so these
graphs are close to but not bit-identical with the eager decode.

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
it. They share one wrapper-private graph pool, so they are never evicted
one at a time: when max_graphs is reached the whole generation is retired
and a fresh pool is started, and no surviving graph can hold an address
that a destroyed graph released.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
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
    """Replay AuKVAE.decode for a batch of clips."""

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
        # Plain graphs keyed by (batch_size, latent_length). Each batch size uses
        # its own private pool handle to prevent PyTorch pool conflicts.
        self._cache: OrderedDict[tuple[int, int], _GraphEntry] = OrderedDict()
        self._plain_pools: dict[int, object] = {}
        # Normalized latent that decodes to raw zero, per device; pads short windows.
        self._pad_latent: torch.Tensor | None = None
        # Startup captures are separate from the bounded runtime graph cache.
        self._compiled: dict[tuple[int, int], _GraphEntry] = {}
        self._compiled_decode: Callable[[torch.Tensor], torch.Tensor] | None = None
        # Which path served the last call: "compiled", "graph" or "eager".
        self.last_mode: str | None = None

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode [B, frames, latent_dim] latents into a [B, frames * hop] waveform."""

        if not self.enabled or latents.ndim != 3:
            self.last_mode = "eager"
            return self.vae.decode(latents)

        B, frames, _ = latents.shape
        if not self.tile_frames or frames <= self.tile_frames:
            return self._decode_window(latents).clone()

        hop = self.vae.hop_size
        wav = latents.new_empty((B, frames * hop))
        for emit_start, chunk in self.decode_tiles(latents):
            wav[:, emit_start * hop : emit_start * hop + chunk.shape[1]].copy_(chunk)
        self.last_mode = "tiled"
        return wav

    def decode_tiles(self, latents: torch.Tensor):
        """Yield ``(emit_start_frame, waveform)`` per tile, in order, for a [B, frames, latent_dim] clip.

        The waveforms are views into graph buffers where a graph served the
        tile; copy before the next tile if they must outlive the iteration.
        """
        frames = int(latents.shape[1])
        tile = self.tile_frames or frames
        left, right = self.context_frames
        hop = self.vae.hop_size
        for start, width, emit_start, emit_end in plan_tiles(
            frames, tile, left, right, sorted({size for _, size in self._compiled})
        ):
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

        B = int(latents.shape[0])
        frames = int(latents.shape[1])
        bucket = self.compiled_bucket(frames)
        if bucket is None:
            bucket = round_up(frames, self.frame_alignment)
        entry, mode = self._lookup_graph(B, bucket)
        if entry is None:
            if len(self._cache) >= self.max_graphs:
                self._retire_plain_graphs()
            pool = self._plain_pools.get(B)
            if pool is None:
                pool = torch.cuda.graph_pool_handle()
                self._plain_pools[B] = pool
            decode = self._compiled_decode if (B > 1 and self._compiled_decode is not None) else self.vae.decode
            warm_iters = 5 if decode is self._compiled_decode else 2
            entry = self._capture(bucket, latents.device, decode, warm_iters=warm_iters, pool=pool, batch_size=B)
            self._cache[(B, bucket)] = entry
            mode = "graph"
        self.last_mode = mode

        graph_b = int(entry.static_latents.shape[0])
        pad = None
        if graph_b == B and bucket == frames:
            entry.static_latents.copy_(latents)
        else:
            entry.static_latents[:B, :frames].copy_(latents)
            if bucket > frames:
                pad = self._pad_for(latents.device)
                entry.static_latents[:B, frames:].copy_(pad.expand(B, bucket - frames, -1))
            if graph_b > B:
                pad = self._pad_for(latents.device) if pad is None else pad
                entry.static_latents[B:].copy_(pad.expand(graph_b - B, bucket, -1))
        entry.graph.replay()
        return entry.static_wav[:B, : frames * self.vae.hop_size]

    def _lookup_graph(self, batch_size: int, bucket: int) -> tuple[_GraphEntry | None, str]:
        """Pick a captured graph for this window: compiled first, then plain, then a slightly larger batch."""
        key = (batch_size, bucket)
        if key in self._compiled:
            return self._compiled[key], "compiled"
        if key in self._cache:
            return self._cache[key], "graph"
        return self._larger_batch_entry(batch_size, bucket)

    def _larger_batch_entry(self, batch_size: int, bucket: int) -> tuple[_GraphEntry | None, str]:
        """Reuse a warmed graph whose batch is only slightly larger than this window.

        Padding 127 into a warmed 128 graph avoids a first-wave capture. Padding
        1 into 128 would run 128x the work, so those misses still capture exact.
        """
        best: tuple[int, _GraphEntry, str] | None = None
        for cached_b, cached_frames, entry, mode in self._iter_batch_graphs():
            if cached_frames != bucket or cached_b < batch_size:
                continue
            if cached_b > batch_size * 2:
                continue
            if best is None or cached_b < best[0]:
                best = (cached_b, entry, mode)
        return (None, "graph") if best is None else (best[1], best[2])

    def _iter_batch_graphs(self) -> Iterator[tuple[int, int, _GraphEntry, str]]:
        """Yield ``(batch, frames, entry, mode)`` for every captured batch graph."""
        for (cached_b, cached_frames), entry in self._compiled.items():
            yield cached_b, cached_frames, entry, "compiled"
        for (cached_b, cached_frames), entry in self._cache.items():
            yield cached_b, cached_frames, entry, "graph"

    def _retire_plain_graphs(self) -> None:
        """Drop every plain graph and their shared pools; the next capture starts a new one."""
        logger.info("AuK codec decode: retiring %d plain CUDA graphs", len(self._cache))
        self._cache.clear()
        self._plain_pools.clear()

    def _pad_for(self, device: torch.device) -> torch.Tensor:
        """The [latent_dim] normalized latent that AuKVAE.decode maps to raw zero."""
        pad = self._pad_latent
        if pad is None or pad.device != device:
            mean = self.vae.global_mean.detach().float().to(device)
            scale = torch.sqrt(self.vae.global_log_std.detach().float().to(device))
            pad = -mean / scale
            self._pad_latent = pad
        return pad

    def compiled_bucket(self, frames: int) -> int | None:
        """The smallest captured compile-bucket that holds frames latents, or None past the largest."""
        for size in self.compile_shapes:
            if frames <= size and any(bucket == size for _, bucket in self._compiled):
                return size
        return None

    def warmup(
        self,
        device: torch.device | str,
        *,
        batch_sizes: Sequence[int] = (1,),
    ) -> None:
        """Compile decode and capture the configured frame and batch buckets at startup."""
        device = torch.device(device)
        if not self.enabled or not self.compile_shapes:
            return
        on_accelerator = current_omni_platform.is_cuda_alike() and device.type == current_omni_platform.device_type
        if not on_accelerator or torch.cuda.is_current_stream_capturing():
            return

        # The traced graph must read exp(alpha) from a buffer, not recompute it.
        for module in self.vae.modules():
            if isinstance(module, SnakeBeta):
                module.precompute_exp_cache()

        if self._compiled_decode is None:
            try:
                self._compiled_decode = torch.compile(self.vae.decode, mode="default", fullgraph=False, dynamic=False)
            except Exception:
                logger.warning("torch.compile of the AuK codec decode failed; using plain CUDA graphs", exc_info=True)
                return

        for batch_size in sorted({1, *(int(b) for b in batch_sizes if int(b) > 0)}):
            pool = torch.cuda.graph_pool_handle()
            for size in self.compile_shapes:
                key = (batch_size, size)
                if key in self._compiled:
                    continue
                try:
                    self._compiled[key] = self._capture(
                        size, device, self._compiled_decode, warm_iters=5, pool=pool, batch_size=batch_size
                    )
                except RuntimeError:
                    logger.warning(
                        "Compiled AuK codec decode failed for batch_size=%d latent_frames=%d; "
                        "the bucket will be captured on demand",
                        batch_size,
                        size,
                        exc_info=True,
                    )

    def _capture(
        self,
        bucket: int,
        device: torch.device,
        decode: Callable[[torch.Tensor], torch.Tensor],
        *,
        warm_iters: int,
        pool=None,
        batch_size: int = 1,
    ) -> _GraphEntry:
        """Capture one graph of decode on zero latents of bucket frames.

        The warm iterations let cuDNN pick its algorithms and, for the
        compiled decode, Inductor finish tracing and autotuning before
        anything is recorded.
        """
        static_latents = torch.zeros(batch_size, bucket, self.vae.latent_dim, device=device, dtype=torch.float32)
        with torch.inference_mode():
            self.vae.decode(static_latents)
            for _ in range(warm_iters):
                decode(static_latents)
            torch.accelerator.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            if pool is None:
                pool = current_omni_platform.get_global_graph_pool()
            with torch.cuda.graph(graph, pool=pool):
                static_wav = decode(static_latents)
        logger.info("Captured AuK codec decode CUDA graph: batch_size=%d, latent_frames=%d", batch_size, bucket)
        return _GraphEntry(graph=graph, static_latents=static_latents, static_wav=static_wav)


__all__ = ["DEFAULT_COMPILE_SHAPES", "AuKVAEDecodeGraph", "plan_tiles"]
