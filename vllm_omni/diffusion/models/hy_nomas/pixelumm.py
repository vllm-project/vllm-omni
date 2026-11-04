# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas@16a4c228 (Copyright 2026 Tencent).

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn

from .pixelumm_model import (
    LATENT_TOKEN_GRID,
    DCAEVPixelUMMDecoder,
    local_attention_layout,
)
from .pixelumm_streaming import (
    PixelUMMStreamingDecodeSession,
    _merge_overlapping,
    _window_starts,
)

# The source's DMD2 schedule (dcaev_pixelumm_dmd2_hq: dmd2.denoising_step_list).
DMD2_DENOISING_STEPS = (0.999, 0.866, 0.634, 0.342)
# Elementwise state updates run in blocks of this many pixel frames (bounds temporaries).
_STATE_BLOCK_FRAMES = 40

XtSource = Callable[[int, int, int, int, int, int], torch.Tensor]


@dataclass
class _DecodeOutput:
    sample: torch.Tensor


class _TemporalBlender:
    """Incremental twin of ``_merge_overlapping(parts, starts, dim=2)`` into one buffer.

    Same arithmetic as the source's ``_merge_into_buffer``: one preallocated clip buffer, each
    window cosine-blended in as soon as it is complete.
    """

    def __init__(self, total_frames: int, *, output_cpu: bool = False):
        self.output_cpu = output_cpu
        self.total = int(total_frames)
        self.out: torch.Tensor | None = None
        self.end = 0

    def add(self, part: torch.Tensor, start: int) -> None:
        length = part.shape[2]
        if self.out is None:
            self.out = part.new_empty(
                (*part.shape[:2], self.total, *part.shape[3:]),
                device="cpu" if self.output_cpu else part.device,
            )
            self.out[:, :, :length] = part
            self.end = length
            return
        overlap = self.end - start
        if overlap <= 0 or start + length <= self.end:
            raise ValueError(
                "overlap windows must be ordered and extend the covered range: "
                f"start={start}, result_end={self.end}, part_end={start + length}"
            )
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = (0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))).view(1, 1, -1, 1, 1)
        weight = weight.to(dtype=part.dtype)
        out = self.out
        # Blend on the decoder device to preserve GPU arithmetic and rounding;
        # only storage of the completed clip moves to CPU.
        previous = out[:, :, start : self.end].to(device=part.device)
        out[:, :, start : self.end] = previous * (1.0 - weight) + part[:, :, :overlap] * weight
        out[:, :, self.end : start + length] = part[:, :, overlap:]
        self.end = start + length

    def result(self) -> torch.Tensor:
        if self.out is None or self.end != self.total:
            raise RuntimeError(f"temporal merge incomplete: covered {self.end} of {self.total}")
        return self.out


class PixelUMMDCAELatentDecoder(nn.Module):
    """Decode DCAE-V c32 latents with the PixelUMM DMD2 student (pipeline ``self.vae``)."""

    spatial_compression_ratio = 32
    time_compression_ratio = 4
    ffactor_spatial = 32
    ffactor_temporal = 4
    scaling_factor = 1.0

    def __init__(
        self,
        decoder: DCAEVPixelUMMDecoder,
        *,
        denoising_step_list: Sequence[float] = DMD2_DENOISING_STEPS,
        num_steps: int = 1,
        seed: int = 42,
        temporal_window_latents: int = 10,
        temporal_overlap_latents: int = 5,
        decoder_tile_pixels: int = 256,
        decoder_tile_overlap_pixels: int = 64,
        tail_hold_latents: int = 1,
        decode_output_cpu: bool = False,
    ):
        super().__init__()
        self.decoder = decoder
        if type(decode_output_cpu) is not bool:
            raise ValueError("decode_output_cpu must be boolean")
        self.decode_output_cpu = decode_output_cpu
        steps = tuple(float(value) for value in denoising_step_list)
        if not steps or not all(0.0 < value <= 1.0 for value in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
            raise ValueError(f"denoising_step_list must be non-empty, in (0, 1] and decreasing, got {steps}")
        if not 1 <= int(num_steps) <= len(steps):
            raise ValueError(f"num_steps={num_steps} must be in [1, {len(steps)}] (distilled schedule length)")
        self.denoising_step_list = steps
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self.temporal_window_latents = int(temporal_window_latents)
        self.temporal_overlap_latents = int(temporal_overlap_latents)
        self.decoder_tile_pixels = int(decoder_tile_pixels)
        self.decoder_tile_overlap_pixels = int(decoder_tile_overlap_pixels)
        self.tail_hold_latents = int(tail_hold_latents)
        if self.tail_hold_latents < 0:
            raise ValueError("tail_hold_latents must be non-negative")
        self.latent_channels = 32
        self.config = SimpleNamespace(
            latent_channels=32,
            scaling_factor=1.0,
            shift_factor=None,
            spatial_compression_ratio=32,
            time_compression_ratio=4,
            block_out_channels=[1] * 6,
        )
        self._noise_cache: tuple[tuple, torch.Tensor] | None = None
        self._validate_tiling()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _validate_tiling(self) -> None:
        if self.temporal_window_latents <= 0:
            raise ValueError("temporal_window_latents must be positive")
        if not 0 <= self.temporal_overlap_latents < self.temporal_window_latents:
            raise ValueError("temporal overlap must be smaller than the window")
        if self.decoder_tile_pixels < 0 or self.decoder_tile_pixels % 32:
            raise ValueError("decoder_tile_pixels must be zero or a multiple of 32")
        if self.decoder_tile_overlap_pixels < 0 or self.decoder_tile_overlap_pixels % 32:
            raise ValueError("decoder overlap must be a non-negative multiple of 32")
        if self.decoder_tile_pixels and self.decoder_tile_overlap_pixels >= self.decoder_tile_pixels:
            raise ValueError("decoder overlap must be smaller than tile size")

    # ------------------------------------------------------------------
    # geometry / noise
    # ------------------------------------------------------------------
    def _tile_grid(self, height: int, width: int) -> tuple[list[int], list[int], int, int]:
        """Tile origins (rows, columns) and tile size of one window, in latent cells."""
        cells = self.decoder_tile_pixels // 32
        if cells <= 0 or (height <= cells and width <= cells):
            return [0], [0], height, width
        stride = cells - self.decoder_tile_overlap_pixels // 32
        tile_h, tile_w = min(cells, height), min(cells, width)
        ys = _window_starts(height, tile_h, min(stride, tile_h))
        xs = _window_starts(width, tile_w, min(stride, tile_w))
        return ys, xs, tile_h, tile_w

    @staticmethod
    def _merge_tiles(tiles: list[torch.Tensor], ys: list[int], xs: list[int]) -> torch.Tensor:
        """Row-major tiles of one window -> its field (columns first, then rows)."""
        if len(tiles) == 1:
            return tiles[0]
        rows = [
            _merge_overlapping(tiles[row * len(xs) : (row + 1) * len(xs)], [x * 32 for x in xs], dim=-1)
            for row in range(len(ys))
        ]
        return _merge_overlapping(rows, [y * 32 for y in ys], dim=-2)

    def _make_noise(self, latent: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """fp32 noise of latent frames ``[start, end)`` at full resolution: [B,3,4(end-start),H,W]."""
        batch, _, _, height, width = latent.shape
        frames = []
        for index in range(start, end):
            generator = torch.Generator(device=latent.device)
            generator.manual_seed(self.seed * 1_000_003 + index)
            frames.append(
                torch.randn(
                    (batch, 3, 4, height * 32, width * 32),
                    generator=generator,
                    device=latent.device,
                    dtype=torch.float32,
                )
            )
        return torch.cat(frames, dim=2)

    def _noise_tile(self, latent: torch.Tensor, *, frame_offset: int = 0) -> XtSource:
        def source(t0: int, t1: int, y: int, x: int, th: int, tw: int) -> torch.Tensor:
            key = (latent.data_ptr(), tuple(latent.shape), t0, t1, frame_offset)
            if self._noise_cache is None or self._noise_cache[0] != key:
                self._noise_cache = None
                self._noise_cache = (key, self._make_noise(latent, t0 + frame_offset, t1 + frame_offset))
            return self._noise_cache[1][..., y * 32 : (y + th) * 32, x * 32 : (x + tw) * 32]

        return source

    # ------------------------------------------------------------------
    # one x0 prediction of the whole clip
    # ------------------------------------------------------------------
    def _predict_x0(
        self,
        latent: torch.Tensor,
        xt_source: XtSource,
        *,
        t_starts: Sequence[int] | None = None,
        pad_short: bool = False,
        output_cpu: bool = False,
    ) -> torch.Tensor:
        """Merged x0 in the decoder dtype.

        By default the whole clip is covered. ``t_starts`` selects explicit latent window starts
        (streaming); the result then covers ``[t_starts[0], t_starts[-1] + window)`` latents.
        """
        frames, height, width = (int(v) for v in latent.shape[2:])
        window = self.temporal_window_latents
        if t_starts is None:
            t_starts = _window_starts(frames, window, window - self.temporal_overlap_latents)
        t_starts = list(t_starts)
        origin = t_starts[0]
        t_len = min(window, frames)
        ys, xs, tile_h, tile_w = self._tile_grid(height, width)
        spatial = [(y, x) for y in ys for x in xs]
        if self.decoder.local_attention_tile:
            # The sliding-tile mask is built per input shape and cached. Build it here so a
            # compiled decoder (``compile_decoder``) only looks it up: building is not fullgraph
            # traceable. Same call the decoder's own forward makes, so eager results are unchanged.
            local_attention_layout(
                (t_len, tile_h * LATENT_TOKEN_GRID, tile_w * LATENT_TOKEN_GRID),
                self.decoder.local_attention_tile,
                latent.device,
            )

        # ``pad_short`` (streaming tail fix): a window on the stride grid may hold fewer than
        # ``window`` real latents at the end of the timeline. Fill it to the full window with noise
        # drawn from the per-channel mean/variance of the real latents, as tailrand training does.
        # The padded frames are decoded only as context; callers read just the real range.
        pad_noise = None
        if pad_short and frames >= window and t_starts[-1] + window > frames:
            variance, mean = torch.var_mean(latent.detach().float(), dim=(2, 3, 4), correction=0, keepdim=True)
            generator = torch.Generator(device=latent.device)
            generator.manual_seed(self.seed * 1_000_003 + frames - 1)
            pad_noise = mean + variance.sqrt() * torch.randn(
                (latent.shape[0], latent.shape[1], window, height, width),
                generator=generator,
                device=latent.device,
                dtype=torch.float32,
            )
        stride = window - self.temporal_overlap_latents

        def run(task) -> torch.Tensor:
            ti, y, x = task
            t0 = t_starts[ti]
            # The slices are views of the whole clip: their strides scale with clip length, and
            # a stride guard recompiles the whole decoder under torch.compile(dynamic=False) per clip.
            real = latent[:, :, t0 : t0 + t_len, y : y + tile_h, x : x + tile_w]
            if pad_noise is not None and real.shape[2] < t_len:
                fill = t_len - real.shape[2]
                offset = (t0 // stride) % (window - fill + 1)
                tile = pad_noise[:, :, offset : offset + fill, y : y + tile_h, x : x + tile_w]
                real = torch.cat([real, tile.to(real.dtype)], dim=2)
            return self.decoder(
                real.contiguous(),
                xt_source(t0, t0 + t_len, y, x, tile_h, tile_w).contiguous(),
            )

        blender = _TemporalBlender((t_starts[-1] + t_len - origin) * 4, output_cpu=output_cpu)
        for index, start in enumerate(t_starts):
            tiles = [run((index, y, x)) for y, x in spatial]
            blender.add(self._merge_tiles(tiles, ys, xs), (start - origin) * 4)
            del tiles
        return blender.result()

    def create_streaming_decode_session(self, *, to_cpu: bool = True) -> PixelUMMStreamingDecodeSession:
        """Open-ended global-window decode timeline (the PixelUMM stable-prefix streaming protocol)."""
        if self.num_steps != 1:
            # Multi-step sampling feeds each step the blended x0 of the whole clip, so a window's
            # result depends on neighbouring windows through the sampler state. Only the 1-step
            # decode is window-local, which is what an append-only timeline can reproduce.
            raise ValueError(f"PixelUMM streaming decode requires num_steps=1, got {self.num_steps}")
        return PixelUMMStreamingDecodeSession(self, to_cpu=to_cpu)

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
        *,
        pixel_frames: int | None = None,
    ):
        del generator  # noise comes from ``self.seed``, identical on every SP rank
        if latent.ndim != 5:
            raise ValueError(f"expected [B,C,T,H,W], got {tuple(latent.shape)}")
        steps = self.denoising_step_list[: self.num_steps]
        try:
            if len(steps) == 1:
                video = self._predict_x0(
                    latent,
                    self._noise_tile(latent),
                    output_cpu=self.decode_output_cpu,
                )
            else:
                video = self._sample_multistep(latent, steps)
        finally:
            self._noise_cache = None
        if pixel_frames is not None:
            target = int(pixel_frames)
            if not 1 <= target <= video.shape[2]:
                raise ValueError(f"pixel_frames={target} is outside decoded T={video.shape[2]}")
            if target < video.shape[2]:
                video = video[:, :, :target]
        return _DecodeOutput(video) if return_dict else (video,)

    def _sample_multistep(self, latent: torch.Tensor, steps: Sequence[float]) -> torch.Tensor:
        """Source ``sample_distilled`` with a full-clip fp32 state on every rank."""
        state = self._make_noise(latent, 0, int(latent.shape[2]))

        def source(t0: int, t1: int, y: int, x: int, th: int, tw: int) -> torch.Tensor:
            return state[:, :, t0 * 4 : t1 * 4, y * 32 : (y + th) * 32, x * 32 : (x + tw) * 32]

        for index, (t_cur, t_next) in enumerate(zip(steps, (*steps[1:], 0.0))):
            if index == len(steps) - 1:
                # t_next == 0: x0 + 0 * (x - x0) is this step's x0.
                return self._predict_x0(
                    latent,
                    source,
                    output_cpu=self.decode_output_cpu,
                )
            x0 = self._predict_x0(latent, source)
            ratio = t_next / t_cur
            for offset in range(0, state.shape[2], _STATE_BLOCK_FRAMES):
                block = slice(offset, offset + _STATE_BLOCK_FRAMES)
                anchor = x0[:, :, block].float()
                state[:, :, block] = anchor + ratio * (state[:, :, block] - anchor)
            del x0
        raise AssertionError("unreachable")
