# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental (per-phase) VAE decoding for TaoMate-H3 streaming.

Video
-----
The MiniMax-H3 video VAE decodes a latent timeline in temporal windows: window
``i`` reads latents ``[5i, 5i+7)``, decodes 28 frames, keeps frames ``[3, 20)``
as its 17 primary frames and frames ``[23, 28)`` as a five-frame overlap that is
linearly blended into the first five frames of window ``i+1``. TaoMate phases
add 12/10/10/5 latents (request 0) and 10/10/10/5 afterwards, so after every
phase all windows with ``5i + 7 <= latents_so_far`` decode exactly as a one-shot
decode would; each phase yields 34/34/34/17 frames and the newest window's
five-frame overlap is held until the next phase (or ``flush``). Frames are
native 24 fps, 119 per steady request.

All ranks of a tile-parallel VAE group call ``push`` collectively because the
checkpoint splits spatial tiles across the group.

Audio
-----
The audio VAE (40 latents/s, 800 samples per latent at 32 kHz) is decoded with
a sliding window of 48 latents of left context and 24 of right context, then
cropped to the samples aligned with the frames of the same phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from vllm_omni.platforms import current_omni_platform

from .geometry import AUDIO_SAMPLE_RATE, AUDIO_SAMPLES_PER_LATENT, VIDEO_FPS

_TOKENS_PER_WINDOW = 5
_WINDOW_TOKENS = 7
_PRE_PAD = 3
_PRIMARY_FRAMES = 17
_OVERLAP_FRAMES = 5


def samples_at_frame(frame: int) -> int:
    return (frame * AUDIO_SAMPLE_RATE) // VIDEO_FPS


class StreamingVideoDecoder:
    """Stateful temporal-window decode through the pipeline's H3 video VAE."""

    def __init__(
        self,
        vae: MiniMaxH3VideoVAE,
        *,
        device: torch.device,
        height: int,
        width: int,
        emit_frames: bool = True,
    ) -> None:
        self.vae = vae
        self.model = vae.model
        self.device = device
        self.height = int(height)
        self.width = int(width)
        # Every rank runs the collective tile decode; only the output-owning
        # rank pays for the pixel conversion and the host copy.
        self.emit_frames = bool(emit_frames)
        expected = (_TOKENS_PER_WINDOW, _WINDOW_TOKENS - _TOKENS_PER_WINDOW, _PRE_PAD, _OVERLAP_FRAMES)
        actual = (
            int(self.model.tokens_chunk_size),
            int(self.model.token_overlap),
            int(self.model.frame_pre_padding),
            int(self.model.frame_overlap),
        )
        if actual != expected or self.model.isolated_first_frame or self.model.isolated_last_frame:
            raise RuntimeError(f"unexpected MiniMax-H3 video VAE temporal windowing {actual}")
        self.reset()

    def reset(self) -> None:
        self.latents: torch.Tensor | None = None  # denormalized [1, 24, T, H, W] fp32 on device
        self.base = 0  # global latent index of self.latents[:, :, 0]
        self.total = 0  # latents pushed so far
        self.next_window = 0
        self.overlap: torch.Tensor | None = None
        self.frames_emitted = 0

    def state_bytes(self, latent_h: int, latent_w: int) -> int:
        """Upper bound of resident latent and overlap buffers for one session."""
        latent_rows = 24 * (_WINDOW_TOKENS + 12) * latent_h * latent_w * 4
        overlap = 3 * _OVERLAP_FRAMES * self.height * self.width * 4
        return latent_rows + overlap

    def _to_uint8_device(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [1, 3, T, H, W] in VAE output space -> [T, H, W, 3] uint8 on the device, cropped to the canvas
        reverted = self.vae._revert_decoded_inplace(frames.float())
        reverted = reverted[..., : self.height, : self.width]
        pixels = reverted[0].permute(1, 2, 3, 0).mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
        return pixels.contiguous()

    @staticmethod
    def to_host(frames: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
        """Fetch device frames from ``push_device`` (a host wait); host arrays pass through."""
        if frames is None or isinstance(frames, np.ndarray):
            return frames
        return frames.cpu().numpy()

    def _to_uint8(self, frames: torch.Tensor) -> np.ndarray:
        return self._to_uint8_device(frames).cpu().numpy()

    def advance(self, latents: torch.Tensor) -> torch.Tensor | None:
        """Append ``[1, 24, n, H, W]`` normalized latents and decode every complete window."""
        visual = self.vae._denormalize_latent(latents.to(device=self.device, dtype=torch.float32))
        self.latents = visual if self.latents is None else torch.cat((self.latents, visual), dim=2)
        self.total += int(visual.shape[2])
        outputs: list[torch.Tensor] = []
        autocast = current_omni_platform.create_autocast_context(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        )
        with torch.inference_mode(), autocast:
            while _TOKENS_PER_WINDOW * self.next_window + _WINDOW_TOKENS <= self.total:
                start = _TOKENS_PER_WINDOW * self.next_window - self.base
                clip = self.latents[:, :, start : start + _WINDOW_TOKENS]
                with self.vae._decode_tiling_context(clip):
                    decoded = self.model._adaptive_decode(clip)
                if int(decoded.shape[2]) != 4 * _WINDOW_TOKENS:
                    raise RuntimeError(f"video VAE window returned {decoded.shape[2]} frames")
                primary = decoded[:, :, _PRE_PAD : _PRE_PAD + _PRIMARY_FRAMES]
                overlap = decoded[:, :, 4 * _TOKENS_PER_WINDOW + _PRE_PAD :].contiguous()
                if self.overlap is not None:
                    primary = self.model.blend(self.overlap, primary, _OVERLAP_FRAMES, dim=-3)
                outputs.append(primary)
                self.overlap = overlap
                self.next_window += 1
                self.frames_emitted += _PRIMARY_FRAMES
            keep_from = _TOKENS_PER_WINDOW * self.next_window
            if keep_from > self.base:
                self.latents = self.latents[:, :, keep_from - self.base :].contiguous()
                self.base = keep_from
        return torch.cat(outputs, dim=2) if outputs else None

    def push_device(self, latents: torch.Tensor) -> tuple[int, torch.Tensor | np.ndarray | None]:
        """Collective: append latents and queue the decode; frames stay on the device.

        Returns ``(frame_start, uint8 [T, H, W, 3] device tensor or None)``; call
        :meth:`to_host` for the array once the caller has nothing left to do
        while the device finishes the decode.
        """
        frame_start = self.frames_emitted
        frames = self.advance(latents)
        if frames is None:
            return frame_start, None
        if not self.emit_frames:
            return frame_start, np.zeros((int(frames.shape[2]), 0, 0, 3), dtype=np.uint8)
        return frame_start, self._to_uint8_device(frames)

    def push(self, latents: torch.Tensor) -> tuple[int, np.ndarray | None]:
        """Collective: append latents; return ``(frame_start, uint8 [T, H, W, 3] or None)``."""
        frame_start, frames = self.push_device(latents)
        return frame_start, self.to_host(frames)

    def flush(self) -> tuple[int, np.ndarray | None]:
        """Emit the held five-frame overlap of the last window (end of a session)."""
        if self.overlap is None:
            return self.frames_emitted, None
        frame_start = self.frames_emitted
        if self.emit_frames:
            frames = self._to_uint8(self.overlap)
        else:
            frames = np.zeros((_OVERLAP_FRAMES, 0, 0, 3), dtype=np.uint8)
        self.frames_emitted += _OVERLAP_FRAMES
        self.overlap = None
        return frame_start, frames


class StreamingAudioDecoder:
    """Sliding-window audio VAE decode aligned to emitted video frames."""

    def __init__(self, vae: MiniMaxH3AudioVAE, *, device: torch.device, left: int = 48, right: int = 24) -> None:
        self.vae = vae
        self.device = device
        self.left = int(left)
        self.right = int(right)
        if int(getattr(vae, "sample_rate", AUDIO_SAMPLE_RATE)) != AUDIO_SAMPLE_RATE:
            raise RuntimeError("MiniMax-H3 audio VAE sample rate differs from 32 kHz")
        self.reset()

    def reset(self) -> None:
        self.latents: torch.Tensor | None = None  # normalized [2, 32, L] fp32 on device
        self.base = 0
        self.total = 0

    def append(self, latents: torch.Tensor) -> None:
        """Append normalized clean audio latents ``[2, 32, L]``."""
        latents = latents.to(device=self.device, dtype=torch.float32)
        self.latents = latents if self.latents is None else torch.cat((self.latents, latents), dim=2)
        self.total += int(latents.shape[2])

    def decode_range(self, s0: int, s1: int) -> np.ndarray:
        """Return samples ``[s0, s1)`` as float32 ``[n, 2]``.

        The video timeline may outrun the generated audio by less than one
        latent at the end of a session (audio boundaries are rounded to 40 Hz
        latents); that tail is padded with silence rather than refused.
        """
        if s1 <= s0:
            return np.zeros((0, 2), dtype=np.float32)
        if self.latents is None:
            raise RuntimeError("no audio latents have been appended")
        available = self.total * AUDIO_SAMPLES_PER_LATENT
        if available < s1 - AUDIO_SAMPLES_PER_LATENT:
            raise RuntimeError("audio latents do not cover the requested samples")
        requested = s1 - s0
        s1 = min(s1, available)
        if s1 <= s0:
            return np.zeros((requested, 2), dtype=np.float32)
        l0 = max(self.base, s0 // AUDIO_SAMPLES_PER_LATENT - self.left)
        l1 = min(self.total, -(-s1 // AUDIO_SAMPLES_PER_LATENT) + self.right)
        window = self.latents[:, :, l0 - self.base : l1 - self.base]
        with torch.inference_mode():
            wave = self.vae.decode_latent(window)  # [1, 2, S]
        if wave.ndim != 3 or int(wave.shape[1]) != 2:
            raise RuntimeError(f"unexpected audio VAE output {tuple(wave.shape)}")
        if int(wave.shape[2]) != (l1 - l0) * AUDIO_SAMPLES_PER_LATENT:
            raise RuntimeError("audio VAE hop differs from 800 samples per latent")
        start = s0 - l0 * AUDIO_SAMPLES_PER_LATENT
        segment = wave[0, :, start : start + (s1 - s0)].float().transpose(0, 1).contiguous()
        keep_from = max(self.base, s1 // AUDIO_SAMPLES_PER_LATENT - self.left - 1)
        if keep_from > self.base:
            self.latents = self.latents[:, :, keep_from - self.base :].contiguous()
            self.base = keep_from
        samples = segment.cpu().numpy()
        if samples.shape[0] < requested:
            samples = np.concatenate((samples, np.zeros((requested - samples.shape[0], 2), np.float32)), axis=0)
        return samples

    def skip_range(self, s1: int) -> None:
        """Advance the retained-latent window as if samples up to ``s1`` were decoded."""
        if self.latents is None:
            return
        keep_from = max(self.base, s1 // AUDIO_SAMPLES_PER_LATENT - self.left - 1)
        if keep_from > self.base and keep_from <= self.total:
            self.latents = self.latents[:, :, keep_from - self.base :].contiguous()
            self.base = keep_from

    def state_bytes(self) -> int:
        return 2 * 32 * (self.left + self.right + 2 * 210) * 4


def video_frames_to_payload(frames: np.ndarray | None) -> np.ndarray:
    if frames is None:
        return np.zeros((0, 1, 1, 3), dtype=np.uint8)
    return frames


__all__: list[Any] = [
    "StreamingAudioDecoder",
    "StreamingVideoDecoder",
    "samples_at_frame",
    "video_frames_to_payload",
]
