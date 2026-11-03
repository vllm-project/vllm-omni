# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-addressed H3 clip latents for continuation across requests.

The video server tags each H3 request with ``extra_args["latent_cache_key"]``
(its job id). A finished request stores its denoised video/audio latents here,
on CPU. A later request naming that key in ``extra_args["continue_from"]``
samples a window whose hidden first frames repeat the cached tail, carried as a
frame-0 timeline guide, so no clip is decoded and re-encoded.
Everything else in that request -- references, timeline guides, latent edits,
driving audio, long-video windows -- addresses the returned frames only; the
encoder maps it past the hidden overlap and the decoder trims the overlap.

Every DiT rank runs the same forward on identical inputs, so each rank stores
an identical copy and nothing is broadcast. The cache is per worker process:
it does not survive a restart and is not shared between data-parallel replicas.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.continuation_request import ContinuationRequest
from vllm_omni.model_executor.models.minimax_h3.long_video import MINIMAX_H3_MAX_CONTINUATION_SECONDS

from .continuation import LatentPrefix, _audio_t, _video_t

LATENT_CACHE_KEY = "latent_cache_key"
# ``--additional-config`` key for the cache size in clips; 0 disables it.
LATENT_CACHE_CLIPS_CONFIG = "latent_cache_clips"
DEFAULT_LATENT_CACHE_CLIPS = 16
_FPS = 24
_MAX_TIMELINE_FRAMES = int(MINIMAX_H3_MAX_CONTINUATION_SECONDS * _FPS)


@dataclass(frozen=True)
class CachedClip:
    """Denoised latents of one clip, placed on the chain's global frame timeline."""

    video: torch.Tensor  # [1, C, latent_t, latent_h, latent_w]
    audio: torch.Tensor  # [2, C, audio_t]
    start_frame: int
    num_frames: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.num_frames


def latent_cache_clips(additional_config: Mapping[str, object] | None) -> int:
    """The cache size from ``--additional-config``, or the default."""
    value = (additional_config or {}).get(LATENT_CACHE_CLIPS_CONFIG, DEFAULT_LATENT_CACHE_CLIPS)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{LATENT_CACHE_CLIPS_CONFIG} must be a non-negative integer, got {value!r}")
    return value


class LatentClipCache:
    """Thread-safe LRU of :class:`CachedClip` keyed by video job id."""

    def __init__(self, capacity: int = DEFAULT_LATENT_CACHE_CLIPS) -> None:
        if capacity < 0:
            raise ValueError(f"{LATENT_CACHE_CLIPS_CONFIG} must be non-negative")
        self.capacity = capacity
        self._clips: OrderedDict[str, CachedClip] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._clips)

    def put(self, key: str, clip: CachedClip) -> None:
        if self.capacity == 0:
            return
        with self._lock:
            self._clips[key] = clip
            self._clips.move_to_end(key)
            while len(self._clips) > self.capacity:
                self._clips.popitem(last=False)

    def get(self, key: str) -> CachedClip | None:
        with self._lock:
            clip = self._clips.get(key)
            if clip is not None:
                self._clips.move_to_end(key)
            return clip

    def clear(self) -> None:
        with self._lock:
            self._clips.clear()


@dataclass(frozen=True)
class LatentContinuation:
    """The hidden prefix of one request and the global AV span it samples."""

    prefix: LatentPrefix
    num_frames: int  # window frames, overlap included

    @property
    def audio_t(self) -> int:
        start = self.prefix.start_frame
        return _audio_t(start + self.num_frames) - _audio_t(start)

    @property
    def overlap_audio_t(self) -> int:
        return int(self.prefix.audio.shape[-1])


def plan_latent_continuation(
    clip: CachedClip,
    request: ContinuationRequest,
    *,
    num_frames: int,
    latent_h: int,
    latent_w: int,
) -> LatentContinuation:
    """Place a ``num_frames`` window so its first ``overlap`` frames repeat the clip tail."""
    overlap = request.overlap_frames
    if tuple(clip.video.shape[-2:]) != (latent_h, latent_w):
        raise OmniClientError(
            "MiniMax H3 continue_from requires the same canvas as the source video: "
            f"source latent {tuple(clip.video.shape[-2:])}, request latent {(latent_h, latent_w)}"
        )
    if overlap > clip.num_frames:
        raise OmniClientError(
            f"MiniMax H3 continue_overlap_frames ({overlap}) exceeds the source video ({clip.num_frames} frames)"
        )
    if overlap >= num_frames:
        raise OmniClientError(
            f"MiniMax H3 continuation window ({num_frames} frames) must exceed continue_overlap_frames ({overlap})"
        )
    start = clip.end_frame - overlap
    if start + num_frames > _MAX_TIMELINE_FRAMES:
        raise OmniClientError(
            f"MiniMax H3 continuation chains are limited to {MINIMAX_H3_MAX_CONTINUATION_SECONDS:g} seconds"
        )
    overlap_audio_t = _audio_t(clip.end_frame) - _audio_t(start)
    if overlap_audio_t > clip.audio.shape[-1]:
        raise OmniClientError("MiniMax H3 cached source audio is shorter than the requested overlap")
    prefix = LatentPrefix(
        video=clip.video[:, :, -_video_t(overlap) :],
        audio=clip.audio[..., -overlap_audio_t:],
        start_frame=start,
        overlap_frames=overlap,
    )
    return LatentContinuation(prefix=prefix, num_frames=num_frames)


def trim_decoded_overlap(
    video: torch.Tensor,
    audio: torch.Tensor | None,
    *,
    frames: int,
    audio_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Drop the hidden overlap from decoded ``[B, C, F, H, W]`` frames and ``[..., S]`` audio."""
    if audio is not None:
        audio = audio[..., round(audio.shape[-1] * audio_fraction) :]
    return video[:, :, frames:], audio
