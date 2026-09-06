# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-side MP4 encoding for Wan VAE temporal chunks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.utils.media_utils import ChunkedMP4Encoder, normalize_video_codec_options

# Matches the fallback the video serving layer applies when a request omits fps,
# so the bytes the worker encodes carry the rate the response advertises.
WAN_DEFAULT_OUTPUT_FPS = 24

# Wan publishes one latent frame group per callback, which is far finer than one
# host transfer is worth. 17 frames matches MiniMax-H3's native clip size.
WAN_DEFAULT_BATCH_FRAMES = 17


def wan_preencoded_mp4_payload(video: Any) -> dict[str, Any] | None:
    """Return a post-process payload when the worker already encoded the MP4.

    Returns ``None`` for the ordinary frame-tensor output so callers fall
    through to their existing post-processing.

    No fps is reported here: :func:`resolve_wan_output_fps` mirrors the fallback
    the serving layer applies, so the rate the response advertises already
    matches the rate these bytes were encoded at.
    """
    if isinstance(video, (bytes, bytearray, memoryview)):
        video = [video]
    if not isinstance(video, list) or not all(isinstance(item, (bytes, bytearray, memoryview)) for item in video):
        return None
    return {"payload": {"video": [bytes(item) for item in video]}, "metadata": {}}


def resolve_wan_preencode_mp4(sampling_params: Any, *, output_type: str) -> bool:
    """Return whether this request wants the worker to pre-encode the MP4.

    Reject rather than silently downgrade the combinations the pre-encoded path
    cannot serve: it emits MP4 bytes, so nothing downstream can interpolate
    frames or hand back an array.
    """
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    if not extra_args.get("preencode_mp4", False):
        return False
    if getattr(sampling_params, "enable_frame_interpolation", False):
        raise ValueError(
            "preencode_mp4 cannot be combined with enable_frame_interpolation: interpolation "
            "needs the decoded frames that the pre-encoded path no longer materializes"
        )
    if output_type != "np":
        raise ValueError(f"preencode_mp4 returns MP4 bytes and cannot serve output_type={output_type!r}")
    return True


def resolve_wan_video_codec_options(sampling_params: Any) -> dict[str, str] | None:
    """Read the request's encoder options for the worker-side encoder."""
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return normalize_video_codec_options(extra_args.get("video_codec_options"))


def resolve_wan_output_fps(sampling_params: Any) -> int:
    """Resolve the fps the worker encodes with.

    The serving layer resolves output fps from the request's ``fps`` field, not
    from ``frame_rate``, so pre-encoding has to read the same field or the bytes
    would carry a rate the response does not advertise.
    """
    fps = getattr(sampling_params, "fps", None)
    if isinstance(fps, list):
        fps = fps[0] if fps else None
    return int(fps) if fps else WAN_DEFAULT_OUTPUT_FPS


def _to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """Quantize ``BCTHW`` in [-1, 1] to ``BTHWC`` uint8, transferring once.

    The quantization runs on the accelerator so a single transfer moves the
    final bytes rather than float frames.
    """
    frames = video.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    return frames.permute(0, 2, 3, 4, 1).cpu().numpy()


class WanClipMP4Session:
    """Encode whole decoded clips as an autoregressive loop produces them.

    Wan S2V already decodes one clip per iteration and needs each clip's
    trailing frames to build the next clip's motion latents, so it has no use
    for the finer temporal callback. Handing each finished clip to a bounded
    encoder still overlaps H.264 encoding with the next clip's denoise and
    decode, and drops the full-video concatenation.

    ``audio_waveforms`` holds one waveform per batch entry, so a caller with
    several outputs per prompt repeats a request's waveform across its entries.
    """

    def __init__(
        self,
        *,
        audio_waveforms: list[np.ndarray | None],
        audio_sample_rate: int | None,
        fps: float,
        video_codec_options: dict[str, str] | None = None,
        max_pending: int = 2,
    ) -> None:
        self._audio_waveforms = audio_waveforms
        self._audio_sample_rate = audio_sample_rate
        self._fps = fps
        self._video_codec_options = video_codec_options
        self._max_pending = max_pending
        self._encoders: list[ChunkedMP4Encoder] = []

    def push_clip(self, clip: torch.Tensor) -> None:
        """Queue one decoded ``BCTHW`` clip, one entry per encoder."""
        frames = _to_uint8_frames(clip)
        if not self._encoders:
            if frames.shape[0] != len(self._audio_waveforms):
                raise ValueError(
                    f"expected one audio waveform per batch entry, got {len(self._audio_waveforms)} "
                    f"for {frames.shape[0]} entries"
                )
            self._encoders = [
                ChunkedMP4Encoder(
                    width=frames.shape[3],
                    height=frames.shape[2],
                    fps=self._fps,
                    audio_waveform=waveform,
                    audio_sample_rate=self._audio_sample_rate,
                    max_pending=self._max_pending,
                    video_codec_options=self._video_codec_options,
                )
                for waveform in self._audio_waveforms
            ]
        for index, encoder in enumerate(self._encoders):
            encoder.push(np.ascontiguousarray(frames[index]))

    def finish(self) -> list[bytes]:
        return [encoder.finish() for encoder in self._encoders]

    def abort(self) -> None:
        for encoder in self._encoders:
            encoder.abort()


def decode_wan_latents_to_mp4(
    vae: Any,
    latents: torch.Tensor,
    *,
    fps: float,
    batch_frames: int = WAN_DEFAULT_BATCH_FRAMES,
    max_pending: int = 2,
    video_codec_options: dict[str, str] | None = None,
) -> list[bytes]:
    """Decode denormalized Wan latents into one progressive MP4 per batch entry.

    The VAE publishes committed temporal chunks while later chunks are still
    decoding, so host transfer and H.264 encoding overlap the remaining decode
    instead of following it. The full video is never materialized.

    Ranks that do not own the decode output receive no chunks and return an
    empty list, matching the empty tensor the full-decode path returns there.
    """
    if batch_frames <= 0:
        raise ValueError("batch_frames must be positive")

    encoders: list[ChunkedMP4Encoder] = []
    pending: list[torch.Tensor] = []
    pending_frames = 0

    def flush() -> None:
        nonlocal pending_frames
        if not pending:
            return
        frames = _to_uint8_frames(torch.cat(pending, dim=2))
        if not encoders:
            encoders.extend(
                ChunkedMP4Encoder(
                    width=frames.shape[3],
                    height=frames.shape[2],
                    fps=fps,
                    max_pending=max_pending,
                    video_codec_options=video_codec_options,
                )
                for _ in range(frames.shape[0])
            )
        for index, encoder in enumerate(encoders):
            encoder.push(np.ascontiguousarray(frames[index]))
        pending.clear()
        pending_frames = 0

    def on_chunk(chunk: torch.Tensor) -> None:
        nonlocal pending_frames
        pending.append(chunk)
        pending_frames += int(chunk.shape[2])
        if pending_frames >= batch_frames:
            flush()

    try:
        vae.decode(latents, return_dict=False, on_chunk=on_chunk)
        flush()
        return [encoder.finish() for encoder in encoders]
    except BaseException:
        for encoder in encoders:
            encoder.abort()
        raise
