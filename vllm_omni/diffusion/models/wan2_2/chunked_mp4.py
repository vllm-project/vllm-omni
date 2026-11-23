# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker-side MP4 encoding for Wan VAE temporal chunks."""

from __future__ import annotations

from typing import Any

from vllm_omni.diffusion.utils.media_utils import normalize_preencode_batch_frames, normalize_video_codec_options

# Matches the fallback the video serving layer applies when a request omits fps,
# so the bytes the worker encodes carry the rate the response advertises.
WAN_DEFAULT_OUTPUT_FPS = 24

# Keep worker-side encoding consistent with the one-shot serving encoder when
# a request does not choose codec options explicitly.
WAN_DEFAULT_VIDEO_CODEC_OPTIONS = {"preset": "ultrafast", "threads": "0"}

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


def resolve_wan_preencode_batch_frames(sampling_params: Any, *, default: int = WAN_DEFAULT_BATCH_FRAMES) -> int:
    """Read the MP4 batching threshold before starting expensive generation."""
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return normalize_preencode_batch_frames(extra_args.get("preencode_batch_frames", default))


def resolve_wan_video_codec_options(sampling_params: Any) -> dict[str, str] | None:
    """Read the request's encoder options, preserving serving defaults.

    An absent key inherits the one-shot serving encoder's defaults. Explicit
    ``None`` and an empty mapping remain distinct caller choices.
    """
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return normalize_video_codec_options(extra_args.get("video_codec_options", WAN_DEFAULT_VIDEO_CODEC_OPTIONS))


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
