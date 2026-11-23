# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax H3 timeline guides: placement rules and media preparation.

A timeline guide is a clean condition anchored at an explicit pixel frame of
the generated clip. It carries a still image or a clip, optionally plus audio,
and is VAE-encoded into the packed sequence without entering the Qwen
presentation. Guides are supplied as ``multi_modal_data["timeline_guides"]``:

.. code-block:: python

    [
        {"frame_index": 0, "image": "first.png"},
        {"frame_index": 36, "image": pil_image, "audio": "tone.flac"},
        {"frame_index": -22, "video": "tail.mp4"},
    ]

``frame_index`` is a strict integer; negative values count back from the end
of the aligned output. ``image`` and ``video`` are mutually exclusive. The
placement rules follow the native MiniMax H3 multi-frame reference workflow.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3TimelineGuide
from vllm_omni.model_executor.models.minimax_h3.preprocessing import load_minimax_h3_images
from vllm_omni.model_executor.models.minimax_h3.reference_video import load_audio_file

MINIMAX_H3_TIMELINE_GUIDES_KEY = "timeline_guides"
_GUIDE_FIELDS = frozenset({"frame_index", "image", "video", "audio"})
_FPS = 24


def _strict_int(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        raise OmniClientError(f"{name} must be a {'positive ' if positive else ''}integer")
    return value


def normalize_visual_frame_count(count: int) -> int:
    """Round a clip length down to the native ``17k + 5`` guide cadence.

    Clips shorter than five frames keep only their first frame.
    """
    _strict_int(count, "guide frame count", positive=True)
    return 1 if count < 5 else 5 + 17 * ((count - 5) // 17)


def resolve_guide_start(index: int, output_frames: int, visual_frames: int = 1) -> int:
    """Resolve a pixel-frame index against the aligned output frame count."""
    _strict_int(index, "timeline guide frame_index")
    _strict_int(output_frames, "output frame count", positive=True)
    _strict_int(visual_frames, "guide frame count", positive=True)
    start = output_frames + index if index < 0 else index
    if start < 0 or start + visual_frames > output_frames:
        raise OmniClientError(
            f"timeline guide at frame_index={index} with {visual_frames} frame(s) does not fit "
            f"the {output_frames}-frame output; trim the source or change frame_index"
        )
    return start


def guide_audio_limit(output_frames: int, start: int) -> int:
    """Audio latent positions available from ``start`` to the end of the output.

    Audio runs at 40 latent positions per second against 24 pixel frames per
    second, so a pixel frame spans 5/3 audio positions. Integer arithmetic keeps
    the floor exact at fractional origins.
    """
    _strict_int(start, "timeline guide start")
    if start < 0:
        raise OmniClientError("timeline guide start must be resolved and non-negative")
    resolve_guide_start(start, output_frames)
    target_audio_t = (5 * output_frames + 1) // 3
    available = (3 * target_audio_t - 5 * start) // 3
    if available < 1:
        raise OmniClientError("timeline guide audio must leave at least one audio latent position")
    return available


def _fit_image(image: Image.Image, width: int, height: int) -> torch.Tensor:
    fitted = ImageOps.fit(image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(fitted, dtype=np.uint8).copy())


def _decode_guide_video(path: str, width: int, height: int, max_frames: int) -> np.ndarray:
    """Decode at 24 FPS, center-crop to the canvas aspect, then scale."""
    crop_w = f"if(gt(iw*{height},ih*{width}),trunc(ih*{width}/{height}),iw)"
    crop_h = f"if(gt(iw*{height},ih*{width}),ih,trunc(iw*{height}/{width}))"
    filters = (
        f"crop=w='{crop_w}':h='{crop_h}',setpts=PTS-STARTPTS,fps={_FPS},scale={width}:{height}:flags=lanczos,setsar=1"
    )
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        path,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filters,
        "-frames:v",
        str(max_frames + 1),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OmniClientError(f"MiniMax H3 could not decode timeline guide video {path!r}") from exc
    frame_bytes = width * height * 3
    count, remainder = divmod(len(result.stdout), frame_bytes)
    if count == 0 or remainder:
        raise OmniClientError(f"MiniMax H3 timeline guide video {path!r} has no complete frames")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, height, width, 3)


def _video_frames(value: Any, width: int, height: int, max_frames: int) -> torch.Tensor:
    if isinstance(value, (str, os.PathLike)):
        frames = _decode_guide_video(str(value), width, height, max_frames)
    else:
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if array.ndim != 4 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise OmniClientError("MiniMax H3 timeline guide video frames must be uint8 [T, H, W, 3] at 24 FPS")
        frames = np.stack([np.asarray(_fit_image(Image.fromarray(frame), width, height)) for frame in array])
    if frames.shape[0] > max_frames:
        raise OmniClientError(
            f"MiniMax H3 timeline guide video is longer than {max_frames} frames at 24 FPS; trim the source"
        )
    count = normalize_visual_frame_count(int(frames.shape[0]))
    # Copy: the ffmpeg buffer is read-only and the trailing frames are dropped.
    return torch.from_numpy(np.array(frames[:count]))


def _guide_audio(value: Any) -> tuple[torch.Tensor, int]:
    if isinstance(value, (str, os.PathLike)):
        try:
            waveform, sample_rate = load_audio_file(str(value))
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            raise OmniClientError(f"MiniMax H3 could not decode timeline guide audio {str(value)!r}") from exc
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        waveform, sample_rate = value
    elif isinstance(value, Mapping):
        waveform = value.get("waveform", value.get("array"))
        sample_rate = value.get("sample_rate", value.get("sampling_rate"))
    else:
        raise OmniClientError("MiniMax H3 timeline guide audio must be a path, (waveform, sample_rate), or a mapping")
    waveform = torch.as_tensor(waveform).float()
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or int(sample_rate) <= 0:
        raise OmniClientError("MiniMax H3 timeline guide audio requires a positive integer sample rate")
    if waveform.ndim not in (1, 2) or waveform.shape[-1] == 0 or not torch.isfinite(waveform).all():
        raise OmniClientError("MiniMax H3 timeline guide audio must be a finite non-empty waveform")
    return waveform.contiguous(), int(sample_rate)


def prepare_timeline_guides(
    value: Any,
    *,
    width: int,
    height: int,
    num_frames: int,
) -> tuple[MiniMaxH3TimelineGuide, ...]:
    """Validate, decode and place ordered guides on the output canvas."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OmniClientError("MiniMax H3 timeline_guides must be a list of guide objects")
    max_frames = normalize_visual_frame_count(num_frames) + 16
    guides: list[MiniMaxH3TimelineGuide] = []
    for index, entry in enumerate(value):
        name = f"timeline_guides[{index}]"
        if not isinstance(entry, Mapping) or not set(entry) <= _GUIDE_FIELDS:
            raise OmniClientError(f"{name} must be an object with frame_index and image, video, or audio")
        frame_index = _strict_int(entry.get("frame_index"), f"{name}.frame_index")
        image, video, audio = entry.get("image"), entry.get("video"), entry.get("audio")
        if image is None and video is None and audio is None:
            raise OmniClientError(f"{name} requires an image, video, or audio source")
        if image is not None and video is not None:
            raise OmniClientError(f"{name} cannot combine image and video")
        frames = None
        if image is not None:
            loaded = load_minimax_h3_images(image)
            if len(loaded) != 1:
                raise OmniClientError(f"{name}.image must be one image")
            frames = _fit_image(loaded[0], width, height)[None]
        elif video is not None:
            frames = _video_frames(video, width, height, max_frames)
        start = resolve_guide_start(frame_index, num_frames, int(frames.shape[0]) if frames is not None else 1)
        guide_audio = _guide_audio(audio) if audio is not None else None
        guides.append(
            MiniMaxH3TimelineGuide(
                start=start,
                is_clip=video is not None,
                frames=frames,
                audio=guide_audio,
                audio_limit=guide_audio_limit(num_frames, start) if guide_audio is not None else 0,
            )
        )
    return tuple(guides)


__all__ = [
    "MINIMAX_H3_TIMELINE_GUIDES_KEY",
    "MiniMaxH3TimelineGuide",
    "guide_audio_limit",
    "normalize_visual_frame_count",
    "prepare_timeline_guides",
    "resolve_guide_start",
]
