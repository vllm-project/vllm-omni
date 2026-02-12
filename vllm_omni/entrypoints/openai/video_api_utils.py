# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible video generation API.
"""

from __future__ import annotations

import base64
import os
import tempfile
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image


def decode_input_reference(input_reference: str | None, input_reference_bytes: bytes | None) -> Image.Image | None:
    """Decode image input from multipart bytes or base64/data URL."""
    if input_reference and input_reference_bytes:
        raise ValueError("Provide input_reference either as file upload or base64, not both.")
    if input_reference_bytes:
        return Image.open(BytesIO(input_reference_bytes)).convert("RGB")
    if input_reference:
        if input_reference.startswith("data:image"):
            _, b64_data = input_reference.split(",", 1)
        else:
            b64_data = input_reference
        try:
            image_bytes = base64.b64decode(b64_data)
        except Exception as exc:  # pragma: no cover - malformed base64
            raise ValueError("Invalid base64 input_reference.") from exc
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    return None


def _normalize_video_tensor(video_tensor: torch.Tensor) -> np.ndarray:
    """Normalize a torch video tensor into a numpy array of frames (F, H, W, C)."""
    video_tensor = video_tensor.detach().cpu()
    if video_tensor.dim() == 5:
        raise ValueError("Batched video tensors are not supported for single-video encoding.")
    elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
        # [C, F, H, W] -> [F, H, W, C]
        video_tensor = video_tensor.permute(1, 2, 3, 0)

    if video_tensor.is_floating_point():
        video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
    else:
        video_tensor = video_tensor.to(torch.float32) / 255.0
    video_array = video_tensor.numpy()
    return _normalize_single_video_array(video_array)


def _normalize_single_video_array(video_array: np.ndarray) -> np.ndarray:
    """Normalize a single video array into shape (F, H, W, C)."""
    if video_array.ndim == 5:
        raise ValueError("Batched video arrays are not supported for single-video encoding.")

    if video_array.ndim == 4:
        # Convert channel-first layouts to channel-last
        if video_array.shape[0] in (3, 4) and video_array.shape[-1] not in (3, 4):
            video_array = np.transpose(video_array, (1, 2, 3, 0))
        elif video_array.shape[1] in (3, 4) and video_array.shape[-1] not in (3, 4):
            video_array = np.transpose(video_array, (0, 2, 3, 1))

    if np.issubdtype(video_array.dtype, np.floating):
        if video_array.min() < 0.0 or video_array.max() > 1.0:
            video_array = np.clip(video_array, -1.0, 1.0) * 0.5 + 0.5
    elif np.issubdtype(video_array.dtype, np.integer):
        video_array = video_array.astype(np.float32) / 255.0
    return video_array


def _normalize_video_array(video_array: np.ndarray) -> list[np.ndarray] | np.ndarray:
    """Normalize a numpy video array into shape (F, H, W, C).

    If a batch dimension is present, returns a list of per-video arrays.
    """
    if video_array.ndim == 5:
        return [_normalize_single_video_array(video_array[i]) for i in range(video_array.shape[0])]
    return _normalize_single_video_array(video_array)


def _normalize_frames(frames: list[Any]) -> list[np.ndarray]:
    """Normalize a list of frames into numpy arrays with values in [0,1]."""
    normalized: list[np.ndarray] = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame_array = frame.detach().cpu().numpy()
        elif isinstance(frame, Image.Image):
            frame_array = np.array(frame)
        elif isinstance(frame, np.ndarray):
            frame_array = frame
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}")

        if frame_array.ndim == 3 and frame_array.shape[0] in (3, 4) and frame_array.shape[-1] not in (3, 4):
            frame_array = np.transpose(frame_array, (1, 2, 0))

        if np.issubdtype(frame_array.dtype, np.floating):
            if frame_array.min() < 0.0 or frame_array.max() > 1.0:
                frame_array = np.clip(frame_array, -1.0, 1.0) * 0.5 + 0.5
        elif np.issubdtype(frame_array.dtype, np.integer):
            frame_array = frame_array.astype(np.float32) / 255.0

        normalized.append(frame_array)
    return normalized


def _coerce_video_to_frames(video: Any) -> list[np.ndarray]:
    """Convert a video payload into a list of frames for export_to_video."""
    if isinstance(video, torch.Tensor):
        video_array = _normalize_video_tensor(video)
        return list(video_array)
    if isinstance(video, np.ndarray):
        video_array = _normalize_video_array(video)
        if isinstance(video_array, list):
            raise ValueError("Batched video arrays must be split before encoding.")
        if video_array.ndim == 4:
            return list(video_array)
        if video_array.ndim == 3:
            return [video_array]
        raise ValueError(f"Unsupported video array shape: {video_array.shape}")
    if isinstance(video, list):
        if not video:
            return []
        # If this looks like a list of frames, normalize directly.
        if all(isinstance(item, (np.ndarray, torch.Tensor, Image.Image)) for item in video):
            # If each item is itself a video (ndim==4), handle elsewhere.
            if all(hasattr(item, "ndim") and item.ndim >= 4 for item in video):
                raise ValueError("Expected a single video, got a list of video tensors/arrays.")
            return _normalize_frames(video)
        raise ValueError("Unsupported list contents for video payload.")
    raise ValueError(f"Unsupported video payload type: {type(video)}")


def encode_video_base64(video: Any, fps: int) -> str:
    """Encode a video (frames/array/tensor) to base64 MP4."""
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("diffusers is required for export_to_video.") from exc

    frames = _coerce_video_to_frames(video)
    if not frames:
        raise ValueError("No frames found to encode.")

    tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_file.close()
    try:
        export_to_video(frames, tmp_file.name, fps=fps)
        with open(tmp_file.name, "rb") as f:
            video_bytes = f.read()
        return base64.b64encode(video_bytes).decode("utf-8")
    finally:
        try:
            os.remove(tmp_file.name)
        except OSError:
            pass
