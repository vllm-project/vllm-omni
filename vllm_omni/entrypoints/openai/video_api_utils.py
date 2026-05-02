# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible video generation API.
"""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Any

import httpx
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.protocol.videos import (
    FileImageReference,
    ImageReference,
    UrlImageReference,
)


def _decode_image_bytes(image_bytes: bytes, *, source: str) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: provided content is not a valid image.") from exc


def _decode_base64_image(input_reference: str, *, source: str) -> Image.Image:
    if input_reference:
        if input_reference.startswith("data:image"):
            _, b64_data = input_reference.split(",", 1)
        else:
            b64_data = input_reference

        try:
            image_bytes = base64.b64decode(b64_data)
        except (binascii.Error, ValueError) as exc:  # pragma: no cover - malformed base64
            raise InvalidInputReferenceError(f"Invalid {source}: image data is not valid base64.") from exc
        return _decode_image_bytes(image_bytes, source=source)
    raise InvalidInputReferenceError(f"Invalid {source}: image data is empty.")


async def decode_image_url(image_url: str) -> Image.Image:
    if image_url.startswith("data:image"):
        return _decode_base64_image(image_url, source="image_reference.image_url")

    if image_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.get(image_url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise InvalidInputReferenceError(
                    "Invalid image_reference.image_url: failed to download image."
                ) from exc
        return _decode_image_bytes(response.content, source="image_reference.image_url")

    raise InvalidInputReferenceError("Invalid image_reference.image_url: must be an http(s) URL or data URL.")


async def decode_input_reference(
    image_reference: ImageReference | None,
    input_reference_bytes: bytes | None,
) -> Image.Image | None:
    """Decode image input from multipart bytes, base64/data URL, or image_reference."""

    if input_reference_bytes is not None and image_reference is not None:
        raise InvalidInputReferenceError("Provide either input_reference or image_reference, not both.")

    if isinstance(input_reference_bytes, bytes):
        return _decode_image_bytes(input_reference_bytes, source="input_reference")

    if isinstance(image_reference, UrlImageReference):
        return await decode_image_url(image_reference.image_url)
    elif isinstance(image_reference, FileImageReference):
        raise InvalidInputReferenceError("Invalid image_reference: file_id is not supported yet.")

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
    """Convert a video payload into a list of normalized float32 frames."""
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


def _coerce_audio_to_numpy(audio: Any) -> np.ndarray:
    """Convert an audio payload into a float32 numpy array for muxing."""
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().cpu().float().numpy()
    elif isinstance(audio, np.ndarray):
        arr = audio
    elif isinstance(audio, list):
        arr = np.array(audio)
    else:
        raise ValueError(f"Unsupported audio payload type: {type(audio)}")

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("Audio payload must contain at least one sample.")

    return arr.astype(np.float32)


def _encode_video_bytes(
    video: Any,
    fps: int,
    audio: Any | None = None,
    audio_sample_rate: int | None = None,
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Encode a video payload into MP4 bytes, optionally muxing audio."""
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    frames = _coerce_video_to_frames(video)
    if not frames:
        raise ValueError("No frames found to encode.")

    frames_np = np.stack(frames, axis=0)
    if frames_np.ndim == 4 and frames_np.shape[-1] == 4:
        frames_np = frames_np[..., :3]

    if frames_np.dtype == np.uint8:
        frames_u8 = frames_np
    else:
        frames_np = np.clip(frames_np, 0.0, 1.0)
        frames_np *= 255.0
        frames_u8 = np.round(frames_np).astype(np.uint8)

    # Ensure contiguous memory layout for faster PyAV muxing
    frames_u8 = np.ascontiguousarray(frames_u8)

    audio_np = _coerce_audio_to_numpy(audio) if audio is not None else None

    return mux_video_audio_bytes(
        frames_u8,
        audio_np,
        fps=float(fps),
        audio_sample_rate=audio_sample_rate or 24000,
        video_codec_options=video_codec_options,
    )


def encode_video_base64(
    video: Any,
    fps: int,
    audio: Any | None = None,
    audio_sample_rate: int | None = None,
    video_codec_options: dict[str, str] | None = None,
) -> str:
    """Encode a video (frames/array/tensor) to base64 MP4."""
    video_bytes = _encode_video_bytes(
        video, fps=fps, audio=audio, audio_sample_rate=audio_sample_rate, video_codec_options=video_codec_options
    )
    return base64.b64encode(video_bytes).decode("utf-8")
