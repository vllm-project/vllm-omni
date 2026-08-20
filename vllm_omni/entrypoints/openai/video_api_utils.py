# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible video generation API.
"""

from __future__ import annotations

import base64
import binascii
import os
import tempfile
from io import BytesIO
from typing import Any, Literal

import httpx
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from vllm import envs
from vllm.multimodal.video import (
    VIDEO_LOADER_REGISTRY,
    VideoBackend,
    VideoSourceMetadata,
    VideoTargetMetadata,
)

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.protocol.videos import (
    FileImageReference,
    FileVideoReference,
    ImageReference,
    UrlImageReference,
    UrlVideoReference,
    VideoReference,
)


class VideoFrames(list[Image.Image]):
    """Decoded video frames plus source metadata."""

    def __init__(
        self,
        frames: list[Image.Image] | None = None,
        *,
        fps: float | None = None,
        source_path: str | None = None,
    ) -> None:
        super().__init__(frames or [])
        self.fps = fps
        self.frame_rate = fps
        self.source_path = source_path


def positive_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str)):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result <= 0:
        return None
    return result


def _decode_image_bytes(image_bytes: bytes, *, source: str) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: provided content is not a valid image.") from exc


@VIDEO_LOADER_REGISTRY.register("omni")
class OmniVideoBackend(VideoBackend):
    """Video backend that selects sequential first-N or last-N frames."""

    @classmethod
    def compute_frames_index_to_sample(
        cls,
        source: VideoSourceMetadata,
        target: VideoTargetMetadata,
        *,
        keep: Literal["first", "last"] = "first",
        **kwargs,
    ) -> list[int]:
        num_frames_to_sample = source.total_frames_num
        if target.num_frames > 0:
            num_frames_to_sample = min(num_frames_to_sample, target.num_frames)
        num_frames_to_sample = max(1, num_frames_to_sample)
        if num_frames_to_sample >= source.total_frames_num:
            return list(range(source.total_frames_num))
        if keep == "last":
            return list(range(source.total_frames_num - num_frames_to_sample, source.total_frames_num))
        return list(range(num_frames_to_sample))


def _decode_video_bytes(
    video_bytes: bytes,
    *,
    source: str,
    max_frames: int | None = None,
    keep: Literal["first", "last"] = "first",
    source_path: str | None = None,
) -> VideoFrames:
    if keep not in {"first", "last"}:
        raise InvalidInputReferenceError(f"Invalid {source}: video frame selection must be 'first' or 'last'.")
    if max_frames is not None and max_frames <= 0:
        raise InvalidInputReferenceError(f"Invalid {source}: max video frames must be positive.")

    loader = VIDEO_LOADER_REGISTRY.load("omni")
    num_frames = max_frames if max_frames is not None else -1
    try:
        frames_array, metadata = loader.load_bytes(
            video_bytes,
            num_frames=num_frames,
            backend="pyav",
            keep=keep,
        )
    except Exception as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: provided content is not a valid video.") from exc

    fps: float | None = positive_float(metadata.get("fps"))

    frames = [Image.fromarray(f, "RGB") for f in frames_array]
    if not frames:
        raise InvalidInputReferenceError(f"Invalid {source}: provided content is not a valid video.")
    return VideoFrames(frames, fps=fps, source_path=source_path)


def _decode_media_bytes(
    media_bytes: bytes,
    *,
    source: str,
    max_video_frames: int | None = None,
    video_keep: Literal["first", "last"] = "first",
) -> Image.Image | VideoFrames:
    try:
        return _decode_image_bytes(media_bytes, source=source)
    except InvalidInputReferenceError:
        try:
            return _decode_video_bytes(
                media_bytes,
                source=source,
                max_frames=max_video_frames,
                keep=video_keep,
            )
        except InvalidInputReferenceError as video_exc:
            raise InvalidInputReferenceError(
                f"Invalid {source}: provided content is not a valid image or video."
            ) from video_exc


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
        allow_redirects = envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS
        async with httpx.AsyncClient(timeout=60, follow_redirects=allow_redirects) as client:
            try:
                response = await client.get(image_url)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.has_redirect_location and not allow_redirects:
                    raise InvalidInputReferenceError(
                        "Invalid image_reference.image_url: redirect response was rejected because "
                        "VLLM_MEDIA_URL_ALLOW_REDIRECTS is disabled."
                    ) from exc
                raise InvalidInputReferenceError(
                    f"Invalid image_reference.image_url: server returned HTTP {exc.response.status_code}."
                ) from exc
            except httpx.RequestError as exc:
                raise InvalidInputReferenceError(
                    "Invalid image_reference.image_url: failed to download image."
                ) from exc
        return _decode_image_bytes(response.content, source="image_reference.image_url")

    raise InvalidInputReferenceError("Invalid image_reference.image_url: must be an http(s) URL or data URL.")


def _decode_base64_video(
    video_reference: str,
    *,
    source: str,
    max_frames: int | None = None,
    keep: Literal["first", "last"] = "first",
    source_path: str | None = None,
) -> VideoFrames:
    if video_reference:
        if video_reference.startswith("data:video"):
            _, b64_data = video_reference.split(",", 1)
        else:
            b64_data = video_reference

        try:
            video_bytes = base64.b64decode(b64_data)
        except (binascii.Error, ValueError) as exc:  # pragma: no cover - malformed base64
            raise InvalidInputReferenceError(f"Invalid {source}: video data is not valid base64.") from exc
        if source_path is not None:
            with open(source_path, "wb") as output:
                output.write(video_bytes)
        return _decode_video_bytes(
            video_bytes,
            source=source,
            max_frames=max_frames,
            keep=keep,
            source_path=source_path,
        )
    raise InvalidInputReferenceError(f"Invalid {source}: video data is empty.")


async def decode_video_url(
    video_url: str,
    *,
    max_frames: int | None = None,
    keep: Literal["first", "last"] = "first",
) -> VideoFrames:
    if video_url.startswith("data:video"):
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        try:
            return _decode_base64_video(
                video_url,
                source="video_reference.video_url",
                max_frames=max_frames,
                keep=keep,
                source_path=path,
            )
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            raise

    if video_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.get(video_url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise InvalidInputReferenceError(
                    "Invalid video_reference.video_url: failed to download video."
                ) from exc
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        try:
            with open(path, "wb") as output:
                output.write(response.content)
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            raise
        try:
            return _decode_video_bytes(
                response.content,
                source="video_reference.video_url",
                max_frames=max_frames,
                keep=keep,
                source_path=path,
            )
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            raise

    raise InvalidInputReferenceError("Invalid video_reference.video_url: must be an http(s) URL or data URL.")


async def decode_audio_url(audio_url: str) -> str:
    """Decode an audio URL or data-URL to a temporary file path."""
    import tempfile

    audio_bytes: bytes | None = None

    if audio_url.startswith("data:audio"):
        _, b64_data = audio_url.split(",", 1)
        try:
            audio_bytes = base64.b64decode(b64_data)
        except (binascii.Error, ValueError) as exc:
            raise InvalidInputReferenceError(
                "Invalid audio_reference.audio_url: audio data is not valid base64."
            ) from exc
    elif audio_url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.get(audio_url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise InvalidInputReferenceError(
                    "Invalid audio_reference.audio_url: failed to download audio."
                ) from exc
        audio_bytes = response.content
    else:
        raise InvalidInputReferenceError("Invalid audio_reference.audio_url: must be an http(s) URL or data URL.")

    if not audio_bytes:
        raise InvalidInputReferenceError("Invalid audio_reference: audio data is empty.")

    suffix = ".wav"
    if audio_url.startswith("data:audio/"):
        mime = audio_url.split(";")[0].removeprefix("data:")
        ext = mime.split("/")[-1]
        if ext in ("mpeg", "mp3"):
            suffix = ".mp3"
        elif ext == "wav":
            suffix = ".wav"
        elif ext.isalnum() and len(ext) <= 8:
            suffix = f".{ext}"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


async def decode_input_reference(
    image_reference: ImageReference | None,
    video_reference: VideoReference | None,
    input_reference_bytes: bytes | None,
    *,
    max_video_frames: int | None = None,
    video_keep: Literal["first", "last"] = "first",
) -> Image.Image | VideoFrames | None:
    """Decode media input from multipart bytes, data URLs, or typed references."""

    provided = sum(item is not None for item in (input_reference_bytes, image_reference, video_reference))
    if provided > 1:
        raise InvalidInputReferenceError("Provide only one of input_reference, image_reference, or video_reference.")

    if isinstance(input_reference_bytes, bytes):
        return _decode_media_bytes(
            input_reference_bytes,
            source="input_reference",
            max_video_frames=max_video_frames,
            video_keep=video_keep,
        )

    if isinstance(image_reference, UrlImageReference):
        return await decode_image_url(image_reference.image_url)
    elif isinstance(image_reference, FileImageReference):
        raise InvalidInputReferenceError("Invalid image_reference: file_id is not supported yet.")

    if isinstance(video_reference, UrlVideoReference):
        return await decode_video_url(
            video_reference.video_url,
            max_frames=max_video_frames,
            keep=video_keep,
        )
    elif isinstance(video_reference, FileVideoReference):
        raise InvalidInputReferenceError("Invalid video_reference: file_id is not supported yet.")

    return None


def _normalize_video_tensor(video_tensor: torch.Tensor) -> np.ndarray:
    """Normalize a torch video tensor into a numpy array of frames (F, H, W, C)."""
    video_tensor = video_tensor.detach().cpu()
    if video_tensor.dim() == 5:
        raise ValueError("Batched video tensors are not supported for single-video encoding.")

    if video_tensor.is_floating_point():
        # Cast to float32 first: bf16 (e.g. SANA-WM's refiner output) has no
        # numpy dtype, so ``.numpy()`` below raises on it.
        video_tensor = video_tensor.float().clamp(-1, 1) * 0.5 + 0.5
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
        # Prefer an explicit channel-last dimension for ambiguous 3/4-frame
        # videos before interpreting a leading dimension as channels.
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


def _coerce_video_to_uint8_frames(video: Any) -> np.ndarray:
    """Convert a video payload into contiguous uint8 frames shaped (F, H, W, 3)."""
    frames = _coerce_video_to_frames(video)
    if not frames:
        raise ValueError("No frames found to encode.")

    frame_shape = frames[0].shape
    has_alpha = len(frame_shape) == 3 and frame_shape[-1] == 4
    output_shape = (*frame_shape[:-1], 3) if has_alpha else frame_shape
    frames_u8 = np.empty((len(frames), *output_shape), dtype=np.uint8)
    common_dtype = np.result_type(*(frame.dtype for frame in frames))

    # Convert one frame at a time instead of stacking the normalized float
    # payload first. Long videos can otherwise require another full-size
    # float array plus conversion temporaries before encoding.
    for index, frame in enumerate(frames):
        if frame.shape != frame_shape:
            raise ValueError("All video frames must have the same shape.")
        frame = frame[..., :3] if has_alpha else frame
        if frame.dtype == np.uint8:
            frames_u8[index] = frame
            continue

        # np.stack(), used by the previous implementation, promoted mixed
        # frame dtypes before scaling. Preserve those rounding semantics
        # without allocating a full-video float buffer.
        scaled = np.clip(frame.astype(common_dtype, copy=False), 0.0, 1.0)
        scaled *= 255.0
        np.rint(scaled, out=scaled)
        frames_u8[index] = scaled

    return frames_u8


def _encode_video_bytes(
    video: Any,
    fps: int,
    audio: Any | None = None,
    audio_sample_rate: int | None = None,
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Encode a video payload into MP4 bytes, optionally muxing audio."""
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    audio_np = _coerce_audio_to_numpy(audio) if audio is not None else None

    return mux_video_audio_bytes(
        _coerce_video_to_uint8_frames(video),
        audio_np,
        fps=float(fps),
        audio_sample_rate=audio_sample_rate or 24000,
        video_codec_options=video_codec_options,
    )


class FragmentedMP4VideoEncoder:
    """Normalize video chunks and append them to one fragmented MP4 stream."""

    def __init__(
        self,
        *,
        fps: int | float,
        video_codec_options: dict[str, str] | None = None,
    ) -> None:
        self._fps = float(fps)
        self._video_codec_options = video_codec_options
        self._muxer: Any | None = None

    def encode(self, video: Any) -> bytes:
        """Encode one generated video chunk and return newly emitted fMP4 bytes."""
        from vllm_omni.diffusion.utils.media_utils import FragmentedMP4Muxer

        frames_u8 = _coerce_video_to_uint8_frames(video)
        if self._muxer is None:
            self._muxer = FragmentedMP4Muxer(
                width=frames_u8.shape[2],
                height=frames_u8.shape[1],
                fps=self._fps,
                video_codec_options=self._video_codec_options,
            )
        return self._muxer.mux_video_frames(frames_u8)

    def close(self) -> bytes:
        """Close the underlying fMP4 muxer and return trailing bytes, if any."""
        if self._muxer is None:
            return b""
        return self._muxer.close()


StreamingVideoFormat = Literal["m4s"]


def create_streaming_video_encoder(
    *,
    output_format: StreamingVideoFormat,
    fps: int | float,
    video_codec_options: dict[str, str] | None = None,
) -> FragmentedMP4VideoEncoder:
    """Create an incremental encoder for the requested WebSocket video format."""
    if output_format == "m4s":
        return FragmentedMP4VideoEncoder(fps=fps, video_codec_options=video_codec_options)
    raise ValueError(f"Unsupported streaming video format: {output_format}")


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
