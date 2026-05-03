import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file as a base64 string."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _looks_like_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def _detect_mime_type(file_path: str, media_kind: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if media_kind == "image":
        if suffix in (".jpg", ".jpeg"):
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix == ".gif":
            return "image/gif"
        if suffix == ".webp":
            return "image/webp"
        return "image/jpeg"

    if media_kind == "audio":
        if suffix in (".mp3", ".mpeg"):
            return "audio/mpeg"
        if suffix == ".wav":
            return "audio/wav"
        if suffix == ".ogg":
            return "audio/ogg"
        if suffix == ".flac":
            return "audio/flac"
        if suffix == ".m4a":
            return "audio/mp4"
        return "audio/wav"

    if media_kind == "video":
        if suffix == ".mp4":
            return "video/mp4"
        if suffix == ".webm":
            return "video/webm"
        if suffix == ".mov":
            return "video/quicktime"
        if suffix == ".avi":
            return "video/x-msvideo"
        if suffix == ".mkv":
            return "video/x-matroska"
        return "video/mp4"

    raise ValueError(f"Unsupported media kind: {media_kind}")


def local_path_to_data_url(file_path: str, media_kind: str) -> str:
    """Convert a local file path to a base64 data URL."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{media_kind.capitalize()} file not found: {file_path}")

    mime_type = _detect_mime_type(file_path, media_kind)
    base64_payload = encode_base64_content_from_file(file_path)
    return f"data:{mime_type};base64,{base64_payload}"


def path_or_url_to_url(value: str | None, media_kind: str, default_url: str) -> str:
    """Resolve a media input into either a remote URL or a data URL."""
    if not value:
        return default_url
    if _looks_like_url(value):
        return value
    return local_path_to_data_url(value, media_kind)


def pil_image_to_jpeg_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a JPEG data URL."""
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_b64}"


def audio_array_to_wav_data_url(audio_data: tuple[np.ndarray, int]) -> str:
    """Convert a numpy audio array to a WAV data URL."""
    audio_np, sample_rate = audio_data
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)

    buffered = io.BytesIO()
    sf.write(buffered, audio_np, sample_rate, format="WAV")
    wav_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"
