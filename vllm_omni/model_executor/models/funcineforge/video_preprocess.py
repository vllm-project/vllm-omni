# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video-front-end helpers for FunCineForge.

The official Gradio demo accepts a whole video plus clip metadata, then
materializes the real FunCineForge inputs: a clipped reference wav, visual face
embeddings, dialogue metadata, and a 25 Hz target speech length.  These helpers
keep that blocking, dependency-heavy preprocessing out of the model hot path
while making the OpenAI speech endpoint able to accept video directly.
"""

from __future__ import annotations

import base64
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile as sf
import torch

from vllm_omni.model_executor.models.funcineforge.config import FunCineForgeConfig
from vllm_omni.model_executor.models.funcineforge.utils import load_face_embedding


@dataclass(slots=True)
class FunCineForgeVideoConditions:
    ref_audio: tuple[np.ndarray, int]
    face_embedding: torch.Tensor
    speech_len: int
    speech_type: str
    dialogue: list[dict[str, object]]
    work_dir: str
    ref_audio_path: str
    video_clip_path: str
    face_path: str


_FRONTEND_CACHE: dict[tuple[str, str, str, str], object] = {}


def materialize_video_source(video: str, work_dir: str) -> str:
    """Resolve a video URI/data URL/local path to a local filesystem path."""
    parsed = urlparse(video)
    if parsed.scheme in {"http", "https"}:
        suffix = Path(parsed.path).suffix or ".mp4"
        dst = Path(work_dir) / f"source{suffix}"
        with urlopen(video) as response:  # noqa: S310 - user-provided media URI
            dst.write_bytes(response.read())
        return str(dst)
    if video.startswith("data:"):
        header, _, payload = video.partition(",")
        media_type = header.split(";", 1)[0].split(":", 1)[-1]
        suffix = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/webm": ".webm",
            "application/octet-stream": ".mp4",
        }.get(media_type, ".mp4")
        dst = Path(work_dir) / f"source{suffix}"
        dst.write_bytes(base64.b64decode(payload))
        return str(dst)
    if parsed.scheme == "file":
        return os.path.abspath(os.path.join(parsed.netloc, parsed.path))
    return os.path.abspath(os.path.expanduser(video))


def validate_video_segment(start: float, end: float, duration: float) -> None:
    if start < 0:
        raise ValueError(f"video_start ({start}s) cannot be negative")
    if end > duration:
        raise ValueError(f"video_end ({end}s) cannot exceed video duration ({duration:.2f}s)")
    segment_duration = end - start
    if segment_duration <= 0:
        raise ValueError("video_start must be smaller than video_end")
    if segment_duration <= 2:
        raise ValueError("FunCineForge video segment must be longer than 2 seconds")
    if segment_duration >= 30:
        raise ValueError("FunCineForge video segment must be shorter than 30 seconds")


def _video_duration(video_path: str) -> float:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError as exc:
        raise ImportError(
            "FunCineForge video preprocessing requires moviepy. "
            "Install the official demo dependencies or set ref_audio/face_path directly."
        ) from exc

    clip = VideoFileClip(video_path)
    try:
        return float(clip.duration)
    finally:
        clip.close()


def _subclip(clip: object, start: float, end: float) -> object:
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def _write_silent_wav(path: str, duration: float, sample_rate: int = 16000) -> None:
    num_samples = max(1, int(sample_rate * duration))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00" * (num_samples * 2))


def _clip_video_segment(video_path: str, start: float, end: float, work_dir: str) -> tuple[str, str]:
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
    except ImportError as exc:
        raise ImportError(
            "FunCineForge video preprocessing requires moviepy. "
            "Install the official demo dependencies or set ref_audio/face_path directly."
        ) from exc

    clip_name = "clip_0"
    video_clip_path = str(Path(work_dir) / f"{clip_name}.mp4")
    audio_clip_path = str(Path(work_dir) / f"{clip_name}.wav")
    original = VideoFileClip(video_path)
    clip = _subclip(original, start, end)
    try:
        clip.write_videofile(video_clip_path, codec="libx264", audio_codec="aac", logger=None)
        if getattr(clip, "audio", None) is not None:
            clip.audio.write_audiofile(audio_clip_path, codec="pcm_s16le", logger=None)
        else:
            _write_silent_wav(audio_clip_path, end - start)
    finally:
        clip.close()
        original.close()
    return video_clip_path, audio_clip_path


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, int(sr)


def _get_demo_root() -> str | None:
    root = os.environ.get("FUNCINEFORGE_DEMO_ROOT")
    if root:
        return root

    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate in (
            parent / "tmp" / "Fun-CineForge-Demo",
            parent / "tmp" / "FunCineForge",
        ):
            if (candidate / "speaker_diarization").exists():
                return str(candidate)
    return None


def _build_frontend(
    *,
    pretrained_dir: str | None = None,
    diar_config_path: str | None = None,
    device: str | None = None,
) -> object:
    """Create/cache the official visual frontend from the Gradio demo."""
    demo_root = _get_demo_root()
    if demo_root is None:
        raise RuntimeError(
            "FunCineForge video preprocessing needs the official demo checkout. "
            "Clone https://huggingface.co/spaces/FunAudioLLM/Fun-CineForge-Demo "
            "and set FUNCINEFORGE_DEMO_ROOT to that directory, or pass face_path directly."
        )

    import sys

    if demo_root not in sys.path:
        sys.path.insert(0, demo_root)

    if pretrained_dir is None:
        pretrained_dir = os.environ.get("FUNCINEFORGE_PRETRAIN_DIR") or str(Path(demo_root) / "pretrained_models")
    if diar_config_path is None:
        diar_config_path = os.environ.get("FUNCINEFORGE_DIAR_CONFIG") or str(
            Path(demo_root) / "decode_conf" / "diar.yaml"
        )
    if device is None:
        device = os.environ.get("FUNCINEFORGE_PREPROCESS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    key = (demo_root, pretrained_dir, diar_config_path, device)
    cached = _FRONTEND_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from speaker_diarization.run import GlobalModels
    except ImportError as exc:
        raise ImportError(
            "Could not import speaker_diarization from the FunCineForge demo checkout. "
            "Set FUNCINEFORGE_DEMO_ROOT to the cloned Space directory."
        ) from exc

    frontend = GlobalModels(
        hf_token=None,
        config_path=diar_config_path,
        pretrained_dir=pretrained_dir,
        device=device,
        pool_sizes={"face": 1, "asd": 1, "fr": 1},
        batch_size=1,
        preload=True,
    )
    _FRONTEND_CACHE[key] = frontend
    return frontend


def _extract_visual_embeddings(
    frontend: object,
    *,
    video_clip_path: str,
    audio_clip_path: str,
    face_path: str,
    duration: float,
) -> None:
    try:
        from speaker_diarization.local.vision_processer import VisionProcesser
    except ImportError as exc:
        raise ImportError("Could not import VisionProcesser from the FunCineForge demo checkout.") from exc

    vp = VisionProcesser(
        video_file_path=video_clip_path,
        audio_file_path=audio_clip_path,
        audio_vad=[[0.0, round(duration, 2)]],
        out_feat_path=face_path,
        visual_models=frontend,
        conf=frontend.conf,
        out_video_path=None,
    )
    try:
        vp.run()
    finally:
        vp.close()


def build_video_conditions(
    *,
    video: str,
    start: float | None = None,
    end: float | None = None,
    age: str | None = None,
    gender: str | None = None,
    speech_type: str | None = None,
    work_dir: str | None = None,
    frontend: object | None = None,
    pretrained_dir: str | None = None,
    diar_config_path: str | None = None,
    device: str | None = None,
) -> FunCineForgeVideoConditions:
    """Preprocess a video segment into direct FunCineForge model conditions."""
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="funcineforge_video_")
    else:
        os.makedirs(work_dir, exist_ok=True)

    video_path = materialize_video_source(video, work_dir)
    duration = _video_duration(video_path)
    seg_start = float(start or 0.0)
    seg_end = float(end) if end is not None else duration
    validate_video_segment(seg_start, seg_end, duration)

    padded_start = max(0.0, seg_start - 0.1)
    padded_end = min(seg_end + 0.1, duration)
    segment_duration = padded_end - padded_start
    video_clip_path, audio_clip_path = _clip_video_segment(video_path, padded_start, padded_end, work_dir)

    face_path = str(Path(work_dir) / "clip_0.pkl")
    frontend = frontend or _build_frontend(
        pretrained_dir=pretrained_dir,
        diar_config_path=diar_config_path,
        device=device,
    )
    _extract_visual_embeddings(
        frontend,
        video_clip_path=video_clip_path,
        audio_clip_path=audio_clip_path,
        face_path=face_path,
        duration=segment_duration,
    )

    cfg = FunCineForgeConfig()
    speech_len = max(1, int(segment_duration * 25))
    face_embedding = load_face_embedding(face_path, speech_len=speech_len, face_size=cfg.face_size)
    ref_audio = _load_wav(audio_clip_path)
    dialogue = [
        {
            "start": 0.0,
            "duration": round(segment_duration, 2),
            "spk": 1,
            "gender": gender or "unknown",
            "age": age or "unknown",
        }
    ]

    return FunCineForgeVideoConditions(
        ref_audio=ref_audio,
        face_embedding=face_embedding,
        speech_len=speech_len,
        speech_type=speech_type or "独白",
        dialogue=dialogue,
        work_dir=work_dir,
        ref_audio_path=audio_clip_path,
        video_clip_path=video_clip_path,
        face_path=face_path,
    )
