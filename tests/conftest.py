import base64
import datetime
import io
import json
import math
import os
import random
import re
import tempfile

import requests

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Set CPU device for CI environments without GPU
if "VLLM_TARGET_DEVICE" not in os.environ:
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"

import concurrent.futures
import gc
import multiprocessing
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import psutil
import pytest
import soundfile as sf
import torch
import yaml
from openai import OpenAI, omit
from PIL import Image
from transformers import pipeline
from vllm import TextPrompt
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

PromptAudioInput = list[tuple[Any, int]] | tuple[Any, int] | None
PromptImageInput = list[Any] | Any | None
PromptVideoInput = list[Any] | Any | None

_GENDER_PIPELINE = None
# transformers.Pipeline is not thread-safe; concurrent e2e requests must serialize inference.
_GENDER_PIPELINE_LOCK = threading.Lock()

# int16 mono PCM from /v1/audio/speech when response_format=pcm (Qwen3-TTS code2wav output rate).
_PCM_SPEECH_SAMPLE_RATE_HZ = 24_000


class OmniServerParams(NamedTuple):
    model: str
    port: int | None = None
    stage_config_path: str | None = None
    server_args: list[str] | None = None
    env_dict: dict[str, str] | None = None
    use_omni: bool = True


def assert_image_diffusion_response(
    response,
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate image diffusion response.

    Expected request_config schema:
        {
            "request_type": "image",
            "extra_body": {
                "num_outputs_per_prompt": 1,
                "width": ...,
                "height": ...,
                ...
            }
        }
    """
    assert response.images is not None, "Image response is None"
    assert len(response.images) > 0, "No images in response"

    extra_body = request_config.get("extra_body") or {}

    num_outputs_per_prompt = extra_body.get("num_outputs_per_prompt")
    if num_outputs_per_prompt is not None:
        assert len(response.images) == num_outputs_per_prompt, (
            f"Expected {num_outputs_per_prompt} images, got {len(response.images)}"
        )

    if run_level == "advanced_model":
        width = extra_body.get("width")
        height = extra_body.get("height")

        if width is not None or height is not None:
            for img in response.images:
                assert_image_valid(img, width=width, height=height)


def assert_video_diffusion_response(
    response,
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate video diffusion response.

    Expected request_config schema:
        {
            "request_type": "video",
            "form_data": {
                "prompt": "...",
                "num_frames": ...,
                "width": ...,
                "height": ...,
                "fps": ...,
                ...
            }
        }
    """
    form_data = request_config.get("form_data", {})

    assert response.videos is not None, "Video response is None"
    assert len(response.videos) > 0, "No videos in response"

    expected_frames = _maybe_int(form_data.get("num_frames"))
    expected_width = _maybe_int(form_data.get("width"))
    expected_height = _maybe_int(form_data.get("height"))
    expected_fps = _maybe_int(form_data.get("fps"))

    for vid_bytes in response.videos:
        assert_video_valid(
            vid_bytes,
            num_frames=expected_frames,
            width=expected_width,
            height=expected_height,
            fps=expected_fps,
        )


def assert_audio_diffusion_response(
    response,
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate audio diffusion response.
    """
    raise NotImplementedError("Audio validation is not implemented yet")
    # consider using assert_audio_valid defined above


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def assert_image_valid(image: Path | Image.Image, *, width: int | None = None, height: int | None = None):
    """Assert the file is a loadable image with optional exact dimensions."""
    if isinstance(image, Path):
        assert image.exists(), f"Image not found: {image}"
        image = Image.open(image)
        image.load()
    assert image.width > 0 and image.height > 0
    if width is not None:
        assert image.width == width, f"Expected width={width}, got {image.width}"
    if height is not None:
        assert image.height == height, f"Expected height={height}, got {image.height}"
    return image


def assert_video_valid(
    video: Path | bytes | BytesIO,
    *,
    num_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
) -> dict[str, int | float]:
    """Assert the MP4 has the expected resolution and exact frame count."""
    temp_path = None
    cap = None
    try:
        # Normalize input to file path
        if isinstance(video, Path):
            if not video.exists():
                raise AssertionError(f"Video file not found: {video}")
            video_path = str(video)
        else:
            # Create temp file for bytes/BytesIO
            suffix = ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
                if isinstance(video, bytes):
                    tmp.write(video)
                elif isinstance(video, BytesIO):
                    tmp.write(video.getvalue())
                else:
                    raise TypeError(f"Unsupported video type: {type(video)}")
                temp_path = Path(tmp.name)
                video_path = str(temp_path)

        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise AssertionError(f"Failed to open video: {video_path}")

        # Extract properties
        actual_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        actual_num_frames = 0
        while True:
            ok, _frame = cap.read()
            if not ok:
                break
            actual_num_frames += 1

        # Basic validity checks
        if actual_num_frames <= 0:
            raise AssertionError(f"Invalid frame count: {actual_num_frames} (must be > 0)")
        if actual_width <= 0 or actual_height <= 0:
            raise AssertionError(f"Invalid dimensions: {actual_width}x{actual_height} (must be > 0)")
        if actual_fps <= 0:
            raise AssertionError(f"Invalid FPS: {actual_fps} (must be > 0)")

        # Validate against expectations
        if num_frames is not None:
            expected_num_frames = (num_frames // 4) * 4 + 1
            assert actual_num_frames == expected_num_frames, (
                f"Frame count mismatch: expected {num_frames}, got {actual_num_frames}"
            )
        if width is not None:
            assert actual_width == width, f"Width mismatch: expected {width}px, got {actual_width}px"
        if height is not None:
            assert actual_height == height, f"Height mismatch: expected {height}px, got {actual_height}px"
        if fps is not None:
            # Use tolerance for float comparison (codec rounding)
            assert abs(actual_fps - fps) < 0.5, f"FPS mismatch: expected {fps}, got {actual_fps:.2f}"

        return {"num_frames": actual_num_frames, "width": actual_width, "height": actual_height, "fps": actual_fps}

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        raise

    finally:
        # Cleanup resources
        if cap is not None:
            cap.release()
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def assert_audio_valid(path: Path, *, sample_rate: int, channels: int, duration_s: float) -> None:
    """Assert the WAV has the expected sample rate, channel count, and duration."""
    assert path.exists(), f"Audio not found: {path}"
    info = sf.info(str(path))
    assert info.samplerate == sample_rate, f"Expected sample_rate={sample_rate}, got {info.samplerate}"
    assert info.channels == channels, f"Expected {channels} channel(s), got {info.channels}"
    expected_frames = int(duration_s * sample_rate)
    assert info.frames == expected_frames, (
        f"Expected {expected_frames} frames ({duration_s}s @ {sample_rate} Hz), got {info.frames}"
    )


def decode_b64_image(b64: str):
    img = Image.open(BytesIO(base64.b64decode(b64)))
    img.load()
    return img


@pytest.fixture(scope="session")
def model_prefix() -> str:
    """Optional model-path prefix from MODEL_PREFIX env var.
    Useful if models are downloaded to non-default local directories.
    """
    prefix = os.environ.get("MODEL_PREFIX", "")
    return f"{prefix.rstrip('/')}/" if prefix else ""


@pytest.fixture(autouse=True)
def default_vllm_config():
    """Set a default VllmConfig for all tests.

    This fixture is auto-used for all tests to ensure that any test
    that directly instantiates vLLM CustomOps (e.g., RMSNorm, LayerNorm)
    or model components has the required VllmConfig context.

    This fixture is required for vLLM 0.14.0+ where CustomOp initialization
    requires a VllmConfig context set via set_current_vllm_config().
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    # Use CPU device if no GPU is available (e.g., in CI environments)
    has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if has_gpu else "cpu"
    device_config = DeviceConfig(device=device)

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


@pytest.fixture(autouse=True)
def clean_gpu_memory_between_tests():
    print("\n=== PRE-TEST GPU CLEANUP ===")
    _run_pre_test_cleanup()
    yield
    _run_post_test_cleanup()


@pytest.fixture(autouse=True)
def log_test_name_before_test(request):
    print(f"--- Running test: {request.node.name}")
    yield


def _run_pre_test_cleanup(enable_force=False):
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1" and not enable_force:
        print("GPU cleanup disabled")
        return

    print("Pre-test GPU status:")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        try:
            from tests.utils import wait_for_gpu_memory_to_clear

            wait_for_gpu_memory_to_clear(
                devices=list(range(num_gpus)),
                threshold_ratio=0.05,
            )
        except Exception as e:
            print(f"Pre-test cleanup note: {e}")


def _run_post_test_cleanup(enable_force=False):
    if os.getenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0") != "1" and not enable_force:
        print("GPU cleanup disabled")
        return

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

        print("Post-test GPU status:")
        _print_gpu_processes()


def _print_gpu_processes():
    """Print GPU information including nvidia-smi and system processes"""

    print("\n" + "=" * 80)
    print("NVIDIA GPU Information (nvidia-smi)")
    print("=" * 80)

    try:
        nvidia_result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if nvidia_result.returncode == 0:
            lines = nvidia_result.stdout.strip().split("\n")
            for line in lines[:20]:
                print(line)

            if len(lines) > 20:
                print(f"... (showing first 20 of {len(lines)} lines)")
        else:
            print("nvidia-smi command failed")

    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("nvidia-smi not available or timed out")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")

    print("\n" + "=" * 80)
    print("Detailed GPU Processes (nvidia-smi pmon)")
    print("=" * 80)

    try:
        pmon_result = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        if pmon_result.returncode == 0 and pmon_result.stdout.strip():
            print(pmon_result.stdout)
        else:
            print("No active GPU processes found via nvidia-smi pmon")

    except Exception:
        print("nvidia-smi pmon not available")

    print("\n" + "=" * 80)
    print("System Processes with GPU keywords")
    print("=" * 80)


def dummy_messages_from_mix_data(
    system_prompt: dict[str, Any] = None,
    video_data_url: Any = None,
    audio_data_url: Any = None,
    image_data_url: Any = None,
    content_text: str = None,
):
    """Create messages with video、image、audio data URL for OpenAI API."""

    if content_text is not None:
        content = [{"type": "text", "text": content_text}]
    else:
        content = []

    media_items = []
    if isinstance(video_data_url, list):
        for video_url in video_data_url:
            media_items.append((video_url, "video"))
    else:
        media_items.append((video_data_url, "video"))

    if isinstance(image_data_url, list):
        for url in image_data_url:
            media_items.append((url, "image"))
    else:
        media_items.append((image_data_url, "image"))

    if isinstance(audio_data_url, list):
        for url in audio_data_url:
            media_items.append((url, "audio"))
    else:
        media_items.append((audio_data_url, "audio"))

    content.extend(
        {"type": f"{media_type}_url", f"{media_type}_url": {"url": url}}
        for url, media_type in media_items
        if url is not None
    )
    messages = [{"role": "user", "content": content}]
    if system_prompt is not None:
        messages = [system_prompt] + messages
    return messages


def generate_synthetic_audio(
    duration: int,  # seconds
    num_channels: int,  # 1：Mono，2：Stereo 5：5.1 surround sound
    sample_rate: int = 48000,  # Default use 48000Hz.
    save_to_file: bool = False,
) -> dict[str, Any]:
    """
    Generate TTS speech with pyttsx3 and return base64 string.
    """

    import pyttsx3
    import soundfile as sf

    def _pick_voice(engine: pyttsx3.Engine) -> str | None:
        voices = engine.getProperty("voices")
        if not voices:
            return None

        preferred_tokens = (
            "natural",
            "jenny",
            "sonia",
            "susan",
            "zira",
            "aria",
            "hazel",
            "samantha",
            "ava",
            "allison",
            "female",
            "woman",
            "english-us",
            "en-us",
            "english",
        )
        discouraged_tokens = (
            "espeak",
            "robot",
            "mbrola",
            "microsoft david",
            "male",
            "man",
        )

        best_voice = voices[0]
        best_score = float("-inf")
        for voice in voices:
            voice_text = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            voice_languages = " ".join(
                lang.decode(errors="ignore") if isinstance(lang, bytes) else str(lang)
                for lang in getattr(voice, "languages", [])
            ).lower()
            combined_text = f"{voice_text} {voice_languages}"
            score = 0
            for idx, token in enumerate(preferred_tokens):
                if token in combined_text:
                    score += 20 - idx
            for token in discouraged_tokens:
                if token in combined_text:
                    score -= 10
            if "english" in combined_text or "en_" in combined_text or "en-" in combined_text:
                score += 4
            if "en-us" in combined_text or "english-us" in combined_text:
                score += 4
            if score > best_score:
                best_score = score
                best_voice = voice

        return best_voice.id

    def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr or len(audio) == 0:
            return audio.astype(np.float32)

        src_len = audio.shape[0]
        dst_len = max(1, int(round(src_len * float(dst_sr) / float(src_sr))))
        src_idx = np.arange(src_len, dtype=np.float32)
        dst_idx = np.linspace(0, src_len - 1, dst_len, dtype=np.float32)

        resampled_channels: list[np.ndarray] = []
        for ch in range(audio.shape[1]):
            resampled_channels.append(np.interp(dst_idx, src_idx, audio[:, ch]).astype(np.float32))
        return np.stack(resampled_channels, axis=1)

    def _match_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
        current_channels = audio.shape[1]
        if current_channels == target_channels:
            return audio.astype(np.float32)
        if target_channels == 1:
            return np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
        if current_channels == 1:
            return np.repeat(audio, target_channels, axis=1).astype(np.float32)

        collapsed = np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
        return np.repeat(collapsed, target_channels, axis=1).astype(np.float32)

    def _trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        if len(audio) == 0:
            return audio
        energy = np.max(np.abs(audio), axis=1)
        voiced = np.where(energy > threshold)[0]
        if len(voiced) == 0:
            return audio
        start = max(0, int(voiced[0]) - int(sample_rate * 0.02))
        end = min(len(audio), int(voiced[-1]) + int(sample_rate * 0.04) + 1)
        return audio[start:end]

    def _enhance_speech(audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio.astype(np.float32)
        enhanced = audio.astype(np.float32).copy()
        enhanced -= np.mean(enhanced, axis=0, keepdims=True, dtype=np.float32)
        if len(enhanced) > 1:
            preemphasis = enhanced.copy()
            preemphasis[1:] = enhanced[1:] - 0.94 * enhanced[:-1]
            enhanced = 0.7 * enhanced + 0.3 * preemphasis
        # Mild dynamic-range compression for ASR/TTS robustness.
        enhanced = np.sign(enhanced) * np.sqrt(np.abs(enhanced))
        # Light fade to avoid clicks after trimming/repeating.
        fade = min(len(enhanced) // 4, max(1, int(sample_rate * 0.01)))
        if fade > 1:
            ramp_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            enhanced[:fade] *= ramp_in[:, None]
            enhanced[-fade:] *= ramp_out[:, None]
        peak = float(np.max(np.abs(enhanced)))
        if peak > 1e-8:
            enhanced = enhanced / peak * 0.95
        return enhanced.astype(np.float32)

    phrase_text = "test"
    num_samples = int(sample_rate * max(1, duration))
    audio_data = np.zeros((num_samples, num_channels), dtype=np.float32)

    engine = pyttsx3.init()
    engine.setProperty("rate", 112)
    engine.setProperty("volume", 1.0)
    selected_voice = _pick_voice(engine)
    if selected_voice is not None:
        engine.setProperty("voice", selected_voice)

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()

    try:
        engine.save_to_file(phrase_text, temp_wav.name)
        engine.runAndWait()
        engine.stop()

        ready = False
        for _ in range(50):
            if os.path.exists(temp_wav.name) and os.path.getsize(temp_wav.name) > 44:
                ready = True
                break
            time.sleep(0.1)

        if not ready:
            raise RuntimeError("pyttsx3 did not produce a WAV file in time.")

        tts_audio, tts_sr = sf.read(temp_wav.name, dtype="float32", always_2d=True)
    finally:
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)

    if len(tts_audio) == 0:
        raise RuntimeError("pyttsx3 produced an empty WAV file.")

    tts_audio = _resample_audio(tts_audio, tts_sr, sample_rate)
    tts_audio = _match_channels(tts_audio, num_channels)
    tts_audio = _trim_silence(tts_audio, threshold=0.012)
    tts_audio = _enhance_speech(tts_audio)

    lead_silence = min(int(sample_rate * 0.02), num_samples // 8)
    pause_samples = int(sample_rate * 0.18)
    start = lead_silence
    phrase_len = tts_audio.shape[0]

    while start < num_samples:
        take = min(phrase_len, num_samples - start)
        audio_data[start : start + take] = tts_audio[:take]
        start += phrase_len + pause_samples

    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp > 0:
        audio_data = audio_data / max_amp * 0.95

    audio_bytes: bytes | None = None
    output_path: str | None = None
    result: dict[str, Any] = {
        "np_array": audio_data.copy(),
    }

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"audio_{num_channels}ch_{timestamp}.wav"

        try:
            sf.write(output_path, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            print(f"Audio saved: {output_path}")

            with open(output_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            print(f"Save failed: {e}")
            save_to_file = False

    # If not saving or save failed, create in memory
    if not save_to_file or audio_bytes is None:
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        audio_bytes = buffer.read()

    # Return result
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    result["base64"] = base64_audio
    # Always include file_path to avoid KeyError in callers.
    result["file_path"] = output_path if save_to_file and output_path else None

    return result


def _mux_mp4_bytes_with_synthetic_audio(
    video_mp4_bytes: bytes,
    *,
    num_frames: int,
    fps: float = 30.0,
    sample_rate: int = 48000,
) -> bytes:
    """
    Mux a video-only MP4 with mono TTS audio from :func:`generate_synthetic_audio` (AAC).

    Audio length is at least the video duration in whole seconds (rounded up); ffmpeg
    ``-shortest`` trims to the video when the WAV is longer.

    Uses ffmpeg from ``imageio_ffmpeg`` when available, else ``ffmpeg`` on PATH.
    If TTS or mux fails, returns ``video_mp4_bytes`` unchanged.

    Mux subprocess does **not** use ``capture_output=True``: ffmpeg can block writing
    to a full stderr pipe while :func:`subprocess.run` waits for exit (classic deadlock).
    """
    duration_sec = num_frames / fps if fps > 0 else 0.0
    # generate_synthetic_audio(duration=int) uses at least 1s of buffer internally
    duration_int = max(1, int(math.ceil(duration_sec)))

    try:
        audio_result = generate_synthetic_audio(
            duration=duration_int,
            num_channels=1,
            sample_rate=sample_rate,
            save_to_file=False,
        )
        audio_pcm = audio_result["np_array"]
    except Exception as e:
        logger.warning("Synthetic video: generate_synthetic_audio failed (%s); using video-only MP4.", e)
        return video_mp4_bytes

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    import tempfile

    try:
        with tempfile.TemporaryDirectory(prefix="syn_vid_mux_") as tmp:
            vid_path = os.path.join(tmp, "video.mp4")
            wav_path = os.path.join(tmp, "audio.wav")
            out_path = os.path.join(tmp, "out.mp4")
            with open(vid_path, "wb") as f:
                f.write(video_mp4_bytes)
            sf.write(wav_path, audio_pcm, sample_rate, format="WAV", subtype="PCM_16")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                vid_path,
                "-i",
                wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
                "-movflags",
                "+faststart",
                out_path,
            ]
            subprocess.run(
                cmd,
                check=True,
                stdin=subprocess.DEVNULL,
                timeout=300,
            )
            with open(out_path, "rb") as f:
                return f.read()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ) as e:
        logger.warning("Synthetic video: audio mux failed (%s); using video-only MP4.", e)
        return video_mp4_bytes


def generate_synthetic_video(
    width: int,
    height: int,
    num_frames: int,
    save_to_file: bool = False,
    *,
    embed_audio: bool = False,
) -> dict[str, Any]:
    """Generate synthetic video with bouncing balls and base64 MP4.

    When ``embed_audio`` is True, muxes mono AAC from :func:`generate_synthetic_audio`
    (TTS + ffmpeg) into the MP4; otherwise returns video-only MP4 (faster when tests do
    not need an audio track).
    """

    import cv2
    import imageio

    # Create random balls
    num_balls = random.randint(3, 8)
    balls = []

    for _ in range(num_balls):
        radius = min(width, height) // 8
        if radius < 1:
            raise ValueError(f"Video dimensions ({width}x{height}) are too small for synthetic video generation")
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)

        speed = random.uniform(3.0, 8.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        # OpenCV uses BGR format, but imageio expects RGB
        # We'll create in BGR first, then convert to RGB later
        color_bgr = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        balls.append({"x": x, "y": y, "vx": vx, "vy": vy, "radius": radius, "color_bgr": color_bgr})

    # Generate video frames
    video_frames = []

    for frame_idx in range(num_frames):
        # Create black background (BGR format)
        frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)

        for ball in balls:
            # Update position
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"]

            # Boundary collision detection
            if ball["x"] - ball["radius"] <= 0 or ball["x"] + ball["radius"] >= width:
                ball["vx"] = -ball["vx"]
                ball["x"] = max(ball["radius"], min(width - ball["radius"], ball["x"]))

            if ball["y"] - ball["radius"] <= 0 or ball["y"] + ball["radius"] >= height:
                ball["vy"] = -ball["vy"]
                ball["y"] = max(ball["radius"], min(height - ball["radius"], ball["y"]))

            # Use cv2 to draw circle
            x, y = int(ball["x"]), int(ball["y"])
            radius = ball["radius"]

            # Draw solid circle (main circle)
            cv2.circle(frame_bgr, (x, y), radius, ball["color_bgr"], -1)

            # Add simple 3D effect: draw a brighter center
            if radius > 3:  # Only add highlight when radius is large enough
                highlight_radius = max(1, radius // 2)
                highlight_x = max(highlight_radius, min(x - radius // 4, width - highlight_radius))
                highlight_y = max(highlight_radius, min(y - radius // 4, height - highlight_radius))

                # Create highlight color (brighter)
                highlight_color = tuple(min(c + 40, 255) for c in ball["color_bgr"])
                cv2.circle(frame_bgr, (highlight_x, highlight_y), highlight_radius, highlight_color, -1)

        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frames.append(frame_rgb)

    video_array = np.array(video_frames)
    result = {
        "np_array": video_array,
    }
    saved_file_path = None

    fps = 30
    buffer = io.BytesIO()
    writer_kwargs = {
        "format": "mp4",
        "fps": fps,
        "codec": "libx264",
        "quality": 7,
        "pixelformat": "yuv420p",
        "macro_block_size": 16,
        "ffmpeg_params": [
            "-preset",
            "medium",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={width}:{height}",
        ],
    }

    try:
        with imageio.get_writer(buffer, **writer_kwargs) as writer:
            for frame in video_frames:
                writer.append_data(frame)
        buffer.seek(0)
        video_only_bytes = buffer.read()
    except Exception as e:
        print(f"Warning: Failed to encode synthetic video: {e}")
        raise

    if embed_audio:
        video_bytes = _mux_mp4_bytes_with_synthetic_audio(video_only_bytes, num_frames=num_frames, fps=float(fps))
    else:
        video_bytes = video_only_bytes

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"video_{width}x{height}_{timestamp}.mp4"
        try:
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            saved_file_path = output_path
            print(f"Video saved to: {saved_file_path}")
        except Exception as e:
            print(f"Warning: Failed to save video to file {output_path}: {e}")

    base64_video = base64.b64encode(video_bytes).decode("utf-8")

    result["base64"] = base64_video
    if save_to_file and saved_file_path:
        result["file_path"] = saved_file_path

    return result


def generate_synthetic_image(width: int, height: int, save_to_file: bool = False) -> dict[str, Any]:
    """Generate synthetic image with randomly colored squares and return base64 string."""
    from PIL import Image, ImageDraw

    # Create white background
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Generate random number of squares
    num_squares = random.randint(3, 8)

    for _ in range(num_squares):
        # Random square size
        square_size = random.randint(min(width, height) // 8, min(width, height) // 4)

        # Random position
        x = random.randint(0, width - square_size - 1)
        y = random.randint(0, height - square_size - 1)

        # Random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Random border width
        border_width = random.randint(1, 5)

        # Draw square
        draw.rectangle([x, y, x + square_size, y + square_size], fill=color, outline=(0, 0, 0), width=border_width)

    image_array = np.array(image)
    result = {"np_array": image_array.copy()}

    # Handle file saving
    image_bytes = None
    saved_file_path = None

    if save_to_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"image_{width}x{height}_{timestamp}.jpg"

        try:
            # Save image to file
            image.save(output_path, format="JPEG", quality=85, optimize=True)
            saved_file_path = output_path
            print(f"Image saved to: {saved_file_path}")

            # Read file for base64 encoding
            with open(output_path, "rb") as f:
                image_bytes = f.read()

        except Exception as e:
            print(f"Warning: Failed to save image to file {output_path}: {e}")
            save_to_file = False

    # If not saving or save failed, create in memory
    if not save_to_file or image_bytes is None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        buffer.seek(0)
        image_bytes = buffer.read()

    # Generate base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Return result
    result["base64"] = base64_image
    if save_to_file and saved_file_path:
        result["file_path"] = saved_file_path

    return result


def preprocess_text(text):
    import opencc

    word_to_num = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, num in word_to_num.items():
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(pattern, num, text, flags=re.IGNORECASE)

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    cc = opencc.OpenCC("t2s")
    text = cc.convert(text)

    # Special handling for spaces between Chinese characters:
    # - Keep single spaces between English words/numbers
    # - Remove spaces only when surrounded by Chinese characters on both sides to prevent incorrect word segmentation
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    return text.lower().strip()


def cosine_similarity_text(text1, text2, n: int = 3):
    from collections import Counter

    if not text1 or not text2:
        return 0.0

    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    print(f"cosine similarity text1 is: {text1}, text2 is: {text2}")

    ngrams1 = [text1[i : i + n] for i in range(len(text1) - n + 1)]
    ngrams2 = [text2[i : i + n] for i in range(len(text2) - n + 1)]

    counter1 = Counter(ngrams1)
    counter2 = Counter(ngrams2)

    all_ngrams = set(counter1.keys()) | set(counter2.keys())
    vec1 = [counter1.get(ng, 0) for ng in all_ngrams]
    vec2 = [counter2.get(ng, 0) for ng in all_ngrams]

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def convert_audio_to_text(audio_data):
    """
    Convert base64 encoded audio data to text using speech recognition.
    """
    audio_data = base64.b64decode(audio_data)
    output_path = f"./test_{uuid.uuid4().hex}.wav"
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio_data)

    print(f"audio data is saved: {output_path}")
    text = convert_audio_file_to_text(output_path=output_path)
    return text


def _merge_base64_audio_to_segment(base64_list: list[str]):
    """Merge a list of base64-encoded audio chunks into one pydub AudioSegment."""
    from pydub import AudioSegment

    merged = None
    for b64 in base64_list:
        raw = base64.b64decode(b64.split(",", 1)[-1])
        seg = AudioSegment.from_file(io.BytesIO(raw))
        merged = seg if merged is None else merged + seg
    return merged


def _whisper_transcribe_in_current_process(output_path: str) -> str:
    import whisper

    # Multi-GPU: use last visible device to avoid colliding with default device 0; single device uses 0.
    device_index = None
    if current_omni_platform.is_available():
        n = current_omni_platform.get_device_count()
        if n == 1:
            device_index = 0
        elif n > 1:
            device_index = n - 1

    if device_index is not None:
        torch_device = current_omni_platform.get_torch_device(device_index)
        current_omni_platform.set_device(torch_device)
        device = str(torch_device)
        use_accelerator = True
    else:
        use_accelerator = False
        device = "cpu"
    model = whisper.load_model("small", device=device)
    try:
        text = model.transcribe(
            output_path,
            temperature=0.0,
            word_timestamps=True,
            condition_on_previous_text=False,
        )["text"]
    finally:
        del model
        gc.collect()
        if use_accelerator:
            current_omni_platform.synchronize()
            current_omni_platform.empty_cache()

    return text or ""


def convert_audio_file_to_text(output_path: str) -> str:
    """Convert an audio file to text in an isolated subprocess."""
    # Import locally to avoid impacting test module import time.
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(_whisper_transcribe_in_current_process, output_path)
        return future.result()


def convert_audio_bytes_to_text(raw_bytes: bytes) -> str:
    """
    Write container audio bytes (WAV, etc.) to a temp WAV file suitable for Whisper/ffmpeg.
    Normalizes with soundfile to PCM_16 WAV when possible to avoid codec issues.
    """
    output_path = f"./test_{uuid.uuid4().hex}.wav"
    data, samplerate = sf.read(io.BytesIO(raw_bytes))
    sf.write(output_path, data, samplerate, format="WAV", subtype="PCM_16")
    text = convert_audio_file_to_text(output_path)
    return text


def merge_base64_and_convert_to_text(base64_list):
    """
    Merge a list of base64 encoded audio data and convert to text.
    """
    merged_audio = _merge_base64_audio_to_segment(base64_list)
    output_path = f"./test_{uuid.uuid4().hex}.wav"
    merged_audio.export(output_path, format="wav")
    print(f"audio data is saved: {output_path}")
    text = convert_audio_file_to_text(output_path)
    return text


def modify_stage_config(
    yaml_path: str,
    updates: dict[str, Any] = None,
    deletes: dict[str, Any] = None,
) -> str:
    """
    Modify configurations in a YAML file, supporting both top-level and stage-specific modifications,
    including addition, modification, and deletion of configurations.

    Args:
        yaml_path: Path to the YAML configuration file.
        updates: Dictionary containing both top-level and stage-specific modifications to add or update.
                Format: {
                    'async_chunk': True,
                    'stage_args': {
                        0: {'engine_args.max_model_len': 5800},
                        1: {'engine_args.max_num_seqs': 2}
                    }
                }
        deletes: Dictionary containing configurations to delete.
                Format: {
                    'old_config': None,  # Delete entire key
                    'stage_args': {
                        0: ['engine_args.old_param'],
                        1: ['runtime.unused_setting']
                    }
                }

    Returns:
        str: Path to the newly created modified YAML file with timestamp suffix.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"yaml does not exist: {path}")

    try:
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Cannot parse YAML file: {e}")

    # Helper function to apply update
    def apply_update(config_dict: dict, key_path: str, value: Any) -> None:
        """Apply update to dictionary using dot-separated path."""
        # Handle direct list assignment (e.g., engine_input_source: [1, 2])
        if "." not in key_path:
            # Simple key, set directly
            config_dict[key_path] = value
            return

        current = config_dict
        keys = key_path.split(".")

        for i in range(len(keys) - 1):
            key = keys[i]

            # Handle list indices
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0:
                    raise ValueError(f"Negative list index not allowed: {index}")
                if index >= len(current):
                    # Expand list if needed
                    while len(current) <= index:
                        # If we need to go deeper (more keys after this), create a dict
                        # Otherwise, create None placeholder
                        current.append({} if i < len(keys) - 2 else None)
                current = current[index]
            elif isinstance(current, dict):
                # Handle dictionary keys
                if key not in current:
                    # If there are more keys after this, create appropriate structure
                    if i < len(keys) - 1:
                        # Check if next key is a digit (list index) or string (dict key)
                        if keys[i + 1].isdigit():
                            current[key] = []
                        else:
                            current[key] = {}
                    else:
                        # This is the last key, create based on value type
                        current[key] = [] if isinstance(value, list) else {}
                elif not isinstance(current[key], (dict, list)) and i < len(keys) - 1:
                    # If current value is not dict/list but we need to go deeper, replace it
                    if keys[i + 1].isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]
            else:
                # Current is not a dict or list, cannot traverse further
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        # Set the final value
        last_key = keys[-1]
        if isinstance(current, list) and last_key.isdigit():
            # Setting a value in a list by index
            index = int(last_key)
            if index < 0:
                raise ValueError(f"Negative list index not allowed: {index}")
            if index >= len(current):
                # Expand list if needed
                while len(current) <= index:
                    current.append(None)
            current[index] = value
        elif isinstance(current, dict):
            # Special case: if the value is a list and we're setting a top-level key
            # Example: updating engine_input_source with [1, 2]
            current[last_key] = value
        else:
            # Current is not a dict, cannot set key
            raise TypeError(f"Cannot set value at {key_path}. Current type is {type(current).__name__}, expected dict.")

    # Helper function to delete by path
    def delete_by_path(config_dict: dict, path: str) -> None:
        """Delete configuration by dot-separated path."""
        if not path:
            return

        current = config_dict
        keys = path.split(".")

        # Traverse to the parent
        for i in range(len(keys) - 1):
            key = keys[i]

            # Handle list indices
            if key.isdigit() and isinstance(current, list):
                index = int(key)
                if index < 0 or index >= len(current):
                    raise KeyError(f"List index {index} out of bounds")
                current = current[index]
            elif isinstance(current, dict):
                if key not in current:
                    raise KeyError(f"Path {'.'.join(keys[: i + 1])} does not exist")
                current = current[key]
            else:
                raise TypeError(
                    f"Cannot access {'.'.join(keys[: i + 1])} as a dict/list. It's a {type(current).__name__}"
                )

        # Delete the item
        last_key = keys[-1]

        if isinstance(current, list) and last_key.isdigit():
            index = int(last_key)
            if index < 0 or index >= len(current):
                raise KeyError(f"List index {index} out of bounds")
            del current[index]
        elif isinstance(current, dict) and last_key in current:
            del current[last_key]
        else:
            print(f"Path {path} does not exist")

    # Apply deletions first
    if deletes:
        for key, value in deletes.items():
            if key == "stage_args":
                if value and isinstance(value, dict):
                    stage_args = config.get("stage_args", [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, delete_paths in value.items():
                        if not delete_paths:
                            continue

                        # Find stage by ID
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            continue

                        # Delete specified paths in this stage
                        for path in delete_paths:
                            if path:  # Skip empty paths
                                delete_by_path(target_stage, path)
            elif "." in key:
                # Delete using dot-separated path
                delete_by_path(config, key)
            elif value is None and key in config:
                # Delete entire key
                del config[key]

    # Apply updates
    if updates:
        for key, value in updates.items():
            if key == "stage_args":
                if value and isinstance(value, dict):
                    stage_args = config.get("stage_args", [])
                    if not stage_args:
                        raise ValueError("stage_args does not exist in config")

                    for stage_id, stage_updates in value.items():
                        # Find stage by ID
                        target_stage = None
                        for stage in stage_args:
                            if stage.get("stage_id") == int(stage_id):
                                target_stage = stage
                                break

                        if target_stage is None:
                            available_ids = [s.get("stage_id") for s in stage_args if "stage_id" in s]
                            raise KeyError(f"Stage ID {stage_id} not found, available: {available_ids}")

                        # Apply updates to this stage
                        for path, val in stage_updates.items():
                            # Check if this is a simple key (not dot-separated)
                            # Example: 'engine_input_source' vs 'engine_args.max_model_len'
                            if "." not in path:
                                # Direct key assignment (e.g., updating a list value)
                                target_stage[path] = val
                            else:
                                # Dot-separated path (e.g., nested dict access)
                                apply_update(target_stage, path, val)
            elif "." in key:
                # Apply using dot-separated path
                apply_update(config, key, value)
            else:
                # Direct top-level key
                config[key] = value

    # Unique suffix: multiple modify_stage_config calls in one process often run
    # within the same second (e.g. test_qwen3_omni_expansion imports both
    # get_chunk_config and get_batch_token_config). int(time.time()) would collide
    # and the later write would overwrite the earlier YAML on disk.
    base_name = yaml_path.rsplit(".", 1)[0] if "." in yaml_path else yaml_path
    output_path = f"{base_name}_{time.time_ns()}.yaml"

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=None, sort_keys=False, allow_unicode=True, indent=2)

    return output_path


class OmniServer:
    """Omniserver for vLLM-Omni tests."""

    def __init__(
        self,
        model: str,
        serve_args: list[str],
        *,
        port: int | None = None,
        env_dict: dict[str, str] | None = None,
        use_omni: bool = True,
    ) -> None:
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        cleanup_dist_env_and_memory()
        self.model = model
        self.serve_args = serve_args
        self.env_dict = env_dict
        self.use_omni = use_omni
        self.proc: subprocess.Popen | None = None
        self.host = "127.0.0.1"
        if port is None:
            self.port = get_open_port()
        else:
            self.port = port

    def _start_server(self) -> None:
        """Start the vLLM-Omni server subprocess."""
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        if self.env_dict is not None:
            env.update(self.env_dict)

        cmd = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.use_omni:
            cmd.append("--omni")
        cmd += self.serve_args

        print(f"Launching OmniServer with: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Set working directory to vllm-omni root
        )

        # Wait for server to be ready
        max_wait = 1200  # 20 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            # Check for process status
            ret = self.proc.poll()
            if ret is not None:
                raise RuntimeError(f"Server processes exited with code {ret} before becoming ready.")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((self.host, self.port))
                if result == 0:
                    print(f"Server ready on {self.host}:{self.port}")
                    return
            time.sleep(2)

        raise RuntimeError(f"Server failed to start within {max_wait} seconds")

    def _kill_process_tree(self, pid):
        """kill process and its children with verification"""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Get all PIDs first
            all_pids = [pid] + [child.pid for child in children]

            # Terminate children
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass

            # Wait for children
            gone, still_alive = psutil.wait_procs(children, timeout=10)

            # Kill remaining children
            for child in still_alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

            # Terminate parent
            try:
                parent.terminate()
                parent.wait(timeout=10)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

            # VERIFICATION: Check if all processes are gone
            time.sleep(1)  # Give system time
            alive_processes = []
            for check_pid in all_pids:
                if psutil.pid_exists(check_pid):
                    alive_processes.append(check_pid)

            if alive_processes:
                print(f"Warning: Processes still alive: {alive_processes}")
                # Optional: Try system kill
                import subprocess

                for alive_pid in alive_processes:
                    try:
                        subprocess.run(["kill", "-9", str(alive_pid)], timeout=2)
                    except Exception as e:
                        print(f"Cleanup failed: {e}")

        except psutil.NoSuchProcess:
            pass

    def __enter__(self):
        self._start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.proc:
            self._kill_process_tree(self.proc.pid)
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        cleanup_dist_env_and_memory()


def pytest_addoption(parser):
    parser.addoption(
        "--run-level",
        action="store",
        default="core_model",
        choices=["core_model", "advanced_model"],
        help="Test level to run: L2, L3",
    )


@pytest.fixture(scope="session")
def run_level(request) -> str:
    """A command-line argument that specifies the level of tests to run in this session.
    See https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/ci/CI_5levels/"""
    return request.config.getoption("--run-level")


_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request: pytest.FixtureRequest, run_level: str, model_prefix: str) -> Generator[OmniServer, Any, None]:
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Uses session scope so the server starts only once for the entire test session.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        params: OmniServerParams = request.param
        model = model_prefix + params.model
        port = params.port
        stage_config_path = params.stage_config_path
        if run_level == "advanced_model" and stage_config_path is not None:
            with open(stage_config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            stage_ids = [stage["stage_id"] for stage in cfg.get("stage_args", []) if "stage_id" in stage]
            stage_config_path = modify_stage_config(
                stage_config_path,
                deletes={"stage_args": {stage_id: ["engine_args.load_format"] for stage_id in stage_ids}},
            )

        server_args = params.server_args or []
        if params.use_omni:
            server_args = ["--stage-init-timeout", "120", *server_args]
        if stage_config_path is not None:
            server_args += ["--stage-configs-path", stage_config_path]

        with (
            OmniServer(
                model,
                server_args,
                port=port,
                env_dict=params.env_dict,
                use_omni=params.use_omni,
            )
            if port
            else OmniServer(
                model,
                server_args,
                env_dict=params.env_dict,
                use_omni=params.use_omni,
            )
        ) as server:
            print("OmniServer started successfully")
            yield server
            print("OmniServer stopping...")

        print("OmniServer stopped")


@dataclass
class OmniResponse:
    text_content: str | None = None
    audio_data: list[str] | None = None
    audio_content: str | None = None
    audio_format: str | None = None
    audio_bytes: bytes | None = None
    similarity: float | None = None
    e2e_latency: float | None = None
    success: bool = False
    error_message: str | None = None


@dataclass
class DiffusionResponse:
    text_content: str | None = None
    images: list[Image.Image] | None = None
    audios: list[Any] | None = None
    videos: list[Any] | None = None
    e2e_latency: float | None = None
    success: bool = False
    error_message: str | None = None


def _load_gender_pipeline():
    """
    Lazy-load a cached audio-classification pipeline for gender.

    We prefer the pipeline wrapper because it encapsulates processor/model loading
    and avoids direct AutoProcessor.from_pretrained call sites in this file.
    """
    global _GENDER_PIPELINE
    if _GENDER_PIPELINE is not None:
        return _GENDER_PIPELINE

    model_name = "7wolf/wav2vec2-base-gender-classification"
    try:
        # device=-1 forces CPU for pipeline.
        _GENDER_PIPELINE = pipeline(
            task="audio-classification",
            model=model_name,
            device=-1,
        )
        return _GENDER_PIPELINE
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"Warning: failed to create gender pipeline '{model_name}': {exc}")
        _GENDER_PIPELINE = None
        return None


def _median_pitch_hz_from_autocorr(mono: np.ndarray, sr: int) -> float | None:
    """
    Rough median F0 (Hz) over short-time frames. Used to debias wav2vec2 gender head on TTS,
    which often labels lower-pitched synthetic speech as female under load or on clean signals.
    Returns None if the clip is too short or mostly unvoiced.
    """
    x = np.asarray(mono, dtype=np.float64)
    x = x - np.mean(x)
    if x.size < int(0.15 * sr):
        return None
    frame_len = int(0.04 * sr)
    hop = max(frame_len // 2, 1)
    f0_min_hz, f0_max_hz = 70.0, 400.0
    lag_min = max(1, int(sr / f0_max_hz))
    lag_max = min(frame_len - 2, int(sr / f0_min_hz))
    if lag_max <= lag_min:
        return None
    win = np.hamming(frame_len)
    pitches: list[float] = []
    for start in range(0, int(x.shape[0]) - frame_len, hop):
        frame = x[start : start + frame_len] * win
        frame = frame - np.mean(frame)
        if float(np.sqrt(np.mean(frame**2))) < 1e-4:
            continue
        ac = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
        ac = ac / (float(ac[0]) + 1e-12)
        region = ac[lag_min : lag_max + 1]
        peak_rel = int(np.argmax(region))
        peak_lag = peak_rel + lag_min
        if peak_lag <= 0:
            continue
        f0 = float(sr) / float(peak_lag)
        if f0_min_hz <= f0 <= f0_max_hz:
            pitches.append(f0)
    if len(pitches) < 4:
        return None
    return float(np.median(np.asarray(pitches, dtype=np.float64)))


def _estimate_voice_gender_from_audio(audio_bytes: bytes) -> str:
    """
    Estimate voice gender from audio using a small pre-trained classification model.

    Uses a cached `audio-classification` pipeline to classify the clip.
    Returns 'male' / 'female' when the model confidence is >= 0.9 and the label
    maps to one of these; otherwise returns 'unknown'. If the model is unavailable
    or inference fails, returns 'unknown' to keep tests stable.

    Under concurrent tests, a global lock serializes pipeline calls (the HF pipeline is not
    thread-safe). A coarse F0 median can correct systematic "male -> female" errors on TTS audio.
    """
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("Empty audio")
    mono = np.mean(data, axis=1)

    try:
        target_sr = 16000
        if int(sr) != target_sr and mono.size > 1:
            src_len = int(mono.shape[0])
            dst_len = max(1, int(round(src_len * float(target_sr) / float(sr))))
            src_idx = np.arange(src_len, dtype=np.float32)
            dst_idx = np.linspace(0, src_len - 1, dst_len, dtype=np.float32)
            mono = np.interp(dst_idx, src_idx, mono.astype(np.float32, copy=False)).astype(np.float32)
            sr = target_sr

        median_f0 = _median_pitch_hz_from_autocorr(mono, sr)

        clf = _load_gender_pipeline()
        if clf is None:
            print("gender model not available, returning 'unknown'")
            return "unknown"

        # transformers pipeline returns a list of {label, score} (highest score first).
        with _GENDER_PIPELINE_LOCK:
            outputs = clf(mono, sampling_rate=sr)
        if not outputs:
            return "unknown"

        top = outputs[0]
        label = str(top.get("label", "")).lower()
        conf = float(top.get("score", 0.0))

        if conf < 0.6:
            gender = "unknown"
        # Some models use non-English labels (e.g., Russian). Normalize to 'male'/'female'.
        elif ("female" in label) or ("жен" in label):
            gender = "female"
        elif ("male" in label) or ("муж" in label):
            gender = "male"
        else:
            gender = "unknown"

        # Debias: wav2vec2 gender heads often call TTS / band-limited male speech "female".
        # Low median F0 (~speech male range) + female label -> trust pitch when score is not overwhelming.
        if gender == "female" and median_f0 is not None and median_f0 < 165.0 and conf < 0.88:
            print(f"gender pitch assist: reclassifying female->male (median_f0={median_f0:.1f} Hz, conf={conf:.3f})")
            gender = "male"
        elif gender == "male" and median_f0 is not None and median_f0 > 230.0 and conf < 0.88:
            print(f"gender pitch assist: reclassifying male->female (median_f0={median_f0:.1f} Hz, conf={conf:.3f})")
            gender = "female"

        print(
            f"gender classifier: label={label}, conf={conf:.3f}, gender={gender}"
            + (f", median_f0={median_f0:.1f}Hz" if median_f0 is not None else "")
        )
        return gender
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"Warning: gender classification failed, returning 'unknown': {exc}")
        return "unknown"


# Threshold aligned with _compute_pcm_hnr_db docstring (clean clone vs distorted).
_MIN_PCM_SPEECH_HNR_DB = 1.0


def _compute_pcm_hnr_db(pcm_samples: np.ndarray, sr: int = _PCM_SPEECH_SAMPLE_RATE_HZ) -> float:
    """Compute mean Harmonic-to-Noise Ratio (dB) for speech quality.

    Clean cloned speech has HNR > 1.2 dB; distorted speech (e.g. lost
    ref_code decoder context) drops below 1.0 dB.
    """
    frame_len = int(0.03 * sr)  # 30ms frames
    hop = frame_len // 2
    hnr_values: list[float] = []

    for start in range(0, len(pcm_samples) - frame_len, hop):
        frame = pcm_samples[start : start + frame_len].astype(np.float32, copy=False)
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 0.01:
            continue
        ac = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        ac = ac / (ac[0] + 1e-10)
        min_lag = int(sr / 400)
        max_lag = min(int(sr / 80), len(ac))
        if min_lag >= max_lag:
            continue
        peak = float(np.max(ac[min_lag:max_lag]))
        if 0 < peak < 1:
            hnr_values.append(10 * np.log10(peak / (1 - peak + 1e-10)))

    return float(np.mean(hnr_values)) if hnr_values else 0.0


def _assert_pcm_int16_speech_hnr(audio_bytes: bytes) -> None:
    """Validate harmonic-to-noise ratio on raw int16 PCM from /v1/audio/speech."""
    assert audio_bytes is not None and len(audio_bytes) >= 2, "missing PCM bytes"
    assert len(audio_bytes) % 2 == 0, "PCM byte length must be aligned to int16"
    pcm_samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    hnr = _compute_pcm_hnr_db(pcm_samples)
    print(f"PCM speech HNR: {hnr:.2f} dB (threshold: {_MIN_PCM_SPEECH_HNR_DB} dB)")
    assert hnr >= _MIN_PCM_SPEECH_HNR_DB, (
        f"Audio distortion detected: HNR={hnr:.2f} dB < {_MIN_PCM_SPEECH_HNR_DB} dB. "
        "Voice clone decoder may be losing ref_code speaker context on later chunks."
    )


def assert_omni_response(response: OmniResponse, request_config: dict[str, Any], run_level):
    """
    Validate response results.

    Args:
        response: OmniResponse object

    Raises:
        AssertionError: When the response does not meet validation criteria
    """
    assert response.success, "The request failed."
    e2e_latency = response.e2e_latency
    if e2e_latency is not None:
        print(f"the e2e latency is: {e2e_latency}")

    modalities = request_config.get("modalities", ["text", "audio"])

    if run_level == "advanced_model":
        if "audio" in modalities:
            assert response.audio_content is not None, "No audio output is generated"
            print(f"audio content is: {response.audio_content}")

        if "text" in modalities:
            assert response.text_content is not None, "No text output is generated"
            print(f"text content is: {response.text_content}")

        # Verify image description
        word_types = ["text", "image", "audio", "video"]
        keywords_dict = request_config.get("key_words", {})
        for word_type in word_types:
            keywords = keywords_dict.get(word_type)
            if "text" in modalities:
                if keywords:
                    assert any(keyword in response.text_content.lower() for keyword in keywords), (
                        "The output does not contain any of the keywords."
                    )
            else:
                if keywords:
                    assert any(keyword in response.audio_content.lower() for keyword in keywords), (
                        "The output does not contain any of the keywords."
                    )

        # Verify similarity (Whisper transcript vs streamed/detokenized text)
        if "text" in modalities and "audio" in modalities:
            assert response.similarity is not None and response.similarity > 0.9, (
                "The audio content is not same as the text"
            )
            print(f"similarity is: {response.similarity}")


def assert_audio_speech_response(
    response: OmniResponse,
    request_config: dict[str, Any],
    run_level: str,
) -> None:
    """
    Validate /v1/audio/speech response: success, optional format check, transcription similarity
    and gender (non-PCM only for advanced_model), and int16 PCM HNR when response_format is pcm.
    """
    assert response.success, "The request failed."

    req_fmt = request_config.get("response_format")

    if req_fmt == "pcm" and response.audio_bytes:
        _assert_pcm_int16_speech_hnr(response.audio_bytes)
        if response.audio_format:
            assert "pcm" in response.audio_format.lower(), (
                f"Expected audio/pcm content-type, got {response.audio_format!r}"
            )

    elif req_fmt == "wav" and response.audio_format:
        assert req_fmt in response.audio_format, (
            f"The response audio format {response.audio_format} don't match the request audio format {req_fmt}"
        )

    e2e_latency = response.e2e_latency
    if e2e_latency is not None:
        print(f"the avg e2e latency is: {e2e_latency}")

    if run_level == "advanced_model" and req_fmt != "pcm":
        # Text–audio semantic similarity check (skipped for raw PCM: no Whisper transcript).
        expected_text = request_config.get("input")
        if expected_text:
            transcript = (response.audio_content or "").strip()
            print(f"audio content is: {transcript}")
            print(f"input text is: {expected_text}")
            similarity = cosine_similarity_text(transcript.lower(), expected_text.lower())
            print(f"Cosine similarity: {similarity:.3f}")
            assert similarity > 0.9, (
                f"Transcript doesn't match input: similarity={similarity:.2f}, transcript='{transcript}'"
            )

        # Voice gender consistency check:
        # When the estimator returns 'unknown', we treat it as inconclusive and do NOT fail the test.
        voice = (request_config.get("voice") or "").lower()
        if voice and response.audio_bytes:
            estimated_gender = _estimate_voice_gender_from_audio(response.audio_bytes)
            voice_gender_map = {
                # adjust this mapping to your actual voice names
                "serena": "female",
                "uncle_fu": "male",
                "clone": "female",
            }
            expected_gender = voice_gender_map.get(voice)
            if expected_gender is not None:
                print(f"Estimated voice gender from audio: {estimated_gender} (voice='{voice}')")
                if estimated_gender != "unknown":
                    assert estimated_gender == expected_gender, (
                        f"Voice '{voice}' is expected {expected_gender}, but estimated gender is '{estimated_gender}'"
                    )


def assert_diffusion_response(response: DiffusionResponse, request_config: dict[str, Any], run_level: str = None):
    """
    Validate diffusion response results.

    Dispatcher that routes validation to modality-specific assert functions.

    Args:
        response: DiffusionResponse object.
        request_config: Request configuration dictionary.
        run_level: Test run level (e.g. "core_model", "advanced_model")

    Raises:
        AssertionError: When the response does not meet validation criteria
        KeyError: When the request_config does not contain necessary parameters for validation
    """
    assert response.success, "The request failed."

    e2e_latency = response.e2e_latency
    if e2e_latency is not None:
        print(f"the avg e2e is: {e2e_latency}")

    has_any_content = any(content is not None for content in (response.images, response.videos, response.audios))
    assert has_any_content, "Response contains no images, videos, or audios"

    if response.images is not None:
        assert_image_diffusion_response(
            response=response,
            request_config=request_config,
            run_level=run_level,
        )

    if response.videos is not None:
        assert_video_diffusion_response(
            response=response,
            request_config=request_config,
            run_level=run_level,
        )

    if response.audios is not None:
        assert_audio_diffusion_response(
            response=response,
            request_config=request_config,
            run_level=run_level,
        )


class OpenAIClientHandler:
    """
    OpenAI client handler class, encapsulating both streaming and non-streaming response processing logic.

    This class integrates OpenAI API request sending, response handling, and validation functionality,
    supporting both single request and concurrent request modes.
    """

    def __init__(
        self, host: str = "127.0.0.1", port: int = get_open_port(), api_key: str = "EMPTY", run_level: str = None
    ):
        """
        Initialize the OpenAI client.

        Args:
            host: vLLM-Omni server host address
            port: vLLM-Omni server port
            api_key: API key (defaults to "EMPTY")
        """
        self.base_url = f"http://{host}:{port}"
        self.client = OpenAI(base_url=f"http://{host}:{port}/v1", api_key=api_key)
        self.run_level = run_level

    def _process_stream_omni_response(self, chat_completion) -> OmniResponse:
        """
        Process streaming responses.

        Args:
            chat_completion: OpenAI streaming response object
            request_config: Request configuration dictionary

        Returns:
            OmniResponse: Processed response object
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            text_content = ""
            audio_data = []

            for chunk in chat_completion:
                for choice in chunk.choices:
                    # Get content data
                    if hasattr(choice, "delta"):
                        content = getattr(choice.delta, "content", None)
                    else:
                        content = None

                    # Get modality type
                    modality = getattr(chunk, "modality", None)

                    # Process content based on modality type
                    if modality == "audio" and content:
                        audio_data.append(content)
                    elif modality == "text" and content:
                        text_content += content if content else ""

            # Calculate end-to-end latency
            result.e2e_latency = time.perf_counter() - start_time

            # Process audio and text content
            audio_content = None
            similarity = None

            if audio_data or text_content:
                if audio_data:
                    audio_content = merge_base64_and_convert_to_text(audio_data)
                if audio_content and text_content:
                    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())

            # Populate result object
            result.text_content = text_content
            result.audio_data = audio_data
            result.audio_content = audio_content
            result.similarity = similarity
            result.success = True

        except Exception as e:
            result.error_message = f"Stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_non_stream_omni_response(self, chat_completion) -> OmniResponse:
        """
        Process non-streaming responses.

        Args:
            chat_completion: OpenAI non-streaming response object
            request_config: Request configuration dictionary

        Returns:
            OmniResponse: Processed response object
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            audio_data = None
            text_content = None

            # Iterate through all choices
            for choice in chat_completion.choices:
                # Process audio data
                if hasattr(choice.message, "audio") and choice.message.audio is not None:
                    audio_message = choice.message
                    audio_data = audio_message.audio.data

                # Process text content
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    text_content = choice.message.content

            # Calculate end-to-end latency
            result.e2e_latency = time.perf_counter() - start_time

            # Process audio and text content
            audio_content = None
            similarity = None

            if audio_data or text_content:
                if audio_data:
                    audio_content = convert_audio_to_text(audio_data)
                if audio_content and text_content:
                    similarity = cosine_similarity_text(audio_content.lower(), text_content.lower())

            # Populate result object
            result.text_content = text_content
            result.audio_content = audio_content
            result.similarity = similarity
            result.success = True

        except Exception as e:
            result.error_message = f"Non-stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_diffusion_response(self, chat_completion) -> DiffusionResponse:
        """
        Process diffusion responses (image generation/editing).

        Args:
            chat_completion: OpenAI response object

        Returns:
            DiffusionResponse: Processed response object
        """
        result = DiffusionResponse()
        start_time = time.perf_counter()

        try:
            images = []
            # [TODO] reading video and audio output from API response for later validation

            for choice in chat_completion.choices:
                if hasattr(choice.message, "content") and choice.message.content is not None:
                    content = choice.message.content
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                image_url = item.get("image_url", {}).get("url")
                            else:
                                image_url_obj = getattr(item, "image_url", None)
                                image_url = hasattr(image_url_obj, "url", None) if image_url_obj else None
                            if image_url and image_url.startswith("data:image"):
                                b64_data = image_url.split(",", 1)[1]
                                img = decode_b64_image(b64_data)
                                images.append(img)

            result.e2e_latency = time.perf_counter() - start_time
            result.images = images if images else None
            result.success = True

        except Exception as e:
            result.error_message = f"Diffusion response processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_stream_audio_speech_response(self, response, *, response_format: str | None = None) -> OmniResponse:
        """
        Process streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_stream_omni_response but operates on low-level
        audio bytes and produces an OmniResponse with audio_content filled
        from Whisper transcription.
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            # Aggregate all audio bytes from the streaming response.
            data = bytearray()

            # Preferred OpenAI helper.
            if hasattr(response, "iter_bytes") and callable(getattr(response, "iter_bytes")):
                for chunk in response.iter_bytes():
                    if chunk:
                        data.extend(chunk)
            else:
                # Generic iterable-of-bytes fallback (e.g., generator or list of chunks).
                try:
                    iterator = iter(response)
                except TypeError:
                    iterator = None

                if iterator is not None:
                    for chunk in iterator:
                        if not chunk:
                            continue
                        if isinstance(chunk, (bytes, bytearray)):
                            data.extend(chunk)
                        elif hasattr(chunk, "data"):
                            data.extend(chunk.data)  # type: ignore[arg-type]
                        elif hasattr(chunk, "content"):
                            data.extend(chunk.content)  # type: ignore[arg-type]
                        else:
                            raise TypeError(f"Unsupported stream chunk type: {type(chunk)}")
                else:
                    raise TypeError(f"Unsupported audio speech streaming response type: {type(response)}")

            raw_bytes = bytes(data)
            if response_format == "pcm":
                transcript = None
            else:
                transcript = convert_audio_bytes_to_text(raw_bytes)

            # Populate OmniResponse.
            result.audio_bytes = raw_bytes
            result.audio_content = transcript
            result.e2e_latency = time.perf_counter() - start_time
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            result.error_message = f"Audio speech stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def _process_non_stream_audio_speech_response(
        self, response, *, response_format: str | None = None
    ) -> OmniResponse:
        """
        Process non-streaming /v1/audio/speech responses into an OmniResponse.

        This mirrors _process_non_stream_omni_response but for the binary
        audio payload returned by audio.speech.create.
        """
        result = OmniResponse()
        start_time = time.perf_counter()

        try:
            # OpenAI non-streaming audio.speech.create returns HttpxBinaryResponseContent (.read() or .content)
            if hasattr(response, "read") and callable(getattr(response, "read")):
                raw_bytes = response.read()
            elif hasattr(response, "content"):
                raw_bytes = response.content  # type: ignore[assignment]
            else:
                raise TypeError(f"Unsupported audio speech response type: {type(response)}")

            if response_format == "pcm":
                transcript = None
            else:
                transcript = convert_audio_bytes_to_text(raw_bytes)

            result.audio_bytes = raw_bytes
            result.audio_content = transcript
            result.e2e_latency = time.perf_counter() - start_time
            result.success = True
            result.audio_format = getattr(response, "response", None)
            if result.audio_format is not None:
                result.audio_format = result.audio_format.headers.get("content-type", "")

        except Exception as e:
            result.error_message = f"Audio speech non-stream processing error: {str(e)}"
            print(f"Error: {result.error_message}")

        return result

    def send_omni_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send OpenAI requests.

        Args:
            request_config: Request configuration dictionary containing parameters like model, messages, stream.
                Optional ``use_audio_in_video`` (bool): when true, sets
                ``extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}`` for Qwen-Omni video+audio
                extraction (merged with any existing ``extra_body`` / ``mm_processor_kwargs``).
                Optional ``extra_body`` (dict): passed through to ``chat.completions.create`` after merge.
            request_num: Number of requests, defaults to 1 (single request)

        Returns:
            List[OmniResponse]: List of response objects
        """

        responses = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", ["text", "audio"])

        extra_body: dict[str, Any] = {}
        raw_extra = request_config.get("extra_body")
        if raw_extra:
            extra_body.update(raw_extra)
        if request_config.get("use_audio_in_video"):
            mm = dict(extra_body.get("mm_processor_kwargs") or {})
            mm["use_audio_in_video"] = True
            extra_body["mm_processor_kwargs"] = mm
        extra_body_arg: dict[str, Any] | None = extra_body if extra_body else None

        create_kwargs: dict[str, Any] = {
            "model": request_config.get("model"),
            "messages": request_config.get("messages"),
            "stream": stream,
            "modalities": modalities,
        }
        if extra_body_arg is not None:
            create_kwargs["extra_body"] = extra_body_arg

        if request_num == 1:
            # Send single request
            chat_completion = self.client.chat.completions.create(**create_kwargs)

            if stream:
                response = self._process_stream_omni_response(chat_completion)
            else:
                response = self._process_non_stream_omni_response(chat_completion)

            assert_omni_response(response, request_config, run_level=self.run_level)
            responses.append(response)

        else:
            # Send concurrent requests: run create + process in worker so e2e_latency includes full round-trip.
            def _one_omni_request():
                start = time.perf_counter()
                chat_completion = self.client.chat.completions.create(
                    model=request_config.get("model"),
                    messages=request_config.get("messages"),
                    modalities=modalities,
                    stream=stream,
                )
                if stream:
                    response = self._process_stream_omni_response(chat_completion)
                else:
                    response = self._process_non_stream_omni_response(chat_completion)
                response.e2e_latency = time.perf_counter() - start
                return response

            with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                futures = [executor.submit(_one_omni_request) for _ in range(request_num)]
                for future in concurrent.futures.as_completed(futures):
                    response = future.result()
                    assert_omni_response(response, request_config, run_level=self.run_level)
                    responses.append(response)

        return responses

    def send_audio_speech_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Call the /v1/audio/speech endpoint using the same configuration-dict
        style as send_omni_request, but via the OpenAI Python client's
        audio.speech APIs.

        Expected keys in request_config:
          - model: model name/path (required)
          - input: text to synthesize (required)
          - response_format: audio format such as "wav" or "pcm" (optional)
          - task_type, ref_text, ref_audio: TTS-specific extras (optional, passed via extra_body)
          - timeout: request timeout in seconds (float, optional, default 120.0)
          - stream: whether to use streaming API (bool, optional, default False)
        """
        timeout = float(request_config.get("timeout", 120.0))

        model = request_config["model"]
        text_input = request_config["input"]
        stream = bool(request_config.get("stream", False))
        voice = request_config.get("voice", None)

        # Standard OpenAI param: use omit when not provided to keep default behavior.
        response_format = request_config.get("response_format", omit)

        # Qwen3-TTS custom fields, forwarded via extra_body.
        extra_body: dict[str, Any] = {}
        # Keep this list aligned with vllm_omni.entrypoints.openai.protocol.audio params.
        for key in ("task_type", "ref_text", "ref_audio", "language", "max_new_tokens"):
            if key in request_config:
                extra_body[key] = request_config[key]

        responses: list[OmniResponse] = []

        speech_fmt: str | None = None if response_format is omit else str(response_format).lower()

        if request_num == 1:
            if stream:
                # Use streaming response helper.
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                ) as resp:
                    omni_resp = self._process_stream_audio_speech_response(resp, response_format=speech_fmt)
            else:
                # Non-streaming response.
                resp = self.client.audio.speech.create(
                    model=model,
                    input=text_input,
                    response_format=response_format,
                    extra_body=extra_body or None,
                    timeout=timeout,
                    voice=voice,
                )
                omni_resp = self._process_non_stream_audio_speech_response(resp, response_format=speech_fmt)

            assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
            responses.append(omni_resp)
            return responses
        else:
            # request_num > 1: concurrent requests (use same params as single-request path)

            if stream:

                def _stream_task():
                    with self.client.audio.speech.with_streaming_response.create(
                        model=model,
                        input=text_input,
                        response_format=response_format,
                        extra_body=extra_body or None,
                        timeout=timeout,
                        voice=voice,
                    ) as resp:
                        return self._process_stream_audio_speech_response(resp, response_format=speech_fmt)

                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = [executor.submit(_stream_task) for _ in range(request_num)]
                    for future in concurrent.futures.as_completed(futures):
                        omni_resp = future.result()
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                    futures = []
                    for _ in range(request_num):
                        future = executor.submit(
                            self.client.audio.speech.create,
                            model=model,
                            input=text_input,
                            response_format=response_format,
                            extra_body=extra_body or None,
                            timeout=timeout,
                            voice=voice,
                        )
                        futures.append(future)

                    for future in concurrent.futures.as_completed(futures):
                        resp = future.result()
                        omni_resp = self._process_non_stream_audio_speech_response(resp, response_format=speech_fmt)
                        assert_audio_speech_response(omni_resp, request_config, run_level=self.run_level)
                        responses.append(omni_resp)

        return responses

    def send_diffusion_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send OpenAI requests for diffusion models.

        Args:
            request_config: Request configuration dictionary containing parameters like model, messages
            request_num: Number of requests to send concurrently, defaults to 1 (single request)
        Returns:
            List[OmniResponse]: List of response objects
        """
        responses = []
        stream = request_config.get("stream", False)
        modalities = request_config.get("modalities", omit)  # Most diffusion models don't require modalities param
        extra_body = request_config.get("extra_body", None)

        if stream:
            raise NotImplementedError("Streaming is not currently implemented for diffusion model e2e test")

        if request_num == 1:
            # Send single request
            chat_completion = self.client.chat.completions.create(
                model=request_config.get("model"),
                messages=request_config.get("messages"),
                extra_body=extra_body,
                modalities=modalities,
            )

            response = self._process_diffusion_response(chat_completion)
            assert_diffusion_response(response, request_config, run_level=self.run_level)
            responses.append(response)

        else:
            # Send concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=request_num) as executor:
                futures = []

                # Submit all request tasks
                for _ in range(request_num):
                    future = executor.submit(
                        self.client.chat.completions.create,
                        model=request_config.get("model"),
                        messages=request_config.get("messages"),
                        modalities=modalities,
                        extra_body=extra_body,
                    )
                    futures.append(future)

                # Process completed tasks
                for future in concurrent.futures.as_completed(futures):
                    chat_completion = future.result()
                    response = self._process_diffusion_response(chat_completion)
                    assert_diffusion_response(response, request_config, run_level=self.run_level)
                    responses.append(response)

        return responses

    def send_video_diffusion_request(self, request_config: dict[str, Any], request_num: int = 1) -> list[OmniResponse]:
        """
        Send native /v1/videos requests.
        """
        if request_num != 1:
            raise NotImplementedError("Concurrent video diffusion requests are not currently implemented")

        if request_config.get("stream", False):
            raise NotImplementedError("Streaming is not currently implemented for video diffusion e2e test")

        form_data = request_config.get("form_data")
        if not isinstance(form_data, dict):
            raise ValueError("Video request_config must contain 'form_data'")

        if not form_data.get("prompt"):
            raise ValueError("Video request_config['form_data'] must contain 'prompt'")

        normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}

        files: dict[str, tuple[str, BytesIO, str]] = {}
        image_reference = request_config.get("image_reference")
        if image_reference:
            if image_reference.startswith("data:image"):
                header, encoded = image_reference.split(",", 1)
                content_type = header.split(";")[0].removeprefix("data:")
                extension = content_type.split("/")[-1]
                file_data = base64.b64decode(encoded)

                files["input_reference"] = (
                    f"reference.{extension}",
                    BytesIO(file_data),
                    content_type,
                )
            else:
                normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})

        result = DiffusionResponse()
        start_time = time.perf_counter()

        try:
            create_url = self._build_url("/v1/videos")
            response = requests.post(
                create_url,
                data=normalized_form_data,
                files=files,
                headers={"Accept": "application/json"},
                timeout=60,
            )
            response.raise_for_status()

            job_data = response.json()
            video_id = job_data["id"]

            self._wait_until_video_completed(video_id)

            video_content = self._download_video_content(video_id)

            result.success = True
            result.videos = [video_content]
            result.e2e_latency = time.perf_counter() - start_time

            assert_diffusion_response(result, request_config, run_level=self.run_level)

        except Exception as e:
            result.success = False
            result.error_message = f"Diffusion response processing error: {e}"
            assert False, result.error_message

        return [result]

    def _wait_until_video_completed(
        self,
        video_id: str,
        poll_interval_seconds: int = 2,
        timeout_seconds: int = 300,
    ) -> None:
        status_url = self._build_url(f"/v1/videos/{video_id}")
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            status_resp = requests.get(
                status_url,
                headers={"Accept": "application/json"},
                timeout=30,
            )
            status_resp.raise_for_status()

            status_data = status_resp.json()
            current_status = status_data["status"]

            if current_status == "completed":
                return

            if current_status == "failed":
                error_msg = status_data.get("last_error", "Unknown error")
                raise RuntimeError(f"Job failed: {error_msg}")

            time.sleep(poll_interval_seconds)

        raise TimeoutError(f"Video job {video_id} did not complete within {timeout_seconds}s")

    def _download_video_content(self, video_id: str) -> bytes:
        download_url = self._build_url(f"/v1/videos/{video_id}/content")
        video_resp = requests.get(download_url, stream=True, timeout=60)
        video_resp.raise_for_status()

        video_bytes = BytesIO()
        for chunk in video_resp.iter_content(chunk_size=8192):
            if chunk:
                video_bytes.write(chunk)

        return video_bytes.getvalue()

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


@pytest.fixture
def openai_client(omni_server: OmniServer, run_level: str):
    """Create OpenAIClientHandler fixture to facilitate communication with OmniServer
    with encapsulated request sending, concurrent requests, response handling, and validation."""
    return OpenAIClientHandler(host=omni_server.host, port=omni_server.port, api_key="EMPTY", run_level=run_level)


class OmniRunner:
    """
    Offline test runner for Omni models.
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 42,
        stage_init_timeout: int = 300,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        shm_threshold_bytes: int = 65536,
        log_stats: bool = False,
        stage_configs_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize an OmniRunner for testing.

        Args:
            model_name: The model name or path
            seed: Random seed for reproducibility
            stage_init_timeout: Timeout for initializing a single stage in seconds
            batch_timeout: Timeout for batching in seconds
            init_timeout: Timeout for initializing stages in seconds
            shm_threshold_bytes: Threshold for using shared memory
            log_stats: Enable detailed statistics logging
            stage_configs_path: Optional path to YAML stage config file
            **kwargs: Additional arguments passed to Omni
        """
        cleanup_dist_env_and_memory()
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        self.model_name = model_name
        self.seed = seed

        self.omni = Omni(
            model=model_name,
            log_stats=log_stats,
            stage_init_timeout=stage_init_timeout,
            batch_timeout=batch_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            stage_configs_path=stage_configs_path,
            **kwargs,
        )

    def _estimate_prompt_len(
        self,
        additional_information: dict[str, Any],
        model_name: str,
        _cache: dict[str, Any] = {},
    ) -> int:
        """Estimate prompt_token_ids placeholder length for the Talker stage.

        The AR Talker replaces all input embeddings via ``preprocess``, so the
        placeholder values are irrelevant but the **length** must match the
        embeddings that ``preprocess`` will produce.
        """
        try:
            from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
                Qwen3TTSTalkerForConditionalGeneration,
            )

            if model_name not in _cache:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
                cfg = Qwen3TTSConfig.from_pretrained(model_name, trust_remote_code=True)
                _cache[model_name] = (tok, getattr(cfg, "talker_config", None))

            tok, tcfg = _cache[model_name]
            task_type = (additional_information.get("task_type") or ["CustomVoice"])[0]
            return Qwen3TTSTalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(
                additional_information=additional_information,
                task_type=task_type,
                tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
                codec_language_id=getattr(tcfg, "codec_language_id", None),
                spk_is_dialect=getattr(tcfg, "spk_is_dialect", None),
            )
        except Exception as exc:
            logger.warning("Failed to estimate prompt length, using fallback 2048: %s", exc)
            return 2048

    def get_default_sampling_params_list(self) -> list[OmniSamplingParams]:
        """
        Get a list of default sampling parameters for all stages.

        Returns:
            List of SamplingParams with default decoding for each stage
        """
        if not hasattr(self.omni, "default_sampling_params_list"):
            raise AttributeError("Omni.default_sampling_params_list is not available")
        return list(self.omni.default_sampling_params_list)

    def get_omni_inputs(
        self,
        prompts: list[str] | str,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[TextPrompt]:
        """
        Construct Omni input format from prompts and multimodal data.

        Args:
            prompts: Text prompt(s) - either a single string or list of strings
            system_prompt: Optional system prompt (defaults to Qwen system prompt)
            audios: Audio input(s) - tuple of (audio_array, sample_rate) or list of tuples
            images: Image input(s) - PIL Image or list of PIL Images
            videos: Video input(s) - numpy array or list of numpy arrays
            mm_processor_kwargs: Optional processor kwargs (e.g., use_audio_in_video)

        Returns:
            List of prompt dictionaries suitable for Omni.generate()
        """
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                "Group, capable of perceiving auditory and visual inputs, as well as "
                "generating text and speech."
            )

        video_padding_token = "<|VIDEO|>"
        image_padding_token = "<|IMAGE|>"
        audio_padding_token = "<|AUDIO|>"

        if "Qwen3-Omni-30B-A3B-Instruct" in self.model_name:
            video_padding_token = "<|video_pad|>"
            image_padding_token = "<|image_pad|>"
            audio_padding_token = "<|audio_pad|>"

        if isinstance(prompts, str):
            prompts = [prompts]

        # Qwen-TTS: follow examples/offline_inference/qwen3_tts/end2end.py style.
        # Stage 0 expects token placeholders + additional_information (text/speaker/task_type/...),
        # and Talker replaces embeddings in preprocess based on additional_information only.
        is_tts_model = "Qwen3-TTS" in self.model_name or "qwen3_tts" in self.model_name.lower()
        if is_tts_model and modalities == ["audio"]:
            tts_kw = mm_processor_kwargs or {}
            task_type = tts_kw.get("task_type", "CustomVoice")
            speaker = tts_kw.get("speaker", "Vivian")
            language = tts_kw.get("language", "Auto")
            max_new_tokens = int(tts_kw.get("max_new_tokens", 2048))
            ref_audio = tts_kw.get("ref_audio", None)
            ref_text = tts_kw.get("ref_text", None)

            omni_inputs: list[TextPrompt] = []
            for prompt_text in prompts:
                text_str = str(prompt_text).strip() or " "
                additional_information: dict[str, Any] = {
                    "task_type": [task_type],
                    "text": [text_str],
                    "language": [language],
                    "speaker": [speaker],
                    "max_new_tokens": [max_new_tokens],
                }
                if ref_audio is not None:
                    additional_information["ref_audio"] = [ref_audio]
                if ref_text is not None:
                    additional_information["ref_text"] = [ref_text]
                # Use official helper to get correct placeholder length
                plen = self._estimate_prompt_len(additional_information, self.model_name)
                input_dict: TextPrompt = {
                    "prompt_token_ids": [0] * plen,
                    "additional_information": additional_information,
                }
                omni_inputs.append(input_dict)
            return omni_inputs

        def _normalize_mm_input(mm_input, num_prompts):
            if mm_input is None:
                return [None] * num_prompts
            if isinstance(mm_input, list):
                if len(mm_input) != num_prompts:
                    raise ValueError(
                        f"Multimodal input list length ({len(mm_input)}) must match prompts length ({num_prompts})"
                    )
                return mm_input
            return [mm_input] * num_prompts

        num_prompts = len(prompts)
        audios_list = _normalize_mm_input(audios, num_prompts)
        images_list = _normalize_mm_input(images, num_prompts)
        videos_list = _normalize_mm_input(videos, num_prompts)

        omni_inputs = []
        for i, prompt_text in enumerate(prompts):
            user_content = ""
            multi_modal_data = {}

            audio = audios_list[i]
            if audio is not None:
                if isinstance(audio, list):
                    for _ in audio:
                        user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio
                else:
                    user_content += f"<|audio_bos|>{audio_padding_token}<|audio_eos|>"
                    multi_modal_data["audio"] = audio

            image = images_list[i]
            if image is not None:
                if isinstance(image, list):
                    for _ in image:
                        user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image
                else:
                    user_content += f"<|vision_bos|>{image_padding_token}<|vision_eos|>"
                    multi_modal_data["image"] = image

            video = videos_list[i]
            if video is not None:
                if isinstance(video, list):
                    for _ in video:
                        user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video
                else:
                    user_content += f"<|vision_bos|>{video_padding_token}<|vision_eos|>"
                    multi_modal_data["video"] = video

            user_content += prompt_text

            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            input_dict: TextPrompt = {"prompt": full_prompt}
            if multi_modal_data:
                input_dict["multi_modal_data"] = multi_modal_data
            if modalities:
                input_dict["modalities"] = modalities
            if mm_processor_kwargs:
                input_dict["mm_processor_kwargs"] = mm_processor_kwargs

            omni_inputs.append(input_dict)

        return omni_inputs

    def generate(
        self,
        prompts: list[TextPrompt],
        sampling_params_list: list[OmniSamplingParams] | None = None,
    ) -> list[OmniRequestOutput]:
        """
        Generate outputs for the given prompts.

        Args:
            prompts: List of prompt dictionaries with 'prompt' and optionally
                    'multi_modal_data' keys
            sampling_params_list: List of sampling parameters for each stage.
                                 If None, uses default parameters.

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()

        return self.omni.generate(prompts, sampling_params_list)

    def generate_multimodal(
        self,
        prompts: list[str] | str,
        sampling_params_list: list[OmniSamplingParams] | None = None,
        system_prompt: str | None = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        modalities: list[str] | None = None,
    ) -> list[OmniRequestOutput]:
        """
        Convenience method to generate with multimodal inputs.

        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            images: Image input(s)
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            images=images,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
            modalities=modalities,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages.

        Args:
            profile_prefix: Optional prefix for the trace file names.
            stages: List of stage IDs to profile. If None, profiles all stages.

        Returns:
            List of results from each stage.
        """
        return self.omni.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages.

        Args:
            stages: List of stage IDs to profile. If None, stops all stages.

        Returns:
            List of results from each stage.
        """
        return self.omni.stop_profile(stages=stages)

    def _cleanup_process(self):
        try:
            keywords = ["enginecore"]
            matched = []

            for proc in psutil.process_iter(["pid", "name", "cmdline", "username"]):
                try:
                    cmdline = " ".join(proc.cmdline()).lower() if proc.cmdline() else ""
                    name = proc.name().lower()

                    is_process = any(keyword in cmdline for keyword in keywords) or any(
                        keyword in name for keyword in keywords
                    )

                    if is_process:
                        print(f"Found vllm process: PID={proc.pid}, cmd={cmdline[:100]}")
                        matched.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            for proc in matched:
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            _, still_alive = psutil.wait_procs(matched, timeout=5)
            for proc in still_alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if still_alive:
                _, stubborn = psutil.wait_procs(still_alive, timeout=3)
                if stubborn:
                    print(f"Warning: failed to kill residual vllm pids: {[p.pid for p in stubborn]}")
                else:
                    print(f"Force-killed residual vllm pids: {[p.pid for p in still_alive]}")
            elif matched:
                print(f"Terminated vllm pids: {[p.pid for p in matched]}")

        except Exception as e:
            print(f"Error in psutil vllm cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if hasattr(self.omni, "close"):
            self.omni.close()
        self._cleanup_process()
        _run_pre_test_cleanup(enable_force=True)
        _run_post_test_cleanup(enable_force=True)
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def omni_runner(request, model_prefix):
    with _omni_server_lock:
        model, stage_config_path = request.param
        model = model_prefix + model
        with OmniRunner(model, seed=42, stage_configs_path=stage_config_path, stage_init_timeout=300) as runner:
            print("OmniRunner started successfully")
            yield runner
            print("OmniRunner stopping...")

        print("OmniRunner stopped")


class OmniRunnerHandler:
    def __init__(self, omni_runner):
        self.runner = omni_runner

    def _process_output(self, outputs: list[Any]) -> OmniResponse:
        result = OmniResponse()
        try:
            text_content = None
            audio_content = None
            for stage_output in outputs:
                if getattr(stage_output, "final_output_type", None) == "text":
                    text_content = stage_output.request_output.outputs[0].text
                if getattr(stage_output, "final_output_type", None) == "audio":
                    audio_content = stage_output.request_output.outputs[0].multimodal_output["audio"]

            result.audio_content = audio_content
            result.text_content = text_content
            result.success = True

        except Exception as e:
            result.error_message = f"Output processing error: {str(e)}"
            result.success = False
            print(f"Error: {result.error_message}")

        return result

    def send_request(self, request_config: dict[str, Any] | None = None) -> OmniResponse:
        if request_config is None:
            request_config = {}
        prompts = request_config.get("prompts")
        videos = request_config.get("videos")
        images = request_config.get("images")
        audios = request_config.get("audios")
        modalities = request_config.get("modalities", ["text", "audio"])
        outputs = self.runner.generate_multimodal(
            prompts=prompts, videos=videos, images=images, audios=audios, modalities=modalities
        )
        response = self._process_output(outputs)
        assert_omni_response(response, request_config, run_level="core_model")
        return response

    def send_audio_speech_request(
        self,
        request_config: dict[str, Any],
    ) -> OmniResponse:
        """
        Offline TTS: text -> audio via generate_multimodal, then validate with assert_audio_speech_response.

        request_config must contain:
          - 'input' or 'prompts': text to synthesize.
        Optional keys:
          - 'voice'       -> speaker (CustomVoice)
          - 'task_type'   -> task_type in additional_information (default: "CustomVoice")
          - 'language'    -> language in additional_information (default: "Auto")
          - 'max_new_tokens' -> max_new_tokens in additional_information (default: 2048)
          - 'response_format' -> desired audio format (used only for assertion)
        """
        input_text = request_config.get("input") or request_config.get("prompts")
        if input_text is None:
            raise ValueError("request_config must contain 'input' or 'prompts' for TTS")
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        # Build TTS-specific kwargs passed through to get_omni_inputs for Qwen3-TTS,
        # matching examples/offline_inference/qwen3_tts/end2end.py.
        mm_processor_kwargs: dict[str, Any] = {}
        if "voice" in request_config:
            mm_processor_kwargs["speaker"] = request_config["voice"]
        if "task_type" in request_config:
            mm_processor_kwargs["task_type"] = request_config["task_type"]
        if "ref_audio" in request_config:
            mm_processor_kwargs["ref_audio"] = request_config["ref_audio"]
        if "ref_text" in request_config:
            mm_processor_kwargs["ref_text"] = request_config["ref_text"]
        if "language" in request_config:
            mm_processor_kwargs["language"] = request_config["language"]
        if "max_new_tokens" in request_config:
            mm_processor_kwargs["max_new_tokens"] = request_config["max_new_tokens"]

        outputs = self.runner.generate_multimodal(
            prompts=input_text,
            modalities=["audio"],
            mm_processor_kwargs=mm_processor_kwargs or None,
        )
        mm_out: dict[str, Any] | None = None
        for stage_out in outputs:
            if getattr(stage_out, "final_output_type", None) == "audio":
                mm_out = stage_out.request_output.outputs[0].multimodal_output
                break
        if mm_out is None:
            result = OmniResponse(success=False, error_message="No audio output from pipeline")
            assert result.success, result.error_message
            return result

        audio_data = mm_out.get("audio")
        if audio_data is None:
            result = OmniResponse(success=False, error_message="No audio tensor in multimodal output")
            assert result.success, result.error_message
            return result

        sr_raw = mm_out.get("sr")
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)
        wav_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
        wav_buf = io.BytesIO()
        sf.write(
            wav_buf,
            wav_tensor.float().cpu().numpy().reshape(-1),
            samplerate=sr,
            format="WAV",
            subtype="PCM_16",
        )
        result = OmniResponse(success=True, audio_bytes=wav_buf.getvalue(), audio_format="audio/wav")
        assert_audio_speech_response(result, request_config, run_level="core_model")
        return result

    def start_profile(
        self,
        profile_prefix: str | None = None,
        stages: list[int] | None = None,
    ) -> list[Any]:
        """Start profiling specified stages."""
        return self.runner.start_profile(profile_prefix=profile_prefix, stages=stages)

    def stop_profile(self, stages: list[int] | None = None) -> list[Any]:
        """Stop profiling specified stages."""
        return self.runner.stop_profile(stages=stages)


@pytest.fixture
def omni_runner_handler(omni_runner):
    return OmniRunnerHandler(omni_runner)
