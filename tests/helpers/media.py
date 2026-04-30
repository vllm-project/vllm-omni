"""Synthetic media generation and media/text utilities for tests."""

import base64
import concurrent.futures
import gc
import hashlib
import io
import logging
import math
import multiprocessing
import os
import random
import re
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image

logger = logging.getLogger(__name__)


def _resolve_synthetic_media_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    return Path(tempfile.gettempdir()) / "vllm_omni_test_synthetic_media"


def _np_array_from_mp4_bytes(video_bytes: bytes) -> np.ndarray:
    """Decode MP4 bytes to a (T, H, W, 3) uint8 RGB stack (matches in-memory synthetic frames)."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open cached synthetic video for decode")
        frames: list[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not frames:
            raise RuntimeError("Cached synthetic video has no decodable frames")
        return np.stack(frames, axis=0)
    finally:
        if cap is not None:
            cap.release()
        try:
            os.unlink(path)
        except OSError:
            pass


def generate_synthetic_audio(
    duration: int,
    num_channels: int,
    sample_rate: int = 48000,
    *,
    phrase_text: str = "test",
    force_regenerate: bool = False,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Generate TTS speech with pyttsx3 and return base64 string.

    Caches the WAV under ``cache_dir`` when given, else under the default temp
    subdirectory. Reuses the file when the same
    ``duration`` / ``num_channels`` / ``sample_rate`` / ``phrase_text`` are
    requested unless ``force_regenerate`` is true.

    The cache filename includes a SHA-256 digest of ``phrase_text`` so different
    phrases never share a WAV cache entry.
    """
    root = _resolve_synthetic_media_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    phrase_key = hashlib.sha256(phrase_text.encode("utf-8")).hexdigest()
    cache_path = root / f"synth_audio_d{duration}_ch{num_channels}_sr{sample_rate}_pt{phrase_key}.wav"

    if not force_regenerate and cache_path.is_file():
        data, _sr = sf.read(str(cache_path), dtype="float32", always_2d=True)
        audio_bytes = cache_path.read_bytes()
        return {
            "np_array": np.asarray(data, dtype=np.float32),
            "base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "file_path": str(cache_path.resolve()),
        }

    import pyttsx3

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
        enhanced = np.sign(enhanced) * np.sqrt(np.abs(enhanced))
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

    sf.write(str(cache_path), audio_data, sample_rate, format="WAV", subtype="PCM_16")
    audio_bytes = cache_path.read_bytes()

    return {
        "np_array": audio_data.copy(),
        "base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "file_path": str(cache_path.resolve()),
    }


def _mux_mp4_bytes_with_synthetic_audio(
    video_mp4_bytes: bytes,
    *,
    num_frames: int,
    fps: float = 30.0,
    sample_rate: int = 48000,
) -> bytes:
    duration_sec = num_frames / fps if fps > 0 else 0.0
    duration_int = max(1, int(math.ceil(duration_sec)))

    try:
        audio_result = generate_synthetic_audio(
            duration=duration_int,
            num_channels=1,
            sample_rate=sample_rate,
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
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=300)
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
    *,
    embed_audio: bool = False,
    force_regenerate: bool = False,
    cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Generate synthetic MP4 (optional AAC audio). Caches final bytes by
    ``width`` / ``height`` / ``num_frames`` / ``embed_audio`` unless
    ``force_regenerate`` is true. Cache root: ``cache_dir`` if given, else the
    default temp subdirectory.
    """
    root = _resolve_synthetic_media_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"synth_video_w{width}_h{height}_nf{num_frames}_ea{int(embed_audio)}.mp4"

    if not force_regenerate and cache_path.is_file():
        video_bytes = cache_path.read_bytes()
        return {
            "np_array": _np_array_from_mp4_bytes(video_bytes),
            "base64": base64.b64encode(video_bytes).decode("utf-8"),
            "file_path": str(cache_path.resolve()),
        }

    import cv2
    import imageio

    num_balls = random.randint(3, 8)
    balls = []
    for _ in range(num_balls):
        radius = min(width, height) // 8
        if radius < 1:
            raise ValueError(f"Video dimensions ({width}x{height}) too small")
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)
        speed = random.uniform(3.0, 8.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        color_bgr = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        balls.append({"x": x, "y": y, "vx": vx, "vy": vy, "radius": radius, "color_bgr": color_bgr})

    video_frames = []
    for _ in range(num_frames):
        frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        for ball in balls:
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"]
            if ball["x"] - ball["radius"] <= 0 or ball["x"] + ball["radius"] >= width:
                ball["vx"] = -ball["vx"]
                ball["x"] = max(ball["radius"], min(width - ball["radius"], ball["x"]))
            if ball["y"] - ball["radius"] <= 0 or ball["y"] + ball["radius"] >= height:
                ball["vy"] = -ball["vy"]
                ball["y"] = max(ball["radius"], min(height - ball["radius"], ball["y"]))
            x, y = int(ball["x"]), int(ball["y"])
            radius = int(ball["radius"])
            cv2.circle(frame_bgr, (x, y), radius, ball["color_bgr"], -1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        video_frames.append(frame_rgb)

    fps = 30
    buffer = io.BytesIO()
    writer_kwargs = {
        "format": "mp4",
        "fps": fps,
        "codec": "libx264",
        "quality": 7,
        "pixelformat": "yuv420p",
        "macro_block_size": 16,
        "ffmpeg_params": ["-preset", "medium", "-crf", "23", "-movflags", "+faststart", "-pix_fmt", "yuv420p"],
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
    video_bytes = (
        _mux_mp4_bytes_with_synthetic_audio(video_only_bytes, num_frames=num_frames, fps=float(fps))
        if embed_audio
        else video_only_bytes
    )

    cache_path.write_bytes(video_bytes)

    return {
        "np_array": np.array(video_frames),
        "base64": base64.b64encode(video_bytes).decode("utf-8"),
        "file_path": str(cache_path.resolve()),
    }


def generate_synthetic_image(
    width: int,
    height: int,
    *,
    force_regenerate: bool = False,
    cache_dir: Path | str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Random colored squares on white background. Caches JPEG by ``width`` /
    ``height`` unless ``force_regenerate`` is true. Cache root: ``cache_dir``
    if given, else the default temp subdirectory.
    """
    if seed is not None:
        random.seed(seed)

    root = _resolve_synthetic_media_cache_dir(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"synth_image_w{width}_h{height}.jpg"

    if not force_regenerate and cache_path.is_file():
        from PIL import Image as PILImage

        image = PILImage.open(cache_path)
        image.load()
        image_bytes = cache_path.read_bytes()
        return {
            "np_array": np.array(image).copy(),
            "base64": base64.b64encode(image_bytes).decode("utf-8"),
            "file_path": str(cache_path.resolve()),
        }

    from PIL import ImageDraw

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    num_squares = random.randint(3, 8)
    for _ in range(num_squares):
        square_size = random.randint(max(1, min(width, height) // 8), max(2, min(width, height) // 4))
        x = random.randint(0, max(0, width - square_size - 1))
        y = random.randint(0, max(0, height - square_size - 1))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        border_width = random.randint(1, 5)
        draw.rectangle([x, y, x + square_size, y + square_size], fill=color, outline=(0, 0, 0), width=border_width)

    image.save(str(cache_path), format="JPEG", quality=85, optimize=True)
    image_bytes = cache_path.read_bytes()

    return {
        "np_array": np.array(image).copy(),
        "base64": base64.b64encode(image_bytes).decode("utf-8"),
        "file_path": str(cache_path.resolve()),
    }


_TEST_ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets"

_AUDIO_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def get_asset_path(relative_path: str | os.PathLike) -> Path:
    """Resolve a path under ``tests/assets/`` to its absolute on-disk location."""
    return _TEST_ASSETS_ROOT / Path(relative_path)


def load_test_audio_data_url(relative_path: str | os.PathLike) -> str:
    """Load a vendored test audio file under ``tests/assets/`` as a base64 data URL.

    Used by tests that need real reference audio (e.g. voice cloning) without
    relying on the server's ability to fetch external URLs at request time.
    """
    path = get_asset_path(relative_path)
    mime = _AUDIO_MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def decode_b64_image(b64: str):
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    return img


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
    cosine = dot_product / (norm1 * norm2)
    # Down-weight when lengths differ: repeated/hallucinated transcripts stay
    # high in bag-of-ngrams cosine (e.g. ABCABCABC vs ABC) but should score low.
    len1, len2 = len(text1), len(text2)
    length_harmony = (2.0 * min(len1, len2)) / (len1 + len2)
    return cosine * length_harmony


def _merge_base64_audio_to_segment(base64_list: list[str]):
    from pydub import AudioSegment

    merged = None
    for b64 in base64_list:
        raw = base64.b64decode(b64.split(",", 1)[-1])
        seg = AudioSegment.from_file(io.BytesIO(raw))
        merged = seg if merged is None else merged + seg
    return merged


@contextmanager
def _serialize_whisper_small_model_download():
    """Serialize Whisper ``small`` cache writes across processes (Linux/Unix)."""
    import fcntl

    lock_path = Path.home() / ".cache" / "whisper" / ".small_model_download.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _whisper_transcribe_in_current_process(output_path: str) -> str:
    import whisper

    device_index = None
    from vllm_omni.platforms import current_omni_platform

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

    with _serialize_whisper_small_model_download():
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
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
        future = executor.submit(_whisper_transcribe_in_current_process, output_path)
        return future.result()


def convert_audio_bytes_to_text(raw_bytes: bytes) -> str:
    output_path = f"./test_{uuid.uuid4().hex}.wav"
    data, samplerate = sf.read(io.BytesIO(raw_bytes))
    sf.write(output_path, data, samplerate, format="WAV", subtype="PCM_16")
    print(f"audio data is saved: {output_path}")
    return convert_audio_file_to_text(output_path)


__all__ = [
    "_merge_base64_audio_to_segment",
    "convert_audio_bytes_to_text",
    "convert_audio_file_to_text",
    "cosine_similarity_text",
    "decode_b64_image",
    "generate_synthetic_audio",
    "generate_synthetic_image",
    "generate_synthetic_video",
    "get_asset_path",
    "load_test_audio_data_url",
    "preprocess_text",
]
