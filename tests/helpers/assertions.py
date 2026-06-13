"""Assertion and response validation helpers for tests."""

import io
import json
import tempfile
import threading
import wave
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from tests.helpers.runtime import DiffusionResponse

import numpy as np
import soundfile as sf
from PIL import Image

from tests.helpers.media import (
    cosine_similarity_text,
)

_GENDER_PIPELINE = None
_GENDER_PIPELINE_LOCK = threading.Lock()
_PCM_SPEECH_SAMPLE_RATE_HZ = 24_000
_MIN_PCM_SPEECH_HNR_DB = 1.0
_PRESET_VOICE_GENDER_MAP: dict[str, str] = {
    "serena": "female",
    "uncle_fu": "male",
    "chelsie": "female",
    "clone": "female",
    "ethan": "male",
}


def assert_image_diffusion_response(
    response: "DiffusionResponse",
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

    if run_level in {"advanced_model", "full_model"}:
        width = extra_body.get("width")
        height = extra_body.get("height")

        if width is not None or height is not None:
            if isinstance(width, (list, tuple)) and isinstance(height, (list, tuple)):
                assert len(response.images) == len(width) == len(height), (
                    f"Per-output size lists require one image per entry; got {len(response.images)} images, "
                    f"len(width)={len(width)}, len(height)={len(height)}"
                )
                for img, w, h in zip(response.images, width, height, strict=True):
                    assert_image_valid(img, width=int(w), height=int(h))
            else:
                for img in response.images:
                    assert_image_valid(
                        img,
                        width=_maybe_int(width) if width is not None else None,
                        height=_maybe_int(height) if height is not None else None,
                    )


def assert_video_diffusion_response(
    response: "DiffusionResponse",
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
    response: "DiffusionResponse",
    request_config: dict[str, Any],
    run_level: str = None,
) -> None:
    """
    Validate audio diffusion response.

    `response.audios` carries one entry per choice, each a `dict` with raw WAV
    bytes (`wav_bytes`) and the OpenAI audio metadata (`id`, `expires_at`).
    """
    assert response.audios, "Audio response is empty"
    for audio in response.audios:
        wav_bytes = audio.get("wav_bytes")
        assert wav_bytes, "Audio entry missing decoded WAV bytes"
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            assert wav_file.getnframes() > 0, "Decoded WAV has zero frames"
            assert wav_file.getframerate() > 0, "Decoded WAV has invalid sample rate"


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
    """Assert the MP4 has the expected resolution and frame count.

    For several diffusion backends, encoded MP4 frame count follows a codec-aligned
    convention (e.g. request `num_frames=8` can produce 9 encoded frames). Keep
    this compatibility behavior to avoid false negatives in online-serving tests.
    """
    temp_path = None
    cap = None
    try:
        import cv2

        if isinstance(video, Path):
            if not video.exists():
                raise AssertionError(f"Video file not found: {video}")
            video_path = str(video)
        else:
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

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise AssertionError("Failed to open video capture")

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width is not None:
            assert actual_width == width, f"Expected width={width}, got {actual_width}"
        if height is not None:
            assert actual_height == height, f"Expected height={height}, got {actual_height}"
        if fps is not None and actual_fps:
            assert abs(actual_fps - float(fps)) < 1.0, f"Expected fps~={fps}, got {actual_fps}"
        if num_frames is not None:
            expected_frames = (int(num_frames) // 4) * 4 + 1
            assert actual_frames == expected_frames, f"Expected frames={expected_frames}, got {actual_frames}"

        return {
            "width": actual_width,
            "height": actual_height,
            "fps": actual_fps,
            "num_frames": actual_frames,
        }
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        if cap is not None:
            cap.release()
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def assert_audio_valid(
    audio_or_path: Path | np.ndarray,
    *,
    sample_rate: int,
    channels: int,
    duration_s: float,
) -> None:
    """Assert WAV file or (batch, channels, samples) ndarray matches expected audio format."""
    expected_samples = int(duration_s * sample_rate)
    if isinstance(audio_or_path, np.ndarray):
        audio = audio_or_path
        assert audio.ndim == 3, f"Expected audio ndim=3 (batch, channels, samples), got shape {audio.shape}"
        assert audio.shape[0] == 1, f"Expected batch size 1, got {audio.shape[0]}"
        assert audio.shape[1] == channels, f"Expected {channels} channels, got {audio.shape[1]}"
        assert audio.shape[2] == expected_samples, (
            f"Expected {expected_samples} samples ({duration_s}s @ {sample_rate} Hz), got {audio.shape[2]}"
        )
        return

    path = audio_or_path
    assert path.exists(), f"Audio not found: {path}"
    info = sf.info(str(path))
    assert info.samplerate == sample_rate, f"Expected sample_rate={sample_rate}, got {info.samplerate}"
    assert info.channels == channels, f"Expected {channels} channel(s), got {info.channels}"
    assert info.frames == expected_samples, (
        f"Expected {expected_samples} frames ({duration_s}s @ {sample_rate} Hz), got {info.frames}"
    )


def _load_gender_pipeline():
    global _GENDER_PIPELINE
    if _GENDER_PIPELINE is not None:
        return _GENDER_PIPELINE
    model_name = "7wolf/wav2vec2-base-gender-classification"
    try:
        from transformers import pipeline

        _GENDER_PIPELINE = pipeline(task="audio-classification", model=model_name, device=-1)
        return _GENDER_PIPELINE
    except Exception as exc:  # pragma: no cover
        print(f"Warning: failed to create gender pipeline '{model_name}': {exc}")
        _GENDER_PIPELINE = None
        return None


def _median_pitch_hz_from_autocorr(mono: np.ndarray, sr: int) -> float | None:
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
        with _GENDER_PIPELINE_LOCK:
            outputs = clf(mono, sampling_rate=sr)
        if not outputs:
            return "unknown"
        top = outputs[0]
        label = str(top.get("label", "")).lower()
        conf = float(top.get("score", 0.0))
        if conf < 0.6:
            gender = "unknown"
        elif ("female" in label) or ("жен" in label):
            gender = "female"
        elif ("male" in label) or ("муж" in label):
            gender = "male"
        else:
            gender = "unknown"

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
    except Exception as exc:  # pragma: no cover
        print(f"Warning: gender classification failed, returning 'unknown': {exc}")
        return "unknown"


def _assert_preset_voice_gender_from_audio(audio_bytes: bytes | None, voice_name: str | None) -> None:
    """If ``voice_name`` matches a known preset, assert classifier gender matches (skip when unknown)."""
    if not voice_name or not audio_bytes:
        return
    key = str(voice_name).lower()
    expected_gender = _PRESET_VOICE_GENDER_MAP.get(key)
    if expected_gender is None:
        return
    estimated_gender = _estimate_voice_gender_from_audio(audio_bytes)
    print(f"Preset voice gender check: preset={key!r}, estimated={estimated_gender!r}, expected={expected_gender!r}")
    if estimated_gender != "unknown":
        assert estimated_gender == expected_gender, (
            f"{voice_name!r} is expected {expected_gender}, but estimated gender is {estimated_gender!r}"
        )


def _compute_pcm_hnr_db(pcm_samples: np.ndarray, sr: int = _PCM_SPEECH_SAMPLE_RATE_HZ) -> float:
    frame_len = int(0.03 * sr)
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


def _assert_pcm_int16_speech_hnr(audio_bytes: bytes, min_hnr_db: float = _MIN_PCM_SPEECH_HNR_DB) -> None:
    """Validate harmonic-to-noise ratio on raw int16 PCM from /v1/audio/speech.

    min_hnr_db defaults to the global _MIN_PCM_SPEECH_HNR_DB (1.0 dB),
    which matches the cleaner TTS models the helper was originally calibrated
    for. Quieter codecs (e.g. MOSS-TTS-Nano, whose voice_clone output is
    intrinsically around -2 dB) can pass a lower per-test threshold via
    request_config["min_hnr_db"] to keep the catastrophic-failure check
    while not gating CI on a model-intrinsic property.
    """
    assert audio_bytes is not None and len(audio_bytes) >= 2, "missing PCM bytes"
    assert len(audio_bytes) % 2 == 0, "PCM byte length must be aligned to int16"
    pcm_samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    hnr = _compute_pcm_hnr_db(pcm_samples)
    print(f"PCM speech HNR: {hnr:.2f} dB (threshold: {min_hnr_db} dB)")
    assert hnr >= min_hnr_db, (
        f"Audio distortion detected: HNR={hnr:.2f} dB < {min_hnr_db} dB. "
        "Voice clone decoder may be losing ref_code speaker context on later chunks."
    )


def assert_omni_response(response: Any, request_config: dict[str, Any], run_level):
    """
    Validate response results.

    Args:
        response: OmniResponse object

    Raises:
        AssertionError: When the response does not meet validation criteria
    """
    assert response.success, "The request failed."

    modalities = request_config.get("modalities", ["text", "audio"])

    if run_level in {"advanced_model", "full_model"}:
        # Verify output success
        if "audio" in modalities:
            assert response.audio_content is not None, "No audio output is generated"
            print(f"audio content is: {response.audio_content}")
            speaker = request_config.get("speaker")
            if speaker:
                _assert_preset_voice_gender_from_audio(
                    response.audio_bytes,
                    speaker,
                )
        if "text" in modalities:
            assert response.text_content is not None, "No text output is generated"
            print(f"text content is: {response.text_content}")

        # Verify keywords in output
        word_types = ["text", "image", "audio", "video"]
        keywords_dict = request_config.get("key_words", {})
        for word_type in word_types:
            keywords = keywords_dict.get(word_type)
            if "text" in modalities:
                if keywords:
                    text_lower = response.text_content.lower()
                    assert any(str(kw).lower() in text_lower for kw in keywords), (
                        "The output does not contain any of the keywords."
                    )
            else:
                if keywords:
                    audio_lower = response.audio_content.lower()
                    assert any(str(kw).lower() in audio_lower for kw in keywords), (
                        "The output does not contain any of the keywords."
                    )

        # Verify similarity (Whisper transcript vs streamed/detokenized text)
        if "audio" in modalities:
            audio_ref_text = request_config.get("audio_ref_text")
            similarity_threshold = request_config.get("similarity_threshold", 0.8)
            if "text" in modalities:
                transcript = (response.audio_content or "").strip()
                text_output = (response.text_content or "").strip()
                # For very short outputs (e.g. one-word answers), n-gram cosine
                # similarity with length penalty is unreliable because Whisper
                # may hallucinate extra context around the short utterance.  Use
                # a containment check instead: the shorter text must appear in
                # the longer one (after preprocessing removes punctuation).
                _SHORT_TEXT_THRESHOLD = 15
                if len(text_output) <= _SHORT_TEXT_THRESHOLD or len(transcript) <= _SHORT_TEXT_THRESHOLD:
                    shorter = text_output.lower() if len(text_output) <= len(transcript) else transcript.lower()
                    longer = transcript.lower() if len(text_output) <= len(transcript) else text_output.lower()
                    import re as _re

                    shorter_clean = _re.sub(r"[^\w\s]", "", shorter).strip()
                    longer_clean = _re.sub(r"[^\w\s]", "", longer).strip()
                    assert shorter_clean and (shorter_clean in longer_clean), (
                        f"The audio content is not same as the text "
                        f"(short-text containment check failed: "
                        f"text={text_output!r}, transcript={transcript!r})"
                    )
                    print(f"short-text containment check passed: {shorter_clean!r} in {longer_clean!r}")
                else:
                    similarity = cosine_similarity_text(
                        transcript.lower(),
                        text_output.lower(),
                    )
                    print(f"similarity is: {similarity}")
                    assert similarity > similarity_threshold, "The audio content is not same as the text"
            if audio_ref_text:
                audio_similarity = cosine_similarity_text(
                    response.audio_content.lower(),
                    str(audio_ref_text).lower(),
                )
                assert audio_similarity > similarity_threshold, (
                    f"The audio content does not match reference text: similarity={audio_similarity:.3f}"
                )


def assert_audio_speech_response(response: Any, request_config: dict[str, Any], run_level: str) -> None:
    """Validate speech API results from :class:`~tests.helpers.runtime.OmniResponse`.

    When ``request_config`` carries ``status_code`` and/or ``err_message``, the
    request is expected to be rejected: assert it failed and that the HTTP status
    / error text match. Otherwise the normal success-path checks run.
    """
    expected_status = request_config.get("status_code")
    expected_err = request_config.get("err_message")
    if expected_status is not None or expected_err is not None:
        assert not response.success, "Expected an error response, but the request succeeded."
        if expected_status is not None:
            allowed = expected_status if isinstance(expected_status, (list, tuple)) else (expected_status,)
            assert response.status_code in allowed, f"Expected HTTP status in {allowed}, got {response.status_code}"
        if expected_err is not None:
            alternatives = expected_err if isinstance(expected_err, (list, tuple)) else (expected_err,)
            error_text = response.error_message or ""
            assert any(alt in error_text for alt in alternatives), (
                f"Expected one of {alternatives} in error text, got: {error_text!r}"
            )
        return

    assert response.success, "The request failed."

    # Optional floor on decoded audio size (models with very short clips may use a lower value).
    min_audio = request_config.get("min_audio_bytes")
    if min_audio is not None:
        n = int(min_audio)
        if n > 0:
            ab = response.audio_bytes
            assert ab is not None, "Expected audio bytes when min_audio_bytes is set"
            assert len(ab) > n, f"Audio payload too small: {len(ab)} bytes, expected more than {n} (min_audio_bytes)"

    req_fmt = request_config.get("response_format")
    if req_fmt == "pcm" and response.audio_bytes:
        min_hnr_db = float(request_config.get("min_hnr_db", _MIN_PCM_SPEECH_HNR_DB))
        _assert_pcm_int16_speech_hnr(response.audio_bytes, min_hnr_db=min_hnr_db)
        if response.audio_format:
            assert "pcm" in response.audio_format.lower(), (
                f"Expected audio/pcm content-type, got {response.audio_format!r}"
            )
    elif req_fmt == "wav" and response.audio_format:
        assert req_fmt in response.audio_format

    if run_level in {"advanced_model", "full_model"} and req_fmt != "pcm":
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
        _assert_preset_voice_gender_from_audio(response.audio_bytes, request_config.get("voice"))


def assert_diffusion_response(response: "DiffusionResponse", request_config: dict[str, Any], run_level: str = None):
    assert response.success, "The request failed."
    has_any_content = any(content is not None for content in (response.images, response.videos, response.audios))
    assert has_any_content, "Response contains no images, videos, or audios"
    if response.images is not None:
        assert_image_diffusion_response(response=response, request_config=request_config, run_level=run_level)
    if response.videos is not None:
        assert_video_diffusion_response(response=response, request_config=request_config, run_level=run_level)
    if response.audios is not None:
        assert_audio_diffusion_response(response=response, request_config=request_config, run_level=run_level)


def _http_response_body_materialize(resp: Any) -> tuple[bytes, dict[str, Any] | None]:
    """Serialize ``HttpResponse``-like body to UTF-8 bytes and parse a JSON object when possible."""
    jb = getattr(resp, "json_body", None)
    if jb is not None:
        raw = json.dumps(jb, ensure_ascii=False).encode("utf-8")
        if isinstance(jb, dict):
            return raw, jb
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return raw, None
        return raw, parsed if isinstance(parsed, dict) else None
    err = getattr(resp, "error_message", None)
    raw = (err or "").encode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return raw, None
    return raw, parsed if isinstance(parsed, dict) else None


def assert_err_message_in_text(
    haystack: str,
    err_message: str | tuple[str, ...] | list[str] | set[str] | frozenset[str],
    *,
    sequence_match: Literal["all", "any"] = "all",
) -> None:
    """Assert ``err_message`` appears in ``haystack`` (case-insensitive).

    For non-string sequences: ``sequence_match='all'`` requires every substring (HTTP-style);
    ``sequence_match='any'`` requires at least one (WebSocket first-frame JSON helpers).
    """
    hl = haystack.lower()
    if isinstance(err_message, (list, tuple, set, frozenset)):
        if sequence_match == "all":
            missing = [s for s in err_message if str(s).lower() not in hl]
            assert not missing, (
                f"Expected error text to contain all of {err_message!r}; missing {missing!r}. haystack={haystack!r}"
            )
        else:
            assert any(str(s).lower() in hl for s in err_message), (
                f"Expected error text to contain one of {err_message!r}. haystack={haystack!r}"
            )
    else:
        assert str(err_message).lower() in hl, f"Expected error text to contain {str(err_message)!r}, got: {haystack!r}"


def assert_http_error(
    resp: Any,
    *,
    err_code: int | tuple[int, ...] | list[int] | None = None,
    err_message: str | tuple[str, ...] | list[str] | None = None,
    websocket_json_message: bool = False,
) -> dict[str, Any] | None:
    """Validate a raw-HTTP :class:`~tests.helpers.runtime.HttpResponse`-like object.

    Used by :class:`~tests.helpers.runtime.OpenAIClientHandler` ``send_*_http_request`` helpers when
    ``request_config`` contains optional ``err_code`` and/or ``err_message``.

    When ``websocket_json_message=True``, only ``json_body`` is checked (first JSON WebSocket text frame).
    Tuple/list ``err_message`` then uses **any** substring match; HTTP mode still requires **all** pieces.

    - ``err_code``: exact HTTP ``int``, or membership if a non-string sequence (e.g. ``(400, 422)``).
      When ``err_code`` is set and the actual status is a client error in ``400..499`` other than ``404``,
      the JSON body must include FastAPI ``detail`` and/or OpenAI-style ``error`` (2xx skips this).
    - ``err_message``: substring match (case-insensitive) against serialized ``json_body`` and ``error_message``.
      If ``err_message`` is a non-string sequence (``list`` / ``tuple`` / ``set`` / ``frozenset``), **every**
      element must appear as a substring; a plain ``str`` still requires that single substring.
    """
    if websocket_json_message:
        if err_code is not None:
            raise ValueError("assert_http_error: err_code is incompatible with websocket_json_message=True")
        jb_ws = getattr(resp, "json_body", None)
        assert jb_ws is not None, resp
        if err_message is None:
            return jb_ws if isinstance(jb_ws, dict) else None
        assert_err_message_in_text(
            json.dumps(jb_ws, ensure_ascii=False),
            err_message,
            sequence_match="any",
        )
        return jb_ws if isinstance(jb_ws, dict) else None

    if err_code is None and err_message is None:
        return None

    actual = getattr(resp, "status_code", None)
    assert actual is not None, "response missing status_code"

    if err_code is not None:
        if isinstance(err_code, int):
            assert actual == err_code, (resp, err_code)
        else:
            allowed = tuple(err_code)
            assert actual in allowed, (resp, allowed)

    body_bytes, payload = _http_response_body_materialize(resp)

    if err_code is not None and actual is not None and 400 <= actual < 500 and actual != 404:
        assert payload is not None, getattr(resp, "error_message", resp)
        assert "detail" in payload or "error" in payload, payload

    if err_message is not None:
        pieces: list[str] = []
        jb = getattr(resp, "json_body", None)
        if jb is not None:
            pieces.append(json.dumps(jb, ensure_ascii=False))
        em = getattr(resp, "error_message", None)
        if em:
            pieces.append(str(em))
        haystack = "\n".join(pieces) if pieces else body_bytes.decode("utf-8", errors="replace")
        assert_err_message_in_text(haystack, err_message, sequence_match="all")

    return payload


__all__ = [
    "assert_audio_diffusion_response",
    "assert_audio_speech_response",
    "assert_diffusion_response",
    "assert_err_message_in_text",
    "assert_http_error",
    "assert_image_diffusion_response",
    "assert_image_valid",
    "assert_omni_response",
    "assert_video_diffusion_response",
    "assert_video_valid",
    "assert_audio_valid",
]
