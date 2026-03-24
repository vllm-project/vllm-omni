"""Shared constants, helpers, and payload building for Qwen3-TTS Gradio demos."""

import base64
import io

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

SUPPORTED_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

TASK_TYPES = ["CustomVoice", "VoiceDesign", "Base"]

PCM_SAMPLE_RATE = 24000

DEFAULT_API_BASE = "http://localhost:8000"


def fetch_voices(api_base: str) -> list[str]:
    """Fetch available voices from the server."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{api_base}/v1/audio/voices",
                headers={"Authorization": "Bearer EMPTY"},
            )
        if resp.status_code == 200:
            data = resp.json()
            voices = data.get("voices") or []
            if voices:
                return voices
    except Exception:
        pass
    return ["Vivian", "Ryan"]


def encode_audio_to_base64(audio_data: tuple) -> str:
    """Encode Gradio audio input (sample_rate, numpy_array) to base64 data URL."""
    sample_rate, audio_np = audio_data

    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    wav_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"


def build_payload(
    text: str,
    task_type: str,
    voice: str,
    language: str,
    instructions: str,
    ref_audio: tuple | None,
    ref_audio_url: str,
    ref_text: str,
    x_vector_only: bool,
    response_format: str = "pcm",
    speed: float = 1.0,
    stream: bool = True,
) -> dict:
    """Build the /v1/audio/speech request payload.

    Raises gr.Error for invalid input so callers don't need to validate.
    """
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    payload: dict = {
        "input": text.strip(),
        "response_format": "pcm" if stream else response_format,
        "stream": stream,
    }
    if not stream:
        payload["speed"] = speed

    if task_type:
        payload["task_type"] = task_type
    if language:
        payload["language"] = language

    if task_type == "CustomVoice":
        if voice:
            payload["speaker"] = voice
        if instructions and instructions.strip():
            payload["instructions"] = instructions.strip()

    elif task_type == "VoiceDesign":
        if not instructions or not instructions.strip():
            raise gr.Error("VoiceDesign task requires voice style instructions.")
        payload["instructions"] = instructions.strip()

    elif task_type == "Base":
        ref_audio_url_stripped = ref_audio_url.strip() if ref_audio_url else ""
        if ref_audio_url_stripped:
            payload["ref_audio"] = ref_audio_url_stripped
        elif ref_audio is not None:
            payload["ref_audio"] = encode_audio_to_base64(ref_audio)
        else:
            raise gr.Error("Base (voice clone) task requires reference audio. Upload a file or provide a URL.")
        if ref_text and ref_text.strip():
            payload["ref_text"] = ref_text.strip()
        if x_vector_only:
            payload["x_vector_only_mode"] = True

    return payload


def on_task_type_change(task_type: str):
    """Update UI visibility based on selected task type."""
    if task_type == "CustomVoice":
        return (
            gr.update(visible=True),  # voice dropdown
            gr.update(visible=True, info="Optional style/emotion instructions"),
            gr.update(visible=False),  # ref_audio
            gr.update(visible=False),  # ref_audio_url
            gr.update(visible=False),  # ref_text
            gr.update(visible=False),  # x_vector_only
        )
    elif task_type == "VoiceDesign":
        return (
            gr.update(visible=False),
            gr.update(visible=True, info="Required: describe the voice style"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif task_type == "Base":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def stream_pcm_chunks(api_base: str, payload: dict):
    """Stream raw PCM bytes from the server, yielding int16 numpy arrays.

    Handles odd-byte boundaries between network chunks.
    """
    leftover = b""
    with httpx.Client(timeout=300.0) as client:
        with client.stream(
            "POST",
            f"{api_base}/v1/audio/speech",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
            },
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                raise gr.Error(f"Server error ({resp.status_code}): {resp.text}")
            for chunk in resp.iter_bytes():
                if not chunk:
                    continue
                raw = leftover + chunk
                usable = len(raw) - (len(raw) % 2)
                leftover = raw[usable:]
                if usable == 0:
                    continue
                yield np.frombuffer(raw[:usable], dtype=np.int16).copy()


def add_common_args(parser):
    """Add CLI arguments shared by both demos."""
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Base URL for the vLLM API server (default: {DEFAULT_API_BASE}).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP for Gradio server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for Gradio server (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio demo publicly.",
    )
    return parser
