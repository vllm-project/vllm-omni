"""Gradio demo for MiniCPM-o 4.5 online serving through vllm-omni.

Supports:
- Text / Image / Audio / Video inputs (OpenAI chat-completions multimodal format).
- Text + audio outputs, returned side-by-side in the UI.

Usage:
    # run a 4.5 server on :8099
    python gradio_demo.py \
        --minicpmo45-api-base http://localhost:8099/v1 \
        --minicpmo45-model openbmb/MiniCPM-o-4_5 \
        --port 7862
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf
from openai import OpenAI
from PIL import Image

MINICPMO45 = "MiniCPM-o-4.5"

# Default reference audio per model, relative to the model path. Used to build
# the audio_assistant system prompt so that the LLM adopts the "speak in this
# voice" persona (matches the official MiniCPM-o demo).
_DEFAULT_REF_AUDIO = {
    MINICPMO45: "assets/HT_ref_audio.wav",
}

# audio_assistant prompt text, mirrors MiniCPMO.get_sys_prompt("audio_assistant")
# in the MiniCPM-o-Demo reference implementation.
_AUDIO_ASSISTANT_PROMPT = {
    "zh": {
        "prefix": "模仿音频样本的音色并生成新的内容。",
        "suffix": (
            "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
            "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
        ),
    },
    "en": {
        "prefix": "Use the voice in the audio prompt to synthesize new content.",
        "suffix": "You are a helpful assistant with the above voice style.",
    },
}


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------


def image_to_base64_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def audio_to_base64_data_url(audio_np: np.ndarray, sample_rate: int) -> str:
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def ref_audio_to_data_url(path: str, target_sr: int = 16000, max_s: float | None = 8.0) -> str:
    """Load a file, downmix to mono, resample to 16kHz, optionally truncate, and
    return a data-URL usable as audio_url content in the chat API.

    MiniCPM-o expects 16kHz mono audio input; long reference audios can waste
    tokens so we cap at max_s seconds by default (3-10s is the recommended range).
    """
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    if sr != target_sr:
        try:
            # Demo client only; librosa here resamples audio before sending to
            # the server.  The fallback below covers environments without it.
            import librosa  # noqa: TID251

            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from math import gcd

            g = gcd(sr, target_sr)
            up, down = target_sr // g, sr // g
            idx = (np.arange(len(data) * up) / up).astype(np.int64)
            data = data[idx][::down]
        sr = target_sr
    if max_s is not None and len(data) > int(max_s * sr):
        data = data[: int(max_s * sr)]
    return audio_to_base64_data_url(data, sr)


def video_to_base64_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    ext = p.suffix.lower()
    mime = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "video/mp4")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def process_audio_input(audio_input: Any | None) -> tuple[np.ndarray, int] | None:
    """Normalize Gradio audio input to (np.ndarray mono float32, sample_rate).

    Gradio's `gr.Audio(type="numpy")` yields (sr, ndarray); we also accept
    filesystem paths and a few nested tuple variants that appear in different
    Gradio versions.
    """
    if audio_input is None:
        return None

    def _from_path(p: str) -> tuple[np.ndarray, int] | None:
        if not p:
            return None
        fp = Path(p)
        if not fp.exists():
            return None
        data, sr = sf.read(fp)
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32), int(sr)

    audio_np: np.ndarray | None = None
    sr: int | None = None

    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        a, b = audio_input
        if isinstance(a, int | float) and isinstance(b, np.ndarray):
            sr, audio_np = int(a), b
        elif isinstance(a, str):
            loaded = _from_path(a)
            if loaded is not None:
                audio_np, sr = loaded
    elif isinstance(audio_input, str):
        loaded = _from_path(audio_input)
        if loaded is not None:
            audio_np, sr = loaded

    if audio_np is None or sr is None:
        return None
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    return audio_np.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Per-model inference config
# ---------------------------------------------------------------------------


class ModelEndpoint:
    """Endpoint + TTS-trigger convention for one MiniCPM-o variant."""

    def __init__(self, name: str, api_base: str, model_path: str):
        self.name = name
        self.api_base = api_base
        self.model_path = model_path
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self._ref_audio_data_url: str | None = None

    @property
    def ref_audio_path(self) -> str | None:
        rel = _DEFAULT_REF_AUDIO.get(self.name)
        if not rel:
            return None
        full = os.path.join(self.model_path, rel)
        return full if os.path.exists(full) else None

    def get_ref_audio_data_url(self) -> str | None:
        """Lazily load and cache the default reference audio as a 16kHz mono
        data URL. Returns None if the reference audio file is missing."""
        if self._ref_audio_data_url is not None:
            return self._ref_audio_data_url
        p = self.ref_audio_path
        if not p:
            print(f"[{self.name}] no default reference audio found; falling back to text-only system prompt")
            return None
        try:
            self._ref_audio_data_url = ref_audio_to_data_url(p)
            print(f"[{self.name}] loaded reference audio: {p}")
        except Exception as e:
            print(f"[{self.name}] failed to load reference audio {p}: {e}")
            return None
        return self._ref_audio_data_url

    def build_extras(self) -> dict[str, Any]:
        """Extra kwargs passed to chat.completions.create for this model.

        4.5: use_tts_template=True chat-template flag appends <|tts_bos|>.
        """
        if self.name == MINICPMO45:
            return {
                "extra_body": {
                    "chat_template_kwargs": {"use_tts_template": True},
                    "modalities": ["text", "audio"],
                },
            }
        return {}


# ---------------------------------------------------------------------------
# System prompt (matches MiniCPMO.get_sys_prompt("audio_assistant") in the
# official MiniCPM-o-Demo). When TTS is disabled we fall back to a plain
# text-only system prompt.
# ---------------------------------------------------------------------------


def build_audio_assistant_system(
    ref_audio_data_url: str | None,
    language: str = "zh",
) -> dict[str, Any]:
    """Return an OpenAI-style system message carrying the reference audio."""
    txt = _AUDIO_ASSISTANT_PROMPT.get(language, _AUDIO_ASSISTANT_PROMPT["zh"])
    if ref_audio_data_url is None:
        return {
            "role": "system",
            "content": (
                "You are MiniCPM-o, a helpful multimodal assistant that can "
                "understand images, audio and video, and respond in text and speech."
            ),
        }
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": txt["prefix"]},
            {"type": "audio_url", "audio_url": {"url": ref_audio_data_url}},
            {"type": "text", "text": txt["suffix"]},
        ],
    }


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------


def run_inference(
    endpoint: ModelEndpoint,
    user_prompt: str,
    audio_tuple: tuple[np.ndarray, int] | None,
    image_pil: Image.Image | None,
    video_path: str | None,
    enable_tts: bool,
    max_tokens: int,
    temperature: float,
) -> Iterable[tuple[str, tuple[int, np.ndarray] | None]]:
    user_prompt = user_prompt or ""
    if not user_prompt.strip() and not audio_tuple and not image_pil and not video_path:
        yield "Please provide a text prompt or at least one multimodal input.", None
        return

    try:
        content: list[dict[str, Any]] = []

        if audio_tuple is not None:
            audio_np, sr = audio_tuple
            content.append(
                {"type": "audio_url", "audio_url": {"url": audio_to_base64_data_url(audio_np, sr)}},
            )

        if image_pil is not None:
            img = image_pil if image_pil.mode == "RGB" else image_pil.convert("RGB")
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_base64_data_url(img)}},
            )

        if video_path:
            content.append(
                {"type": "video_url", "video_url": {"url": video_to_base64_data_url(video_path)}},
            )

        if user_prompt.strip():
            content.append({"type": "text", "text": user_prompt})

        if enable_tts:
            ref_url = endpoint.get_ref_audio_data_url()
            system_msg = build_audio_assistant_system(ref_url, language="zh")
        else:
            system_msg = {
                "role": "system",
                "content": (
                    "You are MiniCPM-o, a helpful multimodal assistant that can "
                    "understand images, audio and video, and respond in text and speech."
                ),
            }

        messages = [
            system_msg,
            {"role": "user", "content": content},
        ]

        kwargs: dict[str, Any] = {
            "model": endpoint.model_path,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if enable_tts:
            extras = endpoint.build_extras()
            # Merge extra_body dicts cleanly.
            extra_body = dict(extras.get("extra_body", {}))
            kwargs["extra_body"] = extra_body
        else:
            # Text-only: suppress audio output.
            kwargs["extra_body"] = {"modalities": ["text"]}

        completion = endpoint.client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        audio_out: tuple[int, np.ndarray] | None = None

        for choice in completion.choices:
            msg = choice.message
            if getattr(msg, "content", None):
                text_parts.append(msg.content)
            audio_obj = getattr(msg, "audio", None)
            if audio_obj and getattr(audio_obj, "data", None):
                raw = base64.b64decode(audio_obj.data)
                if len(raw) > 80:
                    data, sr = sf.read(io.BytesIO(raw))
                    if data.ndim > 1:
                        data = data[:, 0]
                    audio_out = (int(sr), data.astype(np.float32))

        text_response = "\n\n".join(t for t in text_parts if t) or "(no text returned)"
        yield text_response, audio_out
    except Exception as exc:  # noqa: BLE001
        yield f"Inference failed: {type(exc).__name__}: {exc}", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "你好，请用一句话介绍北京。",
    "Describe what you see in the image.",
    "Transcribe the audio and translate it into English.",
    "What is happening in this video?",
]


def build_interface(endpoints: dict[str, ModelEndpoint], default_model: str) -> gr.Blocks:
    model_choices = list(endpoints.keys())

    css = """
    #generate-btn button { width: 100%; font-size: 1.05rem; }
    .media-row > div { min-height: 260px; }
    .model-banner { color: #555; font-size: 0.9rem; }
    """

    with gr.Blocks(css=css, title="vLLM-Omni · MiniCPM-o Demo") as demo:
        gr.Markdown(
            "# vLLM-Omni · MiniCPM-o Online Demo\n"
            "Multimodal chat (text / image / audio / video in; text + speech out) for "
            "MiniCPM-o 4.5 served by vllm-omni."
        )

        with gr.Row():
            model_dd = gr.Dropdown(
                label="Model",
                choices=model_choices,
                value=default_model,
                scale=2,
            )
            endpoint_info = gr.Markdown(
                elem_classes="model-banner",
                value=f"`{endpoints[default_model].api_base}`  →  `{endpoints[default_model].model_path}`",
            )

        with gr.Row():
            prompt_box = gr.Textbox(
                label="Text Prompt",
                placeholder="Ask the model anything. Combine with image / audio / video below.",
                lines=3,
                value=DEFAULT_PROMPTS[0],
                scale=3,
            )
            with gr.Column(scale=1):
                tts_checkbox = gr.Checkbox(label="Generate speech output (TTS)", value=True)
                max_tokens_slider = gr.Slider(label="Max tokens", minimum=32, maximum=4096, value=1024, step=32)
                temperature_slider = gr.Slider(label="Temperature", minimum=0.0, maximum=1.5, value=0.7, step=0.05)

        with gr.Row(elem_classes="media-row"):
            image_input = gr.Image(label="Image (optional)", type="pil", sources=["upload"])
            audio_input = gr.Audio(
                label="Audio (optional)",
                type="numpy",
                sources=["upload", "microphone"],
            )
            video_input = gr.Video(label="Video (optional)", sources=["upload"])

        with gr.Row():
            generate_btn = gr.Button("Generate", variant="primary", elem_id="generate-btn")
            clear_btn = gr.Button("Clear")

        with gr.Row():
            text_output = gr.Textbox(label="Text Output", lines=10, scale=2)
            audio_output = gr.Audio(label="Audio Output", interactive=False, scale=1)

        gr.Examples(
            examples=[
                ["你好，请用一句话介绍北京。"],
                ["Describe what you see in detail."],
                ["请先识别这段语音的内容，再用中文回复说话人。"],
                ["What is happening in this video?"],
            ],
            inputs=[prompt_box],
            label="Quick prompts",
        )

        def _on_model_change(name: str) -> str:
            ep = endpoints[name]
            return f"`{ep.api_base}`  →  `{ep.model_path}`"

        model_dd.change(_on_model_change, inputs=[model_dd], outputs=[endpoint_info])

        def _run(
            model_name: str,
            prompt: str,
            image_pil: Image.Image | None,
            audio_in: Any,
            video_path: str | None,
            enable_tts: bool,
            max_tokens: int,
            temperature: float,
        ):
            ep = endpoints[model_name]
            audio_tuple = process_audio_input(audio_in)
            yield from run_inference(
                ep,
                prompt,
                audio_tuple,
                image_pil,
                video_path,
                enable_tts,
                max_tokens,
                temperature,
            )

        generate_btn.click(
            _run,
            inputs=[
                model_dd,
                prompt_box,
                image_input,
                audio_input,
                video_input,
                tts_checkbox,
                max_tokens_slider,
                temperature_slider,
            ],
            outputs=[text_output, audio_output],
        )

        def _clear():
            return "", None, None, None, "", None

        clear_btn.click(
            _clear,
            outputs=[
                prompt_box,
                image_input,
                audio_input,
                video_input,
                text_output,
                audio_output,
            ],
        )

        demo.queue()
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minicpmo45-api-base", default=os.environ.get("MINICPMO45_API_BASE", "http://localhost:8099/v1"))
    p.add_argument(
        "--minicpmo45-model",
        default=os.environ.get("MINICPMO45_MODEL", "openbmb/MiniCPM-o-4_5"),
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--ssl-certfile",
        default=os.environ.get("GRADIO_SSL_CERTFILE", ""),
        help="Path to TLS certificate (PEM). Browsers require HTTPS to access the microphone.",
    )
    p.add_argument(
        "--ssl-keyfile",
        default=os.environ.get("GRADIO_SSL_KEYFILE", ""),
        help="Path to TLS private key (PEM).",
    )
    p.add_argument(
        "--ssl-verify",
        action="store_true",
        default=os.environ.get("GRADIO_SSL_VERIFY", "0") == "1",
        help="Require browser to verify the certificate (off by default; self-signed certs should stay off).",
    )
    return p.parse_args()


def _ping(api_base: str, timeout: float = 3.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(api_base.rstrip("/").replace("/v1", "/health"), timeout=timeout):
            return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    endpoints: dict[str, ModelEndpoint] = {}

    if args.minicpmo45_api_base and args.minicpmo45_model:
        ok = _ping(args.minicpmo45_api_base)
        status = "reachable" if ok else "not reachable (will error on generate)"
        print(f"[4.5] {args.minicpmo45_api_base}  ({status})")
        endpoints[MINICPMO45] = ModelEndpoint(MINICPMO45, args.minicpmo45_api_base, args.minicpmo45_model)

    if not endpoints:
        raise SystemExit("No endpoints configured. Pass --minicpmo45-api-base/--minicpmo45-model.")

    default_model = next(iter(endpoints.keys()))
    demo = build_interface(endpoints, default_model)

    launch_kwargs: dict = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
    }
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs.update(
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
            ssl_verify=args.ssl_verify,
        )
        print(f"[tls] cert={args.ssl_certfile} key={args.ssl_keyfile} verify={args.ssl_verify}")
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
