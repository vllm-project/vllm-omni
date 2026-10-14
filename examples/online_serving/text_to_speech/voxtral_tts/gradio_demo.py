"""
- Make sure to install the following for this example to function correctly:
- `pip install -e .`
- `pip install gradio==5.50 mistral_common=1.10.0`

Example use case:

python examples/online_serving/text_to_speech/voxtral_tts/gradio_demo.py --host slurm-199-077 --port 8000

"""

import argparse
import io
import json
import logging
import socket
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None
import httpx
import numpy as np
import soundfile as sf
from text_preprocess import sanitize_tts_input_text_for_demo

logger = logging.getLogger()

LOGFORMAT = "%(asctime)s - %(levelname)s - %(message)s"
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, force=True, format=LOGFORMAT, datefmt=TIMEFORMAT)
logger.setLevel(logging.INFO)


_BASE_URL = f"{socket.gethostname()}:7860"

# Default fallback voices - comprehensive list
_DEFAULT_VOICES = [
    "casual_female",
    "casual_male",
    "cheerful_female",
    "neutral_female",
    "neutral_male",
    "ar_male",
    "de_female",
    "de_male",
    "es_female",
    "es_male",
    "fr_female",
    "fr_male",
    "hi_female",
    "hi_male",
    "it_female",
    "it_male",
    "nl_female",
    "nl_male",
    "pt_female",
    "pt_male",
]


def organize_voices_by_language(voices: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Organize voices into language categories.

    Args:
        voices: List of voice names (e.g., ["neutral_male", "es_female", "fr_male"])

    Returns:
        Tuple of (sorted list of language categories, dictionary mapping language to voices)
    """
    # Define language prefixes and their display names
    LANGUAGE_PREFIXES = {
        "ar": "Arabic",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "nl": "Dutch",
        "pt": "Portuguese",
        "hi": "Hindi",
    }

    # Initialize language to voices mapping
    language_voices: dict[str, list[str]] = {}

    for voice in voices:
        # Check for language prefix
        found_language = None
        for prefix, lang_name in LANGUAGE_PREFIXES.items():
            if voice.lower().startswith(f"{prefix}_"):
                found_language = lang_name
                break

        if found_language:
            # Add voice to its language category
            if found_language not in language_voices:
                language_voices[found_language] = []
            language_voices[found_language].append(voice)
        else:
            # Add to English (voices without language prefix)
            if "English" not in language_voices:
                language_voices["English"] = []
            language_voices["English"].append(voice)

    # Sort voices within each language category, with neutral_male first for English
    for lang in language_voices:
        if lang == "English":
            language_voices[lang].sort(key=lambda v: (0 if v == "neutral_male" else 1, v))
        else:
            language_voices[lang].sort()

    # Sort language categories (English first, then alphabetically)
    sorted_languages = sorted(language_voices.keys(), key=lambda x: (0 if x == "English" else 1, x.lower()))

    return sorted_languages, language_voices


# Configuration for server health check
_SERVER_CHECK_TIMEOUT = 300.0  # 5 minutes max wait
_SERVER_CHECK_INTERVAL = 5.0  # 5 seconds between retries


def wait_for_server(base_url: str, timeout: float = _SERVER_CHECK_TIMEOUT) -> bool:
    """Block until the server is available or timeout is reached.

    Args:
        base_url: Base URL of the server (e.g., "http://localhost:8091/v1")
        timeout: Maximum time to wait in seconds

    Returns:
        True if server became available, False if timeout reached
    """
    start_time = time.time()
    # Health endpoint is at /health (not /v1/health)
    health_url = base_url.replace("/v1", "") + "/health"

    logger.info(f"Waiting for server at {base_url} to become available...")

    with httpx.Client(timeout=5.0) as client:
        while time.time() - start_time < timeout:
            try:
                resp = client.get(health_url)
                if resp.status_code == 200:
                    logger.info("Server is now available!")
                    return True
            except Exception:
                pass
            elapsed = time.time() - start_time
            logger.info(f"Server not yet available ({elapsed:.1f}s elapsed), retrying in {_SERVER_CHECK_INTERVAL}s...")
            time.sleep(_SERVER_CHECK_INTERVAL)

    logger.warning(f"Server did not become available within {timeout}s timeout")
    return False


def fetch_voices_and_languages(base_url: str, model: str) -> tuple[list[str], dict[str, list[str]]]:
    """Fetch available voices from the server API and organize by language.

    This function blocks until the server is available, then fetches the list
    of available voices from the /v1/audio/voices endpoint and organizes them.

    Args:
        base_url: Base URL of the server API (e.g., "http://localhost:8091/v1")
        model: Model name (used for logging)

    Returns:
        Tuple of (sorted language list, dictionary mapping language to voices)
    """
    # Always wait for server - if server is not up, demo cannot do anything
    if not wait_for_server(base_url):
        logger.warning("Server unavailable, using fallback voices")
        languages, language_voices = organize_voices_by_language(_DEFAULT_VOICES)
        return languages, language_voices

    try:
        with httpx.Client(timeout=10.0) as client:
            # The audio/voices endpoint is directly under /v1, not nested
            resp = client.get(
                f"{base_url}/audio/voices",
                headers={"Authorization": "Bearer EMPTY"},
            )
        if resp.status_code == 200:
            data = resp.json()
            voices = data.get("voices", [])
            if voices:
                logger.info(f"Fetched {len(voices)} voices from server: {voices}")
                languages, language_voices = organize_voices_by_language(sorted(voices))
                return languages, language_voices
            logger.warning("Server returned empty voices list")
    except Exception as e:
        logger.warning(f"Failed to fetch voices from server: {e}")

    # Fallback to default voices
    logger.info(f"Using fallback voices: {_DEFAULT_VOICES}")
    languages, language_voices = organize_voices_by_language(_DEFAULT_VOICES)
    return languages, language_voices


def make_update_voice_dropdown(language_voices: dict[str, list[str]]):
    """Return a callback that updates the voice dropdown when the user
    selects a different language."""

    def update_voice_dropdown(language: str) -> gr.Dropdown:
        voices = language_voices.get(language, [])
        return gr.Dropdown(choices=voices, value=voices[0] if voices else None, interactive=True)

    return update_voice_dropdown


def run_inference(
    voice_name: str,
    text_prompt: str,
    cfg_alpha: float,
    base_url: str,
    model: str,
) -> tuple[int, np.ndarray]:
    """Call /v1/audio/speech and return (sample_rate, audio_array)."""
    user_text_prompt = text_prompt.strip()
    if not user_text_prompt:
        raise gr.Error("Please enter a text prompt.")
    try:
        text_prompt = sanitize_tts_input_text_for_demo(user_text_prompt)
    except Exception as exc:
        raise gr.Error(f"Text preprocessing failed: {exc}") from exc

    payload: dict[str, Any] = {
        "input": text_prompt,
        "model": model,
        "response_format": "wav",
        "voice": voice_name,
        "extra_params": {"cfg_alpha": cfg_alpha},
    }

    response = httpx.post(
        f"{base_url}/audio/speech",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()

    audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    return sr, audio_array


def _save_example(
    outputs_dir: Path,
    voice_name: str,
    text_prompt: str,
    sr: int,
    audio_array: np.ndarray,
) -> tuple[str, str]:
    """
    Save inputs/outputs for sharing.
    Returns (share_id, saved_audio_path)
    """
    share_id = uuid.uuid4().hex

    # Save generated audio
    saved_audio_path = outputs_dir / f"{share_id}_gen.wav"
    sf.write(str(saved_audio_path), audio_array, sr)

    meta = {
        "id": share_id,
        "created_at": datetime.utcnow().isoformat(),
        "voice_name": voice_name,
        "text_prompt": text_prompt,
        "generated_audio_path": str(saved_audio_path),
    }

    with open(outputs_dir / f"{share_id}.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    return share_id, str(saved_audio_path)


def _load_from_share(
    outputs_dir: Path | None,
    request: gr.Request,
    languages: list[str] | None = None,
    language_voices: dict[str, list[str]] | None = None,
) -> tuple[str, str, str | None, str | None, dict[str, Any], str]:
    """
    Called on page load. If ?share_id=... is present, load stored example.
    Returns: (language, voice_name, text_prompt, output_audio, submit_btn_update, share_link_text)
    """
    fallback_language = (
        "English" if languages and "English" in languages else (languages[0] if languages else "English")
    )
    if language_voices:
        voices_list = language_voices.get(fallback_language, [])
        fallback_voice = voices_list[0] if voices_list else None
    else:
        fallback_voice = None

    if outputs_dir is None:
        return fallback_language, fallback_voice, "", None, gr.update(interactive=False), ""

    share_id = None
    if request and request.query_params:
        share_id = request.query_params.get("share_id")

    if not share_id:
        return fallback_language, fallback_voice, "", None, gr.update(interactive=False), ""

    meta_path = outputs_dir / f"{share_id}.json"
    if not meta_path.exists():
        logger.warning("No stored example for share_id=%s", share_id)
        return fallback_language, fallback_voice, "", None, gr.update(interactive=False), ""

    with open(meta_path) as f:
        meta = json.load(f)

    # Determine language and voice from the stored voice name
    voice = meta.get("voice_name", fallback_voice)
    text_prompt = meta.get("text_prompt", "")
    gen_path = meta.get("generated_audio_path")

    # Find which language this voice belongs to
    language = fallback_language
    if language_voices:
        for lang, voices in language_voices.items():
            if voice in voices:
                language = lang
                break

    if _BASE_URL:
        share_link = f"http://{_BASE_URL}?share_id={share_id}"
    else:
        share_link = f"Use this query on your current URL: ?share_id={share_id}"

    return language, voice, text_prompt, gen_path, gr.update(interactive=True), share_link


def main(
    model: str,
    host: str,
    port: str,
    output_dir: str | None = None,
) -> None:
    base_url = f"http://{host}:{port}/v1"
    logger.info(f"Using speech API at: {base_url}/audio/speech")

    outputs_dir: Path | None = None
    if output_dir is not None:
        outputs_dir = Path(output_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load available voices and organize by language
    # This will block until the server is available
    languages, language_voices = fetch_voices_and_languages(base_url, model)

    with gr.Blocks(title="Voxtral TTS", fill_height=True) as demo:
        gr.Markdown("## Voxtral TTS")

        with gr.Row():
            with gr.Column():
                # Language dropdown (first level)
                default_language = "English" if "English" in languages else languages[0]
                language_dropdown = gr.Dropdown(
                    choices=languages,
                    label="Language",
                    value=default_language,
                )
                # Voice dropdown (second level, updates based on language)
                voices_for_default_lang = language_voices.get(default_language, [])
                voice_name = gr.Dropdown(
                    choices=voices_for_default_lang,
                    label="Voice",
                    value=voices_for_default_lang[0] if voices_for_default_lang else None,
                )
                text_prompt = gr.Textbox(
                    label="Text prompt",
                    placeholder="Enter the text you want to synthesize...",
                    lines=4,
                )
                cfg_alpha_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.2,
                    label="CFG Alpha",
                    info="Flow-matching guidance strength (default: 1.2)",
                )
                with gr.Row():
                    reset_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Generate audio", interactive=False)

            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated audio",
                    show_download_button=True,
                    interactive=False,
                    autoplay=True,
                    type="filepath",
                )
                share_link_box = gr.Textbox(
                    label="Shareable link",
                    interactive=False,
                    show_copy_button=True,
                    visible=outputs_dir is not None,
                )

        # --- Cascading dropdown: update voice list when language changes ---
        language_dropdown.change(
            fn=make_update_voice_dropdown(language_voices),
            inputs=[language_dropdown],
            outputs=voice_name,
        )

        # --- UI logic: disable submit until text is non-empty ---
        def _toggle_submit(text: str):
            enabled = bool(text.strip())
            return gr.update(interactive=enabled)

        text_prompt.change(
            fn=_toggle_submit,
            inputs=[text_prompt],
            outputs=submit_btn,
        )

        # --- Wiring inference + persistence to the button ---
        def _on_submit(voice: str, text: str, cfg_alpha: float):
            assert text.strip() != ""
            sr, audio_array = run_inference(voice, text, cfg_alpha, base_url, model)
            if outputs_dir is not None:
                share_id, saved_audio_path = _save_example(
                    outputs_dir,
                    voice_name=voice,
                    text_prompt=text,
                    sr=sr,
                    audio_array=audio_array,
                )
                share_link = f"{_BASE_URL}?share_id={share_id}"
                return saved_audio_path, share_link
            return (sr, audio_array), ""

        submit_btn.click(
            fn=_on_submit,
            inputs=[voice_name, text_prompt, cfg_alpha_slider],
            outputs=[output_audio, share_link_box],
        )

        # --- Clear everything and disable submit again ---
        def make_on_reset(languages: list[str], language_voices: dict[str, list[str]]):
            def _on_reset():
                language = "English" if "English" in languages else languages[0]
                voices_list = language_voices.get(language, [])
                voice = voices_list[0] if voices_list else None
                return (
                    language,  # language_dropdown
                    voice,  # voice_name
                    "",  # text_prompt
                    1.2,  # cfg_alpha_slider
                    None,  # output_audio
                    gr.update(interactive=False),  # submit_btn
                    "",  # share_link_box
                )

            return _on_reset

        reset_btn.click(
            fn=make_on_reset(languages, language_voices),
            inputs=[],
            outputs=[
                language_dropdown,
                voice_name,
                text_prompt,
                cfg_alpha_slider,
                output_audio,
                submit_btn,
                share_link_box,
            ],
        )

        def make_load_from_share(outputs_dir: Path | None, languages: list[str], language_voices: dict[str, list[str]]):
            def _load(request: gr.Request):
                return _load_from_share(outputs_dir, request, languages, language_voices)

            return _load

        demo.load(
            fn=make_load_from_share(outputs_dir, languages, language_voices),
            inputs=[],
            outputs=[language_dropdown, voice_name, text_prompt, output_audio, submit_btn, share_link_box],
        )

    launch_kwargs: dict[str, Any] = {
        "server_name": "0.0.0.0",
        "share": True,
    }
    if outputs_dir is not None:
        launch_kwargs["allowed_paths"] = [str(outputs_dir)]
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxtral TTS Gradio Demo")
    parser.add_argument("--model", type=str, default="mistralai/Voxtral-4B-TTS-2603", help="Name of model repo on HF")
    parser.add_argument("--host", type=str, default="localhost", help="Name of host")
    parser.add_argument("--port", type=str, default="8091", help="port number")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated audio and share links. "
        "If not provided, save/share functionality is disabled.",
    )

    args = parser.parse_args()

    main(
        model=args.model,
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
    )
