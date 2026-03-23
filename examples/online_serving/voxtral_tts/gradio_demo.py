"""
- Make sure to install the following for this example to function correctly:
- `pip install -e .`
- `pip install gradio==5.50 mistral_common=1.10.0`

Example use case:

python examples/online_serving/voxtral_tts/gradio_demo.py --host slurm-199-077 --port 8000

"""

import argparse
import io
import json
import logging
import socket
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger()

LOGFORMAT = "%(asctime)s - %(levelname)s - %(message)s"
TIMEFORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, force=True, format=LOGFORMAT, datefmt=TIMEFORMAT)
logger.setLevel(logging.INFO)


_BASE_URL = f"{socket.gethostname()}:7860"


def run_inference(
    voice_name: str,
    text_prompt: str,
    base_url: str,
    model: str,
) -> tuple[int, np.ndarray]:
    """Call /v1/audio/speech and return (sample_rate, audio_array)."""
    text_prompt = text_prompt.strip()
    if not text_prompt:
        raise gr.Error("Please enter a text prompt.")

    payload: dict[str, Any] = {
        "input": text_prompt,
        "model": model,
        "response_format": "wav",
        "voice": voice_name,
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
) -> tuple[str | None, str, str | None, dict[str, Any], str]:
    """
    Called on page load. If ?share_id=... is present, load stored example.
    Returns: (voice_name, text_prompt, output_audio, submit_btn_update, share_link_text)
    """
    if outputs_dir is None:
        return "Neutral_Male", "", None, gr.update(interactive=False), ""

    share_id = None
    if request and request.query_params:
        share_id = request.query_params.get("share_id")

    if not share_id:
        return "Neutral_Male", "", None, gr.update(interactive=False), ""

    meta_path = outputs_dir / f"{share_id}.json"
    if not meta_path.exists():
        logger.warning("No stored example for share_id=%s", share_id)
        return "Neutral_Male", "", None, gr.update(interactive=False), ""

    with open(meta_path) as f:
        meta = json.load(f)

    voice = meta.get("voice_name", "Neutral_Male")
    text_prompt = meta.get("text_prompt", "")
    gen_path = meta.get("generated_audio_path")

    if _BASE_URL:
        share_link = f"http://{_BASE_URL}?share_id={share_id}"
    else:
        share_link = f"Use this query on your current URL: ?share_id={share_id}"

    return voice, text_prompt, gen_path, gr.update(interactive=True), share_link


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

    with gr.Blocks(title="Voxtral TTS", fill_height=True) as demo:
        gr.Markdown("## Voxtral TTS")

        with gr.Row():
            with gr.Column():
                voice_name = gr.Dropdown(
                    ["Neutral_Male", "Neutral_Female", "Cheerful_Female", "Casual_Male", "Casual_Female"],
                    label="Voice",
                    value="Neutral_Male",
                )
                text_prompt = gr.Textbox(
                    label="Text prompt",
                    placeholder="Enter the text you want to synthesize...",
                    lines=4,
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
        def _on_submit(voice: str, text: str):
            assert text.strip() != ""
            sr, audio_array = run_inference(voice, text, base_url, model)
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
            inputs=[voice_name, text_prompt],
            outputs=[output_audio, share_link_box],
        )

        # --- Clear everything and disable submit again ---
        def _on_reset():
            return (
                "Neutral_Male",  # voice_name
                "",  # text_prompt
                None,  # output_audio
                gr.update(interactive=False),  # submit_btn
                "",  # share_link_box
            )

        reset_btn.click(
            fn=_on_reset,
            inputs=[],
            outputs=[voice_name, text_prompt, output_audio, submit_btn, share_link_box],
        )

        demo.load(
            fn=lambda request: _load_from_share(outputs_dir, request),
            inputs=[],
            outputs=[voice_name, text_prompt, output_audio, submit_btn, share_link_box],
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
