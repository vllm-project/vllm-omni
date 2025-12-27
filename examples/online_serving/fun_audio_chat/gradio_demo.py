# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gradio demo for Fun-Audio-Chat-8B online inference.

Usage:
    # Start the vLLM server first:
    vllm serve FunAudioLLM/Fun-Audio-Chat-8B --omni --port 8091

    # Then run this demo:
    python gradio_demo.py --api-base http://localhost:8091/v1
"""

import argparse
import base64
import io
import os
import random
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from openai import OpenAI

SEED = 42

# Fun-Audio-Chat sampling parameters
SAMPLING_PARAMS = {
    "main": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 2048,
        "detokenize": True,
        "repetition_penalty": 1.05,
        "seed": SEED,
    },
    "crq": {
        "temperature": 0.9,
        "top_k": 50,
        "max_tokens": 4096,
        "seed": SEED,
        "detokenize": False,
        "repetition_penalty": 1.05,
    },
    "cosyvoice": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 4096 * 16,
        "seed": SEED,
        "detokenize": True,
        "repetition_penalty": 1.1,
    },
}

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for Fun-Audio-Chat-8B online inference.")
    parser.add_argument(
        "--model",
        default="FunAudioLLM/Fun-Audio-Chat-8B",
        help="Model name (should match the server model).",
    )
    parser.add_argument(
        "--api-base",
        default="http://localhost:8091/v1",
        help="Base URL for the vLLM API server.",
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="Host/IP for Gradio.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port for Gradio.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the Gradio demo publicly.",
    )
    parser.add_argument(
        "--s2s",
        action="store_true",
        help="Enable S2S mode (speech-to-speech output).",
    )
    return parser.parse_args()


def build_sampling_params_list(s2s_mode: bool = False) -> list[dict]:
    """Build sampling params list for API call."""
    if s2s_mode:
        return [
            dict(SAMPLING_PARAMS["main"]),
            dict(SAMPLING_PARAMS["crq"]),
            dict(SAMPLING_PARAMS["cosyvoice"]),
        ]
    else:
        return [dict(SAMPLING_PARAMS["main"])]


def audio_to_base64_data_url(audio_data: tuple[np.ndarray, int]) -> str:
    """Convert audio (numpy array, sample_rate) to base64 data URL."""
    audio_np, sample_rate = audio_data

    # Convert to int16 format for WAV
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)

    # Write to WAV bytes
    buffered = io.BytesIO()
    sf.write(buffered, audio_np, sample_rate, format="WAV")
    wav_bytes = buffered.getvalue()
    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"


def process_audio_file(audio_file) -> tuple[np.ndarray, int] | None:
    """Normalize Gradio audio input to (np.ndarray, sample_rate)."""
    if audio_file is None:
        return None

    sample_rate = None
    audio_np = None

    def _load_from_path(path_str: str):
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return None
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32), int(sr)

    if isinstance(audio_file, tuple):
        if len(audio_file) == 2:
            first, second = audio_file
            # (sample_rate, np.ndarray)
            if isinstance(first, (int, float)) and isinstance(second, np.ndarray):
                sample_rate = int(first)
                audio_np = second
            # (filepath, (sample_rate, np.ndarray))
            elif isinstance(first, str):
                if isinstance(second, tuple) and len(second) == 2:
                    sr_candidate, data_candidate = second
                    if isinstance(sr_candidate, (int, float)) and isinstance(data_candidate, np.ndarray):
                        sample_rate = int(sr_candidate)
                        audio_np = data_candidate
                if audio_np is None:
                    loaded = _load_from_path(first)
                    if loaded is not None:
                        audio_np, sample_rate = loaded
    elif isinstance(audio_file, str):
        loaded = _load_from_path(audio_file)
        if loaded is not None:
            audio_np, sample_rate = loaded

    if audio_np is None or sample_rate is None:
        return None

    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]

    return audio_np.astype(np.float32), sample_rate


def run_inference(
    client: OpenAI,
    model: str,
    user_prompt: str,
    audio_file,
    s2s_mode: bool = False,
):
    """Run inference using OpenAI API client."""
    if not user_prompt.strip() and audio_file is None:
        return "Please provide a text prompt or audio input.", None

    try:
        # Build message content
        content_list = []

        # Process audio
        audio_data = process_audio_file(audio_file)
        if audio_data is not None:
            audio_url = audio_to_base64_data_url(audio_data)
            content_list.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": audio_url},
                }
            )

        # Add text prompt
        if user_prompt.strip():
            content_list.append(
                {
                    "type": "text",
                    "text": user_prompt,
                }
            )
        else:
            content_list.append(
                {
                    "type": "text",
                    "text": "Please respond to the audio.",
                }
            )

        # Build messages
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful voice assistant that can understand speech and respond naturally.",
                    }
                ],
            },
            {
                "role": "user",
                "content": content_list,
            },
        ]

        # Build sampling params
        sampling_params_list = build_sampling_params_list(s2s_mode)

        # Call API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            extra_body={
                "sampling_params_list": sampling_params_list,
            },
        )

        # Extract outputs
        text_outputs = []
        audio_output = None

        for choice in chat_completion.choices:
            if choice.message.content:
                text_outputs.append(choice.message.content)
            if hasattr(choice.message, "audio") and choice.message.audio:
                # Decode base64 audio
                audio_bytes = base64.b64decode(choice.message.audio.data)
                audio_np, sample_rate = sf.read(io.BytesIO(audio_bytes))
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]
                audio_output = (int(sample_rate), audio_np.astype(np.float32))

        text_response = "\n\n".join(text_outputs) if text_outputs else "No text output."
        return text_response, audio_output

    except Exception as exc:
        return f"Inference failed: {exc}", None


def build_interface(client: OpenAI, model: str, s2s_mode: bool = False):
    """Build Gradio interface."""

    def inference_fn(user_prompt: str, audio_file):
        return run_inference(client, model, user_prompt, audio_file, s2s_mode)

    with gr.Blocks(title="Fun-Audio-Chat Demo") as demo:
        gr.Markdown(
            """
            # 🎤 Fun-Audio-Chat-8B Demo

            Upload an audio file or record audio, then get a response from the model.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                text_input = gr.Textbox(
                    label="Text Prompt (optional)",
                    placeholder="Enter additional instructions...",
                    lines=3,
                )
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                text_output = gr.Textbox(
                    label="Text Response",
                    lines=10,
                    interactive=False,
                )
                if s2s_mode:
                    audio_output = gr.Audio(
                        label="Audio Response",
                        type="numpy",
                        interactive=False,
                    )
                else:
                    audio_output = gr.Textbox(
                        label="Audio Response",
                        value="(S2S mode disabled - text only)",
                        interactive=False,
                        visible=False,
                    )

        # Set up event handler
        if s2s_mode:
            submit_btn.click(
                fn=inference_fn,
                inputs=[text_input, audio_input],
                outputs=[text_output, audio_output],
            )
        else:
            submit_btn.click(
                fn=lambda p, a: (inference_fn(p, a)[0], None),
                inputs=[text_input, audio_input],
                outputs=[text_output, audio_output],
            )

        gr.Markdown(
            """
            ---
            ### Notes
            - Supported audio formats: WAV, MP3, FLAC, OGG
            - Audio will be resampled to 16kHz for processing
            - S2S mode requires the server to be running with S2S stage config
            """
        )

    return demo


def main():
    args = parse_args()

    print(f"[Info] Connecting to API: {args.api_base}")
    print(f"[Info] Model: {args.model}")
    print(f"[Info] S2S mode: {args.s2s}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=args.api_base,
        api_key="EMPTY",  # vLLM doesn't require API key
    )

    # Build and launch interface
    demo = build_interface(client, args.model, args.s2s)
    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
