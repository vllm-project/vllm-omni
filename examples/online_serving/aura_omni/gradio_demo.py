# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradio demo for AURA Omni online serving."""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None

import numpy as np
import soundfile as sf
from openai import OpenAI

from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    DEFAULT_QWEN3_TTS_REF_TEXT,
    default_qwen3_tts_ref_audio_path,
)

SEED = 42
DEFAULT_TTS_REF_AUDIO = default_qwen3_tts_ref_audio_path()


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for AURA Omni online inference.")
    parser.add_argument("--model", default="aurateam/AURA", help="Model name served by vLLM.")
    parser.add_argument("--api-base", default="http://localhost:8091/v1", help="OpenAI-compatible API base.")
    parser.add_argument("--ip", default="127.0.0.1", help="Gradio host.")
    parser.add_argument("--port", type=int, default=7862, help="Gradio port.")
    parser.add_argument("--share", action="store_true", help="Share the Gradio demo publicly.")
    return parser.parse_args()


def _audio_to_data_url(audio_file: Any | None) -> str | None:
    if audio_file is None:
        return None

    def _path_to_data_url(path_str: str) -> str | None:
        if not path_str:
            return None
        path = Path(path_str)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('utf-8')}"

    if isinstance(audio_file, str):
        return _path_to_data_url(audio_file)

    # Gradio >=5 may return dict payloads with local temp path.
    if isinstance(audio_file, dict):
        path_val = audio_file.get("path") or audio_file.get("name")
        if isinstance(path_val, str):
            return _path_to_data_url(path_val)
        return None

    if isinstance(audio_file, tuple):
        # Common shape: (sample_rate, np.ndarray)
        if len(audio_file) == 2 and isinstance(audio_file[0], (int, float)):
            sample_rate, audio_np = audio_file
            audio_np = np.asarray(audio_np)
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            if audio_np.dtype != np.int16:
                audio_np = np.clip(audio_np.astype(np.float32), -1.0, 1.0)
            buf = io.BytesIO()
            sf.write(buf, audio_np, int(sample_rate), format="WAV")
            return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        # Compatibility shape from some gradio builds: (filepath, (sr, np.ndarray))
        if len(audio_file) >= 1 and isinstance(audio_file[0], str):
            path_url = _path_to_data_url(audio_file[0])
            if path_url is not None:
                return path_url
    return None


def _video_to_data_url(video_file: str | None) -> str | None:
    if not video_file:
        return None
    path = Path(video_file)
    mime = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(path.suffix.lower(), "video/mp4")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def _sampling_params_list() -> list[dict[str, Any]]:
    return [
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 256, "seed": SEED},
        {
            "temperature": 0.5,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 256,
            "seed": SEED,
            "repetition_penalty": 1.0,
        },
        {
            "temperature": 0.9,
            "top_k": 50,
            "max_tokens": 4096,
            "seed": SEED,
            "detokenize": False,
            "repetition_penalty": 1.05,
            "stop_token_ids": [2150],
        },
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 65536,
            "seed": SEED,
            "repetition_penalty": 1.0,
        },
    ]


def _decode_audio_choice(audio_data: str | None) -> tuple[int, np.ndarray] | None:
    if not audio_data:
        return None
    wav_bytes = base64.b64decode(audio_data)
    audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes))
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    return int(sample_rate), audio_np.astype(np.float32)


def build_interface(client: OpenAI, model: str):
    def run(
        audio_file,
        video_file,
        prompt: str,
        aura_system_prompt: str,
        tts_task_type: str,
        tts_language: str,
        tts_speaker: str,
        tts_instruct: str,
        tts_ref_audio: str,
        tts_ref_text: str,
        tts_x_vector_only_mode: bool,
        tts_pass_token_ids: bool,
    ):
        audio_url = _audio_to_data_url(audio_file)
        video_url = _video_to_data_url(video_file)
        if not audio_url:
            return "Please provide audio input for the ASR stage.", None
        if tts_task_type == "Base" and (not (tts_ref_audio or "").strip() or not (tts_ref_text or "").strip()):
            return "Base TTS requires both reference audio and reference transcript.", None
        content: list[dict[str, Any]] = [{"type": "audio_url", "audio_url": {"url": audio_url}}]
        if video_url:
            content.append({"type": "video_url", "video_url": {"url": video_url}})
        content.append(
            {"type": "text", "text": prompt or "Use the audio and video together to decide whether a reply is needed."}
        )

        additional_information: dict[str, Any] = {
            "aura_system_prompt": aura_system_prompt,
            "tts_task_type": tts_task_type,
            "tts_instruct": tts_instruct,
            "tts_pass_token_ids": bool(tts_pass_token_ids),
        }
        if tts_task_type == "CustomVoice":
            additional_information.update(
                {
                    "tts_language": tts_language or "English",
                    "tts_speaker": tts_speaker or "Vivian",
                }
            )
        else:
            additional_information.update(
                {
                    "tts_ref_audio": tts_ref_audio.strip() if tts_ref_audio else None,
                    "tts_ref_text": tts_ref_text.strip() if tts_ref_text else None,
                    "tts_x_vector_only_mode": bool(tts_x_vector_only_mode),
                }
            )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            modalities=["text", "audio"],
            extra_body={
                "sampling_params_list": _sampling_params_list(),
                "additional_information": additional_information,
            },
            timeout=600.0,
        )

        text_parts: list[str] = []
        audio_output = None
        for choice in response.choices:
            message = choice.message
            if message.content:
                text_parts.append(str(message.content))
            if getattr(message, "audio", None):
                audio_output = _decode_audio_choice(message.audio.data)
        return "\n\n".join(text_parts) or "No text output.", audio_output

    default_system = (
        "You are receiving a live video stream where the final frame is the present moment. "
        "Respond only when a response is needed. Otherwise output '<|silent|>'. Respond in English."
    )

    def _toggle_tts_advanced(task_type: str):
        is_base = task_type == "Base"
        return (
            gr.update(visible=not is_base),
            gr.update(visible=is_base),
        )

    with gr.Blocks() as demo:
        gr.Markdown("# AURA Omni")
        gr.Markdown("ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Code2Wav")
        with gr.Row():
            audio_input = gr.Audio(label="Audio input", sources=["upload", "microphone"], type="numpy")
            video_input = gr.Video(label="Video input (optional)", sources=["upload"])
        prompt = gr.Textbox(
            label="Instruction",
            value="Use the audio and video together to decide whether a reply is needed. If needed, respond briefly in English.",
            lines=2,
        )
        with gr.Accordion("Advanced", open=False):
            aura_system_prompt = gr.Textbox(label="AURA system prompt", value=default_system, lines=4)
            tts_task_type = gr.Radio(choices=["Base", "CustomVoice"], value="Base", label="TTS task type")
            tts_instruct = gr.Textbox(label="TTS instruction", value="")
            tts_pass_token_ids = gr.Checkbox(
                label="Pass AURA token ids directly to TTS",
                value=False,
            )

            with gr.Group(visible=False) as customvoice_config_group:
                tts_language = gr.Dropdown(
                    choices=["Chinese", "English", "Japanese", "Korean", "Cantonese"],
                    value="Chinese",
                    label="CustomVoice language",
                )
                tts_speaker = gr.Textbox(label="CustomVoice speaker", value="Vivian")

            with gr.Group(visible=True) as base_config_group:
                tts_ref_audio = gr.Textbox(
                    label="Base reference audio path/URL",
                    value=DEFAULT_TTS_REF_AUDIO,
                )
                tts_ref_text = gr.Textbox(
                    label="Base reference transcript",
                    value=DEFAULT_QWEN3_TTS_REF_TEXT,
                    lines=3,
                )
                tts_x_vector_only_mode = gr.Checkbox(
                    label="Base x-vector only mode (disable ICL)",
                    value=False,
                )

            tts_task_type.change(
                _toggle_tts_advanced,
                inputs=[tts_task_type],
                outputs=[customvoice_config_group, base_config_group],
            )
        button = gr.Button("Generate", variant="primary")
        with gr.Row():
            text_output = gr.Textbox(label="Text output", lines=8)
            audio_output = gr.Audio(label="Audio output", interactive=False)
        button.click(
            run,
            inputs=[
                audio_input,
                video_input,
                prompt,
                aura_system_prompt,
                tts_task_type,
                tts_language,
                tts_speaker,
                tts_instruct,
                tts_ref_audio,
                tts_ref_text,
                tts_x_vector_only_mode,
                tts_pass_token_ids,
            ],
            outputs=[text_output, audio_output],
        )
        demo.queue()
    return demo


def main() -> None:
    args = parse_args()
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)
    demo = build_interface(client, args.model)
    demo.launch(server_name=args.ip, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
