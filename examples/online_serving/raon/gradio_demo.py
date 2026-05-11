"""Gradio demo for Raon-Speech online serving.

Supports all 4 task types:
  - TTS: text → audio via /v1/audio/speech
  - TTS with voice clone: text + reference audio → audio
  - STT: audio → transcribed text via /v1/chat/completions
  - TextQA: text question (+ optional audio) → text answer
  - SpeechChat: audio conversation → text response

Usage:
    # Start the server first, then:
    python gradio_demo.py --api-base http://localhost:8091

    # Or use run_gradio_demo.sh to start both server and demo together.
"""

import argparse
import base64
import io

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None
import httpx
import numpy as np
import soundfile as sf

DEFAULT_API_BASE = "http://localhost:8091"
DEFAULT_MODEL = "KRAFTON/Raon-Speech-9B"

TASK_TTS = "TTS"
TASK_STT = "STT"
TASK_TEXT_QA = "TextQA"
TASK_SPEECH_CHAT = "SpeechChat"

TASKS = [TASK_TTS, TASK_STT, TASK_TEXT_QA, TASK_SPEECH_CHAT]

SYSTEM_PROMPT_STT = "You are a helpful assistant capable of generating speech."
SYSTEM_PROMPT_QA = "You are a helpful assistant."
SYSTEM_PROMPT_CHAT = "You are a helpful assistant capable of generating speech."


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


def decode_wav_bytes(wav_bytes: bytes) -> tuple:
    """Decode WAV bytes from API into (sample_rate, numpy_array) for Gradio."""
    audio_np, sample_rate = sf.read(io.BytesIO(wav_bytes))
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    return (int(sample_rate), audio_np.astype(np.float32))


def run_tts(api_base: str, text: str, ref_audio, ref_text: str) -> tuple[str, tuple | None]:
    """Call /v1/audio/speech for TTS (with optional voice clone)."""
    if not text or not text.strip():
        return "Error: Please enter text to synthesize.", None

    payload: dict = {
        "input": text.strip(),
        "response_format": "wav",
    }

    if ref_audio is not None:
        payload["ref_audio"] = encode_audio_to_base64(ref_audio)
        if ref_text and ref_text.strip():
            payload["ref_text"] = ref_text.strip()
        else:
            return "Error: Voice cloning requires a transcript of the reference audio (ref_text).", None

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
    except httpx.TimeoutException:
        return "Error: Request timed out. The server may be busy.", None
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {api_base}. Is the server running?", None

    if resp.status_code != 200:
        return f"Error ({resp.status_code}): {resp.text}", None

    try:
        audio_out = decode_wav_bytes(resp.content)
        return "", audio_out
    except Exception as e:
        return f"Error decoding audio: {e}", None


def run_chat_completions(api_base: str, model: str, messages: list) -> str:
    """Call /v1/chat/completions and return text response."""
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/chat/completions",
                json={"model": model, "messages": messages, "modalities": ["text"]},
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
    except httpx.TimeoutException:
        return "Error: Request timed out. The server may be busy."
    except httpx.ConnectError:
        return f"Error: Cannot connect to server at {api_base}. Is the server running?"

    if resp.status_code != 200:
        return f"Error ({resp.status_code}): {resp.text}"

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error parsing response: {e}"


def run_stt(api_base: str, model: str, audio_input) -> tuple[str, None]:
    """STT: audio → transcribed text."""
    if audio_input is None:
        return "Error: Please provide an audio input.", None

    audio_b64 = encode_audio_to_base64(audio_input)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_STT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_b64}},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        },
    ]
    text = run_chat_completions(api_base, model, messages)
    return text, None


def run_text_qa(api_base: str, model: str, question: str, audio_input) -> tuple[str, None]:
    """TextQA: text question (+ optional audio) → text answer."""
    if not question or not question.strip():
        return "Error: Please enter a question.", None

    user_content: list[dict] = []
    if audio_input is not None:
        audio_b64 = encode_audio_to_base64(audio_input)
        user_content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}})
    user_content.append({"type": "text", "text": question.strip()})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_QA}]},
        {"role": "user", "content": user_content},
    ]
    text = run_chat_completions(api_base, model, messages)
    return text, None


def run_speech_chat(api_base: str, model: str, audio_input) -> tuple[str, None]:
    """SpeechChat: audio → text response."""
    if audio_input is None:
        return "Error: Please provide an audio input.", None

    audio_b64 = encode_audio_to_base64(audio_input)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_CHAT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": audio_b64}},
            ],
        },
    ]
    text = run_chat_completions(api_base, model, messages)
    return text, None


def on_task_change(task: str):
    """Update component visibility based on selected task."""
    show_text_in = task in (TASK_TTS, TASK_TEXT_QA)
    show_audio_in = task in (TASK_STT, TASK_TEXT_QA, TASK_SPEECH_CHAT)
    show_ref_audio = task == TASK_TTS
    show_audio_out = task == TASK_TTS

    return (
        gr.update(visible=show_text_in),
        gr.update(visible=show_audio_in),
        gr.update(visible=show_ref_audio),
        gr.update(visible=show_audio_out),
    )


def build_interface(api_base: str, model: str):
    """Build the Gradio interface."""

    def dispatch(task, text_in, audio_in, ref_audio, ref_text):
        if task == TASK_TTS:
            err, audio_out = run_tts(api_base, text_in, ref_audio, ref_text)
            return err, audio_out
        elif task == TASK_STT:
            text_out, _ = run_stt(api_base, model, audio_in)
            return text_out, None
        elif task == TASK_TEXT_QA:
            text_out, _ = run_text_qa(api_base, model, text_in, audio_in)
            return text_out, None
        elif task == TASK_SPEECH_CHAT:
            text_out, _ = run_speech_chat(api_base, model, audio_in)
            return text_out, None
        return "Error: Unknown task.", None

    with gr.Blocks(title="Raon-Speech Demo") as demo:
        gr.Markdown("# Raon-Speech Demo")
        gr.Markdown(f"**Server:** `{api_base}` | **Model:** `{model}`")

        with gr.Row():
            task_dropdown = gr.Dropdown(
                choices=TASKS,
                value=TASK_TTS,
                label="Task",
                scale=1,
            )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Enter text to synthesize (TTS) or your question (TextQA)...",
                    lines=4,
                    visible=True,
                )
                audio_input = gr.Audio(
                    label="Audio Input",
                    type="numpy",
                    sources=["upload", "microphone"],
                    visible=False,
                )

                with gr.Accordion("Voice Clone (optional)", open=False, visible=True) as ref_audio_accordion:
                    gr.Markdown(
                        "Upload a reference audio clip to clone its voice. "
                        "Providing the transcript of the reference improves quality."
                    )
                    ref_audio = gr.Audio(
                        label="Reference Audio",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    ref_text = gr.Textbox(
                        label="Reference Audio Transcript",
                        placeholder="Exact transcript of the reference audio...",
                        lines=2,
                    )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                text_output = gr.Textbox(
                    label="Text Output",
                    lines=8,
                    interactive=False,
                )
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    autoplay=True,
                    visible=True,
                )

        task_dropdown.input(
            fn=on_task_change,
            inputs=[task_dropdown],
            outputs=[text_input, audio_input, ref_audio_accordion, audio_output],
            queue=False,
            show_progress="hidden",
        )

        generate_btn.click(
            fn=dispatch,
            inputs=[task_dropdown, text_input, audio_input, ref_audio, ref_text],
            outputs=[text_output, audio_output],
        )

        demo.queue()
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for Raon-Speech")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Share publicly via Gradio")
    args = parser.parse_args()

    print(f"Connecting to vLLM server at: {args.api_base}")
    print(f"Model: {args.model}")
    demo = build_interface(args.api_base, args.model)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
