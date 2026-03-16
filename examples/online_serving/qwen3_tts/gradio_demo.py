"""Gradio demo for Qwen3-TTS online serving via /v1/audio/speech API.

Uses Gradio's built-in audio component for playback. Supports streaming
(progressive PCM chunks) and non-streaming (full audio download) modes.

For gapless streaming playback, see gradio_fastrtc_demo.py which uses WebRTC.

Supports all 3 task types:
  - CustomVoice: Predefined speaker with optional style instructions
  - VoiceDesign: Natural language voice description
  - Base: Voice cloning from reference audio (upload or URL)

Usage:
    # Start the server first (see run_server.sh), then:
    python gradio_demo.py --api-base http://localhost:8000

    # Or use run_gradio_demo.sh to start both server and demo together.
"""

import argparse
import io

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
from tts_common import (
    PCM_SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    TASK_TYPES,
    add_common_args,
    build_payload,
    fetch_voices,
    on_task_type_change,
    stream_pcm_chunks,
)


def generate_speech(
    api_base: str,
    text: str,
    task_type: str,
    voice: str,
    language: str,
    instructions: str,
    ref_audio: tuple | None,
    ref_audio_url: str,
    ref_text: str,
    x_vector_only: bool,
    response_format: str,
    speed: float,
):
    """Non-streaming: call /v1/audio/speech and return audio."""
    payload = build_payload(
        text,
        task_type,
        voice,
        language,
        instructions,
        ref_audio,
        ref_audio_url,
        ref_text,
        x_vector_only,
        response_format,
        speed,
        stream=False,
    )

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer EMPTY",
                },
            )
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to server at {api_base}. Make sure the vLLM server is running.")

    if resp.status_code != 200:
        raise gr.Error(f"Server error ({resp.status_code}): {resp.text}")

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            error_data = resp.json()
            raise gr.Error(f"Server error: {error_data}")
        except ValueError:
            pass

    try:
        if response_format == "pcm":
            audio_np = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32767.0
            return (PCM_SAMPLE_RATE, audio_np)
        audio_np, sample_rate = sf.read(io.BytesIO(resp.content))
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        return (sample_rate, audio_np.astype(np.float32))
    except Exception as e:
        raise gr.Error(f"Failed to decode audio response: {e}")


def generate_speech_stream(
    api_base: str,
    text: str,
    task_type: str,
    voice: str,
    language: str,
    instructions: str,
    ref_audio: tuple | None,
    ref_audio_url: str,
    ref_text: str,
    x_vector_only: bool,
    response_format: str,
    speed: float,
    chunk_seconds: float = 0.25,
):
    """Streaming: yield individual PCM chunks for progressive playback."""
    payload = build_payload(
        text,
        task_type,
        voice,
        language,
        instructions,
        ref_audio,
        ref_audio_url,
        ref_text,
        x_vector_only,
        response_format,
        speed,
        stream=True,
    )

    min_chunk_samples = int(PCM_SAMPLE_RATE * chunk_seconds)
    # Pre-buffer 2 chunks before first playback to avoid the gap
    # between chunk 1 and chunk 2 during progressive streaming.
    prebuffer_chunks = 2
    pending: list[np.ndarray] = []
    pending_len = 0
    chunks_yielded = 0
    prebuffer: list[np.ndarray] = []

    try:
        for samples in stream_pcm_chunks(api_base, payload):
            float_samples = samples.astype(np.float32) / 32767.0
            pending.append(float_samples)
            pending_len += len(float_samples)
            if pending_len >= min_chunk_samples:
                audio_chunk = np.concatenate(pending)
                pending.clear()
                pending_len = 0

                if chunks_yielded < prebuffer_chunks:
                    prebuffer.append(audio_chunk)
                    chunks_yielded += 1
                    if chunks_yielded == prebuffer_chunks:
                        yield (PCM_SAMPLE_RATE, np.concatenate(prebuffer))
                        prebuffer.clear()
                else:
                    yield (PCM_SAMPLE_RATE, audio_chunk)
        # Flush: prebuffer (if stream was shorter than 2 chunks) + remaining pending
        remaining = prebuffer + pending
        if remaining:
            yield (PCM_SAMPLE_RATE, np.concatenate(remaining))
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to server at {api_base}. Make sure the vLLM server is running.")


def on_stream_change(stream: bool):
    """When streaming is enabled, lock format to PCM and disable speed."""
    if stream:
        return (
            gr.update(value="pcm", interactive=False),
            gr.update(interactive=False),
        )
    return (
        gr.update(value="wav", interactive=True),
        gr.update(interactive=True),
    )


def build_interface(api_base: str, stream_chunk_seconds: float = 0.25):
    """Build the Gradio interface."""
    voices = fetch_voices(api_base)

    css = """
    #generate-btn button { width: 100%; }
    .task-info { padding: 8px 12px; border-radius: 6px;
                 background: #f0f4ff; margin-bottom: 8px; }
    """

    with gr.Blocks(css=css, title="Qwen3-TTS Demo") as demo:
        gr.Markdown("# Qwen3-TTS Online Serving Demo")
        gr.Markdown(f"**Server:** `{api_base}`")

        with gr.Row():
            # Left column: inputs
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here, e.g., Hello, how are you?",
                    lines=4,
                )

                with gr.Row():
                    task_type = gr.Radio(
                        choices=TASK_TYPES,
                        value="CustomVoice",
                        label="Task Type",
                        scale=2,
                    )
                    language = gr.Dropdown(
                        choices=SUPPORTED_LANGUAGES,
                        value="Auto",
                        label="Language",
                        scale=1,
                    )

                voice = gr.Dropdown(
                    choices=voices,
                    value=voices[0] if voices else None,
                    label="Speaker",
                    visible=True,
                )

                instructions = gr.Textbox(
                    label="Instructions",
                    placeholder=("e.g., Speak with excitement / A warm, friendly female voice"),
                    lines=2,
                    visible=True,
                    info="Optional style/emotion instructions",
                )

                ref_audio = gr.Audio(
                    label="Reference Audio (upload for voice cloning)",
                    type="numpy",
                    sources=["upload", "microphone"],
                    visible=False,
                )
                ref_audio_url = gr.Textbox(
                    label="Reference Audio URL",
                    placeholder=("https://example.com/reference.wav (alternative to uploading)"),
                    lines=1,
                    visible=False,
                )
                ref_text = gr.Textbox(
                    label="Reference Audio Transcript",
                    placeholder=("Transcript of the reference audio (optional, improves quality)"),
                    lines=2,
                    visible=False,
                )
                x_vector_only = gr.Checkbox(
                    label="Use x-vector only",
                    value=False,
                    visible=False,
                    info=("Skip reference transcript, use speaker embedding only (lower quality)"),
                )

                with gr.Row():
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "pcm", "aac", "opus"],
                        value="wav",
                        label="Audio Format",
                        scale=1,
                    )
                    speed = gr.Slider(
                        minimum=0.25,
                        maximum=4.0,
                        value=1.0,
                        step=0.05,
                        label="Speed",
                        scale=1,
                    )
                    stream_checkbox = gr.Checkbox(
                        label="Stream output",
                        value=False,
                        info=(
                            "Enable streaming (PCM format, speed disabled). "
                            "For gapless streaming, use gradio_fastrtc_demo.py"
                        ),
                        scale=1,
                    )

                generate_btn = gr.Button(
                    "Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_id="generate-btn",
                )

            # Right column: output
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    streaming=True,
                    autoplay=True,
                )
                gr.Markdown(
                    "### Task Types\n"
                    "- **CustomVoice**: Use a predefined speaker "
                    "(Vivian, Ryan, etc.) with optional style instructions\n"
                    "- **VoiceDesign**: Describe the desired voice in natural "
                    "language (instructions required)\n"
                    "- **Base**: Clone a voice from reference audio "
                    "(upload a file or provide a URL)"
                )

        # Dynamic UI updates
        task_type.change(
            fn=on_task_type_change,
            inputs=[task_type],
            outputs=[
                voice,
                instructions,
                ref_audio,
                ref_audio_url,
                ref_text,
                x_vector_only,
            ],
        )

        stream_checkbox.change(
            fn=on_stream_change,
            inputs=[stream_checkbox],
            outputs=[response_format, speed],
        )

        all_inputs = [
            text_input,
            task_type,
            voice,
            language,
            instructions,
            ref_audio,
            ref_audio_url,
            ref_text,
            x_vector_only,
            response_format,
            speed,
        ]

        def dispatch(stream_enabled, *args):
            if stream_enabled:
                yield from generate_speech_stream(
                    api_base,
                    *args,
                    chunk_seconds=stream_chunk_seconds,
                )
            else:
                yield generate_speech(api_base, *args)

        generate_btn.click(
            fn=dispatch,
            inputs=[stream_checkbox] + all_inputs,
            outputs=[audio_output],
        )

        demo.queue()
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for Qwen3-TTS online serving.")
    add_common_args(parser)
    parser.add_argument(
        "--stream-chunk-seconds",
        type=float,
        default=0.25,
        help=(
            "Seconds of audio to buffer per streaming chunk (default: 0.25). "
            "Lower = smoother but more overhead, higher = chunkier."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Connecting to vLLM server at: {args.api_base}")
    demo = build_interface(
        args.api_base,
        stream_chunk_seconds=args.stream_chunk_seconds,
    )

    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
