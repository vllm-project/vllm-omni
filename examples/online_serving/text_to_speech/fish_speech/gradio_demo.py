"""Gradio demo for Fish Speech S2 Pro online serving via /v1/audio/speech API.

Supports:
  - Text-to-speech synthesis
  - Voice cloning from reference audio (upload or URL)
  - Streaming (progressive PCM chunks) and non-streaming modes

Usage:
    # Start the server first (see run_server.sh), then:
    python gradio_demo.py --api-base http://localhost:8000

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

PCM_SAMPLE_RATE = 44100
DEFAULT_API_BASE = "http://localhost:8000"


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
    ref_audio: tuple | None,
    ref_audio_url: str,
    ref_text: str,
    response_format: str = "wav",
    stream: bool = False,
) -> dict:
    """Build the /v1/audio/speech request payload."""
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    payload: dict = {
        "input": text.strip(),
        "voice": "default",
        "response_format": "pcm" if stream else response_format,
        "stream": stream,
    }

    # Voice cloning: ref_audio takes priority over URL.
    ref_url_stripped = ref_audio_url.strip() if ref_audio_url else ""
    if ref_audio is not None:
        payload["ref_audio"] = encode_audio_to_base64(ref_audio)
    elif ref_url_stripped:
        payload["ref_audio"] = ref_url_stripped

    if "ref_audio" in payload:
        if not ref_text or not ref_text.strip():
            raise gr.Error("Voice cloning requires a transcript of the reference audio.")
        payload["ref_text"] = ref_text.strip()

    return payload


def generate_speech(api_base: str, text: str, ref_audio, ref_audio_url, ref_text, response_format):
    """Non-streaming: call /v1/audio/speech and return full audio."""
    payload = build_payload(text, ref_audio, ref_audio_url, ref_text, response_format, stream=False)

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to server at {api_base}. Is the server running?")

    if resp.status_code != 200:
        raise gr.Error(f"Server error ({resp.status_code}): {resp.text}")

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            raise gr.Error(f"Server error: {resp.json()}")
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
        raise gr.Error(f"Failed to decode audio: {e}")


def generate_speech_stream(api_base, text, ref_audio, ref_audio_url, ref_text, response_format, chunk_seconds=0.5):
    """Streaming: yield PCM chunks for progressive playback."""
    payload = build_payload(text, ref_audio, ref_audio_url, ref_text, response_format, stream=True)

    min_chunk_samples = int(PCM_SAMPLE_RATE * chunk_seconds)
    prebuffer_chunks = 2
    pending: list[np.ndarray] = []
    pending_len = 0
    chunks_yielded = 0
    prebuffer: list[np.ndarray] = []

    try:
        leftover = b""
        with httpx.Client(timeout=300.0) as client:
            with client.stream(
                "POST",
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
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
                    float_samples = np.frombuffer(raw[:usable], dtype=np.int16).copy().astype(np.float32) / 32767.0
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
        remaining = prebuffer + pending
        if remaining:
            yield (PCM_SAMPLE_RATE, np.concatenate(remaining))
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to server at {api_base}. Is the server running?")


def on_stream_change(stream: bool):
    """Lock format to PCM when streaming."""
    if stream:
        return gr.update(value="pcm", interactive=False)
    return gr.update(value="wav", interactive=True)


def build_interface(api_base: str, stream_chunk_seconds: float = 0.5):
    """Build the Gradio interface."""
    with gr.Blocks(title="Fish Speech S2 Pro Demo") as demo:
        gr.Markdown("# Fish Speech S2 Pro - Text to Speech")
        gr.Markdown(f"**Server:** `{api_base}` | **Model:** fishaudio/s2-pro | **Output:** 44.1kHz")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here...",
                    lines=4,
                )

                with gr.Accordion("Voice Cloning (optional)", open=False):
                    gr.Markdown(
                        "Upload or link a short reference audio (10-30s) and provide its transcript to clone the voice."
                    )
                    ref_audio = gr.Audio(
                        label="Reference Audio",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    ref_audio_url = gr.Textbox(
                        label="Reference Audio URL (alternative to upload)",
                        placeholder="https://example.com/reference.wav",
                        lines=1,
                    )
                    ref_text = gr.Textbox(
                        label="Reference Audio Transcript (required for cloning)",
                        placeholder="Exact transcript of the reference audio...",
                        lines=2,
                    )

                with gr.Row():
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "pcm"],
                        value="wav",
                        label="Audio Format",
                        scale=1,
                    )
                    stream_checkbox = gr.Checkbox(
                        label="Stream output",
                        value=False,
                        info="Progressive PCM streaming",
                        scale=1,
                    )

                generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    streaming=True,
                    autoplay=True,
                )
                gr.Markdown(
                    "### About\n"
                    "- **Fish Speech S2 Pro** by FishAudio: 4B dual-AR model\n"
                    "- **Voice cloning**: upload 10-30s reference + transcript\n"
                    "- **Streaming**: real-time PCM output\n"
                    "- **44.1kHz** output via DAC codec"
                )

        stream_checkbox.change(fn=on_stream_change, inputs=[stream_checkbox], outputs=[response_format])

        all_inputs = [text_input, ref_audio, ref_audio_url, ref_text, response_format]

        def dispatch(stream_enabled, *args):
            if stream_enabled:
                yield from generate_speech_stream(api_base, *args, chunk_seconds=stream_chunk_seconds)
            else:
                yield generate_speech(api_base, *args)

        generate_btn.click(fn=dispatch, inputs=[stream_checkbox] + all_inputs, outputs=[audio_output])
        demo.queue()
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for Fish Speech S2 Pro")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Share publicly")
    parser.add_argument("--stream-chunk-seconds", type=float, default=0.5, help="Seconds per streaming chunk")
    args = parser.parse_args()

    print(f"Connecting to vLLM server at: {args.api_base}")
    demo = build_interface(args.api_base, stream_chunk_seconds=args.stream_chunk_seconds)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
