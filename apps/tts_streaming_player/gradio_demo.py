# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Play a speech server's streamed PCM through Gradio's native streaming audio player."""

import argparse
import functools
import io

import gradio as gr
import httpx
import numpy as np
import pybase64
import soundfile as sf


def main():
    """Connect the browser player to a running ``/v1/audio/speech`` server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default=None, help="Served model id; omitted for a single-model server")
    parser.add_argument("--voice", default="default", help="Voice name the server accepts")
    parser.add_argument("--sample-rate", type=int, default=24000, help="PCM sample rate of the served model")
    args = parser.parse_args()
    with gr.Blocks(title="Speech streaming player") as demo:
        gr.Markdown(
            "# Speech streaming player\nPlays PCM as the server streams it; a reference upload clones that voice."
        )
        api = gr.Textbox(value=args.api_base, label="Speech server URL")
        text = gr.Textbox(value="Hello, this is a streamed sentence.", label="Text")
        reference = gr.Audio(type="filepath", label="Reference voice (optional)")
        audio = gr.Audio(streaming=True, autoplay=True, label="Generated speech")
        # A partial of the generator function keeps it one for Gradio, which
        # streams a handler only when inspect.isgeneratorfunction says so.
        gr.Button("Speak").click(
            functools.partial(stream_speech, model=args.model, voice=args.voice, sample_rate=args.sample_rate),
            [api, text, reference],
            audio,
        )
    demo.launch()


def stream_speech(api_base: str, text: str, reference: str | None, *, model: str | None, voice: str, sample_rate: int):
    """Yield independent WAV chunks without replaying previously received PCM."""
    payload = {
        "input": text,
        "voice": voice,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    if model is not None:
        payload["model"] = model
    if reference is not None:
        # Gradio's audio upload is a WAV file; use an inline URL for API access policy.
        waveform, rate = sf.read(reference, dtype="float32", always_2d=True)
        encoded = io.BytesIO()
        sf.write(encoded, waveform.mean(axis=1), rate, format="WAV")
        payload["ref_audio"] = "data:audio/wav;base64," + pybase64.b64encode(encoded.getvalue()).decode("ascii")
    with httpx.stream("POST", f"{api_base.rstrip('/')}/v1/audio/speech", json=payload, timeout=300.0) as response:
        response.raise_for_status()
        # Native Gradio playback benefits from consistent one-second chunks.
        for chunk in response.iter_bytes(chunk_size=sample_rate * 2):
            waveform = np.frombuffer(chunk, dtype="<i2")
            encoded = io.BytesIO()
            sf.write(encoded, waveform, sample_rate, format="WAV", subtype="PCM_16")
            yield encoded.getvalue()


if __name__ == "__main__":
    main()
