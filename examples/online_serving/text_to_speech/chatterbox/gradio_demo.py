# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Play Chatterbox PCM deltas through Gradio's native streaming audio player."""

import argparse
import io

import gradio as gr
import httpx
import numpy as np
import pybase64
import soundfile as sf

SAMPLE_RATE = 24000


def main():
    """Connect the browser player to an existing Chatterbox speech server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8091")
    args = parser.parse_args()
    with gr.Blocks(title="Chatterbox speech") as demo:
        gr.Markdown("# Chatterbox Turbo / Original\nEnglish speech; optional reference, no transcript.")
        api = gr.Textbox(value=args.api_base, label="Speech server URL")
        text = gr.Textbox(value="Hello, this is Chatterbox speaking.", label="Text")
        reference = gr.Audio(type="filepath", label="Reference voice (optional)")
        audio = gr.Audio(streaming=True, autoplay=True, label="Generated speech")
        gr.Button("Speak").click(stream_speech, [api, text, reference], audio)
    demo.launch()


def stream_speech(api_base: str, text: str, reference: str | None):
    """Yield independent WAV chunks without replaying previously received PCM."""
    payload = {
        "input": text,
        "voice": "default",
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    if reference is not None:
        # Gradio's audio upload is a WAV file; use an inline URL for API access policy.
        waveform, rate = sf.read(reference, dtype="float32", always_2d=True)
        encoded = io.BytesIO()
        sf.write(encoded, waveform.mean(axis=1), rate, format="WAV")
        payload["ref_audio"] = "data:audio/wav;base64," + pybase64.b64encode(encoded.getvalue()).decode("ascii")
    with httpx.stream("POST", f"{api_base.rstrip('/')}/v1/audio/speech", json=payload, timeout=300.0) as response:
        response.raise_for_status()
        # Native Gradio playback benefits from consistent one-second chunks.
        for chunk in response.iter_bytes(chunk_size=SAMPLE_RATE * 2):
            waveform = np.frombuffer(chunk, dtype="<i2")
            encoded = io.BytesIO()
            sf.write(encoded, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            yield encoded.getvalue()


if __name__ == "__main__":
    main()
