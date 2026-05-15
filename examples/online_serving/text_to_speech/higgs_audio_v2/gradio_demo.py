# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal Gradio demo for higgs-audio v2.

Plain text in, 24 kHz speech out. Voice cloning / multi-speaker / language
overrides are intentionally NOT exposed here because the v1 server rejects
them with a 4xx; that constraint is documented under
``examples/online_serving/text_to_speech/higgs_audio_v2/README.md``.

Usage:

  python examples/online_serving/text_to_speech/higgs_audio_v2/gradio_demo.py \
      --base-url http://localhost:8094 --port 7861
"""

from __future__ import annotations

import argparse
import io
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8094")
    parser.add_argument("--model", default="higgs_audio_v2")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    try:
        import gradio as gr
        import httpx
        import soundfile as sf
    except ImportError as exc:
        print(
            f"missing dependency: {exc.name}. Install with `pip install gradio httpx soundfile`.",
            file=sys.stderr,
        )
        return 2

    url = args.base_url.rstrip("/") + "/v1/audio/speech"

    def synthesize(text: str, seed: int):
        if not text.strip():
            raise gr.Error("Input text is empty.")
        payload = {
            "model": args.model,
            "input": text,
            "response_format": "wav",
            "max_new_tokens": int(args.max_new_tokens),
            "seed": int(seed),
        }
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, json=payload)
        if resp.status_code != 200:
            raise gr.Error(f"server returned {resp.status_code}: {resp.text[:200]}")
        audio, sr = sf.read(io.BytesIO(resp.content), dtype="int16")
        return int(sr), audio

    with gr.Blocks(title="higgs-audio v2 (vllm-omni)") as demo:
        gr.Markdown(
            "# higgs-audio v2 (vllm-omni)\n"
            "Plain text -> 24 kHz speech. Voice cloning, language overrides, and multi-speaker "
            "tags are rejected by the v1 server."
        )
        with gr.Row():
            text_in = gr.Textbox(label="Input text", lines=3, value="Hello world.")
            seed_in = gr.Number(label="Seed", value=42, precision=0)
        audio_out = gr.Audio(label="Output (WAV @ 24 kHz)", interactive=False, type="numpy")
        gr.Button("Synthesize").click(synthesize, inputs=[text_in, seed_in], outputs=audio_out)

    demo.launch(server_name="0.0.0.0", server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
