# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AuK speech generation and audio editing through /v1/audio/speech.

See the online TTS user guide for checkpoint preparation and serving. Local
reference audio is sent as a data URL; it need not exist on the server.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path

import httpx


def reference_audio(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://localhost:8091")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", required=True, help="Model name or assembled bundle path used by the server")
    parser.add_argument(
        "--instructions",
        required=True,
        help=(
            "Complete AuK instruction after filling the cookbook template "
            "(https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md)"
        ),
    )
    parser.add_argument("--ref-audio", help="Reference/source audio: local file, HTTP URL or data URL")
    parser.add_argument("--duration-seconds", type=float, help="Target duration; defaults to source length with audio")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--sway", type=float)
    parser.add_argument("--t-grid", type=float, nargs="+", help="Explicit increasing Euler time grid")
    parser.add_argument("--vae-sample", action="store_true", help="Sample the VAE posterior instead of taking its mean")
    parser.add_argument("--stream", action="store_true", help="Receive raw PCM over HTTP after full generation")
    parser.add_argument("--response-format", choices=["wav", "pcm", "flac", "mp3", "opus"], default="wav")
    parser.add_argument("--output", type=Path, help="Output file; defaults to output.wav or output.pcm")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.duration_seconds is None and args.ref_audio is None:
        parser.error("--duration-seconds is required without --ref-audio")

    response_format = "pcm" if args.stream else args.response_format
    payload = {
        "model": args.model,
        # OpenAI's shared Speech schema still declares input, while AuK's
        # canonical request carries all content in the complete instruction.
        "input": "",
        "voice": "default",
        "response_format": response_format,
    }
    for name in ("duration_seconds", "seed"):
        value = getattr(args, name)
        if value is not None:
            payload[name] = value
    payload["instructions"] = args.instructions
    if args.ref_audio:
        payload["ref_audio"] = reference_audio(args.ref_audio)
    extra = {
        name: getattr(args, name)
        for name in ("num_inference_steps", "guidance_scale", "sway", "t_grid")
        if getattr(args, name) is not None
    }
    if args.vae_sample:
        extra["vae_sample"] = True
    if extra:
        payload["extra_params"] = extra
    if args.stream:
        payload.update(stream=True, stream_format="audio")

    output = args.output or Path(f"output.{response_format}")
    headers = {"Authorization": f"Bearer {args.api_key}"}
    with httpx.Client(timeout=args.timeout) as client:
        with client.stream(
            "POST", f"{args.api_base.rstrip('/')}/v1/audio/speech", json=payload, headers=headers
        ) as response:
            if response.is_error:
                response.read()
                raise RuntimeError(f"Speech API returned {response.status_code}: {response.text}")
            with output.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    print(f"Saved {output} ({output.stat().st_size} bytes). Native audio: 24000 Hz, mono.")
    if response_format == "pcm":
        print("PCM format: signed 16-bit little-endian, 24000 Hz, mono.")


if __name__ == "__main__":
    main()
