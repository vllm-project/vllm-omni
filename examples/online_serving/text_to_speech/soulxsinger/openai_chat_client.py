#!/usr/bin/env python3
"""SoulX-Singer OpenAI-compatible chat client (SVS / SVC).

Sends prompt audio via ``input_audio`` and target accompaniment via
``extra_args['target_audio']`` (server-local path). For integrated preprocess,
also pass ``preprocess_weights_dir`` in ``extra_args``.

Usage:
  python openai_chat_client.py \\
      --prompt-audio /path/on/server/zh_prompt.mp3 \\
      --target-audio /path/on/server/music.mp3 \\
      --preprocess-weights-dir /path/on/server/SoulX-Singer-Preprocess \\
      -o output.wav
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import requests
import soundfile
import torch


def _audio_to_data_url(path: Path) -> str:
    with path.open("rb") as handle:
        data = base64.b64encode(handle.read()).decode("ascii")
    return f"data:audio/mpeg;base64,{data}"


def _save_wav(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    audio = audio.to(torch.float32)
    peak = audio.abs().max().clamp(min=1e-8)
    audio = audio / peak
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio.clamp(-1.0, 1.0).cpu().T.numpy(), sample_rate, subtype="PCM_16")


def _decode_audio_from_response(body: dict) -> tuple[torch.Tensor, int]:
    for choice in body.get("choices", []):
        audio_obj = choice.get("message", {}).get("audio")
        if isinstance(audio_obj, dict) and audio_obj.get("data"):
            data, sr = soundfile.read(
                io.BytesIO(base64.b64decode(audio_obj["data"])),
                dtype="float32",
                always_2d=True,
            )
            return torch.from_numpy(data).transpose(0, 1), sr
    brief = {k: v for k, v in body.items() if k != "choices"}
    raise RuntimeError(f"no audio in response message.audio: {brief}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]
    default_assets = repo_root / "tests" / "assets" / "soulxsinger"

    parser = argparse.ArgumentParser(description="SoulX-Singer online chat client")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--model", default="Soul-AILab/SoulX-Singer")
    parser.add_argument(
        "--prompt-audio",
        default=str(default_assets / "zh_prompt.mp3"),
        help="Prompt vocal audio (path on server if using extra_args, or local for input_audio)",
    )
    parser.add_argument(
        "--target-audio",
        default=str(default_assets / "music.mp3"),
        help="Target accompaniment path on the server (extra_args['target_audio'])",
    )
    parser.add_argument(
        "--prompt-metadata-path",
        default=None,
        help="SVS precomputed prompt metadata.json",
    )
    parser.add_argument(
        "--target-metadata-path",
        default=None,
        help="SVS precomputed target metadata.json",
    )
    parser.add_argument(
        "--audio-path",
        default=None,
        help="SVS prompt vocal wav for precomputed metadata",
    )
    parser.add_argument("--preprocess-weights-dir", default=None)
    parser.add_argument("--output", "-o", default="soulxsinger_out.wav")
    parser.add_argument("--svc", action="store_true", help="Use SVC mode knobs")
    parser.add_argument("--language", default="Mandarin")
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional CFM seed. Omit for non-deterministic sampling.",
    )
    parser.add_argument(
        "--auto-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto pitch shift (default: on, original upstream infer.sh)",
    )
    parser.add_argument(
        "--control",
        default="melody",
        choices=["melody", "score"],
        help="SVS control mode",
    )
    parser.add_argument("--vocal-sep", action="store_true")
    args = parser.parse_args()

    meta_paths = (args.prompt_metadata_path, args.target_metadata_path, args.audio_path)
    if any(meta_paths) and not all(meta_paths):
        print(
            "ERROR: precomputed metadata requires --prompt-metadata-path, "
            "--target-metadata-path, and --audio-path together.",
            file=sys.stderr,
        )
        return 2

    extra_args: dict = {
        "vocal_sep": args.vocal_sep,
        "auto_shift": args.auto_shift,
        "pitch_shift": 0,
    }
    if all(meta_paths):
        extra_args.update(
            {
                "prompt_metadata_path": str(Path(args.prompt_metadata_path).expanduser().resolve()),
                "target_metadata_path": str(Path(args.target_metadata_path).expanduser().resolve()),
                "audio_path": str(Path(args.audio_path).expanduser().resolve()),
            }
        )
        content = [{"type": "text", "text": "soulx-singer"}]
    else:
        prompt_path = Path(args.prompt_audio).expanduser().resolve()
        if not prompt_path.is_file():
            print(f"ERROR: prompt audio not found: {prompt_path}", file=sys.stderr)
            return 2
        extra_args["prompt_audio"] = str(prompt_path)
        extra_args["target_audio"] = str(Path(args.target_audio).expanduser().resolve())
        if args.preprocess_weights_dir:
            extra_args["preprocess_weights_dir"] = str(Path(args.preprocess_weights_dir).expanduser().resolve())
        content = [
            {"type": "text", "text": "soulx-singer"},
            {
                "type": "input_audio",
                "input_audio": {"data": _audio_to_data_url(prompt_path), "format": "mp3"},
            },
        ]
    if not args.svc:
        extra_args["language"] = args.language
        extra_args["control"] = args.control

    payload = {
        "model": args.model,
        "modalities": ["audio"],
        "messages": [{"role": "user", "content": content}],
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "extra_args": extra_args,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    print(f"POST http://localhost:{args.port}/v1/chat/completions")
    response = requests.post(
        f"http://localhost:{args.port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=1800,
    )
    response.raise_for_status()
    audio, sample_rate = _decode_audio_from_response(response.json())
    _save_wav(audio, Path(args.output), sample_rate)
    duration = audio.shape[-1] / sample_rate
    print(f"saved {args.output}  sr={sample_rate}Hz  duration={duration:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
