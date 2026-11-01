# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert music to a score and optionally prepare a YuE2 speech request.

Run this preprocessing tool in the separate environment documented in
recipes/m-a-p/SheetSage2-H200.md. It uses the upstream transcriber, not an Omni
engine, and does not import vLLM or change the serving environment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_MODEL = "m-a-p/SheetSage2"
DEFAULT_REVISION = "cafc0df1021e14f49e928c4b345f5959d414ef64"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Local music recording (decoded by FFmpeg)")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SheetSage2 Hub ID or local snapshot")
    parser.add_argument("--revision", help="Model/code revision; the default Hub model uses a pinned commit")
    parser.add_argument("--base-model-path", type=Path, help="Local MERT-v2 parent for offline loading")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow the upstream model's custom Python code"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument(
        "--dtype", choices=("bf16", "fp32"), default="bf16", help="Inference autocast; weights load in FP32"
    )
    parser.add_argument("--melody-only", action="store_true", help="Retain vocal/instrumental melodies and omit chords")
    parser.add_argument(
        "--max-seconds", type=float, help="Transcribe only this many seconds; otherwise process the song"
    )
    parser.add_argument("--lyrics-file", type=Path, help="With --style, write yue2_request.json for /v1/audio/speech")
    parser.add_argument("--style", help="Target musical arrangement for the optional YuE2 request")
    parser.add_argument("--yue2-model", default="m-a-p/YuE2-3B", help="Served model name in the optional request")
    parser.add_argument("--seed", type=int, help="Optional YuE2 generation seed")
    return parser


def validate_args(args: argparse.Namespace) -> str | None:
    if not args.audio.is_file():
        raise ValueError(f"Audio file does not exist: {args.audio}")
    if not args.trust_remote_code:
        raise ValueError(
            "SheetSage2 uses custom model code; pass --trust-remote-code after reviewing the model repository"
        )
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("--max-seconds must be finite and positive")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise ValueError("--output-dir must be a new or empty directory")
    if args.base_model_path is not None and not args.base_model_path.is_dir():
        raise ValueError(f"MERT-v2 directory does not exist: {args.base_model_path}")
    if (args.lyrics_file is None) != (args.style is None):
        raise ValueError("--lyrics-file and --style must be supplied together")
    if args.lyrics_file is None:
        if args.seed is not None:
            raise ValueError("--seed applies to the YuE2 request; supply --lyrics-file and --style")
        return None
    lyrics = args.lyrics_file.read_text(encoding="utf-8").strip()
    if not lyrics or not args.style.strip():
        raise ValueError("Lyrics and --style must not be blank")
    if not args.yue2_model.strip():
        raise ValueError("--yue2-model must not be blank")
    if args.seed is not None and not 0 <= args.seed < 2**63:
        raise ValueError("--seed must be between 0 and 2**63 - 1")
    return lyrics


def build_yue2_request(abc: str, lyrics: str, args: argparse.Namespace) -> dict[str, object]:
    request: dict[str, object] = {
        "model": args.yue2_model,
        "input": lyrics,
        "instructions": args.style.strip(),
        "response_format": "wav",
        "stream": False,
        "extra_params": {"cot": "melody" if args.melody_only else "full", "abc": abc},
    }
    if args.seed is not None:
        request["seed"] = args.seed
    return request


def transcribe(args: argparse.Namespace) -> Path:
    # Validate before loading dependencies/weights or touching an existing output.
    lyrics = validate_args(args)
    import torch
    from transformers import AutoModel

    if Path(args.model).is_dir():
        # Transformers 4.45 only copies direct imports for local snapshots.
        # Prime transitive dependencies too (e.g. chord_spelling_sheetsage2),
        # using its cache API without modifying the upstream source files.
        # The pinned snapshot keeps the entry point and all relative imports
        # as sibling .py files; local snapshots must preserve that flat layout.
        from transformers.dynamic_module_utils import get_cached_module_file, get_relative_import_files

        source = Path(args.model) / "modeling_sheetsage2.py"
        for dependency in get_relative_import_files(str(source)):
            get_cached_module_file(args.model, Path(dependency).name, local_files_only=True)

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = args.device
    if device == "auto":
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    revision = args.revision
    if revision is None and args.model == DEFAULT_MODEL:
        revision = DEFAULT_REVISION
    load_options = {
        "trust_remote_code": args.trust_remote_code,
        "revision": revision,
        "code_revision": revision,
        "local_files_only": args.local_files_only,
        # Upstream merges the MERT adapters before any reduced-precision cast.
        "torch_dtype": torch.float32,
    }
    if args.base_model_path is not None:
        load_options["base_model_path"] = str(args.base_model_path)
    model = AutoModel.from_pretrained(args.model, **load_options).eval().to(device)
    if model.config.model_type != "sheetsage2":
        raise ValueError("--model must identify a SheetSage2 checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = model.transcribe(
        str(args.audio),
        output_dir=str(args.output_dir),
        melody_only=args.melody_only,
        dtype=args.dtype,
        max_seconds=args.max_seconds,
    )
    abc = result.get("abc")
    if result.get("abc_error") or not isinstance(abc, str) or not abc.strip():
        raise RuntimeError(
            f"ABC export failed: {result.get('abc_error') or 'empty score'}; no YuE2 request was written"
        )
    midi = result.get("midi")
    if not isinstance(midi, bytes) or not midi.startswith(b"MThd"):
        raise RuntimeError("MIDI export failed; no YuE2 request was written")
    if lyrics is not None:
        request = build_yue2_request(abc, lyrics, args)
        (args.output_dir / "yue2_request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "model": args.model,
        "revision": revision,
        "base_model_revision": getattr(model.config, "base_model_revision", None),
        "audio": str(args.audio.resolve()),
        "melody_only": args.melody_only,
        "max_seconds": args.max_seconds,
        "device": str(device),
        "inference_dtype": args.dtype,
        "torch_version": torch.__version__,
        "backend": "upstream SheetSage2 transcribe",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return args.output_dir


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = transcribe(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"SheetSage2: {exc}\nSee recipes/m-a-p/SheetSage2-H200.md for the isolated environment setup.\n")
    print(f"Saved score.abc, transcription.mid and annotations to {output_dir.resolve()}")
    if args.lyrics_file is not None:
        print(f"YuE2 request: {output_dir.resolve() / 'yue2_request.json'}")


if __name__ == "__main__":
    main()
