#!/usr/bin/env python3
"""Run deterministic text/image/audio/video input smoke requests."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from .client import build_payload, run_stream_request


async def _main(args: argparse.Namespace) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("text_only", "text", None, False, "Reply with one short sentence about vLLM-Omni."),
        ("text_audio", "text", None, True, "Say hello and describe vLLM-Omni in one short sentence."),
    ]
    media_cases = [
        ("image_audio", "image", args.image, "Name the two main shapes and their colors."),
        ("audio_audio", "audio", args.audio, "Describe the sound briefly."),
        ("video_audio", "video", args.video, "Describe the main action in this video briefly."),
    ]
    missing = [modality for _, modality, media, _ in media_cases if media is None]
    if missing and args.require_all_modalities:
        raise SystemExit(f"missing required media inputs: {', '.join(missing)}")
    cases.extend((name, modality, media, True, prompt) for name, modality, media, prompt in media_cases if media)

    timeout = httpx.Timeout(args.timeout)
    records = []
    async with httpx.AsyncClient(timeout=timeout, headers={"Authorization": "Bearer EMPTY"}) as client:
        for name, modality, media, with_audio, prompt in cases:
            payload = build_payload(
                model=args.model,
                prompt=prompt,
                input_modality=modality,
                media=media,
                with_audio=with_audio,
                seed=args.seed,
                thinker_max_tokens=args.thinker_max_tokens,
                talker_max_tokens=args.talker_max_tokens,
            )
            record = await run_stream_request(
                client,
                endpoint=f"{args.base_url.rstrip('/')}/chat/completions",
                payload=payload,
                request_name=name,
                input_modality=modality,
                with_audio=with_audio,
                output_wav=output / f"{name}.wav" if with_audio else None,
            )
            records.append(record)
            print(f"{name}: {'PASS' if record['success'] else 'FAIL'}")

    result = {
        "schema_version": 1,
        "metric_scope": "local_proxy",
        "official_schema_status": "UNRESOLVED",
        "records": records,
        "passed": all(record["success"] for record in records),
    }
    result_path = output / "smoke_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result_path)
    return 0 if result["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8099/v1")
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--video")
    parser.add_argument("--require-all-modalities", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thinker-max-tokens", type=int, default=256)
    parser.add_argument("--talker-max-tokens", type=int, default=256)
    raise SystemExit(asyncio.run(_main(parser.parse_args())))


if __name__ == "__main__":
    main()
