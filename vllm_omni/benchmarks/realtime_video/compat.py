# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy flags and JSON projection over the shared serving benchmark lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .client import BenchmarkError
from .workload import Workload

_WORKLOAD_FLAGS = {
    "--workload": "--dataset-path",
    "--sessions": "--num-prompts",
    "--warmup-sessions": "--num-warmups",
    **{
        f"--{name}": f"--realtime-video-{name}"
        for name in (
            "image",
            "prompt",
            "negative-prompt",
            "num-chunks",
            "camera-pattern",
            "width",
            "height",
            "fps",
            "seed",
            "flow-shift",
            "print-chunks",
            "warmup-num-chunks",
            "warmup-chunks",
            "chunk-slo-ms",
            "target-fps",
            "playback-buffer-chunks",
            "first-chunk-timeout",
            "chunk-timeout",
            "ping-interval",
            "save-video",
        )
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Translate old flags; the normal bench serve parser owns validation."""
    from .serving import build_serve_args

    original = list(sys.argv[1:] if argv is None else argv)
    forwarded = []
    output: str | None = None
    index = 0
    while index < len(original):
        token = original[index]
        flag, equal, value = token.partition("=")
        if flag == "--output-json":
            if not equal:
                index += 1
                if index == len(original):
                    raise SystemExit("--output-json requires a path")
                value = original[index]
            output = value
            path = Path(value)
            forwarded.extend(
                [
                    "--save-result",
                    "--result-dir",
                    str(path.parent),
                    "--result-filename",
                    f"{path.stem}-serve-result.json",
                ]
            )
        else:
            mapped = _WORKLOAD_FLAGS.get(flag, flag)
            forwarded.append(f"{mapped}={value}" if equal else mapped)
        index += 1
    args = build_serve_args(forwarded)
    args.realtime_video_legacy_output = output
    return args


def build_workload(args: argparse.Namespace) -> Workload:
    from .serving import workload_from_args

    return workload_from_args(args)


def endpoint_url(args: argparse.Namespace) -> str:
    base = (args.base_url or f"http://{args.host}:{args.port}").rstrip("/")
    base = base.removesuffix("/v1/realtime/video")
    for scheme, websocket in (("http://", "ws://"), ("https://", "wss://")):
        if base.startswith(scheme):
            base = websocket + base[len(scheme) :]
            break
    return base + "/v1/realtime/video"


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Compatibility projection only: common serve owns all request execution."""
    from vllm_omni.benchmarks.serve import main_async

    result = await main_async(args)
    document = result["realtime_video"]
    if not document.get("passed", False):
        raise BenchmarkError(str(document.get("errors", "Realtime video measurement failed")))
    if output := getattr(args, "realtime_video_legacy_output", None):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return document


def main(argv: list[str] | None = None) -> int:
    import asyncio

    try:
        asyncio.run(run_benchmark(parse_args(argv)))
    except (BenchmarkError, OSError, ValueError) as error:
        print(f"Realtime video benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0
