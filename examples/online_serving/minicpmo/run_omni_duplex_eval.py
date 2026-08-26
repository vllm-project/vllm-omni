#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate Omni-DuplexEval response artifacts."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import (  # noqa: E402
    DEFAULT_DATASET,
    load_samples,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import generate_sample  # noqa: E402


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="all")
    parser.add_argument("--family", choices=["all", "rtd", "pr"], default="all")
    parser.add_argument("--media-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    _common(generate)
    generate.add_argument("--url", default="ws://localhost:8099/v1/realtime?duplex=1")
    generate.add_argument("--model", required=True)
    generate.add_argument("--ref-audio", required=True)
    generate.add_argument("--response-root", required=True)
    generate.add_argument("--fps", type=float, default=1.0)
    generate.add_argument("--mix", default="question")
    generate.add_argument("--pace", default="realtime")
    generate.add_argument("--clock", default="media")
    generate.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    samples = load_samples(
        args.dataset, split=args.split, family=args.family, media_root=args.media_root, limit=args.limit, ids=args.ids
    )
    if args.command == "generate":

        async def run() -> None:
            for sample in samples:
                await generate_sample(
                    sample,
                    url=args.url,
                    model=args.model,
                    ref_audio=args.ref_audio,
                    output_root=args.response_root,
                    fps=args.fps,
                    mix=args.mix,
                    pace=args.pace,
                    clock=args.clock,
                    overwrite=args.overwrite,
                )

        asyncio.run(run())
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
