#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generate, evaluate, or summarize Omni-DuplexEval artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import (  # noqa: E402
    DEFAULT_DATASET,
    load_samples,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import (  # noqa: E402
    evaluate_sample,
    summarize_scores,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge  # noqa: E402
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
    evaluate = sub.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--response-root", required=True)
    evaluate.add_argument("--score-root", required=True)
    evaluate.add_argument("--judge-base-url", default="http://127.0.0.1:8000")
    evaluate.add_argument("--judge-model", required=True)
    evaluate.add_argument("--judge-api-key", default="EMPTY")
    evaluate.add_argument("--judge-video-mode", choices=["video_url", "frame-sample"], default="video_url")
    evaluate.add_argument("--judge-fps", type=int, default=2)
    evaluate.add_argument("--window-size", type=float, default=10.0)
    evaluate.add_argument("--allow-invalid-clock", action="store_true")
    evaluate.add_argument("--overwrite", action="store_true")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--score-root", required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        result = summarize_scores(args.score_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
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
    judge = DuplexJudge(args.judge_base_url, args.judge_model, api_key=args.judge_api_key)
    for sample in samples:
        response_path = Path(args.response_root) / sample.split / f"{sample.id}.json"
        score_path = Path(args.score_root) / sample.split / f"{sample.id}.json"
        if score_path.exists() and not args.overwrite:
            continue
        evaluate_sample(
            sample,
            response_path,
            score_path,
            judge,
            judge_fps=args.judge_fps,
            judge_video_mode=args.judge_video_mode,
            window_size=args.window_size,
            allow_invalid_clock=args.allow_invalid_clock,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
