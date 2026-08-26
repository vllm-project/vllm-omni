# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CLI implementation shared by the example and ``vllm-omni bench``."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .omni_duplex_eval_dataset import DEFAULT_DATASET, load_samples
from .omni_duplex_eval_eval import evaluate_sample, summarize_scores
from .omni_duplex_eval_judge import DuplexJudge
from .omni_duplex_eval_runner import generate_sample


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="all")
    parser.add_argument("--family", choices=("all", "rtd", "pr"), default="all")
    parser.add_argument("--media-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="action", required=True)
    generate = actions.add_parser("generate")
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
    evaluate = actions.add_parser("evaluate")
    _common(evaluate)
    evaluate.add_argument("--response-root", required=True)
    evaluate.add_argument("--score-root", required=True)
    evaluate.add_argument("--judge-base-url", default="http://127.0.0.1:8000")
    evaluate.add_argument("--judge-model", required=True)
    evaluate.add_argument("--judge-api-key", default="EMPTY")
    evaluate.add_argument("--judge-video-mode", choices=("video_url", "frame-sample"), default="video_url")
    evaluate.add_argument("--judge-fps", type=int, default=2)
    evaluate.add_argument("--window-size", type=float, default=10.0)
    evaluate.add_argument("--allow-invalid-clock", action="store_true")
    evaluate.add_argument("--overwrite", action="store_true")
    summarize = actions.add_parser("summarize")
    summarize.add_argument("--score-root", required=True)


def run(args: argparse.Namespace) -> int:
    if args.action == "summarize":
        print(json.dumps(summarize_scores(args.score_root), ensure_ascii=False, indent=2))
        return 0

    samples = load_samples(
        args.dataset,
        split=args.split,
        family=args.family,
        media_root=args.media_root,
        limit=args.limit,
        ids=args.ids,
    )
    if args.action == "generate":

        async def generate() -> None:
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

        asyncio.run(generate())
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_args(parser)
    return run(parser.parse_args(argv))
