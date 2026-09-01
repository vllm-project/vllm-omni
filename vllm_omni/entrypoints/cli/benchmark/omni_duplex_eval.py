# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio
import concurrent.futures
import json
from pathlib import Path

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DEFAULT_DATASET, load_samples
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import evaluate_sample, summarize_scores
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import generate_sample
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


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
    generate.add_argument("--mix", choices=("question",), default="question")
    generate.add_argument("--pace", choices=("realtime", "as-fast-as-possible"), default="realtime")
    generate.add_argument("--clock", choices=("media",), default="media")
    generate.add_argument("--concurrency", type=int, default=1)
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
    evaluate.add_argument("--eval-workers", type=int, default=1)
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
        if args.concurrency < 1:
            raise ValueError("--concurrency must be at least 1")

        async def generate() -> None:
            semaphore = asyncio.Semaphore(args.concurrency)

            async def generate_one(sample) -> None:
                async with semaphore:
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

            await asyncio.gather(*(generate_one(sample) for sample in samples))

        asyncio.run(generate())
        return 0

    if args.eval_workers < 1:
        raise ValueError("--eval-workers must be at least 1")
    judge = DuplexJudge(args.judge_base_url, args.judge_model, api_key=args.judge_api_key)

    def evaluate_one(sample) -> None:
        response_path = Path(args.response_root) / sample.split / f"{sample.id}.json"
        score_path = Path(args.score_root) / sample.split / f"{sample.id}.json"
        if score_path.exists() and not args.overwrite:
            return
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.eval_workers) as executor:
        list(executor.map(evaluate_one, samples))
    return 0


class OmniDuplexEvalSubcommand(OmniBenchmarkSubcommandBase):
    """Run the Omni-DuplexEval generation or scoring workflow."""

    name = "omni-duplex-eval"
    help = "Generate, evaluate, or summarize Omni-DuplexEval artifacts."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        run(args)
