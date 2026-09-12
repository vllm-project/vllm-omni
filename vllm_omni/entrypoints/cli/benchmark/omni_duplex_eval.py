# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio
import concurrent.futures
import json
from pathlib import Path

from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import normalize_exclude_ids
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DEFAULT_DATASET, load_samples
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import (
    CONTENT_FRAME_LIMIT,
    evaluate_sample,
    summarize_scores,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import generate_sample
from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "Hugging Face dataset id, a JSON/JSONL manifest, or a local Hugging Face "
            "dataset directory. A local directory must be a dataset layout (one "
            "data/<config> subfolder per RTD_*/PR_* split) or a single .parquet file; "
            "a bare data/ directory collapses to a single 'train' split and only works "
            "if every row already carries split/family/task_type identity."
        ),
    )
    parser.add_argument("--split", default="all", help="Restrict to one split (e.g. RTD_OCR) or 'all'.")
    parser.add_argument("--family", choices=("all", "rtd", "pr"), default="all")
    parser.add_argument("--media-root")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument(
        "--exclude-ids",
        nargs="*",
        default=None,
        help=(
            "Exclusion list ('<id>' or '<split>/<id>'), applied before --limit. Same "
            "semantics as `vllm bench serve --duplex-eval-exclude-ids`: bare ids are "
            "universal excludes, `split/id` entries only match that split."
        ),
    )


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
    evaluate.add_argument(
        "--judge-modalities",
        nargs="+",
        default=None,
        help=(
            "Request body `modalities` for the judge (e.g. `text`). Omitted by default so "
            "the request body stays byte-identical to single-choice judges."
        ),
    )
    evaluate.add_argument(
        "--content-frame-limit",
        type=int,
        default=CONTENT_FRAME_LIMIT,
        help=(
            "Maximum number of sampled frames for the judge content pass. Defaults to "
            "CONTENT_FRAME_LIMIT so the frame sequence is unchanged for legacy judges."
        ),
    )
    evaluate.add_argument("--window-size", type=float, default=10.0)
    evaluate.add_argument("--allow-invalid-clock", action="store_true")
    evaluate.add_argument("--eval-workers", type=int, default=1)
    evaluate.add_argument("--overwrite", action="store_true")
    summarize = actions.add_parser("summarize")
    summarize.add_argument("--score-root", required=True)
    summarize.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Also write the summary to this path (used by the DFX runner for a deterministic read).",
    )


def _apply_exclude_ids(samples, exclude_ids):
    """Drop excluded samples, mirroring ``DuplexEvalDataset.sample`` semantics.

    ``normalize_exclude_ids`` keeps bare ids and ``split/id`` tokens; a sample is
    dropped when either its bare id or its ``split/id`` key is listed.
    """
    exclude = normalize_exclude_ids(exclude_ids or ())
    if not exclude:
        return list(samples)
    return [sample for sample in samples if f"{sample.split}/{sample.id}" not in exclude and sample.id not in exclude]


def _select_samples(args: argparse.Namespace):
    """Load samples and apply the shared ``exclude -> limit`` ordering.

    ``limit`` is applied *after* exclusion so the standalone CLI matches the
    serving adapter's per-split ``exclude`` then ``limit`` semantics
    (design §5.1 / §8.5 D-2).
    """
    samples = load_samples(
        args.dataset,
        split=args.split,
        family=args.family,
        media_root=args.media_root,
        limit=None,
        ids=args.ids,
    )
    samples = _apply_exclude_ids(samples, getattr(args, "exclude_ids", None))
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]
    return samples


def run(args: argparse.Namespace) -> int:
    if args.action == "summarize":
        summary = summarize_scores(args.score_root)
        if getattr(args, "output_json", None) is not None:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    samples = _select_samples(args)
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
    judge = DuplexJudge(
        args.judge_base_url,
        args.judge_model,
        api_key=args.judge_api_key,
        modalities=getattr(args, "judge_modalities", None),
    )

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
            content_frame_limit=getattr(args, "content_frame_limit", CONTENT_FRAME_LIMIT),
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
