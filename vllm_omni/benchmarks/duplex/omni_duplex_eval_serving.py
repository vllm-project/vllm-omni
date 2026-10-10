# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving-benchmark adapter for the existing Omni-DuplexEval loader/runner.

Dataset selection and post-timing publication stay here; the session lifecycle
and media-clock semantics are shared with the standalone generate command.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, RequestFuncOutput

from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DEFAULT_DATASET, DuplexSample, load_samples
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import evaluate_sample, summarize_scores
from vllm_omni.benchmarks.duplex.omni_duplex_eval_judge import DuplexJudge
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import (
    GenerateSampleResult,
    PreparedSample,
    prepare_sample,
    write_sample_result,
)
from vllm_omni.benchmarks.duplex_session_metrics import build_duplex_metrics_report

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DuplexEvalEvaluation:
    base_url: str
    model: str
    api_key: str = field(default="EMPTY", repr=False)
    timeout_s: float = 600.0
    video_mode: str = "video_url"
    fps: int = 2
    window_size: float = 10.0
    workers: int = 1


@dataclass
class OmniDuplexEvalSampleRequest(SampleRequest):
    sample: DuplexSample | None = None
    prepared: PreparedSample | None = field(default=None, repr=False)
    response_root: Path = Path("omni-duplex-eval-output")
    ref_audio: str = ""
    fps: float = 1.0
    evaluation: DuplexEvalEvaluation | None = field(default=None, repr=False)


def get_duplex_eval_samples(args: argparse.Namespace) -> list[SampleRequest]:
    samples = load_samples(
        args.dataset_path or DEFAULT_DATASET,
        split=args.duplex_eval_split,
        family=args.duplex_eval_family,
        media_root=args.duplex_eval_media_root,
        ids=args.duplex_eval_ids,
    )
    if not args.disable_shuffle:
        random.Random(args.seed).shuffle(samples)
    if args.num_prompts:
        samples = samples[: args.num_prompts]
    if not samples:
        raise ValueError("No Omni-DuplexEval sessions were selected")
    identities = [(sample.split, sample.id) for sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError("Omni-DuplexEval samples must have unique (split, id) identities")
    for split, sample_id in identities:
        if any(not value or value in {".", ".."} or Path(value).name != value for value in (split, sample_id)):
            raise ValueError("Omni-DuplexEval split/id must be single path components")
    artifact_paths = [
        Path(split) / f"{sample_id}{suffix}" for split, sample_id in identities for suffix in (".json", ".meta.json")
    ]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ValueError("Omni-DuplexEval response and metadata paths must not collide")
    root = Path(args.duplex_eval_response_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Isolate judge inputs for every invocation, including concurrency/QPS
    # sweeps that reuse the same dataset arguments and parent directory.
    root = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    logger.info("Omni-DuplexEval response directory: %s", root)
    evaluation = None
    if args.duplex_eval_evaluate:
        evaluation = DuplexEvalEvaluation(
            base_url=args.duplex_eval_judge_base_url,
            model=args.duplex_eval_judge_model,
            api_key=args.duplex_eval_judge_api_key,
            timeout_s=args.duplex_eval_judge_timeout_s,
            video_mode=args.duplex_eval_judge_video_mode,
            fps=args.duplex_eval_judge_fps,
            window_size=args.duplex_eval_window_size,
            workers=args.duplex_eval_eval_workers,
        )
    requests: list[SampleRequest] = []
    for index, sample in enumerate(samples):
        prepared = prepare_sample(
            sample,
            media_dir=root / sample.split / ".media" / sample.id,
            ref_audio=args.duplex_eval_ref_audio,
            fps=args.duplex_eval_fps,
        )
        requests.append(
            OmniDuplexEvalSampleRequest(
                prompt="",
                prompt_len=0,
                expected_output_len=0,
                request_id=f"{args.request_id_prefix or ''}{index}",
                sample=sample,
                prepared=prepared,
                response_root=root,
                ref_audio=args.duplex_eval_ref_audio,
                fps=args.duplex_eval_fps,
                evaluation=evaluation,
            )
        )
    args.num_prompts = len(requests)
    return requests


def attach_duplex_eval(sample: SampleRequest, request: RequestFuncInput) -> None:
    if isinstance(sample, OmniDuplexEvalSampleRequest):
        setattr(request, "duplex_eval_sample", sample)


def _evaluate_published_samples(
    samples: list[OmniDuplexEvalSampleRequest], root: Path, options: DuplexEvalEvaluation, *, total: int
) -> dict[str, object]:
    score_root = root / "evaluation"
    score_root.mkdir()
    judge = DuplexJudge(options.base_url, options.model, api_key=options.api_key, timeout=options.timeout_s)

    def evaluate_one(request: OmniDuplexEvalSampleRequest) -> str | None:
        sample = request.sample
        assert sample is not None
        relative = Path(sample.split) / f"{sample.id}.json"
        try:
            evaluate_sample(
                sample,
                root / relative,
                score_root / relative,
                judge,
                judge_fps=options.fps,
                judge_video_mode=options.video_mode,
                window_size=options.window_size,
            )
        except Exception as exc:  # noqa: BLE001 - preserve other cases and completed service measurements
            logger.exception("Omni-DuplexEval judge failed for %s", relative)
            return f"{relative}: {exc}"
        return None

    with ThreadPoolExecutor(max_workers=options.workers) as executor:
        errors = [error for error in executor.map(evaluate_one, samples) if error is not None]
    evaluated = len(samples) - len(errors)
    summary = summarize_scores(score_root)
    summary.update(
        status="completed" if evaluated == total else "partial" if evaluated else "failed",
        total=total,
        evaluated=evaluated,
        skipped=total - len(samples),
        failed=len(errors),
        errors=errors,
        judge_model=options.model,
        score_root=str(score_root),
    )
    (score_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def finalize_duplex_eval(samples: list[SampleRequest], outputs: list[RequestFuncOutput]) -> dict[str, object] | None:
    """Publish and optionally score measured sessions, off the timed event-loop path."""
    rows = [
        (sample, output)
        for sample, output in zip(samples, outputs, strict=True)
        if isinstance(sample, OmniDuplexEvalSampleRequest)
    ]
    if not rows:
        return None
    request_metrics: list[dict[str, object]] = []
    session_metrics: list[dict[str, object]] = []
    errors: list[str] = []
    published: list[OmniDuplexEvalSampleRequest] = []
    for sample, output in rows:
        result = getattr(output, "duplex_eval_result", None)
        if not output.success or not isinstance(result, GenerateSampleResult):
            continue
        request_metrics.extend(result.request_metrics)
        session_metrics.append(result.session_metrics)
        try:
            write_sample_result(result)
            published.append(sample)
        except OSError as exc:
            message = f"Cannot publish {result.output}: {exc}"
            errors.append(message)
            logger.exception(message)
    root = rows[0][0].response_root
    report = build_duplex_metrics_report(request_metrics=request_metrics, session_metrics=session_metrics)
    try:
        (root / "duplex_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        errors.append(f"Cannot publish duplex metrics: {exc}")
        logger.exception("Cannot publish duplex metrics")
    summary: dict[str, object] = {"response_root": str(root), "published": len(published), "artifact_errors": errors}
    if (options := rows[0][0].evaluation) is not None:
        try:
            summary["accuracy"] = _evaluate_published_samples(published, root, options, total=len(rows))
        except Exception as exc:  # noqa: BLE001 - post-hoc accuracy must not discard service measurements
            logger.exception("Omni-DuplexEval evaluation failed")
            summary["accuracy"] = {"status": "failed", "error": str(exc)}
    return summary
