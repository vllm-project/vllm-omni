# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving adapter and post-run scoring for continuous visual-session QA."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import regex as re
from vllm.benchmarks.datasets import SampleRequest

from vllm_omni.benchmarks.data_modules.long_video_dataset import VideoQuestion, load_video_questions


@dataclass
class LongVideoSampleRequest(SampleRequest):
    case: VideoQuestion | None = None
    output_dir: Path | None = None
    fps: float = 1.0
    max_edge: int = 336
    timeout_s: float = 180.0


def get_long_video_samples(args: argparse.Namespace) -> list[SampleRequest]:
    from vllm_omni.benchmarks.data_modules.long_video_source import resolve_long_video_source

    annotation, media_root = resolve_long_video_source(
        args.dataset_path, dataset=args.dataset_name, media_root=args.long_video_media_root
    )
    cases = load_video_questions(
        annotation,
        media_root,
        dataset=args.dataset_name,
        ids=args.long_video_ids,
    )
    expected_digest = args.long_video_annotation_sha256
    if expected_digest and any(case.annotation_sha256 != expected_digest for case in cases):
        raise ValueError("Annotation SHA256 differs from the pinned dataset revision")
    if not args.disable_shuffle:
        random.Random(args.seed).shuffle(cases)
    if args.num_prompts:
        cases = cases[: args.num_prompts]
    root = Path(args.long_video_output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    args.num_prompts = len(cases)
    return [
        LongVideoSampleRequest(
            prompt="",
            prompt_len=0,
            expected_output_len=0,
            request_id=f"{args.request_id_prefix or ''}{i}",
            case=case,
            output_dir=root,
            fps=args.long_video_fps,
            max_edge=args.long_video_max_edge,
            timeout_s=args.long_video_timeout_s,
        )
        for i, case in enumerate(cases)
    ]


def attach_long_video(sample, request) -> None:
    if isinstance(sample, LongVideoSampleRequest):
        request.long_video_sample = sample


async def request_long_video(request, output, *, headers, pbar, total_timeout_s: float):
    from vllm_omni.benchmarks.duplex.long_video_runner import run_video_question
    from vllm_omni.clients.duplex import metric_mean

    sample = request.long_video_sample
    output.start_time = time.perf_counter()
    output.tpot_measured = False
    output.long_video_events = []
    try:
        result = await asyncio.wait_for(
            run_video_question(
                sample.case,
                url=request.api_url,
                model=request.model_name or request.model,
                fps=sample.fps,
                max_edge=sample.max_edge,
                timeout_s=sample.timeout_s,
                extra_body=dict(request.extra_body or {}),
                headers=headers,
                events=output.long_video_events,
            ),
            timeout=total_timeout_s,
        )
        output.generated_text = result["text"]
        output.output_tokens = result["output_tokens"]
        output.duplex_request_metrics = result["request_metrics"]
        output.duplex_session_metrics = result["session_metrics"]
        # Serve TTFT covers available responses across the session, including
        # observations; final-question latency is saved separately as query_ttft_ms.
        output.ttft = (metric_mean(result["session_metrics"].get("ttft_ms")) or 0.0) / 1000
        output.success = True
        output.long_video_result = result
    except TimeoutError:
        output.timed_out = True
        output.error = traceback.format_exc()
    except Exception:
        output.error = traceback.format_exc()
    output.latency = time.perf_counter() - output.start_time
    if pbar:
        pbar.update(1)
    return output


def extract_choice(text: str) -> str | None:
    """Require an unambiguous MCQ answer; verbose/invalid replies count as wrong."""
    cleaned = text.replace("</response>", "").strip()
    match = re.fullmatch(r"(?:Answer\s*:\s*)?\(?([ABCD])\)?[.!]?", cleaned, flags=re.IGNORECASE)
    return match[1].upper() if match else None


def finalize_long_video(samples, outputs) -> dict | None:
    rows = []
    root = None
    artifact_errors = []
    for index, (sample, output) in enumerate(zip(samples, outputs, strict=True)):
        if not isinstance(sample, LongVideoSampleRequest):
            continue
        case = sample.case
        root = sample.output_dir
        assert case is not None and root is not None
        result = getattr(output, "long_video_result", {})
        prediction = extract_choice(output.generated_text) if output.success else None
        rows.append(
            {
                "id": case.id,
                "dataset": case.dataset,
                "task": case.task,
                "annotation_sha256": case.annotation_sha256,
                "question": case.question,
                "options": case.options,
                "answer": case.answer,
                "prediction": prediction,
                "correct": bool(output.success and prediction == case.answer),
                "success": output.success,
                "error": output.error,
                "fps": sample.fps,
                "max_edge": sample.max_edge,
                **result,
            }
        )
        try:
            directory = root / case.dataset / f"{index:04d}-{case.id}"
            directory.mkdir(parents=True, exist_ok=True)
            _write_json(directory / "result.json", rows[-1])
            _write_json(directory / "events.json", getattr(output, "long_video_events", []))
            _write_json(directory / (".done" if output.success else ".failed.json"), {"error": output.error})
        except OSError as exc:
            artifact_errors.append(str(exc))
    if not rows:
        return None
    assert root is not None
    summary = {
        "protocol": "continuous-visual-mcq-v1",
        "modalities": ["video", "text"],
        "subtitles": False,
        "samples": len(rows),
        "completed": sum(row["success"] for row in rows),
        "valid_answers": sum(row["prediction"] is not None for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "output_dir": str(root),
        "items": rows,
    }
    if artifact_errors:
        summary["artifact_error"] = "; ".join(dict.fromkeys(artifact_errors))
    try:
        _write_json(root / "accuracy.json", summary)
    except OSError as exc:
        artifact_errors.append(str(exc))
    if artifact_errors:
        summary["artifact_error"] = "; ".join(dict.fromkeys(artifact_errors))
    return summary


def _write_json(path: Path, value) -> None:
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        staged.replace(path)
