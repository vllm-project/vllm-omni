# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Serving-benchmark adapter for native Full-Duplex-Bench replay."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, RequestFuncOutput

from vllm_omni.benchmarks.duplex.full_duplex_bench_dataset import (
    FullDuplexBenchInput,
    FullDuplexBenchSample,
    discover_full_duplex_bench,
)
from vllm_omni.benchmarks.duplex.omni_duplex_eval_dataset import DuplexSample
from vllm_omni.benchmarks.duplex.omni_duplex_eval_media import read_audio_pcm16
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import (
    DEFAULT_INSTRUCTIONS,
    MIX_QUESTION,
    GenerateSampleResult,
    PreparedReferenceAudio,
    PreparedSample,
    write_failed_sample_diagnostics,
    write_sample_result,
)
from vllm_omni.benchmarks.duplex_session_inputs import TURN_DETECTION_NONE
from vllm_omni.benchmarks.duplex_session_metrics import build_duplex_metrics_report
from vllm_omni.experimental.fullduplex.client import PCM16_BYTES_PER_SAMPLE, PCM16_SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class FullDuplexBenchSampleRequest(SampleRequest):
    sample: DuplexSample | None = None
    source_id: str = ""
    version: str = ""
    subset: str = ""
    variant: str = ""
    input_audio: Path | None = None
    input_duration_s: float = 0.0
    response_root: Path = Path("full-duplex-bench-output")
    ref_audio: PreparedReferenceAudio | None = field(default=None, repr=False)
    fps: float = 1.0
    mix: str = MIX_QUESTION
    turn_detection: str = TURN_DETECTION_NONE
    instructions: str | None = DEFAULT_INSTRUCTIONS


def _runner_sample(source: FullDuplexBenchSample, audio: FullDuplexBenchInput) -> DuplexSample:
    # ``generate_sample`` receives a decoded PreparedSample below, so only this
    # object's stable artifact identity is used; its video fields stay empty.
    return DuplexSample(
        id=f"{source.name}--{audio.variant}",
        split=f"v{source.version}_{source.subset}",
        family="fdb",
        task_type=None,
        video=None,
        question_audio=str(audio.path),
        video_duration=audio.duration_s,
    )


def get_full_duplex_bench_samples(args: argparse.Namespace) -> list[SampleRequest]:
    requested_sessions = args.num_prompts
    sessions_per_source = 2 if args.fdb_version == "1.5" else 1
    if requested_sessions and requested_sessions % sessions_per_source:
        raise ValueError("Full-Duplex-Bench v1.5 --num-prompts must be even so input/clean_input pairs stay complete")
    sources = discover_full_duplex_bench(
        args.dataset_path,
        version=args.fdb_version,
        subsets=tuple(args.fdb_subsets) if args.fdb_subsets else None,
        sample_ids=tuple(args.fdb_sample_ids) if args.fdb_sample_ids else None,
        max_per_subset=args.fdb_max_per_subset,
        source_limit=requested_sessions // sessions_per_source if requested_sessions else None,
        shuffle_seed=None if args.disable_shuffle else args.seed,
    )
    if not sources:
        raise ValueError("No Full-Duplex-Bench source samples were selected")

    result_dir = getattr(args, "result_dir", None)
    root = Path(result_dir or "full-duplex-bench-output").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    logger.info("Full-Duplex-Bench response directory: %s", root)
    reference = PreparedReferenceAudio.from_path(args.fdb_ref_audio) if args.fdb_ref_audio else None
    requests: list[SampleRequest] = []
    for source in sources:
        for audio in source.inputs:
            runner_sample = _runner_sample(source, audio)
            requests.append(
                FullDuplexBenchSampleRequest(
                    prompt="",
                    prompt_len=0,
                    expected_output_len=0,
                    request_id=f"{args.request_id_prefix or ''}{len(requests)}",
                    sample=runner_sample,
                    source_id=source.id,
                    version=source.version,
                    subset=source.subset,
                    variant=audio.variant,
                    input_audio=audio.path,
                    input_duration_s=audio.duration_s,
                    response_root=root,
                    ref_audio=reference,
                    turn_detection=args.fdb_turn_detection,
                    instructions=(
                        DEFAULT_INSTRUCTIONS if args.fdb_instructions is None else args.fdb_instructions or None
                    ),
                )
            )
    args.num_prompts = len(requests)
    return requests


def prepare_full_duplex_bench_sample(request: FullDuplexBenchSampleRequest) -> PreparedSample:
    if request.input_audio is None:
        raise ValueError("Full-Duplex-Bench request has no input audio")
    pcm = read_audio_pcm16(request.input_audio)
    if not pcm:
        raise ValueError(f"Full-Duplex-Bench input decoded to no audio: {request.input_audio}")
    reference = request.ref_audio
    return PreparedSample(
        pcm=pcm,
        frames=(),
        duration=len(pcm) / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE),
        ref_audio=reference.data_url if reference is not None else None,
        ref_audio_sha256=reference.sha256 if reference is not None else None,
        mix=MIX_QUESTION,
    )


def attach_full_duplex_bench(sample: SampleRequest, request: RequestFuncInput) -> None:
    if isinstance(sample, FullDuplexBenchSampleRequest):
        setattr(request, "duplex_eval_sample", sample)


def annotate_full_duplex_bench_result(request: FullDuplexBenchSampleRequest, result: GenerateSampleResult) -> None:
    result.metadata.update(
        {
            "dataset": "full-duplex-bench",
            "dataset_version": request.version,
            "source_id": request.source_id,
            "subset": request.subset,
            "variant": request.variant,
            "input_audio": str(request.input_audio),
            "input_duration_s": request.input_duration_s,
        }
    )


def finalize_full_duplex_bench(
    samples: list[SampleRequest], outputs: list[RequestFuncOutput]
) -> dict[str, object] | None:
    rows = [
        (sample, output)
        for sample, output in zip(samples, outputs, strict=True)
        if isinstance(sample, FullDuplexBenchSampleRequest)
    ]
    if not rows:
        return None
    request_metrics: list[dict[str, object]] = []
    session_metrics: list[dict[str, object]] = []
    errors: list[str] = []
    published = 0
    failed_diagnostics = 0
    for sample, output in rows:
        result = getattr(output, "duplex_eval_result", None)
        if not isinstance(result, GenerateSampleResult):
            continue
        if not output.success:
            try:
                write_failed_sample_diagnostics(result)
                failed_diagnostics += 1
            except OSError as exc:
                errors.append(f"Cannot publish failed-session diagnostics for {result.output}: {exc}")
            continue
        request_metrics.extend(result.request_metrics)
        session_metrics.append(result.session_metrics)
        try:
            write_sample_result(result)
            published += 1
        except OSError as exc:
            errors.append(f"Cannot publish {result.output}: {exc}")
    root = rows[0][0].response_root
    report = build_duplex_metrics_report(request_metrics=request_metrics, session_metrics=session_metrics)
    try:
        (root / "duplex_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        errors.append(f"Cannot publish duplex metrics: {exc}")
    return {
        "version": rows[0][0].version,
        "response_root": str(root),
        "source_samples": len({sample.source_id for sample, _ in rows}),
        "sessions": len(rows),
        "published": published,
        "failed_diagnostics": failed_diagnostics,
        "artifact_errors": errors,
    }
