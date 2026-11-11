# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Realtime-video protocol adapter for the shared serving benchmark lifecycle."""

from __future__ import annotations

import argparse
import copy
import math
import platform
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, RequestFuncOutput

from vllm_omni.benchmarks.realtime_video.client import (
    SessionResult,
    git_revision,
    realtime_video_url,
    resolve_image_reference,
    run_session,
    save_video,
)
from vllm_omni.benchmarks.realtime_video.metrics import performance_metrics
from vllm_omni.benchmarks.realtime_video.workload import (
    DEFAULT_FLOW_SHIFT,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_PROMPT,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    Workload,
    aggregate_metrics,
    compute_metrics,
    default_workload,
    load_workload,
    warmup_workload,
)


@dataclass(frozen=True)
class RealtimeVideoOptions:
    warmup_num_chunks: int | None = None
    warmup_chunks: int = 6
    chunk_slo_ms: float | None = None
    target_fps: float | None = None
    playback_buffer_chunks: int = 1
    first_chunk_timeout: float = 900.0
    chunk_timeout: float = 300.0
    ping_interval: float = 20.0
    print_chunks: bool = False
    save_video: str | None = None
    workload_file: str | None = None
    camera_pattern: str = "forward"
    insecure: bool = False


@dataclass
class RealtimeVideoSampleRequest(SampleRequest):
    workload: Workload | None = None
    options: RealtimeVideoOptions = field(default_factory=RealtimeVideoOptions)


@dataclass
class RealtimeVideoRequestOutput(RequestFuncOutput):
    session_result: SessionResult | None = None


def build_serve_args(argv: list[str], *, workload: Workload | None = None) -> argparse.Namespace:
    """Parse the actual serve CLI for programmatic and compatibility callers."""
    from vllm_omni.entrypoints.cli.benchmark.cli_args import preprocess_serve_args
    from vllm_omni.entrypoints.cli.benchmark.serve import OmniBenchmarkServingSubcommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser()
    OmniBenchmarkServingSubcommand.add_cli_args(parser)
    args = parser.parse_args(["--backend", "realtime-video", "--dataset-name", "realtime-video", *argv])
    if workload is not None:
        args.realtime_video_workload = workload
    preprocess_serve_args(args)
    return args


def workload_from_args(args: argparse.Namespace) -> Workload:
    supplied = getattr(args, "realtime_video_workload", None)
    if supplied is not None:
        if not isinstance(supplied, Workload):
            raise TypeError("realtime_video_workload must be a Workload")
        return supplied
    overrides = {
        "width": args.realtime_video_width,
        "height": args.realtime_video_height,
        "fps": args.realtime_video_fps,
        "seed": args.realtime_video_seed,
        "prompt": args.realtime_video_prompt,
        "flow_shift": args.realtime_video_flow_shift,
        "negative_prompt": args.realtime_video_negative_prompt,
    }
    if args.dataset_path is not None:
        return load_workload(
            args.dataset_path,
            image_resolver=lambda value: resolve_image_reference(
                value if args.realtime_video_image is None else args.realtime_video_image
            ),
            overrides=overrides,
        )
    return default_workload(
        image_reference=resolve_image_reference(args.realtime_video_image),
        num_chunks=args.realtime_video_num_chunks,
        camera_pattern=args.realtime_video_camera_pattern,
        prompt=args.realtime_video_prompt if args.realtime_video_prompt is not None else DEFAULT_PROMPT,
        width=args.realtime_video_width if args.realtime_video_width is not None else DEFAULT_WIDTH,
        height=args.realtime_video_height if args.realtime_video_height is not None else DEFAULT_HEIGHT,
        fps=args.realtime_video_fps if args.realtime_video_fps is not None else DEFAULT_FPS,
        seed=args.realtime_video_seed if args.realtime_video_seed is not None else DEFAULT_SEED,
        flow_shift=args.realtime_video_flow_shift if args.realtime_video_flow_shift is not None else DEFAULT_FLOW_SHIFT,
        negative_prompt=args.realtime_video_negative_prompt,
    )


def get_realtime_video_samples(args: argparse.Namespace) -> list[SampleRequest]:
    workload = workload_from_args(args)
    options = RealtimeVideoOptions(
        warmup_num_chunks=args.realtime_video_warmup_num_chunks,
        warmup_chunks=args.realtime_video_warmup_chunks,
        chunk_slo_ms=args.realtime_video_chunk_slo_ms,
        target_fps=args.realtime_video_target_fps,
        playback_buffer_chunks=args.realtime_video_playback_buffer_chunks,
        first_chunk_timeout=args.realtime_video_first_chunk_timeout,
        chunk_timeout=args.realtime_video_chunk_timeout,
        ping_interval=args.realtime_video_ping_interval,
        print_chunks=args.realtime_video_print_chunks,
        save_video=args.realtime_video_save_video,
        workload_file=args.dataset_path,
        camera_pattern=args.realtime_video_camera_pattern,
        insecure=args.insecure,
    )
    return [
        RealtimeVideoSampleRequest(
            prompt="",
            prompt_len=0,
            expected_output_len=0,
            workload=workload,
            options=options,
            request_id=f"{args.request_id_prefix or ''}{index}",
        )
        for index in range(args.num_prompts)
    ]


def bind_realtime_video(sample: SampleRequest, request: RequestFuncInput) -> RequestFuncInput:
    if not isinstance(sample, RealtimeVideoSampleRequest):
        return request
    bound = copy.copy(request)
    setattr(bound, "video_sample", sample)
    setattr(bound, "measured", True)
    return bound


def make_realtime_video_warmup(request: RequestFuncInput) -> RequestFuncInput:
    sample = getattr(request, "video_sample", None)
    if not isinstance(sample, RealtimeVideoSampleRequest):
        return request
    assert sample.workload is not None
    warmup = copy.copy(request)
    warmup_sample = copy.copy(sample)
    warmup_sample.workload = warmup_workload(sample.workload, sample.options.warmup_num_chunks)
    setattr(warmup, "measured", False)
    setattr(warmup, "video_sample", warmup_sample)
    return warmup


async def async_request_realtime_video(
    request_func_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Any = None,
) -> RealtimeVideoRequestOutput:
    """Execute one WS rollout; the shared benchmark owns scheduling and timing."""
    sample = getattr(request_func_input, "video_sample", None)
    if not isinstance(sample, RealtimeVideoSampleRequest):
        raise TypeError("realtime-video requires its typed dataset sample")
    assert sample.workload is not None
    options = sample.options
    output = RealtimeVideoRequestOutput(start_time=time.perf_counter())

    def retain_partial(result: SessionResult, error: BaseException | None) -> None:
        output.session_result = result

    url = realtime_video_url(request_func_input.api_url)
    measured = getattr(request_func_input, "measured", True)
    try:
        output.session_result = await run_session(
            url,
            request_func_input.model,
            sample.workload,
            first_chunk_timeout=options.first_chunk_timeout,
            chunk_timeout=options.chunk_timeout,
            ping_interval=options.ping_interval,
            collect_media=measured and options.save_video is not None,
            print_chunks=options.print_chunks and measured,
            on_failure=retain_partial,
            extra_headers=request_func_input.extra_headers,
            insecure=options.insecure,
        )
        output.success = True
    except Exception as error:
        # A failed request remains a failed central result, with its partial raw
        # session retained. Cancellation still propagates through the scheduler.
        output.error = f"{type(error).__name__}: {error}"
    finally:
        output.latency = time.perf_counter() - output.start_time
        if pbar is not None:
            pbar.update(1)
    return output


def _session_payload(output: RealtimeVideoRequestOutput, index: int, *, measured: bool) -> dict[str, Any]:
    result = output.session_result
    return {
        "session_index": index,
        "measured": measured,
        "success": output.success,
        "error": output.error,
        "request_id": result.request_id if result is not None else None,
        "video_start_ms": result.video_start_s * 1000.0
        if result is not None and result.video_start_s is not None
        else None,
        "ttfc_ms": result.ttfc_s * 1000.0 if result is not None and result.ttfc_s is not None else None,
        "wall_seconds": result.wall_s if result is not None else output.latency,
        "chunks": [record.to_dict() for record in result.records] if result is not None else [],
        "events": result.events if result is not None else [],
        "done_event": result.done_event if result is not None else {},
    }


def finalize_realtime_video(
    requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    warmups: list[RequestFuncOutput],
    *,
    duration: float,
    endpoint: str,
    model: str,
) -> dict[str, Any]:
    """Assemble video metrics after the shared measured interval has finished."""
    sample = requests[0]
    assert isinstance(sample, RealtimeVideoSampleRequest) and sample.workload is not None
    workload, options = sample.workload, sample.options
    target_fps = options.target_fps if options.target_fps is not None else float(workload.fps)
    session_payloads: list[dict[str, Any]] = []
    warmup_payloads: list[dict[str, Any]] = []
    session_metrics: list[dict[str, Any]] = []
    finalization_errors: list[str] = []
    for index, output in enumerate(outputs):
        assert isinstance(output, RealtimeVideoRequestOutput)
        payload = _session_payload(output, index, measured=True)
        if output.success and output.session_result is not None:
            result = output.session_result
            try:
                metrics = compute_metrics(
                    result.records,
                    fps=target_fps,
                    warmup_chunks=options.warmup_chunks,
                    slo_ms=options.chunk_slo_ms,
                    playback_buffer_chunks=options.playback_buffer_chunks,
                    ttfc_s=result.ttfc_s,
                    session_wall_s=result.wall_s,
                )
                payload["metrics"] = metrics
                session_metrics.append(metrics)
            except Exception as error:
                # Keep the raw measurement available to the standard writer,
                # while making an invalid metric document visibly fail.
                message = f"session {index} metrics: {type(error).__name__}: {error}"
                payload["metrics_error"] = message
                finalization_errors.append(message)
        session_payloads.append(payload)
    for index, output in enumerate(warmups):
        assert isinstance(output, RealtimeVideoRequestOutput)
        warmup_payloads.append(_session_payload(output, index, measured=False))
    completed = sum(output.success for output in outputs)
    failed_warmups = sum(not output.success for output in warmups)
    if options.save_video is not None and outputs:
        last = outputs[-1]
        assert isinstance(last, RealtimeVideoRequestOutput)
        if last.success and last.session_result is not None:
            try:
                save_video(last.session_result.media, Path(options.save_video), fps=workload.fps)
            except Exception as error:
                finalization_errors.append(f"save video: {type(error).__name__}: {error}")
    aggregate: dict[str, Any] = {}
    if session_metrics:
        try:
            aggregate = aggregate_metrics(session_metrics, fps=target_fps)
        except Exception as error:
            finalization_errors.append(f"aggregate metrics: {type(error).__name__}: {error}")
    passed = completed == len(requests) and not failed_warmups and not finalization_errors
    document = {
        "benchmark": "lingbot_world_realtime",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": realtime_video_url(endpoint),
        "model": model,
        "git_revision": git_revision(),
        "hostname": platform.node(),
        "workload": workload.describe(),
        "config": {
            "sessions": len(requests),
            "warmup_sessions": len(warmups),
            "warmup_num_chunks": options.warmup_num_chunks,
            "warmup_chunks": options.warmup_chunks,
            "camera_pattern": options.camera_pattern,
            "chunk_slo_ms": options.chunk_slo_ms,
            "target_fps": target_fps,
            "mux_fps": workload.fps,
            "playback_buffer_chunks": options.playback_buffer_chunks,
            "workload_file": options.workload_file,
            "insecure": options.insecure,
        },
        "warmups": warmup_payloads,
        "sessions": session_payloads,
        "metrics": aggregate,
        "passed": passed,
        "completed": completed,
        "failed": len(requests) - completed,
        "failed_warmups": failed_warmups,
        "finalization_errors": finalization_errors,
        "errors": [payload["error"] for payload in [*warmup_payloads, *session_payloads] if not payload["success"]]
        + finalization_errors,
    }
    print(f"Realtime video: {completed}/{len(requests)} measured sessions completed; {failed_warmups} failed warmups")
    result_json: dict[str, Any] = {
        "duration": duration,
        "completed": completed,
        "failed": len(requests) - completed,
        "dataset_name": "realtime-video",
        "realtime_video": document,
    }
    if completed == len(requests) and not failed_warmups:
        try:
            scalars = {
                # These are full measured requests, including connection/setup,
                # distinct from steady chunk cadence. Warmups enter neither.
                "mean_e2el_ms": statistics.mean(output.latency for output in outputs) * 1000.0,
                "request_throughput": completed / duration,
                "mean_video_ttfc_ms": statistics.mean(row["ttfc_ms"] for row in session_metrics),
                "min_video_fps": min(row["frames_per_second"] for row in session_metrics),
                "mean_video_rtf": statistics.mean(row["video_rtf"] for row in session_metrics),
            }
            steady_per_session = max(0, workload.num_chunks - max(1, options.warmup_chunks) - 1)
            if steady_per_session:
                summary = performance_metrics(
                    document, num_chunks=workload.num_chunks, min_steady=len(requests) * steady_per_session
                )
                scalars.update(
                    mean_video_steady_interval_ms=summary["mean_ms"],
                    p95_video_steady_interval_ms=summary["p95_ms"],
                )
            if any(not math.isfinite(value) or value < 0 for value in scalars.values()):
                raise ValueError("Nonfinite or negative realtime-video summary")
            result_json.update(scalars)
            result_json["video_metric_scope"] = (
                "e2el/request_throughput: whole measured rollouts; video TTFC/FPS/RTF: session records; "
                "steady intervals: attention warmup and terminal excluded; session warmups excluded throughout"
            )
        except Exception as error:
            message = f"recorded metrics: {type(error).__name__}: {error}"
            finalization_errors.append(message)
            document["errors"].append(message)
            document["passed"] = False
    return result_json
