# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark the five multiview encoding paths on one prerecorded uint8 array.

The input must be ``(views, frames, height, width, 3)``. Parent mode starts a
fresh process for every run, alternates variant order, performs one warm-up,
and prints median/p95 wall time, CPU time, bytes, and RSS above the loaded input
baseline. Pin the parent to fourteen CPUs (for example with ``taskset``) for the
primary 7-view 720p measurements described in the Cosmos recipe.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

VARIANTS = ("diffusers-serial", "streaming-serial", "streaming-parallel", "http-monolithic", "http-parallel")


def _rss_tree() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    process = psutil.Process()
    return process.memory_info().rss + sum(
        (child.memory_info().rss for child in process.children(recursive=True) if child.is_running()), start=0
    )


def _measure_peak(stop: threading.Event, peak: list[int]) -> None:
    while not stop.wait(0.01):
        peak[0] = max(peak[0], _rss_tree())


def _run_child(args: argparse.Namespace) -> None:
    frames = np.load(args.input, mmap_mode="r")
    if (
        frames.ndim != 5
        or frames.shape[-1] != 3
        or not (frames.dtype == np.uint8 or np.issubdtype(frames.dtype, np.floating))
    ):
        raise ValueError("benchmark input must be uint8 or normalized float (V,F,H,W,3)")
    baseline_rss = _rss_tree()
    peak = [baseline_rss]
    stop = threading.Event()
    monitor = threading.Thread(target=_measure_peak, args=(stop, peak), daemon=True)
    monitor.start()
    cpu_started = time.process_time()
    started = time.perf_counter()
    output_bytes = 0
    queue_wait_seconds = 0.0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if args.variant == "diffusers-serial":
            from diffusers.utils import export_to_video

            for index, camera in enumerate(frames):
                path = root / f"{index}.mp4"
                baseline_camera = camera.astype(np.float32) / 255.0 if camera.dtype == np.uint8 else camera
                export_to_video(list(baseline_camera), str(path), fps=args.fps)
                output_bytes += path.stat().st_size
        elif args.variant.startswith("streaming-"):
            from vllm_omni.diffusion.utils.video_encoding import run_ordered_encoding_jobs, write_imageio_video

            jobs = [(index, camera) for index, camera in enumerate(frames)]

            def encode(job, threads):
                index, camera = job
                path = root / f"{index}.mp4"
                write_imageio_video(camera, path, fps=args.fps, encoder_threads=threads)
                return path.stat().st_size

            sizes, _ = run_ordered_encoding_jobs(jobs, encode, parallel=args.variant.endswith("parallel"))
            output_bytes = sum(sizes)
        elif args.variant == "http-monolithic":
            from vllm_omni.diffusion.utils.video_encoding import VideoEncodingScheduler
            from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes, _PlanarFrameConverter

            scheduler = VideoEncodingScheduler()
            converter = _PlanarFrameConverter(max_workers=8)
            try:
                futures = [
                    scheduler.submit(
                        f"request-{request}",
                        lambda _cancel: _encode_video_bytes(
                            frames.reshape((-1, *frames.shape[2:])),
                            fps=args.fps,
                            video_codec_options={"preset": args.preset, "threads": "0"},
                            frame_converter=converter,
                        ),
                        tokens=scheduler.cpu_count,
                    )
                    for request in range(args.http_requests)
                ]
                results = [future.result() for future in futures]
                output_bytes = sum(len(result.value) for result in results)
                queue_wait_seconds = max(result.queue_wait_seconds for result in results)
            finally:
                scheduler.shutdown()
                converter.shutdown()
        else:
            from vllm_omni.diffusion.utils.video_encoding import (
                VideoEncodingScheduler,
                calculate_encoding_allocation,
                encode_video_segment,
                remux_video_segments,
            )

            scheduler = VideoEncodingScheduler()
            allocation = calculate_encoding_allocation(len(frames), scheduler.cpu_count)
            segment_futures = {request: [] for request in range(args.http_requests)}
            for camera in frames:
                for request in range(args.http_requests):
                    segment_futures[request].append(
                        scheduler.submit(
                            f"request-{request}",
                            lambda cancel, camera=camera: encode_video_segment(
                                camera,
                                fps=args.fps,
                                encoder_threads=allocation.encoder_threads,
                                video_codec_options={"preset": args.preset},
                                cancel_event=cancel,
                            ),
                            tokens=allocation.encoder_threads,
                        )
                    )
            segments_by_request = {}
            try:
                for request, futures in segment_futures.items():
                    results = [future.result() for future in futures]
                    queue_wait_seconds = max(
                        queue_wait_seconds,
                        *(result.queue_wait_seconds for result in results),
                    )
                    segments_by_request[request] = [result.value for result in results]
                remux_futures = [
                    scheduler.submit(
                        f"request-{request}",
                        lambda cancel, segments=segments: remux_video_segments(
                            segments, fps=args.fps, cancel_event=cancel
                        ),
                        tokens=1,
                    )
                    for request, segments in segments_by_request.items()
                ]
                output_bytes = sum(len(future.result().value) for future in remux_futures)
            finally:
                for segments in segments_by_request.values():
                    for segment in segments:
                        segment.close()
                scheduler.shutdown()
    wall = time.perf_counter() - started
    cpu = time.process_time() - cpu_started
    stop.set()
    monitor.join()
    print(
        json.dumps(
            {
                "variant": args.variant,
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "queue_wait_seconds": queue_wait_seconds,
                "output_bytes": output_bytes,
                "peak_rss_above_input": max(0, peak[0] - baseline_rss),
            }
        )
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(np.ceil(percentile * len(ordered))) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--http-requests", type=int, choices=(1, 2), default=1)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.variant:
        _run_child(args)
        return

    measurements = {variant: [] for variant in VARIANTS}
    for iteration in range(args.runs + 1):
        order = VARIANTS[iteration % len(VARIANTS) :] + VARIANTS[: iteration % len(VARIANTS)]
        for variant in order:
            command = [
                sys.executable,
                __file__,
                "--input",
                str(args.input),
                "--fps",
                str(args.fps),
                "--preset",
                args.preset,
                "--http-requests",
                str(args.http_requests),
                "--variant",
                variant,
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            record = json.loads(result.stdout.strip().splitlines()[-1])
            if iteration:
                measurements[variant].append(record)
    report = {}
    for variant, records in measurements.items():
        walls = [record["wall_seconds"] for record in records]
        report[variant] = {
            "median_wall_seconds": statistics.median(walls),
            "p95_wall_seconds": _percentile(walls, 0.95),
            "median_cpu_seconds": statistics.median(record["cpu_seconds"] for record in records),
            "median_queue_wait_seconds": statistics.median(record["queue_wait_seconds"] for record in records),
            "median_output_bytes": statistics.median(record["output_bytes"] for record in records),
            "peak_rss_above_input": max(record["peak_rss_above_input"] for record in records),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
