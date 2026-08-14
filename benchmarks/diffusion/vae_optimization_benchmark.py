# SPDX-License-Identifier: Apache-2.0
"""Qualify VAE runtime profiles against an identical-seed reference server.

Run this once per server configuration.  A candidate run may point at a JSON
report and media artifacts from the safe/eager profile to enforce latency and
quality gates without changing optimization settings per request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class RunResult:
    index: int
    cold: bool
    http_status: int
    end_to_end_s: float
    inference_s: float
    stage_durations: dict[str, float]
    peak_memory_mb: float
    media_path: str
    sha256: str


def _json_header(response: httpx.Response, name: str) -> dict[str, float]:
    value = response.headers.get(name)
    if not value:
        return {}
    parsed = json.loads(value)
    return {str(key): float(duration) for key, duration in parsed.items()}


def _request(
    client: httpx.Client,
    args: argparse.Namespace,
    *,
    index: int,
    steps: int,
    output_path: Path,
) -> RunResult:
    fields = {
        "prompt": args.prompt,
        "width": str(args.width),
        "height": str(args.height),
        "aspect_ratio": args.aspect_ratio,
        "fps": str(args.fps),
        "num_inference_steps": str(steps),
        "flow_shift": str(args.flow_shift),
        "seed": str(args.seed),
        "extra_params": json.dumps(
            {
                "task": args.task,
                "duration": args.duration,
                "audio_flow_shift": args.audio_flow_shift,
            },
            separators=(",", ":"),
        ),
    }
    started = time.perf_counter()
    multipart: list[tuple[str, tuple[Any, ...]]] = [(key, (None, value)) for key, value in fields.items()]
    references = list(args.input_reference or [])
    reference_field = "input_reference" if len(references) == 1 else "input_references"
    for path in references:
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        multipart.append((reference_field, (path.name, path.read_bytes(), media_type)))
    if args.audio_reference:
        multipart.append(("audio_reference", (None, json.dumps({"audio_url": args.audio_reference}))))
    response = client.post(args.endpoint, files=multipart)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return RunResult(
        index=index,
        cold=index == 0,
        http_status=response.status_code,
        end_to_end_s=elapsed,
        inference_s=float(response.headers.get("x-inference-time-s", elapsed)),
        stage_durations=_json_header(response, "x-stage-durations"),
        peak_memory_mb=float(response.headers.get("x-peak-memory-mb", 0.0)),
        media_path=str(output_path),
        sha256=hashlib.sha256(response.content).hexdigest(),
    )


def _decode_media(path: str) -> tuple[Any, Any, float, int]:
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("media quality gates require PyAV and NumPy") from exc

    video_frames = []
    audio_frames = []
    with av.open(path) as container:
        video_rate = float(container.streams.video[0].average_rate or 0)
        for frame in container.decode(video=0):
            video_frames.append(frame.to_ndarray(format="rgb24"))
    with av.open(path) as container:
        audio_rate = int(container.streams.audio[0].rate) if container.streams.audio else 0
        if container.streams.audio:
            for frame in container.decode(audio=0):
                samples = frame.to_ndarray()
                if samples.ndim == 1:
                    samples = samples[None, :]
                audio_frames.append(samples)
    video = np.stack(video_frames).astype(np.float32)
    audio = np.concatenate(audio_frames, axis=-1).astype(np.float32) if audio_frames else np.empty((0, 0))
    return video, audio, video_rate, audio_rate


def _tile_boundaries(length: int, tile_size: int, overlap_min: int, vae_ratio: int) -> list[int]:
    if tile_size >= length:
        return []
    count = math.ceil(length / tile_size)
    while True:
        overlaps = [overlap_min] * (count - 1)
        remaining = tile_size * count - sum(overlaps) - length
        if remaining >= 0:
            break
        count += 1
    for index in range(remaining // vae_ratio):
        overlaps[index % (count - 1)] += vae_ratio
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + tile_size - overlap)
    return starts[1:]


def compare_media(
    reference: str,
    candidate: str,
    *,
    decoder_tile_size: int = 256,
    decoder_tile_overlap_min: int = 64,
    vae_ratio: int = 16,
    seam_band_width: int = 4,
) -> dict[str, float | int | list[int]]:
    import numpy as np

    ref_video, ref_audio, ref_video_rate, ref_audio_rate = _decode_media(reference)
    cand_video, cand_audio, cand_video_rate, cand_audio_rate = _decode_media(candidate)
    if ref_video.shape != cand_video.shape:
        raise ValueError(f"video shape mismatch: reference={ref_video.shape}, candidate={cand_video.shape}")
    if ref_audio.shape != cand_audio.shape:
        raise ValueError(f"audio shape mismatch: reference={ref_audio.shape}, candidate={cand_audio.shape}")
    if (ref_video_rate, ref_audio_rate) != (cand_video_rate, cand_audio_rate):
        raise ValueError(
            "media rate mismatch: "
            f"reference={(ref_video_rate, ref_audio_rate)}, candidate={(cand_video_rate, cand_audio_rate)}"
        )
    video_delta = ref_video - cand_video
    video_abs = np.abs(video_delta)
    video_mse = float(np.mean(video_delta**2))
    audio_delta = ref_audio - cand_audio
    video_duration_s = float(ref_video.shape[0]) / ref_video_rate if ref_video_rate else 0.0
    audio_duration_s = float(ref_audio.shape[-1]) / ref_audio_rate if ref_audio_rate and ref_audio.size else 0.0
    spatial_error = np.mean(video_abs, axis=(0, 3))
    seam_mask = np.zeros(spatial_error.shape, dtype=bool)
    y_boundaries = _tile_boundaries(int(ref_video.shape[1]), decoder_tile_size, decoder_tile_overlap_min, vae_ratio)
    x_boundaries = _tile_boundaries(int(ref_video.shape[2]), decoder_tile_size, decoder_tile_overlap_min, vae_ratio)
    for boundary in y_boundaries:
        seam_mask[max(0, boundary - seam_band_width) : boundary + seam_band_width, :] = True
    for boundary in x_boundaries:
        seam_mask[:, max(0, boundary - seam_band_width) : boundary + seam_band_width] = True
    seam_mae = float(np.mean(spatial_error[seam_mask])) if np.any(seam_mask) else 0.0
    nonseam_mae = float(np.mean(spatial_error[~seam_mask])) if np.any(~seam_mask) else 0.0
    return {
        "video_frames": int(ref_video.shape[0]),
        "video_rate": ref_video_rate,
        "video_duration_s": video_duration_s,
        "video_mse": video_mse,
        "video_mae": float(np.mean(video_abs)),
        "video_psnr_db": math.inf if video_mse == 0 else 10 * math.log10((255.0**2) / video_mse),
        "video_max_abs": float(np.max(np.abs(video_delta))),
        "video_seam_band_mae": seam_mae,
        "video_nonseam_mae": nonseam_mae,
        "video_seam_excess_ratio": seam_mae / nonseam_mae if nonseam_mae else 1.0,
        "video_tile_y_boundaries": y_boundaries,
        "video_tile_x_boundaries": x_boundaries,
        "audio_shape": list(ref_audio.shape),
        "audio_rate": ref_audio_rate,
        "audio_duration_s": audio_duration_s,
        "av_sync_delta_s": abs(video_duration_s - audio_duration_s) if ref_audio.size else 0.0,
        "audio_mae": float(np.mean(np.abs(audio_delta))) if audio_delta.size else 0.0,
        "audio_max_abs": float(np.max(np.abs(audio_delta))) if audio_delta.size else 0.0,
    }


def _read_log_suffix(path: Path, start_offset: int = 0) -> str:
    with path.open("rb") as stream:
        stream.seek(start_offset)
        return stream.read().decode("utf-8", errors="replace")


def parse_rank_timings(path: Path, start_offset: int = 0) -> dict[str, Any]:
    marker = "[VAE component timing] "
    by_metric: dict[str, dict[int, list[float]]] = {}
    for line in _read_log_suffix(path, start_offset).splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1]
        match = re.match(r"(\{.*?\})(?:\x1b|$)", payload)
        if match is None:
            continue
        event = json.loads(match.group(1))
        metric = str(event["metric"])
        rank = int(event["rank"])
        by_metric.setdefault(metric, {}).setdefault(rank, []).append(float(event["duration_s"]))

    result: dict[str, Any] = {}
    for metric, per_rank in by_metric.items():
        medians = {str(rank): statistics.median(values) for rank, values in per_rank.items()}
        values = list(medians.values())
        imbalance = 0.0 if not values or min(values) == 0 else (max(values) / min(values) - 1.0) * 100.0
        result[metric] = {"median_s_by_rank": medians, "imbalance_pct": imbalance}
    return result


def parse_vae_diagnostics(path: Path, start_offset: int = 0) -> dict[str, Any]:
    """Return the final per-rank decode metadata from a diagnostic server log."""

    marker = "[VAE diagnostics] "
    last_by_rank: dict[str, dict[str, Any]] = {}
    for line in _read_log_suffix(path, start_offset).splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1]
        match = re.match(r"(\{.*?\})(?:\x1b|$)", payload)
        if match is None:
            continue
        event = json.loads(match.group(1))
        last_by_rank[str(int(event["rank"]))] = event
    return {"last_decode_by_rank": last_by_rank}


def _warm_median(report: dict[str, Any], field: str) -> float:
    runs = report["runs"]
    warm = [float(run[field]) for run in runs if not run["cold"]]
    values = warm or [float(run[field]) for run in runs]
    return statistics.median(values)


def _warm_vae_median(report: dict[str, Any]) -> float:
    runs = report["runs"]
    warm = [run for run in runs if not run["cold"]] or runs
    values = [
        float(run.get("stage_durations", {}).get("video_vae.decode_latent", 0.0))
        + float(run.get("stage_durations", {}).get("audio_vae.decode_latent", 0.0))
        for run in warm
    ]
    return statistics.median(values)


def _vae_time(run: RunResult) -> float:
    return float(run.stage_durations.get("video_vae.decode_latent", 0.0)) + float(
        run.stage_durations.get("audio_vae.decode_latent", 0.0)
    )


def _apply_gates(
    report: dict[str, Any],
    reference: dict[str, Any] | None,
    quality: dict[str, Any] | None,
    args: argparse.Namespace,
) -> list[str]:
    failures = []
    if reference is not None:
        reference_s = _warm_median(reference, "end_to_end_s")
        candidate_s = _warm_median(report, "end_to_end_s")
        regression_pct = (candidate_s / reference_s - 1.0) * 100.0
        report["end_to_end_regression_pct"] = regression_pct
        reference_vae_s = _warm_vae_median(reference)
        candidate_vae_s = _warm_vae_median(report)
        if reference_vae_s > 0 and candidate_vae_s > 0:
            vae_share = min(1.0, reference_vae_s / reference_s)
            vae_speedup = reference_vae_s / candidate_vae_s
            report["amdahl"] = {
                "reference_vae_share": vae_share,
                "vae_speedup": vae_speedup,
                "predicted_end_to_end_speedup": 1.0 / ((1.0 - vae_share) + vae_share / vae_speedup),
                "observed_end_to_end_speedup": reference_s / candidate_s,
            }
        if regression_pct > args.max_end_to_end_regression_pct:
            failures.append(
                f"end-to-end regression {regression_pct:.3f}% exceeds {args.max_end_to_end_regression_pct:.3f}%"
            )
        reference_fingerprints = {
            str(rank): str(event["latent_sha256"])
            for rank, event in reference.get("vae_diagnostics", {}).get("last_decode_by_rank", {}).items()
            if event.get("latent_sha256")
        }
        candidate_fingerprints = {
            str(rank): str(event["latent_sha256"])
            for rank, event in report.get("vae_diagnostics", {}).get("last_decode_by_rank", {}).items()
            if event.get("latent_sha256")
        }
        if reference_fingerprints and candidate_fingerprints and reference_fingerprints != candidate_fingerprints:
            failures.append("reference and candidate VAE latent fingerprints differ")
    if quality is not None:
        if float(quality["video_psnr_db"]) < args.min_video_psnr_db:
            failures.append(f"video PSNR {quality['video_psnr_db']:.3f} dB is below {args.min_video_psnr_db:.3f} dB")
        if float(quality.get("video_mae", 0.0)) > args.max_video_mae:
            failures.append(f"video MAE {quality['video_mae']:.6f} exceeds {args.max_video_mae:.6f}")
        if float(quality["audio_mae"]) > args.max_audio_mae:
            failures.append(f"audio MAE {quality['audio_mae']:.8f} exceeds {args.max_audio_mae:.8f}")
        if float(quality["video_seam_band_mae"]) > args.max_video_seam_band_mae:
            failures.append(
                f"video seam-band MAE {quality['video_seam_band_mae']:.8f} exceeds {args.max_video_seam_band_mae:.8f}"
            )
        if float(quality.get("video_seam_excess_ratio", 1.0)) > args.max_video_seam_excess_ratio:
            failures.append(
                f"video seam excess ratio {quality['video_seam_excess_ratio']:.6f} exceeds "
                f"{args.max_video_seam_excess_ratio:.6f}"
            )
        if float(quality.get("av_sync_delta_s", 0.0)) > args.max_av_sync_delta_s:
            failures.append(
                f"audio/video duration delta {quality['av_sync_delta_s']:.6f}s exceeds {args.max_av_sync_delta_s:.6f}s"
            )
    rank_timings = report.get("rank_timings", {})
    video_ranks = rank_timings.get("video_vae.decode_latent")
    if video_ranks and float(video_ranks["imbalance_pct"]) > args.max_rank_imbalance_pct:
        failures.append(
            f"video VAE rank imbalance {video_ranks['imbalance_pct']:.3f}% exceeds {args.max_rank_imbalance_pct:.3f}%"
        )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/videos/sync")
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--prompt", default="Three cats play tiny brass instruments in a bedroom at night.")
    parser.add_argument("--task", default="t2va")
    parser.add_argument("--input-reference", action="append", type=Path)
    parser.add_argument("--audio-reference")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--postcheck-steps", type=int, default=2)
    parser.add_argument("--max-end-to-end-regression-pct", type=float, default=5.0)
    parser.add_argument("--min-video-psnr-db", type=float, default=40.0)
    parser.add_argument("--max-video-mae", type=float, default=1.0)
    parser.add_argument("--max-video-seam-band-mae", type=float, default=1.0)
    parser.add_argument("--max-video-seam-excess-ratio", type=float, default=1.25)
    parser.add_argument("--max-audio-mae", type=float, default=0.01)
    parser.add_argument("--max-av-sync-delta-s", type=float, default=0.1)
    parser.add_argument("--decoder-tile-size", type=int, default=256)
    parser.add_argument("--decoder-tile-overlap-min", type=int, default=64)
    parser.add_argument("--vae-ratio", type=int, default=16)
    parser.add_argument("--seam-band-width", type=int, default=4)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--max-rank-imbalance-pct", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_log_start = args.server_log.stat().st_size if args.server_log and args.server_log.exists() else 0
    runs = []
    with httpx.Client(timeout=args.timeout) as client:
        for index in range(args.runs):
            runs.append(
                _request(
                    client,
                    args,
                    index=index,
                    steps=args.steps,
                    output_path=args.output_dir / f"{args.profile_name}-run-{index}.mp4",
                )
            )
        postcheck = _request(
            client,
            args,
            index=args.runs,
            steps=args.postcheck_steps,
            output_path=args.output_dir / f"{args.profile_name}-postcheck.mp4",
        )

    report: dict[str, Any] = {
        "profile": args.profile_name,
        "request": {
            "task": args.task,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "duration": args.duration,
            "steps": args.steps,
            "input_references": [str(path) for path in args.input_reference or []],
            "audio_reference": args.audio_reference,
        },
        "runs": [asdict(run) for run in runs],
        "postcheck": asdict(postcheck),
        "summary": {
            "warm_end_to_end_median_s": statistics.median(
                [run.end_to_end_s for run in runs[1:]] or [run.end_to_end_s for run in runs]
            ),
            "warm_inference_median_s": statistics.median(
                [run.inference_s for run in runs[1:]] or [run.inference_s for run in runs]
            ),
            "max_peak_memory_mb": max(run.peak_memory_mb for run in runs),
            "vae_decode_s": [_vae_time(run) for run in runs],
        },
    }
    if args.server_log:
        report["rank_timings"] = parse_rank_timings(args.server_log, server_log_start)
        report["vae_diagnostics"] = parse_vae_diagnostics(args.server_log, server_log_start)
    reference = None
    quality = None
    if args.reference_report:
        reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
        reference_media = reference["runs"][-1]["media_path"]
        quality = compare_media(
            reference_media,
            runs[-1].media_path,
            decoder_tile_size=args.decoder_tile_size,
            decoder_tile_overlap_min=args.decoder_tile_overlap_min,
            vae_ratio=args.vae_ratio,
            seam_band_width=args.seam_band_width,
        )
        report["quality"] = quality
    failures = _apply_gates(report, reference, quality, args)
    report["gate_failures"] = failures
    report_path = args.output_dir / f"{args.profile_name}-report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
