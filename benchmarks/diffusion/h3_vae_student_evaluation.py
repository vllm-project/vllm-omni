# SPDX-License-Identifier: Apache-2.0
"""Offline same-latent qualification for a MiniMax-H3 decoder student.

This tool deliberately does not integrate a student into the serving runtime.
Both reference and candidate artifacts expose a ``module:callable`` runner in
a versioned manifest. The callable receives the same latent tensor and returns
decoded video as a tensor.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DecoderArtifact:
    schema_version: int
    base_model: str
    component: str
    latent_channels: int
    spatial_ratio: int
    temporal_ratio: int
    runner: str
    checkpoint: str
    post_training_provenance: str

    @classmethod
    def load(cls, path: Path) -> DecoderArtifact:
        raw = json.loads(path.read_text(encoding="utf-8"))
        artifact = cls(**raw)
        artifact.validate(path)
        return artifact

    def validate(self, path: Path | None = None) -> None:
        location = f" in {path}" if path is not None else ""
        if self.schema_version != 1:
            raise ValueError(f"unsupported decoder artifact schema_version={self.schema_version}{location}")
        if self.base_model != "MiniMaxAI/MiniMax-H3":
            raise ValueError(f"student base_model must be MiniMaxAI/MiniMax-H3{location}")
        if self.component != "video_vae.decoder":
            raise ValueError(f"student component must be video_vae.decoder{location}")
        if (self.latent_channels, self.spatial_ratio, self.temporal_ratio) != (24, 16, 4):
            raise ValueError(
                f"student latent contract must be channels=24, spatial_ratio=16, temporal_ratio=4{location}"
            )
        if ":" not in self.runner:
            raise ValueError(f"student runner must use module:callable syntax{location}")
        if not self.checkpoint or not self.post_training_provenance:
            raise ValueError(f"student manifest requires checkpoint and post_training_provenance{location}")


def _load_runner(artifact: DecoderArtifact) -> Callable[[torch.Tensor], torch.Tensor]:
    module_name, callable_name = artifact.runner.split(":", 1)
    factory = getattr(importlib.import_module(module_name), callable_name)
    runner = factory(artifact.checkpoint)
    if not callable(runner):
        raise TypeError(f"{artifact.runner} did not return a callable decoder")
    return runner


def _sync(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize()


def _run(
    runner: Callable[[torch.Tensor], torch.Tensor],
    latent: torch.Tensor,
    *,
    warmups: int,
    runs: int,
) -> tuple[torch.Tensor, list[float], float]:
    for _ in range(warmups):
        # Decoder artifacts are allowed to mutate their input; isolate every
        # offline measurement so both artifacts always receive identical data.
        runner(latent.clone())
    if latent.device.type != "cpu":
        torch.accelerator.reset_peak_memory_stats()
    durations = []
    output = None
    for _ in range(runs):
        _sync(latent.device)
        started = time.perf_counter()
        output = runner(latent.clone())
        _sync(latent.device)
        duration = time.perf_counter() - started
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"decoder produced invalid measured duration: {duration}")
        durations.append(duration)
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"decoder returned {type(output).__name__}; expected torch.Tensor")
    peak_mb = 0.0
    if latent.device.type != "cpu":
        peak_mb = float(torch.accelerator.max_memory_reserved()) / (1024**2)
    output = output.detach().float().cpu()
    if not torch.isfinite(output).all():
        raise ValueError("decoder output contains NaN or Infinity")
    if not math.isfinite(peak_mb) or peak_mb < 0:
        raise ValueError(f"decoder produced invalid peak memory: {peak_mb}")
    return output, durations, peak_mb


def evaluate(
    reference: DecoderArtifact,
    candidate: DecoderArtifact,
    latent: torch.Tensor,
    *,
    warmups: int,
    runs: int,
    data_range: float,
) -> dict[str, Any]:
    if runs < 1 or warmups < 0:
        raise ValueError("runs must be >= 1 and warmups must be >= 0")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and greater than zero")
    if not torch.isfinite(latent).all():
        raise ValueError("latent contains NaN or Infinity")
    reference_output, reference_times, reference_peak = _run(
        _load_runner(reference), latent, warmups=warmups, runs=runs
    )
    candidate_output, candidate_times, candidate_peak = _run(
        _load_runner(candidate), latent, warmups=warmups, runs=runs
    )
    if reference_output.shape != candidate_output.shape:
        raise ValueError(
            f"student output shape mismatch: reference={tuple(reference_output.shape)}, "
            f"candidate={tuple(candidate_output.shape)}"
        )
    if reference_output.numel() == 0:
        raise ValueError("decoder output must not be empty")
    delta = reference_output.double() - candidate_output.double()
    mse = float(torch.mean(delta.square()))
    mae = float(torch.mean(delta.abs()))
    max_abs = float(torch.max(delta.abs()))
    if not all(math.isfinite(metric) for metric in (mse, mae, max_abs)):
        raise ValueError("decoder comparison produced a non-finite quality metric")
    reference_median = statistics.median(reference_times)
    candidate_median = statistics.median(candidate_times)
    exact_match = mse == 0
    return {
        "latent_shape": list(latent.shape),
        "output_shape": list(reference_output.shape),
        "reference_median_s": reference_median,
        "candidate_median_s": candidate_median,
        "decoder_speedup": reference_median / candidate_median,
        "reference_peak_memory_mb": reference_peak,
        "candidate_peak_memory_mb": candidate_peak,
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "exact_match": exact_match,
        "psnr_db": None if exact_match else 10 * math.log10((data_range**2) / mse),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--data-range", type=float, default=2.0)
    parser.add_argument("--min-psnr-db", type=float, default=50.0)
    parser.add_argument("--min-decoder-speedup", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise ValueError("--runs must be >= 1 and --warmups must be >= 0")
    for name in ("data_range", "min_psnr_db", "min_decoder_speedup"):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.data_range <= 0 or args.min_decoder_speedup < 0:
        raise ValueError("--data-range must be > 0 and --min-decoder-speedup must be >= 0")
    reference = DecoderArtifact.load(args.reference_manifest)
    candidate = DecoderArtifact.load(args.candidate_manifest)
    latent = torch.load(args.latent, map_location=args.device, weights_only=True)
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5 or int(latent.shape[1]) != 24:
        raise ValueError("--latent must contain a 5D MiniMax-H3 video latent with 24 channels")
    report = evaluate(
        reference,
        candidate,
        latent,
        warmups=args.warmups,
        runs=args.runs,
        data_range=args.data_range,
    )
    failures = []
    if not report["exact_match"] and report["psnr_db"] < args.min_psnr_db:
        failures.append(f"PSNR {report['psnr_db']:.3f} dB is below {args.min_psnr_db:.3f} dB")
    if report["decoder_speedup"] < args.min_decoder_speedup:
        failures.append(f"decoder speedup {report['decoder_speedup']:.3f}x is below {args.min_decoder_speedup:.3f}x")
    report["gate_failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
