# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline MiniMax-H3 Sol-Attn speed/quality sweep.

The parent process launches one short-lived offline worker per attention
configuration.  This isolates CUDA/distributed state between configurations,
keeps the experiment resumable, and avoids running an API server.  Every Sol
output is compared with the same-seed CUDNN_ATTN output using full-video SSIM
and PSNR plus uniformly sampled LPIPS frames.

Example (run from the directory containing ``MiniMax-H3/FL2VA``)::

    source .venv/bin/activate
    export PYTHONPATH="$PWD/vllm-omni-pr5851"
    python vllm-omni-pr5851/benchmarks/diffusion/sol_attn_offline_sweep.py \
        --model MiniMax-H3/FL2VA \
        --gpus 4,6,5,7 \
        --suite full \
        --output-dir results/sol-sm120-offline-sweep \
        --lpips-device cuda:4

Re-running the same command resumes completed configurations.  Use
``--list-cases`` to inspect the matrix or ``--phase quality`` to recompute only
the metrics/report from existing videos.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "At night, while their owner sleeps, three cats march into a bedroom "
    "playing tiny brass instruments, freeze, and quietly march out."
)
SSIM_RE = re.compile(r"All:(?P<score>[-+0-9.eE]+)")
PSNR_RE = re.compile(r"average:(?P<score>inf|[-+0-9.eE]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    group: str
    description: str
    backend: str = "SOL_ATTN"
    tau: float | None = 1.0
    thresh_type: str | None = "diag"
    sink_tokens: int | None = 951
    sink_start: int | None = 0
    dense_steps: int | None = 10
    dense_layers: str | None = "0,1"
    kv_splits: int | str | None = 1

    def attention_config(self) -> dict[str, Any]:
        default: dict[str, Any] = {"backend": self.backend}
        if self.backend == "SOL_ATTN":
            default["sol_attn"] = {
                "tau": self.tau,
                "thresh_type": self.thresh_type,
                "sink_tokens": self.sink_tokens,
                "sink_start": self.sink_start,
                "dense_steps": self.dense_steps,
                "dense_layers": self.dense_layers,
                "kv_splits": self.kv_splits,
            }
        return {"default": default}


DENSE_CASE = ExperimentCase(
    name="dense_cudnn",
    group="baseline",
    description="Exact dense cuDNN reference",
    backend="CUDNN_ATTN",
    tau=None,
    thresh_type=None,
    sink_tokens=None,
    sink_start=None,
    dense_steps=None,
    dense_layers=None,
    kv_splits=None,
)

RECOMMENDED_CASE = ExperimentCase(
    name="sol_recommended",
    group="preset",
    description="Recommended quality-first Sol-Attn configuration",
)


def build_cases(suite: str) -> list[ExperimentCase]:
    """Return an ordered, de-duplicated offline experiment matrix."""
    medium = replace(
        RECOMMENDED_CASE,
        name="sol_medium",
        description="Balanced preset: tau=1.5, dense_steps=8",
        tau=1.5,
        dense_steps=8,
    )
    aggressive = replace(
        RECOMMENDED_CASE,
        name="sol_aggressive",
        description="Aggressive preset: tau=2.0, dense_steps=5",
        tau=2.0,
        dense_steps=5,
    )
    quick = [DENSE_CASE, RECOMMENDED_CASE, medium, aggressive]
    if suite == "quick":
        return quick

    cases = [*quick]
    cases.extend(
        replace(
            RECOMMENDED_CASE,
            name=f"sol_tau_{str(tau).replace('.', 'p')}",
            group="tau",
            description=f"Tau-only sweep: tau={tau}",
            tau=tau,
        )
        for tau in (0.0, 0.5, 1.5, 2.0)
    )
    cases.extend(
        replace(
            RECOMMENDED_CASE,
            name=f"sol_dense_steps_{steps}",
            group="dense_steps",
            description=f"Dense-step-only sweep: dense_steps={steps}",
            dense_steps=steps,
        )
        for steps in (0, 5, 8, 15, 20)
    )
    layer_cases = (("none", ""), ("0", "0"), ("0_3", "0-3"))
    cases.extend(
        replace(
            RECOMMENDED_CASE,
            name=f"sol_dense_layers_{label}",
            group="dense_layers",
            description=f"Dense-layer-only sweep: dense_layers={layers!r}",
            dense_layers=layers,
        )
        for label, layers in layer_cases
    )
    cases.extend(
        replace(
            RECOMMENDED_CASE,
            name=f"sol_sink_{tokens}",
            group="sink_tokens",
            description=f"Exact-sink-only sweep: sink_tokens={tokens}",
            sink_tokens=tokens,
        )
        for tokens in (0, 256, 512)
    )
    cases.extend(
        replace(
            RECOMMENDED_CASE,
            name=f"sol_kv_splits_{splits}",
            group="kv_splits",
            description=f"KV split-only sweep: kv_splits={splits}",
            kv_splits=splits,
        )
        for splits in ("auto", 2, 4)
    )
    cases.append(
        replace(
            RECOMMENDED_CASE,
            name="sol_thresh_exact",
            group="thresh_type",
            description="Exact rather than diagonal routing threshold",
            thresh_type="exact",
        )
    )

    unique: list[ExperimentCase] = []
    seen: set[tuple[Any, ...]] = set()
    for case in cases:
        signature = (
            case.backend,
            case.tau,
            case.thresh_type,
            case.sink_tokens,
            case.sink_start,
            case.dense_steps,
            case.dense_layers,
            case.kv_splits,
        )
        if signature not in seen:
            unique.append(case)
            seen.add(signature)
    return unique


def _parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _case_complete(case_dir: Path, seeds: list[int], repeats: int) -> bool:
    timing_path = case_dir / "timings.json"
    timings = _read_json(timing_path, {})
    measured = {
        (int(row["seed"]), int(row["repeat"]))
        for row in timings.get("runs", [])
        if not row.get("warmup") and row.get("status") == "ok"
    }
    expected = {(seed, repeat) for seed in seeds for repeat in range(1, repeats + 1)}
    if measured != expected:
        return False
    outputs = [case_dir / f"seed_{seed}_run_{repeat}.mp4" for seed, repeat in expected]
    return all(path.exists() and path.stat().st_size > 0 for path in outputs)


def _normalise_video_frames(value: Any) -> Any:
    import numpy as np

    frames = np.asarray(value)
    if frames.ndim == 5 and frames.shape[0] == 1:
        frames = frames[0]
    if frames.ndim != 4:
        raise RuntimeError(f"Expected four-dimensional video frames, got shape={frames.shape}")
    if frames.shape[-1] not in (3, 4) and frames.shape[0] in (3, 4):
        frames = frames.transpose(1, 2, 3, 0)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.shape[-1] != 3:
        raise RuntimeError(f"Expected RGB video frames, got shape={frames.shape}")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.float32)
        if float(frames.min()) < 0.0:
            frames = frames * 0.5 + 0.5
        frames = (frames.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(frames)


def _extract_peak_memory_mb(outputs: Any) -> float:
    result = outputs[0] if isinstance(outputs, list) and outputs else outputs
    value = getattr(result, "peak_memory_mb", 0.0) if result is not None else 0.0
    if not value and result is not None:
        inner = getattr(result, "request_output", None)
        if isinstance(inner, list):
            inner = inner[0] if inner else None
        value = getattr(inner, "peak_memory_mb", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _save_output(outputs: Any, output_path: Path) -> float:
    import numpy as np

    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    if not outputs:
        raise RuntimeError("Offline generation returned no outputs")
    result = outputs[0]
    if not result.images:
        raise RuntimeError("Offline generation returned no video frames")
    frames = _normalise_video_frames(result.images[0])
    multimodal = result.multimodal_output or {}
    audio = multimodal.get("audio")
    audio_array = None if audio is None else np.squeeze(np.asarray(audio)).astype(np.float32)
    fps = float(multimodal.get("fps", 24))
    sample_rate = int(multimodal.get("audio_sample_rate", 32000))
    start = time.perf_counter()
    payload = mux_video_audio_bytes(
        frames,
        audio_array,
        fps=fps,
        audio_sample_rate=sample_rate,
    )
    temporary = output_path.with_suffix(".mp4.tmp")
    temporary.write_bytes(payload)
    temporary.replace(output_path)
    return time.perf_counter() - start


def _worker_sampling_params(config: dict[str, Any], seed: int) -> Any:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(
        height=int(config["height"]),
        width=int(config["width"]),
        fps=int(config["fps"]),
        num_inference_steps=int(config["num_inference_steps"]),
        seed=seed,
        output_type="np",
        extra_args={
            "task": "t2va",
            "duration": float(config["duration"]),
            "aspect_ratio": str(config["aspect_ratio"]),
            "flow_shift": float(config["flow_shift"]),
            "audio_flow_shift": float(config["audio_flow_shift"]),
        },
    )


def run_worker(config_path: Path) -> int:
    """Run one offline attention configuration in an isolated process."""
    config = _read_json(config_path, None)
    if config is None:
        raise FileNotFoundError(config_path)
    case = ExperimentCase(**config["case"])
    case_dir = Path(config["case_dir"])
    timing_path = case_dir / "timings.json"
    timings = _read_json(timing_path, {"case": asdict(case), "runs": []})

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=int(config["tensor_parallel_size"]),
        text_encoder_tp_size=int(config["text_encoder_tp_size"]),
    )
    omni_kwargs: dict[str, Any] = {
        "model": config["model"],
        "parallel_config": parallel_config,
        "num_gpus": int(config["num_gpus"]),
        "trust_remote_code": True,
        "vae_use_tiling": True,
        "diffusion_attention_config": case.attention_config(),
    }
    if config.get("enforce_eager"):
        omni_kwargs["enforce_eager"] = True

    load_start = time.perf_counter()
    engine = Omni(**omni_kwargs)
    timings["load_seconds"] = time.perf_counter() - load_start
    _atomic_json(timing_path, timings)
    try:
        first_seed = int(config["seeds"][0])
        for warmup_index in range(1, int(config["warmups"]) + 1):
            start = time.perf_counter()
            outputs = engine.generate(
                config["prompt"],
                _worker_sampling_params(config, first_seed),
                use_tqdm=False,
            )
            timings["runs"].append(
                {
                    "warmup": True,
                    "warmup_index": warmup_index,
                    "seed": first_seed,
                    "generation_seconds": time.perf_counter() - start,
                    "peak_memory_mb": _extract_peak_memory_mb(outputs),
                    "status": "ok",
                }
            )
            _atomic_json(timing_path, timings)
            del outputs
            gc.collect()

        completed = {
            (int(row["seed"]), int(row["repeat"]))
            for row in timings["runs"]
            if not row.get("warmup") and row.get("status") == "ok"
        }
        for seed in map(int, config["seeds"]):
            for repeat_index in range(1, int(config["repeats"]) + 1):
                output_path = case_dir / f"seed_{seed}_run_{repeat_index}.mp4"
                if (seed, repeat_index) in completed and output_path.exists() and output_path.stat().st_size > 0:
                    continue
                start = time.perf_counter()
                outputs = engine.generate(
                    config["prompt"],
                    _worker_sampling_params(config, seed),
                    use_tqdm=False,
                )
                generation_seconds = time.perf_counter() - start
                encode_seconds = _save_output(outputs, output_path)
                timings["runs"].append(
                    {
                        "warmup": False,
                        "seed": seed,
                        "repeat": repeat_index,
                        "generation_seconds": generation_seconds,
                        "encode_seconds": encode_seconds,
                        "peak_memory_mb": _extract_peak_memory_mb(outputs),
                        "output": str(output_path),
                        "status": "ok",
                    }
                )
                _atomic_json(timing_path, timings)
                del outputs
                gc.collect()
    finally:
        engine.close()
    return 0


def _worker_payload(args: argparse.Namespace, case: ExperimentCase, case_dir: Path) -> dict[str, Any]:
    return {
        "case": asdict(case),
        "case_dir": str(case_dir.resolve()),
        "model": args.model,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "duration": args.duration,
        "aspect_ratio": args.aspect_ratio,
        "flow_shift": args.flow_shift,
        "audio_flow_shift": args.audio_flow_shift,
        "seeds": args.seeds,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "num_gpus": args.num_gpus,
        "tensor_parallel_size": args.tensor_parallel_size,
        "text_encoder_tp_size": args.text_encoder_tp_size,
        "enforce_eager": args.enforce_eager,
    }


def run_sweep(args: argparse.Namespace, cases: list[ExperimentCase]) -> list[str]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    selected_gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if len(selected_gpus) != args.num_gpus:
        raise ValueError(f"--gpus exposes {len(selected_gpus)} devices but --num-gpus={args.num_gpus}")

    _atomic_json(
        output_dir / "manifest.json",
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "cases": [asdict(case) for case in cases],
            "settings": {
                key: value
                for key, value in vars(args).items()
                if key not in {"worker_config"} and isinstance(value, (str, int, float, bool, list, type(None)))
            },
        },
    )
    for index, case in enumerate(cases, start=1):
        case_dir = output_dir / "cases" / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = _worker_payload(args, case, case_dir)
        config_path = case_dir / "worker_config.json"
        existing_payload = _read_json(config_path, None)
        if existing_payload is not None and existing_payload != payload:
            raise ValueError(
                f"{case.name} already has results from different experiment settings. "
                f"Use a new --output-dir or remove that case directory: {case_dir}"
            )
        if _case_complete(case_dir, args.seeds, args.repeats):
            print(f"[{index}/{len(cases)}] {case.name}: complete, skipping", flush=True)
            continue
        _atomic_json(config_path, payload)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-config", str(config_path)]
        print(f"[{index}/{len(cases)}] {case.name}: {case.description}", flush=True)
        if args.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = args.gpus
        environment.setdefault("PYTHONUNBUFFERED", "1")
        log_path = case_dir / "worker.log"
        with log_path.open("a") as log_file:
            log_file.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)} ===\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(f"[{case.name}] {line}", end="", flush=True)
            return_code = process.wait()
        if return_code != 0:
            failures.append(case.name)
            print(f"  FAILED (exit={return_code}); see {log_path}", flush=True)
            if not args.keep_going:
                break
        else:
            print(f"  complete; outputs: {case_dir}", flush=True)
    _atomic_json(output_dir / "failures.json", failures)
    return failures


def _run_ffmpeg_metric(reference: Path, candidate: Path, metric: str) -> float:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            str(reference),
            "-i",
            str(candidate),
            "-lavfi",
            f"[0:v]setpts=PTS-STARTPTS[ref];[1:v]setpts=PTS-STARTPTS[test];[ref][test]{metric}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stderr + result.stdout
    pattern = SSIM_RE if metric == "ssim" else PSNR_RE
    match = pattern.search(output)
    if result.returncode != 0 or match is None:
        raise RuntimeError(f"Could not compute {metric} for {candidate}:\n{output[-4000:]}")
    value = match.group("score")
    return math.inf if value.lower() == "inf" else float(value)


def _probe_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _sample_video(path: Path, indices: set[int]) -> dict[int, Any]:
    import av
    import numpy as np

    selected: dict[int, Any] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index in indices:
                selected[index] = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            if len(selected) == len(indices):
                break
    missing = indices.difference(selected)
    if missing:
        raise RuntimeError(f"Could not decode frame indices {sorted(missing)} from {path}")
    return selected


class LPIPSEvaluator:
    def __init__(self, device: str, size: int, max_frames: int, batch_size: int) -> None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError(
                "LPIPS evaluation requires lpips==0.1.4; install it in the active vLLM-Omni environment"
            ) from error
        import torch

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.size = size
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.model = lpips.LPIPS(net="alex").eval().to(self.device)

    def __call__(self, reference: Path, candidate: Path) -> tuple[float, int]:
        import numpy as np
        import torch
        import torch.nn.functional as torch_functional

        frame_count = min(_probe_frame_count(reference), _probe_frame_count(candidate))
        sample_count = min(frame_count, self.max_frames)
        indices = set(np.linspace(0, frame_count - 1, sample_count, dtype=int).tolist())
        reference_frames = _sample_video(reference, indices)
        candidate_frames = _sample_video(candidate, indices)
        scores: list[float] = []
        ordered = sorted(indices)
        for offset in range(0, len(ordered), self.batch_size):
            batch_indices = ordered[offset : offset + self.batch_size]
            ref = np.stack([reference_frames[index] for index in batch_indices])
            cand = np.stack([candidate_frames[index] for index in batch_indices])
            ref_tensor = torch.from_numpy(ref).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            cand_tensor = torch.from_numpy(cand).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            if self.size > 0:
                target = (self.size, self.size)
                ref_tensor = torch_functional.interpolate(ref_tensor, target, mode="bilinear", align_corners=False)
                cand_tensor = torch_functional.interpolate(cand_tensor, target, mode="bilinear", align_corners=False)
            with torch.inference_mode():
                values = self.model(ref_tensor.to(self.device), cand_tensor.to(self.device))
            scores.extend(values.detach().float().cpu().reshape(-1).tolist())
        return float(statistics.mean(scores)), len(ordered)


def _measured_runs(case_dir: Path) -> list[dict[str, Any]]:
    timings = _read_json(case_dir / "timings.json", {})
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in timings.get("runs", []):
        if not row.get("warmup") and row.get("status") == "ok":
            latest[(int(row["seed"]), int(row["repeat"]))] = row
    return [latest[key] for key in sorted(latest)]


def _summary_rows(output_dir: Path, cases: list[ExperimentCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_dir = output_dir / "cases" / case.name
        measured = _measured_runs(case_dir)
        if not measured:
            continue
        latencies = [float(row["generation_seconds"]) for row in measured]
        peaks = [float(row.get("peak_memory_mb", 0.0)) for row in measured]
        rows.append(
            {
                **asdict(case),
                "samples": len(latencies),
                "median_generation_s": statistics.median(latencies),
                "mean_generation_s": statistics.mean(latencies),
                "stdev_generation_s": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                "min_generation_s": min(latencies),
                "max_generation_s": max(latencies),
                "max_peak_memory_mb": max(peaks),
            }
        )
    dense = next((row for row in rows if row["name"] == DENSE_CASE.name), None)
    if dense is not None:
        baseline = float(dense["median_generation_s"])
        for row in rows:
            latency = float(row["median_generation_s"])
            row["speedup_x"] = baseline / latency
            row["latency_reduction_pct"] = (1.0 - latency / baseline) * 100.0
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_quality(args: argparse.Namespace, cases: list[ExperimentCase]) -> list[dict[str, Any]]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required for SSIM/PSNR")
    output_dir = args.output_dir.resolve()
    lpips_evaluator = None
    if not args.skip_lpips:
        lpips_evaluator = LPIPSEvaluator(
            args.lpips_device,
            args.lpips_size,
            args.lpips_max_frames,
            args.lpips_batch_size,
        )
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        reference = output_dir / "cases" / DENSE_CASE.name / f"seed_{seed}_run_{args.quality_repeat}.mp4"
        if not reference.exists():
            raise FileNotFoundError(f"Dense same-seed reference is missing: {reference}")
        for case in cases:
            candidate = output_dir / "cases" / case.name / f"seed_{seed}_run_{args.quality_repeat}.mp4"
            if not candidate.exists():
                continue
            print(f"quality: {case.name}, seed={seed}", flush=True)
            row: dict[str, Any] = {
                "case": case.name,
                "group": case.group,
                "seed": seed,
                "reference": str(reference),
                "candidate": str(candidate),
            }
            try:
                row["ssim"] = _run_ffmpeg_metric(reference, candidate, "ssim")
                row["psnr_db"] = _run_ffmpeg_metric(reference, candidate, "psnr")
                if lpips_evaluator is not None:
                    row["lpips"], row["lpips_sampled_frames"] = lpips_evaluator(reference, candidate)
                else:
                    row["lpips"] = None
                    row["lpips_sampled_frames"] = 0
                row["quality_pass"] = (
                    float(row["ssim"]) >= args.min_ssim
                    and float(row["psnr_db"]) >= args.min_psnr
                    and (row["lpips"] is None or float(row["lpips"]) <= args.max_lpips)
                )
                row["error"] = ""
            except Exception as error:
                row["quality_pass"] = False
                row["error"] = str(error)
            rows.append(row)
            _write_csv(output_dir / "quality.csv", rows)
    return rows


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "-"
    number = float(value)
    if math.isinf(number):
        return "inf"
    return f"{number:.{digits}f}"


def write_report(
    args: argparse.Namespace,
    cases: list[ExperimentCase],
    quality_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir = args.output_dir.resolve()
    summary = _summary_rows(output_dir, cases)
    _write_csv(output_dir / "summary.csv", summary)
    if quality_rows is None:
        quality_path = output_dir / "quality.csv"
        quality_rows = list(csv.DictReader(quality_path.open())) if quality_path.exists() else []
    quality_by_case: dict[str, dict[str, Any]] = {}
    for row in quality_rows:
        quality_by_case.setdefault(str(row["case"]), row)

    lines = [
        "# Sol-Attn offline sweep",
        "",
        f"Dense reference: `{DENSE_CASE.name}` (CUDNN_ATTN).",
        f"Prompt seeds: `{','.join(map(str, args.seeds))}`; measured repeats per seed: `{args.repeats}`.",
        (
            f"SSIM/PSNR use every decoded video frame at native resolution. LPIPS uses up to "
            f"`{args.lpips_max_frames}` uniformly sampled frames resized to "
            f"`{args.lpips_size}x{args.lpips_size}` (0 means native resolution)."
        ),
        "",
        "| Case | Group | Median (s) | Speedup | Latency reduction | SSIM | PSNR (dB) | LPIPS | Quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        quality = quality_by_case.get(str(row["name"]), {})
        quality_pass = quality.get("quality_pass")
        if isinstance(quality_pass, str):
            quality_pass = quality_pass.lower() == "true"
        quality_label = "PASS" if quality_pass else ("FAIL" if quality else "-")
        row_template = (
            "| {name} | {group} | {latency} | {speedup}x | {reduction}% | {ssim} | {psnr} | {lpips} | {quality} |"
        )
        lines.append(
            row_template.format(
                name=row["name"],
                group=row["group"],
                latency=_format_number(row["median_generation_s"], 2),
                speedup=_format_number(row.get("speedup_x"), 3),
                reduction=_format_number(row.get("latency_reduction_pct"), 2),
                ssim=_format_number(quality.get("ssim"), 5),
                psnr=_format_number(quality.get("psnr_db"), 2),
                lpips=_format_number(quality.get("lpips"), 5),
                quality=quality_label,
            )
        )
    lines.extend(
        [
            "",
            "Quality gates",
            "",
            f"- SSIM >= {args.min_ssim}",
            f"- PSNR >= {args.min_psnr} dB",
            f"- LPIPS <= {args.max_lpips}",
            "",
            "Raw data: `summary.csv`, `quality.csv`, and `cases/*/timings.json`.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="MiniMax-H3/FL2VA")
    parser.add_argument("--output-dir", type=Path, default=Path("results/sol-sm120-offline-sweep"))
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--phase", choices=("all", "run", "quality", "report"), default="all")
    parser.add_argument("--case", action="append", dest="case_names", help="Run only this case (repeatable)")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--gpus", default="4,6,5,7")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--text-encoder-tp-size", type=int, default=4)
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[1101])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)

    parser.add_argument("--quality-repeat", type=int, default=1)
    parser.add_argument("--min-ssim", type=float, default=0.82)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--max-lpips", type=float, default=0.20)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-device", default="auto")
    parser.add_argument("--lpips-size", type=int, default=256)
    parser.add_argument("--lpips-max-frames", type=int, default=16)
    parser.add_argument("--lpips-batch-size", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config is not None:
        return run_worker(args.worker_config)
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("--warmups must be >= 0 and --repeats must be >= 1")
    cases = build_cases(args.suite)
    if args.quality_repeat < 1 or args.quality_repeat > args.repeats:
        raise ValueError("--quality-repeat must be between 1 and --repeats")
    if args.lpips_size < 0 or args.lpips_max_frames < 1 or args.lpips_batch_size < 1:
        raise ValueError("LPIPS size must be >= 0 and frame/batch counts must be >= 1")
    if args.case_names:
        requested = set(args.case_names)
        cases = [case for case in cases if case.name in requested]
        missing = requested.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"Unknown case(s): {sorted(missing)}")
        if args.phase in ("quality", "all") and DENSE_CASE.name not in {case.name for case in cases}:
            cases.insert(0, DENSE_CASE)
    if args.list_cases:
        for case in cases:
            print(f"{case.name:28} {case.group:14} {case.description}")
        return 0

    failures: list[str] = []
    if args.phase in ("all", "run"):
        failures = run_sweep(args, cases)
    quality_rows = None
    if not args.dry_run and args.phase in ("all", "quality"):
        quality_rows = compute_quality(args, cases)
    if not args.dry_run and args.phase in ("all", "quality", "report"):
        write_report(args, cases, quality_rows)
        print(f"Report: {args.output_dir.resolve() / 'report.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
