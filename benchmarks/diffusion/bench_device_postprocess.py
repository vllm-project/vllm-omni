# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare WAN device postprocessing against an unmodified main checkout.

The driver starts one fresh process per checkout, reports cold startup separately,
performs warmup requests, and then measures at least three same-seed requests. It
samples aggregate process-tree host RSS and GPU memory during each measured run.

Example:

    python benchmarks/diffusion/bench_device_postprocess.py \
      --baseline-checkout /path/to/vllm-omni-main \
      --candidate-checkout . --warmup-runs 1 --rounds 3
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

DEFAULT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_PROMPT = "A red fox walking through a snowy pine forest at sunrise, cinematic"


class RunMetrics(TypedDict):
    generate_s: float
    payload_mib: float
    peak_tree_rss_mib: float
    peak_gpu_allocated_mib: float
    frame_sha256: str


class CheckoutMetrics(TypedDict):
    mode: str
    commit: str
    startup_s: float
    warmup_s: list[float]
    runs: list[RunMetrics]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkout", type=Path)
    parser.add_argument("--candidate-checkout", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", default=os.environ.get("VLLM_OMNI_BENCH_MODEL", DEFAULT_MODEL))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("video_benchmark"))
    parser.add_argument("--child-mode", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("--rounds must be at least 3")
    if args.warmup_runs < 1:
        parser.error("--warmup-runs must be positive")
    if args.child_mode is None and args.baseline_checkout is None:
        parser.error("--baseline-checkout is required")
    return args


def _process_tree_pids(root_pid: int) -> set[int]:
    pending = [root_pid]
    seen: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        pending.extend(int(child) for child in children)
    return seen


def _tree_rss_mib(root_pid: int) -> float:
    total = 0.0
    for pid in _process_tree_pids(root_pid):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += float(line.split()[1]) / 1024.0
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return total


class ProcessTreeSampler(threading.Thread):
    def __init__(self, root_pid: int) -> None:
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self.stop_event = threading.Event()
        self.peak_rss_mib = 0.0

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.peak_rss_mib = max(self.peak_rss_mib, _tree_rss_mib(self.root_pid))
            self.stop_event.wait(0.05)


def _quantize_frames(video: np.ndarray) -> np.ndarray:
    if video.dtype == np.uint8:
        return np.ascontiguousarray(video)
    return np.ascontiguousarray(np.rint(np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8))


def _extract_video(outputs: object) -> np.ndarray:
    video = outputs[0].images[0]  # type: ignore[index,union-attr]
    array = video.numpy() if hasattr(video, "numpy") else np.asarray(video)
    if array.ndim == 5:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one video, got batch shape {array.shape}")
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"Expected [T, H, W, C] video, got shape {array.shape}")
    return array


def _generate(engine: object, args: argparse.Namespace) -> object:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        output_type="np",
        seed=args.seed,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        guidance_scale=args.guidance_scale,
    )
    return engine.generate({"prompt": args.prompt}, sampling)  # type: ignore[attr-defined]


def run_child(args: argparse.Namespace) -> None:
    if args.checkout is None or args.result_path is None:
        raise ValueError("child mode requires --checkout and --result-path")
    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))

    import torch

    from vllm_omni.entrypoints.omni import Omni

    engine_kwargs: dict[str, object] = {"model": args.model, "num_gpus": 1}
    if args.child_mode == "candidate":
        engine_kwargs["video_output_transport"] = {"enable_device_postprocess": True}

    started = time.perf_counter()
    engine = Omni(**engine_kwargs)
    startup_s = time.perf_counter() - started
    warmup_s: list[float] = []
    runs: list[RunMetrics] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for _ in range(args.warmup_runs):
            started = time.perf_counter()
            outputs = _generate(engine, args)
            warmup_s.append(time.perf_counter() - started)
            del outputs
            gc.collect()

        for round_index in range(args.rounds):
            torch.accelerator.memory.reset_peak_memory_stats()
            sampler = ProcessTreeSampler(os.getpid())
            sampler.start()
            started = time.perf_counter()
            outputs = _generate(engine, args)
            generate_s = time.perf_counter() - started
            sampler.stop_event.set()
            sampler.join()
            peak_gpu_allocated_mib = torch.accelerator.memory.max_memory_allocated() / 1024**2

            video = _extract_video(outputs)
            payload_mib = video.nbytes / 1024**2
            frames = _quantize_frames(video)
            digest = hashlib.sha256(memoryview(frames)).hexdigest()
            if round_index == 0:
                np.save(args.output_dir / f"{args.child_mode}_frames.npy", frames)
            runs.append(
                {
                    "generate_s": round(generate_s, 3),
                    "payload_mib": round(payload_mib, 3),
                    "peak_tree_rss_mib": round(sampler.peak_rss_mib, 1),
                    "peak_gpu_allocated_mib": round(peak_gpu_allocated_mib, 1),
                    "frame_sha256": digest,
                }
            )
            del frames, video, outputs
            gc.collect()
    finally:
        engine.close()

    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True).strip():
        commit += "-dirty"
    result: CheckoutMetrics = {
        "mode": args.child_mode,
        "commit": commit,
        "startup_s": round(startup_s, 3),
        "warmup_s": [round(value, 3) for value in warmup_s],
        "runs": runs,
    }
    args.result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _run_checkout(mode: str, checkout: Path, args: argparse.Namespace) -> CheckoutMetrics:
    checkout = checkout.resolve()
    result_path = args.output_dir / f"{mode}_results.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-mode",
        mode,
        "--checkout",
        str(checkout),
        "--result-path",
        str(result_path.resolve()),
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--frames",
        str(args.frames),
        "--steps",
        str(args.steps),
        "--guidance-scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--warmup-runs",
        str(args.warmup_runs),
        "--rounds",
        str(args.rounds),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(checkout) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=checkout, env=env, check=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def _mean_range(rows: list[RunMetrics], field: str) -> str:
    values = [float(row[field]) for row in rows]  # type: ignore[literal-required]
    return f"{np.mean(values):.1f} ({min(values):.1f}-{max(values):.1f})"


def _compare_frames(output_dir: Path) -> tuple[int, float]:
    baseline = np.load(output_dir / "baseline_frames.npy", mmap_mode="r")
    candidate = np.load(output_dir / "candidate_frames.npy", mmap_mode="r")
    if baseline.shape != candidate.shape:
        raise ValueError(f"Frame shapes differ: {baseline.shape} vs {candidate.shape}")

    max_diff = 0
    differing = 0
    total = baseline.size
    baseline_flat = baseline.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    for start in range(0, total, 8_000_000):
        stop = min(start + 8_000_000, total)
        delta = np.abs(baseline_flat[start:stop].astype(np.int16) - candidate_flat[start:stop].astype(np.int16))
        max_diff = max(max_diff, int(delta.max(initial=0)))
        differing += int(np.count_nonzero(delta))
    return max_diff, differing / total


def run_driver(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _run_checkout("baseline", args.baseline_checkout, args)
    candidate = _run_checkout("candidate", args.candidate_checkout, args)
    max_diff, differing_fraction = _compare_frames(args.output_dir)

    result = {
        "baseline": baseline,
        "candidate": candidate,
        "frame_max_diff": max_diff,
        "frame_differing_fraction": differing_fraction,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"baseline commit:  {baseline['commit']}")
    print(f"candidate commit: {candidate['commit']}")
    print(f"cold startup (s): {baseline['startup_s']} -> {candidate['startup_s']}")
    print(f"frame max diff:   {max_diff}/255 ({differing_fraction:.2%} values differ)")
    print(f"{'metric':<24}{'main':>28}{'candidate':>28}")
    for field in ("payload_mib", "peak_tree_rss_mib", "peak_gpu_allocated_mib", "generate_s"):
        print(f"{field:<24}{_mean_range(baseline['runs'], field):>28}{_mean_range(candidate['runs'], field):>28}")

    if max_diff > 1 or differing_fraction >= 0.35:
        raise RuntimeError("Candidate output exceeds the documented WAN precision bound")


def main() -> None:
    args = parse_args()
    if args.child_mode is not None:
        run_child(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
