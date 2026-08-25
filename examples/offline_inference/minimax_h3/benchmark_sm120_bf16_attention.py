#!/usr/bin/env python3
"""Matched RTX PRO 5000 (SM120) BF16 cuDNN vs FA4 benchmark for MiniMax-H3.

Run with exactly four visible SM120 GPUs. The first request for each backend is
regional-compile warmup; the following five requests are the reported samples.
MP4 encoding is deliberately deferred until after each backend's timed requests.

Example:
  CUDA_VISIBLE_DEVICES=4,6,5,7 numactl --cpunodebind=1 --membind=1 \
    python examples/offline_inference/minimax_h3/benchmark_sm120_bf16_attention.py \
      --model /path/to/MiniMax-H3/FL2VA --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

import vllm_omni.diffusion.diffusion_engine as diffusion_engine
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PROMPT = """integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with glowing amber displays flank a massive, curved observation window. A female captain, in her late 40s with an athletic build and short silver-streaked black hair, stands in the center midground. She wears a structured, high-collared dark navy military tunic with silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient starlight pouring through the thick glass. She stands perfectly still with her hands clasped tightly behind her back. Outside the window, a massive armada of jagged, dark grey dreadnoughts hovers in tight formation against a deep purple space nebula. The fleet's massive rear thrusters begin to glow with an intense, escalating bright blue light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and shakes strongly. The brilliant blue-white light from the fleet's gathering energy reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the window, completely washing out the background as the fleet jumps to hyperspace. The sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to stagger slightly forward, her shoulders tensing as she visibly braces herself against the physical tremors. As the intense white light fades abruptly, leaving only the dim, empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches, and she slowly closes her eyes in the newly emptied space.
overall_soundscape: A low, resonant hum of the ship's ambient life support systems serves as the baseline, soon drowned out by an audible, escalating, high-pitched electronic whine as the fleet outside charges its hyperdrives. A massive, deafening, bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating under immense physical stress. The intense roaring impact then cuts abruptly back to a hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.
non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a solitary, mournful French horn melody over deep, sustained string dissonances that build rapidly in volume and intensity, swelling to a massive orchestral peak before snapping immediately into silence right after the jump."""

DIFFUSE_KEY = "MiniMaxH3Pipeline.diffuse"
MODE_CONFIGS = {
    "cudnn_bf16": {"default": {"backend": "CUDNN_ATTN"}},
    # FA4 SM120 currently fails on MiniMax-H3's packed/ragged token refiner.
    # Keep that small dense component on cuDNN and benchmark FA4 where it
    # matters: the main DiT self-attention path.
    "fa4_main_dit_bf16": {
        "default": {"backend": "FLASH_ATTN"},
        "per_role": {"minimax_h3.token_refiner": {"backend": "CUDNN_ATTN"}},
    },
}
TELEMETRY_FIELDS = (
    "timestamp,index,temperature.gpu,utilization.gpu,clocks.sm,power.draw,"
    "clocks_event_reasons.sw_thermal_slowdown,"
    "clocks_event_reasons_counters.sw_thermal_slowdown"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="MiniMax-H3 FL2VA directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=6, help="One warmup plus measured requests")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help="Use 5s by default: the 10s/243-frame B300 input exceeds the validated 5K memory envelope.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-encoder-tp-size", type=int, default=4)
    parser.add_argument("--video-run", type=int, default=2, help="Request number saved as MP4")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_CONFIGS),
        default=list(MODE_CONFIGS),
        help="Run a subset while debugging; the default runs the matched cuDNN and FA4-main-DiT pair.",
    )
    args = parser.parse_args()
    if args.num_runs < 6:
        parser.error("--num-runs must be at least 6 (one warmup plus five measured requests)")
    if not args.model.is_dir():
        parser.error(f"--model is not a directory: {args.model}")
    return args


def visible_physical_gpus() -> list[int]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly four physical SM120 GPU indices.")
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain numeric physical GPU indices.") from exc
    if len(devices) != 4 or len(set(devices)) != 4:
        raise RuntimeError(f"Expected four distinct visible GPUs, got {value!r}")
    return devices


def hardware() -> list[dict[str, Any]]:
    device_count = torch.accelerator.device_count()
    if device_count != 4:
        raise RuntimeError(f"Expected four visible GPUs, found {device_count}")
    result = []
    for index in range(device_count):
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        if capability != (12, 0):
            raise RuntimeError(f"Logical GPU {index} must be SM120, got {capability[0]}.{capability[1]}")
        result.append(
            {
                "logical_index": index,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gib": round(props.total_memory / 2**30, 3),
            }
        )
    return result


def require_fa4() -> dict[str, str]:
    try:
        from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    except Exception as exc:
        raise RuntimeError(
            "FA4 is required. Install with: python -m pip install 'flash-attn-4[cu13]==4.0.0b18'"
        ) from exc
    return {
        "flash_attn_4": metadata.version("flash-attn-4"),
        "dense_module": flash_attn_func.__module__,
        "varlen_module": flash_attn_varlen_func.__module__,
    }


class Telemetry:
    def __init__(self, output_dir: Path, physical_gpus: list[int]) -> None:
        self.path = output_dir / "gpu_telemetry.csv"
        self.physical_gpus = physical_gpus
        self._stream = self.path.open("w", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={TELEMETRY_FIELDS}",
                "--format=csv,noheader,nounits",
                "--loop-ms=5000",
            ],
            stdout=self._stream,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)
        self._stream.close()

    def audit(self) -> dict[str, Any]:
        samples: dict[int, list[dict[str, Any]]] = {index: [] for index in self.physical_gpus}
        with self.path.open(encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) != 8:
                    continue
                index = int(row[1])
                if index not in samples:
                    continue
                samples[index].append(
                    {
                        "temperature_c": int(row[2]),
                        "utilization_percent": int(row[3]),
                        "sm_clock_mhz": int(row[4]),
                        "thermal_active": row[6].strip() == "Active",
                        "thermal_counter_us": int(row[7]),
                    }
                )

        per_gpu: dict[str, dict[str, Any]] = {}
        accepted = True
        for index, values in samples.items():
            if not values:
                per_gpu[str(index)] = {"accepted": False, "reason": "no telemetry samples"}
                accepted = False
                continue
            loaded = [sample for sample in values if sample["utilization_percent"] >= 90]
            counters = [sample["thermal_counter_us"] for sample in values]
            active = sum(sample["thermal_active"] for sample in values)
            gpu_accepted = bool(loaded) and active == 0 and max(counters) == min(counters)
            per_gpu[str(index)] = {
                "sample_count": len(values),
                "loaded_sample_count": len(loaded),
                "maximum_temperature_c": max(sample["temperature_c"] for sample in values),
                "minimum_loaded_sm_clock_mhz": min((sample["sm_clock_mhz"] for sample in loaded), default=None),
                "thermal_slowdown_active_samples": active,
                "thermal_slowdown_counter_delta_us": max(counters) - min(counters),
                "accepted": gpu_accepted,
            }
            accepted &= gpu_accepted
        return {"accepted": accepted, "gpus": per_gpu}


def sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sampling_params(args: argparse.Namespace) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        fps=24,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        output_type="np",
        extra_args={
            "task": "t2va",
            "aspect_ratio": "16:9",
            "duration": args.duration_seconds,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    )


def run_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    mode_dir = args.output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    engine = Omni(
        model=str(args.model),
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=2,
            ulysses_degree=2,
            ring_degree=1,
            text_encoder_tp_size=args.text_encoder_tp_size,
            vae_patch_parallel_size=4,
            vae_parallel_mode="tile",
        ),
        trust_remote_code=True,
        enforce_eager=False,
        diffusion_attention_config=MODE_CONFIGS[mode],
        enable_diffusion_pipeline_profiler=True,
    )
    records: list[dict[str, Any]] = []
    video: tuple[Path, np.ndarray, int, np.ndarray, int] | None = None
    try:
        for run_index in range(args.num_runs):
            torch.accelerator.synchronize()
            started = time.perf_counter()
            outputs = engine.generate(PROMPT, sampling_params(args), use_tqdm=False)
            torch.accelerator.synchronize()
            wall_time = time.perf_counter() - started
            if len(outputs) != 1:
                raise RuntimeError(f"Expected one output, got {len(outputs)}")

            result = outputs[0]
            frames = np.asarray(result.images[0])
            multimodal = result.multimodal_output
            if multimodal is None:
                raise RuntimeError("MiniMax-H3 returned no audio metadata")
            audio = np.asarray(multimodal["audio"])
            fps = int(multimodal["fps"])
            sample_rate = int(multimodal["audio_sample_rate"])
            if frames.ndim != 4 or tuple(frames.shape[1:]) != (args.height, args.width, 3):
                raise RuntimeError(f"Unexpected video shape: {frames.shape}")
            if fps != 24 or sample_rate != 32000:
                raise RuntimeError(f"Unexpected media rates: fps={fps}, sample_rate={sample_rate}")

            run_number = run_index + 1
            if run_number == args.video_run:
                video = (mode_dir / f"{mode}_run{run_number}.mp4", frames.copy(), fps, audio.copy(), sample_rate)
            record = {
                "run": run_number,
                "warmup": run_index == 0,
                "wall_time_s": wall_time,
                "stage_durations": dict(getattr(result, "stage_durations", {}) or {}),
                "worker_peak_memory_mb": float(getattr(result, "peak_memory_mb", 0.0) or 0.0),
                "frames_shape": list(frames.shape),
                "audio_shape": list(audio.shape),
                "frames_sha256": sha256(frames),
                "audio_sha256": sha256(audio),
            }
            records.append(record)
            print("RUN_RESULT " + json.dumps({"mode": mode, **record}, sort_keys=True), flush=True)
    finally:
        engine.close()

    if video is not None:
        output_path, frames, fps, audio, sample_rate = video
        output_path.write_bytes(_encode_video_bytes(frames, fps=fps, audio=audio, audio_sample_rate=sample_rate))

    measured = [record for record in records if not record["warmup"]]
    diffuse = [float(record["stage_durations"][DIFFUSE_KEY]) for record in measured]
    median = statistics.median(diffuse)
    mean = statistics.fmean(diffuse)
    stdev = statistics.stdev(diffuse)
    output_hashes = {(record["frames_sha256"], record["audio_sha256"]) for record in measured}
    timing = {
        "values_s": diffuse,
        "median_s": median,
        "mean_s": mean,
        "cv": stdev / mean,
        "span_over_median": (max(diffuse) - min(diffuse)) / median,
    }
    return {
        "mode": mode,
        "attention_config": MODE_CONFIGS[mode],
        "parallel_config": "tp2_ulysses2_ring1_text_encoder_tp4_vae_tile4",
        "runs": records,
        "timing": timing,
        "steady_output_deterministic": len(output_hashes) == 1,
        "video": str(video[0]) if video is not None else None,
        "passed_timing_gate": timing["cv"] <= 0.02 and timing["span_over_median"] <= 0.05,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    physical_gpus = visible_physical_gpus()
    fa4 = require_fa4()
    diffusion_engine._ASYNC_OUTPUT_TIMEOUT = 1800
    telemetry = Telemetry(args.output_dir, physical_gpus)
    try:
        results = []
        for mode in args.modes:
            result = run_mode(args, mode)
            (args.output_dir / mode / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            results.append(result)
    finally:
        telemetry.close()

    thermal = telemetry.audit()
    by_mode = {result["mode"]: result for result in results}
    speedup = None
    if set(MODE_CONFIGS).issubset(by_mode):
        speedup = float(by_mode["cudnn_bf16"]["timing"]["median_s"]) / float(
            by_mode["fa4_main_dit_bf16"]["timing"]["median_s"]
        )
    passed = (
        all(result["passed_timing_gate"] and result["steady_output_deterministic"] for result in results)
        and thermal["accepted"]
    )
    summary = {
        "protocol": "one warmup plus five measured requests per serial mode; deferred MP4 encoding",
        "model": str(args.model),
        "physical_gpus": physical_gpus,
        "hardware": hardware(),
        "fa4": fa4,
        "height": args.height,
        "width": args.width,
        "duration_seconds": args.duration_seconds,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "modes": results,
        "fa4_speedup_vs_cudnn": speedup,
        "thermal_audit": thermal,
        "passed": passed,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(f"Benchmark gate failed; inspect {summary_path}")


if __name__ == "__main__":
    main()
