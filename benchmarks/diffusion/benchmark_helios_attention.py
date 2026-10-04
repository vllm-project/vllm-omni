# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure one Helios attention backend per process and save float-frame evidence.

Use a pinned local Helios-Distilled checkpoint. Run each backend in a fresh
process with the same arguments; initialization, warmup and video encoding are
excluded from the reported request timings. See recipes/Helios/Helios-Distilled-H20.md.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

PROMPTS = [
    "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train.",
    "A golden retriever runs through a green meadow, its fur moving in the breeze, cinematic tracking shot.",
    "Ocean waves roll onto a sandy beach at sunset, warm orange light reflecting on the water, steady camera.",
]


def describe_flash_bindings(provider) -> dict:
    """Record resolved callables, not just the attention wrapper or wheel version."""
    return {
        name: None
        if (func := getattr(provider, name)) is None
        else {"module": func.__module__, "qualname": func.__qualname__}
        for name in ("flash_attn_func", "flash_attn_varlen_func")
    }


class HeliosBenchmarkWorker:
    """Worker extension that measures transformer forwards on the device stream."""

    def reset_helios_request_cache(self) -> dict:
        # Retain within-request KV reuse, but never reuse a previous prompt's KV.
        self.model_runner.pipeline.transformer.clear_cross_attention_cache()
        return {"cache_cleared": True}

    def install_helios_timers(self) -> dict:
        import torch

        from vllm_omni.diffusion.attention.layer import Attention
        from vllm_omni.platforms import current_omni_platform

        transformer = self.model_runner.pipeline.transformer
        self._helios_events = []

        def before_forward(module, args, kwargs):
            hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            start = torch.Event(device=current_omni_platform.device_type, enable_timing=True)
            end = torch.Event(device=current_omni_platform.device_type, enable_timing=True)
            self._helios_events.append((start, end, list(hidden_states.shape)))
            start.record()

        def after_forward(module, args, output):
            self._helios_events[-1][1].record()

        self._helios_timer_handles = [
            transformer.register_forward_pre_hook(before_forward, with_kwargs=True),
            transformer.register_forward_hook(after_forward),
        ]
        implementations = {}
        for module in transformer.modules():
            if isinstance(module, Attention):
                name = f"{type(module.attention).__module__}.{type(module.attention).__qualname__}"
                implementations[name] = implementations.get(name, 0) + 1
        if "vllm_omni.diffusion.attention.backends.flash_attn.FlashAttentionImpl" in implementations:
            from vllm_omni.diffusion.attention.backends.utils import fa

            return {**implementations, "flash_attention_bindings": describe_flash_bindings(fa)}
        return implementations

    def read_helios_timers(self) -> dict:
        if not self._helios_events:
            raise RuntimeError("No transformer forwards were recorded")
        self._helios_events[-1][1].synchronize()
        rows = [{"latent_shape": shape, "gpu_ms": start.elapsed_time(end)} for start, end, shape in self._helios_events]
        self._helios_events.clear()
        return {"steps": rows}


def one_worker_result(result: object) -> dict:
    """Unwrap stage/replica/worker RPC containers for this single-GPU benchmark."""
    while isinstance(result, list) and len(result) == 1:
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError(f"Expected one worker RPC result, got {type(result).__name__}")
    if result.get("supported") is False:
        raise RuntimeError(f"Worker benchmark RPC failed: {result}")
    return result


def normalize_stage_durations(durations: dict[str, float]) -> dict[str, float]:
    """Pipeline timers use seconds; orchestrator fields ending in _ms do not."""
    return {key: value if key.endswith("_ms") else value * 1000 for key, value in durations.items()}


def summarize_runs(records: list[dict]) -> dict[int, dict]:
    """Keep shape groups separate and exclude every warmup from all statistics."""
    measured = [row for row in records if not row["warmup"]]
    summary = {}
    for frames in sorted({row["num_frames"] for row in measured}):
        rows = [row for row in measured if row["num_frames"] == frames]
        times = [row["wall_ms"] for row in rows]
        summary[frames] = {
            "count": len(rows),
            "median_wall_ms": statistics.median(times),
            "min_wall_ms": min(times),
            "max_wall_ms": max(times),
            "peak_worker_reserved_mib": max(row["worker_peak_reserved_mib"] for row in rows),
            "median_transformer_gpu_ms": statistics.median(row["transformer_gpu_ms"] for row in rows),
            "median_transformer_forward_ms": statistics.median(row["mean_transformer_forward_ms"] for row in rows),
        }
    return summary


def expected_transformer_forwards(frames: int, extra_args: dict) -> int:
    """Count forwards for the Distilled stage-2, 33-frame-chunk workload."""
    base_steps = sum(extra_args["pyramid_num_inference_steps_list"])
    return base_steps * (frames // 33 + int(extra_args["is_amplify_first_chunk"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Pinned local Helios-Distilled checkpoint directory.")
    parser.add_argument("--backend", required=True, choices=["TORCH_SDPA", "FLASH_ATTN", "CUDNN_ATTN"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[33, 66])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("repeats and warmup must both be positive")
    if any(count < 33 or count % 33 for count in args.frames):
        parser.error("frame counts must be positive multiples of 33")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.frames)) != len(args.frames):
        parser.error("seed and frame lists must not contain duplicates")
    if not Path(args.model).is_dir():
        parser.error("model must be a local checkpoint directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty to preserve previous evidence")
    return args


def main() -> None:
    args = parse_args()
    # Set the selector before importing the engine and before spawning workers.
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = args.backend
    import numpy as np
    import torch
    from diffusers.utils import export_to_video

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("torch", "vllm", "vllm-omni", "transformers", "diffusers", "triton", "kernels", "fa3-fwd"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    extra_args = {
        "is_enable_stage2": True,
        "pyramid_num_inference_steps_list": [2, 2, 2],
        "is_amplify_first_chunk": True,
    }
    metadata = {
        "backend": args.backend,
        "model": str(Path(args.model).resolve()),
        "versions": versions,
        "python": platform.python_version(),
        "gpu": current_omni_platform.get_device_name(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "height": 384,
        "width": 640,
        "frames": args.frames,
        "seeds": args.seeds,
        "repeats": args.repeats,
        "warmup_per_shape": args.warmup,
        "dit_text_encoder_dtype": "bfloat16",
        "vae_dtype": "float32",
        "guidance_scale": 1.0,
        "extra_args": extra_args,
        "enforce_eager": True,
        "cache_backend": None,
        "cross_attention_cache_reset_per_request": True,
        "offload": False,
        "step_execution": False,
        "pipeline_stage_timers": True,
        "transformer_event_timers": True,
    }
    records = []

    def save_results() -> None:
        payload = {"metadata": metadata, "records": records, "summary": summarize_runs(records)}
        (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    save_results()
    load_start = time.perf_counter()
    omni = Omni(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        enable_diffusion_pipeline_profiler=True,
        step_execution=False,
        worker_extension_cls="benchmarks.diffusion.benchmark_helios_attention.HeliosBenchmarkWorker",
    )
    metadata["engine_startup_ms"] = (time.perf_counter() - load_start) * 1000
    try:
        metadata["attention_implementations"] = one_worker_result(
            omni.engine.collective_rpc(method="install_helios_timers")
        )
        if not metadata["attention_implementations"]:
            raise ValueError("No generic attention modules found in the transformer")
        for count in args.frames:
            cases = [(True, 0, 42, PROMPTS[0])] * args.warmup
            cases += [
                (False, repeat, seed, PROMPTS[index % len(PROMPTS)])
                for repeat in range(args.repeats)
                for index, seed in enumerate(args.seeds)
            ]
            for warmup, repeat, seed, prompt in cases:
                one_worker_result(omni.engine.collective_rpc(method="reset_helios_request_cache"))
                sampling = OmniDiffusionSamplingParams(
                    height=384,
                    width=640,
                    num_frames=count,
                    num_inference_steps=6,
                    guidance_scale=1.0,
                    generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(seed),
                    extra_args=dict(extra_args),
                )
                start = time.perf_counter()
                outputs = omni.generate({"prompt": prompt, "negative_prompt": "", "modalities": ["video"]}, sampling)
                elapsed_ms = (time.perf_counter() - start) * 1000
                transformer_timings = one_worker_result(omni.engine.collective_rpc(method="read_helios_timers"))[
                    "steps"
                ]
                expected_forwards = expected_transformer_forwards(count, extra_args)
                if len(transformer_timings) != expected_forwards:
                    raise ValueError(f"Expected {expected_forwards} forwards, got {len(transformer_timings)}")
                transformer_gpu_ms = sum(step["gpu_ms"] for step in transformer_timings)
                if not isinstance(outputs, list) or len(outputs) != 1:
                    raise ValueError("Expected exactly one completed video request")
                result = outputs[0]
                # Helios's VideoProcessor returns normalized float RGB frames.
                frames = result.images
                while isinstance(frames, list) and len(frames) == 1:
                    frames = frames[0]
                video = np.asarray(frames)
                if video.ndim == 5 and video.shape[0] == 1:
                    video = video[0]
                if video.shape != (count, 384, 640, 3):
                    raise ValueError(f"Unexpected decoded video shape: {video.shape}")
                if not np.isfinite(video).all() or video.min() < 0 or video.max() > 1:
                    raise ValueError("Expected finite video pixels in [0, 1]")
                video = np.ascontiguousarray(video, dtype=np.float32)
                record = {
                    "warmup": warmup,
                    "repeat": repeat,
                    "seed": seed,
                    "prompt": prompt,
                    "num_frames": count,
                    "wall_ms": elapsed_ms,
                    "transformer_timings": transformer_timings,
                    "transformer_gpu_ms": transformer_gpu_ms,
                    "transformer_forward_count": len(transformer_timings),
                    "mean_transformer_forward_ms": transformer_gpu_ms / len(transformer_timings),
                    "stage_durations_ms": normalize_stage_durations(result.stage_durations),
                    "worker_peak_reserved_mib": result.peak_memory_mb,
                    "shape": list(video.shape),
                    "sha256": hashlib.sha256(video.tobytes()).hexdigest(),
                }
                if not warmup:
                    name = f"f{count}-s{seed}-r{repeat}"
                    np.save(args.output_dir / f"{name}.npy", video, allow_pickle=False)
                    record["array"] = f"{name}.npy"
                    if repeat == 0:
                        export_to_video(list(video), str(args.output_dir / f"{name}.mp4"), fps=16)
                records.append(record)
                save_results()
                print(json.dumps(record), flush=True)
    finally:
        omni.close()
        save_results()


if __name__ == "__main__":
    main()
