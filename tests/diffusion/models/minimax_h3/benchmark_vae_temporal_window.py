# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-checkpoint OFF/ON component benchmark; excludes DiT, audio and MP4.

Run with torchrun, a trusted local video-VAE checkpoint, and an Ascend NPU runtime.
Results are per rank; compare the maximum rank latency for each repetition.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import vllm_omni
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.models.minimax_h3.vae_temporal import install_temporal_stream_patches
from vllm_omni.platforms import current_omni_platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted local video_vae checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-shape", type=int, nargs=5, default=[1, 24, 20, 30, 52])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 2 or min(args.latent_shape) < 1:
        parser.error("Use positive latent dimensions, at least one warmup and two repetitions")

    platform = current_omni_platform
    if platform.device_type != "npu":
        parser.error("This benchmark requires an Ascend NPU runtime")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"{platform.device_type}:{local_rank}")
    platform.set_device(device)
    device_module = torch.get_device_module(device)
    dist.init_process_group(platform.dist_backend, timeout=timedelta(seconds=300))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        vae = MiniMaxH3VideoVAE(
            args.model, device=device, decode_only=True, trust_remote_code=True, enable_temporal_scheduling=True
        )
        vae.set_parallel_size(world_size, process_group=dist.group.WORLD)
        latent = torch.randn(args.latent_shape, generator=torch.Generator(device="cpu").manual_seed(1101))
        input_sha = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
        latent = latent.to(device)
        records = []
        reference_sha = None
        args.output.mkdir(parents=True, exist_ok=True)

        # Both modes warm up before measurement; alternating order limits
        # systematic first-mode bias. Cache release and hashing are not timed.
        for repetition in range(-args.warmup, args.repeat):
            for window in (1, 2) if repetition % 2 == 0 else (2, 1):
                install_temporal_stream_patches(
                    vae.model,
                    pixel_transform=vae._pixel_transform,
                    window_decode=None if window == 1 else vae._decode_temporal_window,
                    window_plan=None if window == 1 else vae._plan_temporal_schedule,
                )
                vae.enable_temporal_scheduling = window != 1
                device_module.empty_cache()
                device_module.reset_peak_memory_stats()
                platform.synchronize()
                dist.barrier()
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast(platform.device_type, dtype=torch.float16):
                    output = vae.decode_latent(latent)
                platform.synchronize()
                seconds = time.perf_counter() - start
                peak_bytes = device_module.max_memory_allocated()
                cpu = output.cpu()
                sha = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
                if reference_sha is None:
                    reference_sha = sha
                if sha != reference_sha:
                    raise AssertionError(f"Output differs on rank {rank}, window {window}, repetition {repetition}")
                if repetition >= 0:
                    records.append(
                        {
                            "enabled": window != 1,
                            "repetition": repetition,
                            "seconds": seconds,
                            "peak_bytes": peak_bytes,
                            "sha256": sha,
                            "output_shape": list(cpu.shape),
                        }
                    )
                del output, cpu
        summary = {}
        for window in (1, 2):
            selected = [row for row in records if row["enabled"] == (window != 1)]
            times = [row["seconds"] for row in selected]
            summary["on" if window != 1 else "off"] = {
                "mean_seconds": statistics.mean(times),
                "min_seconds": min(times),
                "max_seconds": max(times),
                "peak_bytes": max(row["peak_bytes"] for row in selected),
            }
        report = {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": str(device),
            "import": vllm_omni.__file__,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "vllm-omni")},
            "model": args.model,
            "latent_shape": args.latent_shape,
            "input_sha256": input_sha,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "stack_tiling": vae.model.stack_tiling,
            "records": records,
            "summary": summary,
        }
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"rank": rank, "summary": summary}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
