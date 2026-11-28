# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure MiniMax-H3 staged Video/Audio decode in a same-process A/B/A.

Formal timings exclude output copies and precision/media collection. An
additional request checks actual outputs after repeated load/offload cycles.
"""

import argparse
import json
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 -- register the NPU device backend

from vllm_omni.diffusion.models.minimax_h3.component_cache_policy import resolve_component_cache_reuse
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from vllm_omni.diffusion.utils.chunked_video import quantize_chunk
from vllm_omni.platforms import current_omni_platform


def main() -> None:
    """Benchmark fixed latents without starting an inference server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("off", "on", "aba"), default="aba")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--formals", type=int, default=5)
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--model", type=Path, required=True, help="Task partition containing video_vae and audio_vae")
    parser.add_argument("--inputs", type=Path, required=True, help="torch.save((video_latent, audio_latent))")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.formals < 1:
        parser.error("--warmups must be nonnegative and --formals must be positive")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    if world > 1:
        dist.init_process_group("hccl", timeout=timedelta(minutes=10))
    video_z, audio_z = torch.load(args.inputs, map_location="cpu", weights_only=True)
    video_z = video_z.to(device)
    audio_z = audio_z.to(device) if rank == 0 else None

    def device_memory() -> dict[str, int]:
        """Sample allocator and device memory outside the timed interval."""
        free, total = torch.npu.mem_get_info(device)
        return {
            "allocated_bytes": torch.npu.memory_allocated(device),
            "reserved_bytes": torch.npu.memory_reserved(device),
            "free_bytes": free,
            "total_bytes": total,
        }

    def run_arm(arm: str, reuse: bool) -> None:
        """Measure one freshly constructed component pair."""
        reuse = resolve_component_cache_reuse({"component_cache_reuse": reuse})
        out = args.output_dir / arm
        if rank == 0:
            out.mkdir(parents=True, exist_ok=False)
        if world > 1:
            dist.barrier()

        def save(name: str, data: dict[str, object]) -> None:
            tmp = out / (name + ".tmp")
            tmp.write_text(json.dumps(data, indent=2) + "\n")
            tmp.replace(out / name)

        video = MiniMaxH3VideoVAE(
            str(args.model / "video_vae"),
            device=device,
            load_device=torch.device("cpu"),
            decode_only=True,
            trust_remote_code=True,
            component_cache_reuse=reuse,
        )
        if world > 1:
            video.set_parallel_size(world, process_group=dist.group.WORLD)
        audio = (
            MiniMaxH3AudioVAE(
                str(args.model / "audio_vae"),
                device=device,
                load_device=torch.device("cpu"),
                decode_only=True,
                trust_remote_code=True,
                component_cache_reuse=reuse,
            )
            if rank == 0
            else None
        )
        purges = 0
        original_empty_cache = current_omni_platform.empty_cache

        def counted_empty_cache() -> None:
            nonlocal purges
            purges += 1
            original_empty_cache()

        save(
            f"residency-rank{rank}.json",
            {
                "video_stager": video._decoder_stager is not None or video._stager is not None,
                "video_bounded_owner": getattr(video, "_omni_component_cache", None) is not None,
                "retain_live_video_storage": (video._decoder_stager or video._stager).retain_device_storage,
                "component_cache_reuse": reuse,
            },
        )
        current_omni_platform.empty_cache = counted_empty_cache
        try:
            times = []
            for i in range(args.warmups + args.formals):
                torch.npu.synchronize()
                if world > 1:
                    dist.barrier()
                torch.npu.reset_peak_memory_stats()
                prior_purges = purges
                start = time.perf_counter()
                video._load_part_to_device("decoder")
                loaded = time.perf_counter()
                with (
                    torch.inference_mode(),
                    current_omni_platform.create_autocast_context(
                        device_type="npu", dtype=video.decode_dtype, enabled=True
                    ),
                ):
                    decoded = video.decode_latent(video_z)
                torch.npu.synchronize()
                decoded_at = time.perf_counter()
                del decoded
                video._offload_part_to_cpu("decoder")
                video_done = time.perf_counter()
                audio_load_s = audio_compute_s = audio_offload_s = 0.0
                if rank == 0:
                    audio.load_to_device()
                    audio_loaded = time.perf_counter()
                    waveform = audio.decode_latent(audio_z)
                    torch.npu.synchronize()
                    audio_decoded = time.perf_counter()
                    del waveform
                    audio.offload_to_cpu()
                    audio_done = time.perf_counter()
                    audio_load_s = audio_loaded - video_done
                    audio_compute_s = audio_decoded - audio_loaded
                    audio_offload_s = audio_done - audio_decoded
                torch.npu.synchronize()
                if world > 1:
                    dist.barrier()
                elapsed = time.perf_counter() - start
                kind = "warmup" if i < args.warmups else "formal"
                if kind == "formal":
                    times.append(elapsed)
                row = {
                    "rank": rank,
                    "kind": kind,
                    "iteration": i,
                    "lifecycle_host_s": elapsed,
                    "video_load_submit_s": loaded - start,
                    "video_compute_and_copy_wait_s": decoded_at - loaded,
                    "video_offload_s": video_done - decoded_at,
                    "audio_load_submit_s": audio_load_s,
                    "audio_compute_and_copy_wait_s": audio_compute_s,
                    "audio_offload_s": audio_offload_s,
                    "platform_purge_calls": purges - prior_purges,
                    "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                    "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                    "offloaded": device_memory(),
                    "component_cache_reuse": reuse,
                    "scope": "staged Video+Audio load/decode/offload; excludes MP4 and DiT",
                }
                save(f"lifecycle-{i}-rank{rank}.json", row)
                print(row, flush=True)

            if args.precision:
                video._load_part_to_device("decoder")
                with (
                    torch.inference_mode(),
                    current_omni_platform.create_autocast_context(
                        device_type="npu", dtype=video.decode_dtype, enabled=True
                    ),
                ):
                    decoded = video.decode_latent(video_z)
                    if rank == 0:
                        torch.save(decoded.permute(0, 2, 3, 4, 1).contiguous().cpu(), out / "video-plain-rgb.pt")
                    del decoded
                    floats, rgbs = [], []

                    def collect(chunk: torch.Tensor) -> None:
                        floats.append(chunk.cpu())
                        rgbs.append(quantize_chunk(chunk, (0.0, 1.0)).cpu())

                    video.decode_with_chunks(video_z, on_chunk=collect)
                    if rank == 0:
                        torch.save(torch.cat(floats, dim=2), out / "video-float.pt")
                        torch.save(torch.cat(rgbs, dim=1), out / "video-rgb.pt")
                    del floats, rgbs
                video._offload_part_to_cpu("decoder")
                if rank == 0:
                    audio.load_to_device()
                    waveform = audio.decode_latent(audio_z)
                    torch.save(waveform.cpu(), out / "audio-float.pt")
                    del waveform
                    audio.offload_to_cpu()
                torch.npu.synchronize()
                if world > 1:
                    dist.barrier()
            if rank == 0:
                save(
                    "summary.json",
                    {
                        "arm": arm,
                        "times_s": times,
                        "mean_s": statistics.mean(times) if times else None,
                        "std_s": statistics.stdev(times) if len(times) > 1 else None,
                        "complete": len(times) == args.formals,
                        "scope": "staged Video+Audio load/decode/offload; excludes MP4 and DiT",
                    },
                )
        finally:
            current_omni_platform.empty_cache = original_empty_cache
            del video, audio
            torch.npu.synchronize()
            before_purge = device_memory()
            original_empty_cache()
            torch.npu.synchronize()
            save(
                f"teardown-rank{rank}.json",
                {
                    "before_forced_purge": before_purge,
                    "after_forced_purge": device_memory(),
                },
            )
        if world > 1:
            dist.barrier()

    try:
        arms = (
            (("a1-off", False), ("b-on", True), ("a2-off", False))
            if args.arm == "aba"
            else ((args.arm, args.arm == "on"),)
        )
        for arm, reuse in arms:
            run_arm(arm, reuse)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
