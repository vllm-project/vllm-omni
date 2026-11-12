# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the main/PR VAE adapters on identical captured decoder inputs."""

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

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE
from vllm_omni.diffusion.utils.chunked_video import quantize_chunk
from vllm_omni.platforms import current_omni_platform

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True)
parser.add_argument("--warmups", type=int, default=3)
parser.add_argument("--formals", type=int, default=3)
parser.add_argument("--precision", action="store_true")
parser.add_argument("--model", type=Path, required=True, help="Task partition containing video_vae and audio_vae")
parser.add_argument("--inputs", type=Path, required=True, help="torch.save((video_latent, audio_latent))")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
torch.npu.set_device(rank)
device = torch.device("npu", rank)
if world > 1:
    dist.init_process_group("hccl", timeout=timedelta(minutes=10))
root = args.output_dir
if rank == 0:
    root.mkdir(parents=True, exist_ok=False)
if world > 1:
    dist.barrier()


def save(name, data):
    tmp = root / (name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(root / name)


def synchronize():
    torch.npu.synchronize()
    if world > 1:
        dist.barrier()


video_input, audio_input = torch.load(args.inputs, map_location="cpu", weights_only=True)
save(
    f"input-rank{rank}.json",
    {
        "video_shape": list(video_input.shape),
        "audio_shape": list(audio_input.shape),
        "arm": args.arm,
        "device": str(device),
        "world_size": world,
        "precision": args.precision,
    },
)
video = MiniMaxH3VideoVAE(
    str(args.model / "video_vae"),
    device=device,
    load_device=torch.device("cpu"),
    decode_only=True,
    trust_remote_code=True,
)
if world > 1:
    video.set_parallel_size(world, process_group=dist.group.WORLD)
video._load_part_to_device("decoder")
loaded = {
    "native_code_module": video.model.__class__.__module__,
    "decoder_class": video.model.decoder.__class__.__name__,
    "component_group_marker": getattr(video.model, "_h3_component_group_supported", False),
    "native_stack_tiling": getattr(video.model, "stack_tiling", None),
    "reshape_installed": getattr(video, "_h3_vae_reshape_installed", False),
    "fia_installed": hasattr(video.model.decoder, "_h3_npu_fia_stats"),
}
save(f"video-loaded-rank{rank}.json", loaded)
video_input = video_input.to(device)
times = []
output = None
for i in range(args.warmups + args.formals + int(args.precision)):
    synchronize()
    torch.npu.reset_peak_memory_stats()
    start = time.perf_counter()
    label = "warmup" if i < args.warmups else "formal" if i < args.warmups + args.formals else "precision"
    floats, rgbs = [], []
    with (
        torch.inference_mode(),
        current_omni_platform.create_autocast_context(device_type="npu", dtype=video.decode_dtype, enabled=True),
    ):
        if label == "precision":

            def collect(chunk):
                floats.append(chunk.cpu())
                rgbs.append(quantize_chunk(chunk, (0.0, 1.0)).cpu())

            video.decode_with_chunks(video_input, on_chunk=collect)
            decoded = torch.cat(floats, dim=2) if rank == 0 else video_input.new_empty((0,))
        else:
            decoded = video.decode_latent(video_input)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - start
    if world > 1:
        value = torch.tensor([elapsed], device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        max_elapsed = value.item()
    else:
        max_elapsed = elapsed
    label = "warmup" if i < args.warmups else "formal" if i < args.warmups + args.formals else "precision"
    row = {
        "kind": label,
        "iteration": i,
        "rank_host_s": elapsed,
        "max_rank_host_s": max_elapsed,
        "shape": list(decoded.shape),
        "dtype": str(decoded.dtype),
        "fia": getattr(video.model.decoder, "_h3_npu_fia_stats", None),
        "reshape": getattr(video.model, "_h3_vae_reshape_last_stats", None),
    }
    row.update(
        peak_allocated_bytes=torch.npu.max_memory_allocated(), peak_reserved_bytes=torch.npu.max_memory_reserved()
    )
    save(f"video-{i}-rank{rank}.json", row)
    print(args.arm, rank, label, row, flush=True)
    if label == "formal":
        times.append(max_elapsed)
    if label == "precision" and rank == 0:
        torch.save(torch.cat(rgbs, dim=1), root / "video-rgb.pt")
        torch.save(decoded, root / "video-float.pt")
    if label == "formal" and rank == 0 and i == args.warmups + args.formals - 1:
        rgb = (
            decoded.permute(0, 2, 3, 4, 1).contiguous()
            if decoded.dtype == torch.uint8
            else quantize_chunk(decoded, (0.0, 1.0))
        )
        torch.save(rgb.cpu(), root / "video-plain-rgb.pt")
    del decoded
video.offload_to_cpu()
del video, video_input
synchronize()

audio_times = []
if rank == 0:
    audio = MiniMaxH3AudioVAE(
        str(args.model / "audio_vae"),
        device=device,
        load_device=torch.device("cpu"),
        decode_only=True,
        trust_remote_code=True,
    )
    audio.load_to_device()
    modules = [m for m in audio.model.decoder.modules() if hasattr(m, "_h3_dense_fir")]
    save(
        "audio-loaded-rank0.json",
        {
            "fir_modules": len(modules),
            "derived_devices": sorted(
                {str(b.device) for m in modules for n, b in m.named_buffers() if n.startswith("_h3_")}
            ),
            "derived_dtype": sorted(
                {str(b.dtype) for m in modules for n, b in m.named_buffers() if n.startswith("_h3_")}
            ),
        },
    )
    audio_input = audio_input.to(device)
    for i in range(args.warmups + args.formals + int(args.precision)):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        start = time.perf_counter()
        waveform = audio.decode_latent(audio_input)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        label = "warmup" if i < args.warmups else "formal" if i < args.warmups + args.formals else "precision"
        row = {
            "kind": label,
            "iteration": i,
            "host_s": elapsed,
            "shape": list(waveform.shape),
            "dtype": str(waveform.dtype),
            "fir_c16_calls": sum(getattr(m, "_h3_fir_c16_calls", 0) for m in modules),
        }
        row.update(
            peak_allocated_bytes=torch.npu.max_memory_allocated(), peak_reserved_bytes=torch.npu.max_memory_reserved()
        )
        save(f"audio-{i}-rank0.json", row)
        print(args.arm, "audio", row, flush=True)
        if label == "formal":
            audio_times.append(elapsed)
        if label == "precision":
            torch.save(waveform.cpu(), root / "audio-float.pt")
        del waveform
    audio.offload_to_cpu()
    del audio
synchronize()
if rank == 0:
    save(
        "summary.json",
        {
            "arm": args.arm,
            "video_max_rank_s": times,
            "video_mean_s": statistics.mean(times) if times else None,
            "audio_rank0_s": audio_times,
            "audio_mean_s": statistics.mean(audio_times) if audio_times else None,
            "full_checkpoint": True,
            "inputs": str(args.inputs),
            "model": str(args.model),
            "scope": "resident decoder compute; excludes staging, DiT and media encoding",
            "status": "complete",
        },
    )
if world > 1:
    dist.destroy_process_group()
