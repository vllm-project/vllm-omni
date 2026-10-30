# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 NPU image-encoder latency and latent exports (not E2E).

Run each configuration in a fresh process on the same physical devices.
Environment flags select opt-in optimizations before model construction.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import vllm
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import _prepare_encoder_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local video_vae checkpoint directory")
    parser.add_argument("--images", nargs="+", required=True, help="Reference images in request order")
    parser.add_argument("--output", type=Path, required=True, help="Unique output directory for this arm")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="After timing, extra warmup + one L1 rank0 capture")
    parser.add_argument("--offload", action="store_true", help="Load/offload encoder around each six-image request")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeat < 3:
        parser.error("use at least one warmup and three measured requests")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    torch.npu.set_device(device)
    if world > 1:
        dist.init_process_group("hccl")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"rank{rank}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    assert torch.npu.conv.allow_hf32, "benchmark requires the default HF32 convolution policy"
    vae = MiniMaxH3VideoVAE(
        args.model,
        device=device,
        load_device=torch.device("cpu") if args.offload else None,
        encode_only=True,
        trust_remote_code=True,
    ).eval()
    if world > 1:
        vae.set_parallel_size(world, process_group=dist.group.WORLD)
    installed = dict(
        temporal_conv=sum(hasattr(module, "_omni_image_conv_forward") for module in vae.model.modules()),
        norm_silu=sum(bool(getattr(module, "_omni_image_norm_installed", False)) for module in vae.model.modules()),
        weight_cache=sum(
            bool(getattr(module, "_omni_image_conv_cache_enabled", False)) for module in vae.model.modules()
        ),
    )
    images = [Image.open(path).convert("RGB") for path in args.images]
    prepared = _prepare_encoder_images("ref2va", images, height=args.height, width=args.width)

    def synchronize():
        if world > 1:
            dist.barrier()
        torch.npu.synchronize()

    def encode():
        if args.offload:
            vae._load_part_to_device("encoder")
        try:
            return [vae.encode_image(image) for image in prepared]
        finally:
            if args.offload:
                vae._offload_part_to_cpu("encoder")

    def cache_snapshot():
        entries = [getattr(module, "_omni_image_conv_weight", None) for module in vae.model.modules()]
        entries = [entry for entry in entries if entry is not None]
        return dict(
            entries=len(entries),
            bytes=sum(entry[2].numel() * entry[2].element_size() for entry in entries),
        )

    rows, outputs = [], []
    with torch.inference_mode():
        for repetition in range(args.warmup + args.repeat):
            synchronize()
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            latents = encode()
            torch.npu.synchronize()
            end = time.perf_counter()
            rows.append(
                dict(
                    repetition=repetition,
                    warmup=repetition < args.warmup,
                    wall_s=end - start,
                    start=start,
                    end=end,
                    peak_allocated_bytes=torch.npu.max_memory_allocated(),
                    peak_reserved_bytes=torch.npu.max_memory_reserved(),
                    weight_cache=cache_snapshot(),
                )
            )
            if args.offload:
                assert rows[-1]["weight_cache"]["entries"] == 0, "derived weights survived encoder offload"
            outputs.append(latents)
        # Hashing, host transfer and I/O are excluded from every timer.
        arrays = []
        for repetition, latents in enumerate(outputs):
            for index, latent in enumerate(latents):
                array = latent.detach().cpu().contiguous().numpy()
                assert np.isfinite(array).all()
                name = f"rank{rank}-call{repetition}-image{index}.npy"
                np.save(args.output / name, array)
                arrays.append(
                    dict(
                        path=name,
                        shape=list(array.shape),
                        dtype=str(array.dtype),
                        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                    )
                )
        if args.profile:
            synchronize()
            encode()  # independent profile warmup
            synchronize()
            if rank == 0:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                    with_stack=True,
                    with_modules=True,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.output / "profile"), analyse_flag=False
                    ),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1, data_simplification=False
                    ),
                ):
                    with torch.profiler.record_function("H3/six-reference-image-encode"):
                        encode()
                    torch.npu.synchronize()
            else:
                encode()
                torch.npu.synchronize()
            synchronize()
    formal = rows[args.warmup :]
    versions = {name: importlib.metadata.version(name) for name in ["torch", "torch-npu", "vllm", "vllm-omni"]}
    result = dict(
        scope=(
            "VAE encoder residency lifecycle used by DLO, no DiT or full pipeline"
            if args.offload
            else "resident VAE image encoder, no text/DiT/decoder/MP4/DLO"
        ),
        offload=args.offload,
        rank=rank,
        world_size=world,
        installed=installed,
        rows=rows,
        mean_s=statistics.mean(row["wall_s"] for row in formal),
        min_s=min(row["wall_s"] for row in formal),
        max_s=max(row["wall_s"] for row in formal),
        continuous_s=formal[-1]["end"] - formal[0]["start"],
        latents=arrays,
        versions=versions,
        vllm_source=vllm.__file__,
        hf32=bool(torch.npu.conv.allow_hf32),
        input_sha256=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.images],
        prepared_rgb_sha256=[hashlib.sha256(np.asarray(image).tobytes()).hexdigest() for image in prepared],
        adapter_source=__import__(MiniMaxH3VideoVAE.__module__, fromlist=["__file__"]).__file__,
        flags={key: value for key, value in os.environ.items() if key.startswith("VLLM_OMNI_H3_NPU_IMAGE_")},
        prepared_sizes=[list(image.size) for image in prepared],
        profile=dict(
            enabled=args.profile,
            rank=0,
            warmup=1,
            formal=1,
            level="L1",
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            offline_analysis=True,
        ),
    )
    result_path.write_text(json.dumps(result, indent=2))
    if world > 1:
        dist.destroy_process_group()
    print(json.dumps(dict(rank=rank, mean_s=result["mean_s"], result=str(result_path))), flush=True)


if __name__ == "__main__":
    main()
