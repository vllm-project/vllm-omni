# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Multi-rank Wan VAE encode/decode: tile vs rank split vs spatial shard (P2P / all-gather halos).

Times the Wan VAE encode (I2V condition: a seeded image as frame 0, zeros after)
and decode (seeded latents) across every rank, and compares each output with a
1-card untiled encode/decode of the same input computed on rank 0. Launch one
process per device with ``torchrun``; only rank 0 prints::

    torchrun --nproc-per-node 4 benchmarks/diffusion/bench_wan_vae_spatial_shard.py \
        --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --json wan_vae_4r.json

Cases (``--cases``, all by default):

- ``tile``: ``vae_parallel_mode=tile`` with native tiling, the default.
- ``tile_rank_split``: ``tile`` with ``VLLM_OMNI_WAN_VAE_ENCODE_SPLIT=rank``
  (one encode piece per rank); decode is the same as ``tile``.
- ``shard_p2p``: ``spatial_shard_<--split-dim>`` without native tiling (as the
  registry configures it) with point-to-point halos, the pre-all-gather path.
- ``shard_allgather``: the same with all-gather halos (the XPU default).

The two shard cases force the halo path by patching
``wan_spatial_shard._halo_exchange_uses_all_gather``, so both run the same code
revision. Each timed sample is bracketed by a device synchronize and a barrier
on every rank, so it is the slowest rank's wall time. Inputs, warm-up and the
reference are outside timing; rel_l2 = ||out - ref|| / ||ref|| in fp32.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders import wan_spatial_shard
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform as plat

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
CASES = ("tile", "tile_rank_split", "shard_p2p", "shard_allgather")
ENCODE_SPLIT_ENV = "VLLM_OMNI_WAN_VAE_ENCODE_SPLIT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF id or local path of the pipeline")
    parser.add_argument("--subfolder", default="vae")
    parser.add_argument("--size", default="1280x720", help="Video WxH")
    parser.add_argument("--frames", type=int, default=81, help="Video frame count (1 + 4k)")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--split-dim", choices=("width", "height"), default="width")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--stages", default="encode,decode")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None, help="Write config, environment and every sample here")
    return parser.parse_args()


def sync(all_ranks: bool) -> None:
    plat.synchronize()
    if all_ranks:
        dist.barrier()


def timed(fn, warmup: int, iters: int, all_ranks: bool = True) -> tuple[torch.Tensor, list[float]]:
    out = None
    for _ in range(warmup):
        out = fn()
    samples = []
    for _ in range(iters):
        sync(all_ranks)
        start = time.perf_counter()
        out = fn()
        sync(all_ranks)
        samples.append(time.perf_counter() - start)
    return out, samples


def compare(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    diff = out.float() - ref
    return {"rel_l2": (diff.norm() / ref.norm()).item(), "max_abs": diff.abs().max().item()}


def environment() -> dict[str, str]:
    import diffusers
    import vllm

    src = Path(wan_spatial_shard.__file__).resolve().parent
    try:
        rev = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(src), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        rev = "unknown"
    return {
        "vllm_omni_revision": rev,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "device": getattr(torch, plat.device_type).get_device_name(0),
        "host": platform.node(),
    }


def configure(vae: DistributedAutoencoderKLWan, case: str, world_size: int, split_dim: str) -> None:
    os.environ[ENCODE_SPLIT_ENV] = "rank" if case == "tile_rank_split" else "tile"
    if case.startswith("shard_"):
        use_all_gather = case == "shard_allgather"
        wan_spatial_shard._halo_exchange_uses_all_gather = lambda: use_all_gather
        vae.use_tiling = False
        vae.set_parallel_size(world_size, mode=f"spatial_shard_{split_dim}")
    else:
        vae.use_tiling = True
        vae.set_parallel_size(world_size, mode="tile")


def main() -> None:
    args = parse_args()
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = plat.get_torch_device(local_rank)
    plat.set_device(device)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank, backend=plat.dist_backend)
    initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size, backend=plat.dist_backend)
    dtype = DTYPES[args.dtype]
    width, height = (int(v) for v in args.size.split("x"))
    stages = args.stages.split(",")

    vae = DistributedAutoencoderKLWan.from_pretrained(args.model, subfolder=args.subfolder, torch_dtype=dtype)
    vae = vae.to(device).eval()
    gen = torch.Generator().manual_seed(args.seed)
    x = torch.zeros((1, 3, args.frames, height, width), dtype=dtype)
    x[:, :, :1] = torch.rand((1, 3, 1, height, width), generator=gen) * 2 - 1
    x = x.to(device)
    ratio_t, ratio_s = vae.config.scale_factor_temporal, vae.config.scale_factor_spatial
    latent_shape = (1, vae.config.z_dim, (args.frames - 1) // ratio_t + 1, height // ratio_s, width // ratio_s)
    z = torch.randn(latent_shape, generator=gen, dtype=dtype).to(device)

    def encode() -> torch.Tensor:
        return vae.encode(x).latent_dist.mode()

    def decode() -> torch.Tensor:
        return vae.decode(z, return_dict=False)[0]

    stage_fns = {"encode": encode, "decode": decode}
    results: dict = {"config": vars(args) | {"world_size": world_size}, "reference": {}, "cases": {}}
    refs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        if rank == 0:
            results["environment"] = environment()
            vae.use_tiling = False
            vae.set_parallel_size(1, mode="tile")
            for stage in stages:
                out, samples = timed(stage_fns[stage], args.warmup, args.iters, all_ranks=False)
                refs[stage] = out.float()
                results["reference"][stage] = {"samples_s": samples, "median_s": statistics.median(samples)}
                print(
                    f"reference 1-card untiled {stage}: median {statistics.median(samples):.3f}s "
                    f"samples {[round(s, 3) for s in samples]}",
                    flush=True,
                )
        dist.barrier()

        for case in args.cases.split(","):
            configure(vae, case, world_size, args.split_dim)
            for stage in stages:
                out, samples = timed(stage_fns[stage], args.warmup, args.iters)
                if rank != 0:
                    continue
                row = {"samples_s": samples, "median_s": statistics.median(samples)} | compare(out, refs[stage])
                if stage == "encode" and case.startswith("shard_"):
                    row["sharded_encode_taken"] = vae._spatial_shard_encode_enabled(x)
                results["cases"].setdefault(case, {})[stage] = row
                print(
                    f"{stage:6s} {case:16s} {world_size} ranks: median {row['median_s']:.3f}s "
                    f"samples {[round(s, 3) for s in samples]} rel_l2 {row['rel_l2']:.3e} "
                    f"max_abs {row['max_abs']:.3e}",
                    flush=True,
                )
            dist.barrier()

    if rank == 0 and args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
