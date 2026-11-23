# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare sparse Ring K/V demand planners on one NCCL backend."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec
from vllm_omni.diffusion.attention.sparsp.placement import interval_periodic_layout
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform


@torch.inference_mode()
def benchmark(args, device):
    world, rank = dist.get_world_size(), dist.get_rank()
    archive = torch.load(args.qkv, map_location="cpu", weights_only=True)
    q, k, v = (archive[name] for name in ("q", "k", "v"))
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[1] % (world * 64):
        raise ValueError("capture must contain equal BSHD tensors with 64-token K/V blocks on every rank")
    _, _, heads, dim = q.shape
    if args.placement_grid is None:
        inputs = [tensor.chunk(world, dim=1)[rank] for tensor in (q, k, v)]
    else:
        layout = interval_periodic_layout(tuple(args.placement_grid), world, torch.device("cpu"))
        if layout.permutation.numel() != q.shape[1]:
            raise ValueError("placement grid does not match the captured video sequence")
        order = layout.permutation.chunk(world)[rank]
        inputs = [tensor.index_select(1, order) for tensor in (q, k, v)]
    inputs = [tensor.to(device).contiguous() for tensor in inputs]
    full = SparSPSpec(cdf=args.cdf)
    demand = replace(full, routing="selective_direct")
    manifest = replace(demand, demand_planner="manifest")
    per_hop = replace(demand, demand_planner="per_hop")
    policy_order = [full, per_hop, demand, manifest, manifest, demand, per_hop, full]
    reference = None
    results = []
    for spec in policy_order:
        # The capture supplies Q/K/V directly and does not need model weights.
        config = OmniDiffusionConfig.from_kwargs(
            dtype=q.dtype,
            enforce_eager=True,
            parallel_config=DiffusionParallelConfig(ring_degree=world),
            diffusion_attention_config={"default": {"backend": "SPARSP_ATTN", "sparsp": asdict(spec)}},
        )
        samples = []
        with set_forward_context(omni_diffusion_config=config), set_current_diffusion_config(config):
            layer = Attention(num_heads=heads, head_size=dim, softmax_scale=dim**-0.5, causal=False)
            for iteration in range(args.warmup + args.iterations):
                current_omni_platform.synchronize()
                dist.barrier()
                start = time.perf_counter()
                result = layer(*inputs)
                current_omni_platform.synchronize()
                elapsed = torch.tensor((time.perf_counter() - start) * 1e3, device=device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if iteration >= args.warmup:
                    samples.append(elapsed.item())
        executor = get_sp_group().sparsp_executors.pop((heads, dim, spec))
        counters = torch.tensor(
            [executor.payload_bytes, executor.dense_payload_bytes], dtype=torch.int64, device=device
        )
        dist.all_reduce(counters)
        executor.close()
        if reference is None:
            reference = result
        same_bits = torch.equal(reference.view(torch.int16), result.view(torch.int16))
        equal = torch.tensor(same_bits and torch.isfinite(result).all().item(), device=device)
        dist.all_reduce(equal, op=dist.ReduceOp.MIN)
        if not equal.item():
            raise RuntimeError(f"SparSP {spec} differs from full-ring NCCL or produced nonfinite output")
        results.append(
            {
                **asdict(spec),
                "sample_ms": samples,
                "mean_ms": statistics.mean(samples),
                "kv_payload_bytes_global": counters[0].item(),
                "relay_kv_payload_bytes_global": counters[1].item(),
                "bitwise_equal": True,
            }
        )
    direct_bytes = {case["kv_payload_bytes_global"] for case in results if case["routing"] == "selective_direct"}
    if len(direct_bytes) != 1:
        raise RuntimeError("SparSP demand planners transferred different numbers of K/V bytes")
    return {
        "input_shape": list(q.shape),
        "placement_grid": args.placement_grid,
        "kv_wire_dtype": str(k.dtype),
        "nccl_version": NCCLLibrary().ncclGetVersion(),
        "world_size": world,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qkv", type=Path, required=True)
    parser.add_argument(
        "--placement-grid",
        type=int,
        nargs=3,
        metavar=("FRAMES", "HEIGHT", "WIDTH"),
        help="latent video grid for interval-periodic placement of the captured tokens",
    )
    parser.add_argument("--cdf", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.iterations) < 1:
        parser.error("warmup and iterations must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    current_omni_platform.set_device(device)
    init_distributed_environment()
    world = dist.get_world_size()
    try:
        if world not in (2, 3, 4):
            raise ValueError("run on two to four GPUs")
        initialize_model_parallel(sequence_parallel_size=world, ring_degree=world)
        report = benchmark(args, device)
        if dist.get_rank() == 0:
            checksum = hashlib.sha256()
            with args.qkv.open("rb") as capture:
                for chunk in iter(lambda: capture.read(1024 * 1024), b""):
                    checksum.update(chunk)
            report.update(
                qkv_sha256=checksum.hexdigest(),
                warmup=args.warmup,
                iterations=args.iterations,
                source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        destroy_distributed_env()


if __name__ == "__main__":
    main()
