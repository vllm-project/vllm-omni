# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sweep the strict Ulysses a2a transports over message sizes and print the tables for ``ulysses_a2a_backend="auto"``.

Like vLLM's ``benchmarks/kernels/benchmark_device_communicators.py``: for each Ulysses world size it times a full
attention layer (Q/K/V exchange, attention, output exchange) with every transport forced explicitly (``nccl``,
``copy_engine``, ``copy_engine_overlap``, ``flashinfer``), over a range of sequence lengths, then prints the entries
for ``vllm_omni/diffusion/distributed/ulysses_transport_tables.py`` keyed by this GPU's compute capability.

A transport's threshold is the smallest swept size from which it beats the next transport in the dispatch chain
(FlashInfer -> copy-engine overlap -> copy engine -> NCCL) by at least ``--min-gain`` at every larger swept size.
Sweep sizes around the workloads you serve: dispatch outside the swept range falls through to the next transport.

    python benchmarks/diffusion/benchmark_ulysses_transports.py --world-sizes 2,4,8 \\
        --heads 56 --head-dim 128 --seqs 8192,16384,24576,37888,49152
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.ulysses_transport import overlap_chunk_count
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.platforms import current_omni_platform

TRANSPORTS = ("nccl", "copy_engine", "copy_engine_overlap", "flashinfer")


def _worker(rank: int, world_size: int, args: argparse.Namespace, result_file: str) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(args.port + world_size),
        }
    )
    dtype = torch.bfloat16
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    init_distributed_environment()
    initialize_model_parallel(
        data_parallel_size=1,
        cfg_parallel_size=1,
        sequence_parallel_size=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    layers = {}
    for transport in args.transports:
        parallel_config = DiffusionParallelConfig(
            sequence_parallel_size=world_size,
            ulysses_degree=world_size,
            ulysses_a2a_backend=transport,
            ulysses_overlap_chunks=args.overlap_chunks if transport == "copy_engine_overlap" else 0,
        )
        kwargs = {"diffusion_attention_backend": args.attention_backend} if args.attention_backend else {}
        od_config = OmniDiffusionConfig.from_kwargs(
            model="benchmark", dtype=dtype, parallel_config=parallel_config, **kwargs
        )
        with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
            try:
                layers[transport] = (
                    od_config,
                    Attention(
                        num_heads=args.heads,
                        head_size=args.head_dim,
                        causal=False,
                        softmax_scale=args.head_dim**-0.5,
                        num_kv_heads=None,
                        scatter_idx=2,
                        gather_idx=1,
                        use_sync=False,
                    ),
                )
            except Exception as error:  # e.g. FlashInfer without UlyssesCommunicator
                if rank == 0:
                    print(f"{world_size} GPUs: {transport} unavailable: {error}", flush=True)
    if rank == 0:
        name = next(iter(layers.values()))[1].attn_backend.get_name()
        print(f"{world_size} GPUs: attention backend {name}", flush=True)

    def timed(transport: str, q, k, v, iters: int) -> float:
        od_config, attn = layers[transport]
        with set_forward_context(omni_diffusion_config=od_config), set_current_diffusion_config(od_config):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.accelerator.synchronize()
            dist.barrier()
            start.record()
            for _ in range(iters):
                attn(q, k, v)
            end.record()
            torch.accelerator.synchronize()
        elapsed = torch.tensor([start.elapsed_time(end) / iters], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)  # the slowest rank sets the layer time
        return elapsed.item()

    results: dict[str, dict[int, float]] = {t: {} for t in layers}
    for seq in args.seqs:
        s_local = seq // world_size
        q, k, v = (torch.randn(1, s_local, args.heads, args.head_dim, device=device, dtype=dtype) for _ in range(3))
        for transport in layers:
            timed(transport, q, k, v, args.warmup)
        # Interleave the transports so clock and power drift hit all of them alike; keep the median round.
        rounds: dict[str, list[float]] = {t: [] for t in layers}
        for _ in range(args.rounds):
            for transport in layers:
                rounds[transport].append(timed(transport, q, k, v, args.iters))
        for transport, times in rounds.items():
            results[transport][seq] = sorted(times)[len(times) // 2]
    if rank == 0:
        with open(result_file, "w") as f:
            json.dump({t: {str(s): ms for s, ms in v.items()} for t, v in results.items()}, f)
    destroy_distributed_env()


def _threshold(sizes: list[int], candidate: list[float | None], reference: list[float | None], min_gain: float):
    """Smallest size from which ``candidate`` beats ``reference`` by ``min_gain`` at every larger size, else None."""
    best = None
    for size, c, r in sorted(zip(sizes, candidate, reference), reverse=True):
        if c is None or r is None or c > r * (1 - min_gain):
            break
        best = size
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-sizes", default="2,4,8")
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seqs", default="8192,16384,24576,37888,49152", help="total sequence lengths (tokens)")
    parser.add_argument("--transports", default=",".join(TRANSPORTS))
    parser.add_argument("--overlap-chunks", type=int, default=0, help="0: two heads per chunk")
    parser.add_argument("--attention-backend", default=None, help="default: the platform's automatic choice")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10, help="back-to-back layer calls per timed round")
    parser.add_argument("--rounds", type=int, default=7, help="interleaved rounds per size (median kept)")
    parser.add_argument("--min-gain", type=float, default=0.01, help="relative gain required to enable a transport")
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--json", default=None, help="write raw timings here")
    args = parser.parse_args()
    args.seqs = [int(s) for s in args.seqs.split(",")]
    args.transports = args.transports.split(",")
    major, minor = torch.cuda.get_device_capability(0)
    capability = f"{major}.{minor}"
    itemsize = torch.finfo(torch.bfloat16).bits // 8

    all_results = {}
    tables: dict[str, dict[int, int]] = {"FLASHINFER": {}, "COPY_ENGINE": {}, "COPY_ENGINE_OVERLAP_CHUNK": {}}
    for world_size in (int(n) for n in args.world_sizes.split(",")):
        seqs = [s for s in args.seqs if s % world_size == 0]
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            mp.spawn(
                _worker,
                args=(world_size, argparse.Namespace(**{**vars(args), "seqs": seqs}), tmp.name),
                nprocs=world_size,
            )
            with open(tmp.name) as f:
                results = {t: {int(s): ms for s, ms in v.items()} for t, v in json.load(f).items()}
        all_results[world_size] = results
        hpr = args.heads // world_size
        chunks = overlap_chunk_count(hpr, args.overlap_chunks)  # 0: the overlap does not apply to this head count
        q_bytes = [s // world_size * args.heads * args.head_dim * itemsize for s in seqs]
        chunk_bytes = [s // world_size * (hpr // max(chunks, 1)) * args.head_dim * itemsize for s in seqs]
        series = {t: [results.get(t, {}).get(s) for s in seqs] for t in TRANSPORTS}

        print(f"\n## {world_size} GPUs, sm{capability}, {args.heads} heads x {args.head_dim}, {chunks} overlap chunks")
        print("| tokens | Q MiB/rank | chunk copy MiB | " + " | ".join(f"{t} ms" for t in TRANSPORTS) + " |")
        print("|---" * (3 + len(TRANSPORTS)) + "|")
        for i, s in enumerate(seqs):
            cells = [f"{series[t][i]:.3f}" if series[t][i] is not None else "-" for t in TRANSPORTS]
            print(f"| {s} | {q_bytes[i] / 2**20:.1f} | {chunk_bytes[i] / 2**20:.2f} | " + " | ".join(cells) + " |")

        ce = _threshold(q_bytes, series["copy_engine"], series["nccl"], args.min_gain)
        if ce is not None:
            tables["COPY_ENGINE"][world_size] = ce
        # The faster of the copy-engine paths is what FlashInfer has to beat.
        ce_best = [
            min(x for x in (o, c) if x is not None) if (o, c) != (None, None) else None
            for o, c in zip(series["copy_engine_overlap"], series["copy_engine"])
        ]
        if chunks:
            overlap = _threshold(chunk_bytes, series["copy_engine_overlap"], series["copy_engine"], args.min_gain)
            if overlap is not None:
                tables["COPY_ENGINE_OVERLAP_CHUNK"][world_size] = overlap
        fi = _threshold(q_bytes, series["flashinfer"], ce_best, args.min_gain)
        if fi is not None:
            tables["FLASHINFER"][world_size] = fi

    print("\n# Entries for vllm_omni/diffusion/distributed/ulysses_transport_tables.py")
    for name, entries in (
        ("FLASHINFER_MIN_BYTES", tables["FLASHINFER"]),
        ("COPY_ENGINE_MIN_BYTES", tables["COPY_ENGINE"]),
        ("COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES", tables["COPY_ENGINE_OVERLAP_CHUNK"]),
    ):
        print(f'{name}["{capability}"] = {{{", ".join(f"{n}: {b}" for n, b in sorted(entries.items()))}}}')
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"capability": capability, "results": all_results, "tables": tables}, f, indent=1)


if __name__ == "__main__":
    main()
