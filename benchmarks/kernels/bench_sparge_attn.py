# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
r"""Benchmark dense FlashAttention and SPARGE INT8 attention on XPU.

The default shape matches Wan2.1-VACE 480P self-attention:
batch 1, sequence length 32760, 12 heads, and head dimension 128 in BF16.

Example:

    python benchmarks/kernels/bench_sparge_attn.py \
        --device xpu:0 --topks 0.1,0.25,0.5,0.75,1.0 \
        --warmup 2 --iters 5 --rounds 3
"""

import argparse
import json
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

for candidate in Path(__file__).resolve().parents:
    if (candidate / "vllm_omni").is_dir():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32760)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topks", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--q-block-tokens", type=int, default=None)
    parser.add_argument("--k-block-tokens", type=int, default=64)
    parser.add_argument("--q-tile-override", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results" / "bench_sparge_attn.json"),
    )
    args = parser.parse_args()
    try:
        args.topk_values = [float(value) for value in args.topks.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"invalid --topks value: {exc}")
    if not args.topk_values:
        parser.error("--topks must contain at least one value")
    if args.batch <= 0 or args.seq_len <= 0 or args.num_heads <= 0 or args.head_dim <= 0:
        parser.error("--batch, --seq-len, --num-heads, and --head-dim must be positive")
    if args.warmup < 0 or args.iters <= 0 or args.rounds <= 0:
        parser.error("--warmup must be non-negative; --iters and --rounds must be positive")
    return args


def _resolve_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def _resolve_tiling(args: argparse.Namespace) -> tuple[int, int, int]:
    q_block_tokens = args.q_block_tokens
    if q_block_tokens is None:
        q_block_tokens = 256 if args.head_dim == 128 else 64

    q_tile_override = args.q_tile_override
    if q_tile_override is None:
        if args.head_dim == 64:
            q_tile_override = 128
        else:
            q_tile_override = 0 if q_block_tokens == 64 else 256
    return q_block_tokens, args.k_block_tokens, q_tile_override


def summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def measure_once(fn: Callable[[], object], iters: int) -> tuple[float, float]:
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    torch.xpu.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(iters):
        out = fn()
        del out
    end_event.record()
    torch.xpu.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iters
    event_ms = start_event.elapsed_time(end_event) / iters
    return event_ms, wall_ms


def run_task(task: dict, *, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        out = task["fn"]()
        del out
    torch.xpu.synchronize()

    event_ms, wall_ms = measure_once(task["fn"], iters)
    print(
        f"  {task['name']}: event={event_ms:.3f} ms wall={wall_ms:.3f} ms",
        flush=True,
    )

    result = dict(task["metadata"])
    result["event"] = summarize([event_ms])
    result["wall"] = summarize([wall_ms])
    return result


def main() -> None:
    args = parse_args()
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available")

    device = torch.device(args.device)
    if device.type != "xpu":
        raise RuntimeError(f"--device must select an XPU device, got {device}")

    try:
        import auto_round_kernel as ark
    except ImportError as exc:
        raise RuntimeError(
            "SPARGE attention requires auto-round-lib with the ARK XPU sparse attention kernel. "
            "Install it with `pip install auto-round-lib`."
        ) from exc

    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
    from vllm_omni.diffusion.attention.backends.sparge_attn import SpargeAttentionImpl
    from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

    if flash_attn_varlen_func is None:
        raise RuntimeError("The vLLM XPU FlashAttention varlen kernel is unavailable")

    q_block_tokens, k_block_tokens, q_tile_override = _resolve_tiling(args)
    dtype = _resolve_dtype(args.dtype)
    topks = args.topk_values

    torch.xpu.set_device(device)
    torch.manual_seed(args.seed)
    torch.xpu.manual_seed(args.seed)
    random.seed(args.seed)

    shape = (args.batch, args.seq_len, args.num_heads, args.head_dim)
    scale = args.head_dim**-0.5
    print(f"Host: {platform.node()}", flush=True)
    print(f"GPU: {torch.xpu.get_device_name(device)}", flush=True)
    print(f"Torch: {torch.__version__}", flush=True)
    print(f"Shape (B,S,H,D): {shape}", flush=True)
    print(f"Dtype: {args.dtype}", flush=True)
    print(f"Scale: {scale}", flush=True)
    print(f"Top-k sweep: {topks}", flush=True)
    print(
        f"SPARGE tiling: q_tile_override={q_tile_override}, "
        f"q_block_tokens={q_block_tokens}, k_block_tokens={k_block_tokens}",
        flush=True,
    )

    q = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    k = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    v = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    torch.xpu.synchronize()

    dense_impl = FlashAttentionImpl(
        num_heads=args.num_heads,
        head_size=args.head_dim,
        softmax_scale=scale,
        causal=False,
    )

    q_flat = q.flatten(0, 1)
    k_flat = k.flatten(0, 1)
    v_flat = v.flatten(0, 1)
    cu_seqlens = torch.arange(
        0,
        (args.batch + 1) * args.seq_len,
        args.seq_len,
        dtype=torch.int32,
        device=device,
    )

    def dense_backend():
        return dense_impl.forward_xpu(q, k, v, None)

    def dense_kernel():
        return flash_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=args.seq_len,
            max_seqlen_k=args.seq_len,
            causal=False,
            softmax_scale=scale,
        )

    sparse_impls = {}
    sparse_metadata = {}
    for topk in topks:
        sparse_impls[topk] = SpargeAttentionImpl(
            num_heads=args.num_heads,
            head_size=args.head_dim,
            softmax_scale=scale,
            causal=False,
            backend_kwargs={
                "sparge_topk": topk,
                "sparge_q_tile_override": q_tile_override,
                "sparge_q_block_tokens": q_block_tokens,
                "sparge_k_block_tokens": k_block_tokens,
            },
        )
        out, _, metadata = ark.sparge_sage2_attn_meansim_topk_xpu(
            q,
            k,
            v,
            is_causal=False,
            scale=scale,
            smooth_k=True,
            simthreshd1=-1.0,
            topk=topk,
            attention_sink=False,
            tensor_layout="NHD",
            q_tile_override=q_tile_override,
            sparse_q_block_tokens=q_block_tokens,
            sparse_k_block_tokens=k_block_tokens,
            return_sparsity=True,
            return_metadata=True,
        )
        torch.xpu.synchronize()
        stats = dict(metadata["stats"])
        sparse_metadata[topk] = stats
        del out, metadata
        print(
            f"topk={topk:g}: selected={stats['selected_ratio']:.6f}, blocks/row={stats['selected_blocks_per_row']:.3f}",
            flush=True,
        )

    flops_per_call = 4.0 * args.batch * args.num_heads * args.seq_len * args.seq_len * args.head_dim
    print(f"Dense-equivalent work/call: {flops_per_call / 1e12:.6f} TFLOP", flush=True)

    tasks = [
        {
            "name": "dense_flash_backend",
            "fn": dense_backend,
            "metadata": {"mode": "dense_flash_backend", "topk": 1.0, "selected_ratio": 1.0},
        },
        {
            "name": "dense_flash_kernel",
            "fn": dense_kernel,
            "metadata": {"mode": "dense_flash_kernel", "topk": 1.0, "selected_ratio": 1.0},
        },
    ]
    for topk in topks:
        impl = sparse_impls[topk]

        def sparse_backend(impl=impl):
            return impl.forward_xpu(q, k, v, None)

        tasks.append(
            {
                "name": f"sparge_topk_{topk:g}",
                "fn": sparse_backend,
                "metadata": {
                    "mode": "sparge_full_backend",
                    "topk": topk,
                    "selected_ratio": sparse_metadata[topk]["selected_ratio"],
                    "sparsity_ratio": sparse_metadata[topk]["sparsity_ratio"],
                    "selected_blocks_per_row": sparse_metadata[topk]["selected_blocks_per_row"],
                },
            }
        )

    results = {}
    with torch.inference_mode():
        for round_index in range(args.rounds):
            order = list(tasks)
            random.Random(args.seed + round_index).shuffle(order)
            print(f"Round {round_index + 1}/{args.rounds}", flush=True)
            for task in order:
                print(f"Benchmarking {task['name']}", flush=True)
                results.setdefault(task["name"], []).append(run_task(task, warmup=args.warmup, iters=args.iters))

    summarized = {}
    for name, rounds_data in results.items():
        event_samples = [item["event"]["median_ms"] for item in rounds_data]
        wall_samples = [item["wall"]["median_ms"] for item in rounds_data]
        metadata = rounds_data[0]
        selected_ratio = metadata["selected_ratio"]
        event_ms = statistics.median(event_samples)
        wall_ms = statistics.median(wall_samples)
        dense_equiv_tops = flops_per_call / (event_ms * 1e9)
        executed_tops = dense_equiv_tops * selected_ratio
        summarized[name] = {
            "mode": metadata["mode"],
            "topk": metadata["topk"],
            "selected_ratio": selected_ratio,
            "sparsity_ratio": metadata.get("sparsity_ratio"),
            "selected_blocks_per_row": metadata.get("selected_blocks_per_row"),
            "event_ms": event_ms,
            "event_min_ms": min(event_samples),
            "event_max_ms": max(event_samples),
            "wall_ms": wall_ms,
            "dense_equiv_tops": dense_equiv_tops,
            "executed_tops": executed_tops,
        }

    dense_ms = summarized["dense_flash_backend"]["event_ms"]
    for value in summarized.values():
        value["speedup_vs_dense_backend"] = dense_ms / value["event_ms"]

    payload = {
        "host": platform.node(),
        "gpu": torch.xpu.get_device_name(device),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": args.dtype,
        "layout": "NHD",
        "shape": {
            "batch": args.batch,
            "seq_len": args.seq_len,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        },
        "benchmark": {
            "eager": True,
            "warmup": args.warmup,
            "iters": args.iters,
            "rounds": args.rounds,
            "seed": args.seed,
            "dense_metadata": "one varlen segment per batch",
            "sparge_tiling": {
                "q_tile_override": q_tile_override,
                "sparse_q_block_tokens": q_block_tokens,
                "sparse_k_block_tokens": k_block_tokens,
                "smooth_k": True,
                "simthreshd1": -1.0,
                "attention_sink": False,
            },
        },
        "arithmetic": {
            "flops_per_call": flops_per_call,
            "formula": "4 * B * H * Sq * Sk * D",
            "note": (
                "Dense is FP16/BF16 FlashAttention. SPARGE uses INT8 QK and BF16/FP16 PV and softmax; "
                "executed TOPS scales dense-equivalent work by the measured selected-block ratio "
                "and includes SPARGE preprocessing in latency."
            ),
        },
        "results": summarized,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("\nSummary (event time; full SPARGE path includes preprocessing)", flush=True)
    print(
        f"{'mode':26s} {'topk':>6s} {'selected':>9s} {'latency_ms':>11s} "
        f"{'dense-eq TOPS':>14s} {'exec TOPS':>10s} {'speedup':>8s}",
        flush=True,
    )
    for name in ["dense_flash_backend", "dense_flash_kernel"] + [f"sparge_topk_{topk:g}" for topk in topks]:
        row = summarized[name]
        print(
            f"{row['mode']:26s} {row['topk']:6.3f} {row['selected_ratio']:9.5f} "
            f"{row['event_ms']:11.3f} {row['dense_equiv_tops']:14.3f} "
            f"{row['executed_tops']:10.3f} {row['speedup_vs_dense_backend']:8.3f}",
            flush=True,
        )
    print(f"\nSaved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
