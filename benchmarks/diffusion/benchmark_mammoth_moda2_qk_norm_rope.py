# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark MammothModa2's Q/K RMSNorm + adjacent-pair real RoPE.

Example:

    CUDA_VISIBLE_DEVICES=0 python \
      benchmarks/diffusion/benchmark_mammoth_moda2_qk_norm_rope.py \
      --tokens 512 4173 8346 --warmup 50 --iters 200
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
from collections.abc import Callable

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from vllm_omni.diffusion.layers.fused_qk_norm_rope import (
    _fused_cuda_supported,
    fused_qk_norm_rope,
)
from vllm_omni.diffusion.models.mammoth_moda2.rope_real import (
    apply_real_rotary_emb,
)

_HEAD_DIM = 120
_Q_HEADS = 21
_KV_HEADS = 7
_EPS = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[512, 4173, 8346])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--attention-sequences", nargs="*", type=int, default=[])
    return parser.parse_args()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _measure(
    fn: Callable[[], object],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize()

    torch.accelerator.reset_peak_memory_stats()
    baseline_bytes = torch.accelerator.memory_allocated()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "stddev_ms": statistics.pstdev(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "extra_peak_mib": (torch.accelerator.max_memory_allocated() - baseline_bytes) / 2**20,
    }


@torch.inference_mode()
def _run_shape(tokens: int, warmup: int, iters: int, *, fused_first: bool = False) -> dict[str, object]:
    dtype = torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(42)
    query = torch.randn(tokens, _Q_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    key = torch.randn(tokens, _KV_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    norm_q = Qwen2RMSNorm(_HEAD_DIM, eps=_EPS).to(device=device, dtype=dtype)
    norm_k = Qwen2RMSNorm(_HEAD_DIM, eps=_EPS).to(device=device, dtype=dtype)
    norm_q.weight.normal_(mean=1.0, std=0.2)
    norm_k.weight.normal_(mean=1.0, std=0.2)
    angles = torch.rand(tokens, _HEAD_DIM // 2, device=device) * (2 * torch.pi)
    cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)
    sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)

    def native() -> tuple[torch.Tensor, torch.Tensor]:
        q_out = apply_real_rotary_emb(norm_q(query).unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)).squeeze(0)
        k_out = apply_real_rotary_emb(norm_k(key).unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0)).squeeze(0)
        return q_out, k_out

    def fused() -> tuple[torch.Tensor, torch.Tensor]:
        rope_table = torch.cat((cos[..., 0::2], sin[..., 0::2]), dim=-1)
        return fused_qk_norm_rope(
            query,
            key,
            norm_q.weight,
            norm_k.weight,
            rope_table,
            _EPS,
            head_dim=_HEAD_DIM,
            rotary_dim=_HEAD_DIM,
            interleaved=True,
        )

    expected_q, expected_k = native()
    actual_q, actual_k = fused()
    errors = {
        "q_max_abs": (actual_q.float() - expected_q.float()).abs().max().item(),
        "q_mean_abs": (actual_q.float() - expected_q.float()).abs().mean().item(),
        "k_max_abs": (actual_k.float() - expected_k.float()).abs().max().item(),
        "k_mean_abs": (actual_k.float() - expected_k.float()).abs().mean().item(),
    }
    torch.testing.assert_close(actual_q, expected_q, atol=0.0625, rtol=0.02)
    torch.testing.assert_close(actual_k, expected_k, atol=0.0625, rtol=0.02)

    functions = {"native": native, "fused": fused}
    order = ("fused", "native") if fused_first else ("native", "fused")
    measured = {name: _measure(functions[name], warmup, iters) for name in order}
    native_stats, fused_stats = measured["native"], measured["fused"]
    return {
        "tokens": tokens,
        "native": native_stats,
        "fused_including_rope_pack": fused_stats,
        "median_speedup": native_stats["median_ms"] / fused_stats["median_ms"],
        "errors": errors,
    }


@torch.inference_mode()
def _run_attention(seq: int, warmup: int, iters: int, *, fused_first: bool = False) -> dict[str, object]:
    from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import TransformerBlock

    torch.manual_seed(42)
    block = (
        TransformerBlock(
            2520,
            _Q_HEADS,
            _KV_HEADS,
            multiple_of=256,
            ffn_dim_multiplier=1.0,
            norm_eps=_EPS,
            rope_repeats_pairs=True,
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    block.attn.norm_q.weight.normal_(1.0, 0.2)
    block.attn.norm_k.weight.normal_(1.0, 0.2)
    hidden = torch.randn(2, seq, 2520, device="cuda", dtype=torch.bfloat16)
    mask = torch.ones(2, seq, device="cuda", dtype=torch.bool)
    mask[0, seq - min(300, seq // 2) :] = False
    mask[1, seq - min(17, seq // 2) :] = False
    angles = torch.rand(2, seq, _HEAD_DIM // 2, device="cuda") * (2 * torch.pi)
    # The production producer supplies FP32 tables; each example has its own positions.
    rotary = angles.cos().repeat_interleave(2, dim=-1), angles.sin().repeat_interleave(2, dim=-1)
    env = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"
    previous = os.environ.get(env)

    def run():
        return block.attn(hidden, hidden, attention_mask=mask, image_rotary_emb=rotary)

    try:
        outputs = {}
        timings = {}
        order = ("fused", "native") if fused_first else ("native", "fused")
        for name in order:
            os.environ[env] = "0" if name == "fused" else str(2 * seq + 1)
            outputs[name] = run()
            timings[name] = _measure(run, warmup, iters)
        diff = (outputs["native"].float() - outputs["fused"].float()).abs()[mask]
        max_diff, mean_diff = diff.max().item(), diff.mean().item()
        assert max_diff < 2e-2 and mean_diff < 1e-3, (max_diff, mean_diff)
        assert torch.count_nonzero(outputs["fused"][~mask]) == 0
        return {
            "sequence": seq,
            "batch": 2,
            "backend": block.attn.omni_attn.attn_backend.__name__,
            "native": timings["native"],
            "fused": timings["fused"],
            "median_speedup": timings["native"]["median_ms"] / timings["fused"]["median_ms"],
            "max_abs_diff": max_diff,
            "mean_abs_diff": mean_diff,
        }
    finally:
        if previous is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = previous


def main() -> None:
    args = _parse_args()
    if (
        args.rounds < 1
        or args.iters < 1
        or args.warmup < 0
        or any(value < 1 for value in [*args.tokens, *args.attention_sequences])
    ):
        raise ValueError("rounds, iters and sequence lengths must be positive; warmup must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    probe_q = torch.empty(1, _Q_HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    probe_k = torch.empty(1, _KV_HEADS, _HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    if not _fused_cuda_supported(probe_q, probe_k, _HEAD_DIM, _HEAD_DIM, interleaved=True):
        raise RuntimeError("The fused CUDA QK norm/RoPE path is unavailable")

    result = {
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": _version("triton"),
            "vllm": _version("vllm"),
            "vllm_omni": _version("vllm-omni"),
        },
        "config": {
            "dtype": "bfloat16",
            "q_heads": _Q_HEADS,
            "kv_heads": _KV_HEADS,
            "head_dim": _HEAD_DIM,
            "warmup": args.warmup,
            "iters": args.iters,
            "rounds": args.rounds,
            "inference_mode": True,
            "alternating_order": True,
            "norm_weights": "normal(mean=1.0, std=0.2)",
        },
        "results": [
            {"round": round_id, **_run_shape(tokens, args.warmup, args.iters, fused_first=bool(round_id % 2))}
            for round_id in range(args.rounds)
            for tokens in args.tokens
        ],
        "attention_results": [
            {"round": round_id, **_run_attention(seq, args.warmup, args.iters, fused_first=bool(round_id % 2))}
            for round_id in range(args.rounds)
            for seq in args.attention_sequences
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
