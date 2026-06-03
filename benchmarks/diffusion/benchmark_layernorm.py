# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: SGLang Triton LayerNorm kernel vs torch.compile(forward_native).

Run with:
    python benchmarks/diffusion/benchmark_layernorm.py
"""

import torch

from vllm_omni.diffusion.layers.norm import LayerNorm

SHAPES = [
    # (batch, seq_len, hidden_dim)
    (1, 4096, 1536),  # Wan2.2-1.3B
    (4, 4096, 1536),
    (1, 16384, 1536),
    (1, 4096, 5120),  # Wan2.2-14B
    (4, 4096, 5120),
]

WARMUP = 50
REPEATS = 200
DTYPE = torch.bfloat16
DEVICE = "cuda"


def bench(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats * 1e3


def check(ref, out, atol=1e-2, rtol=1e-2):
    max_diff = (ref.float() - out.float()).abs().max().item()
    ok = torch.allclose(ref.float(), out.float(), atol=atol, rtol=rtol)
    return ok, max_diff


for B, S, D in SHAPES:
    print(f"\nShape  batch={B}  seq={S}  dim={D}")
    print("-" * 56)

    x = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    ln = LayerNorm(D).to(DEVICE).to(DTYPE)

    compiled_fn = torch.compile(ln.forward_native)
    ref = compiled_fn(x)

    t_compile = bench(lambda: compiled_fn(x))
    print(f"  torch.compile (native)  : {t_compile:8.2f} µs")

    t_cuda = bench(lambda: ln.forward_cuda(x))
    out_cuda = ln.forward_cuda(x)
    ok, md = check(ref, out_cuda)
    print(f"  SGLang Triton kernel    : {t_cuda:8.2f} µs  max_diff={md:.2e}  {'OK' if ok else 'FAIL'}")
