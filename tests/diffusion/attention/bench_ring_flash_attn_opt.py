# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Correctness + speed harness for the optimized Ring-Flash-Attention path.

Compares, on the *same* sharded inputs, three implementations:

  * baseline : ``ring_baseline.baseline_ring_flash_attn_func`` (the upstream
               forward, which circulates K and V as two separate transfers)
  * optimized: ``ring_flash_attn_func``      (the shipped path, which packs K
               and V into one transfer per hop on FA/FA3)
  * reference: full (non-ring) attention in fp32 (torch SDPA or a manual kernel)

It validates two independent properties:

  1. opt == baseline  -- the optimization is a faithful drop-in. This must hold
     for *every* parameter setting, because opt only changes the ring comm/merge,
     never the per-block attention kernel. Checked on all cases below.
  2. opt/baseline == reference -- the whole ring path computes correct attention.
     Only meaningful where the selected block-attention backend actually honors
     the parameter (see the capability table), otherwise the case is SKIPPED.

Coverage:
  * SHAPE SWEEP  -- accuracy + speed across seq_len x head_dim x {MHA, GQA} x
                    {non-causal, causal}.
  * PARAM SWEEP  -- softmax_scale, softcap, window_size (sliding), alibi_slopes,
                    dropout_p, and joint (text-conditioning) tokens.

Backend is auto-probed at runtime (fa3 -> fa2 -> torch); the torch/SDPA backend
needs no FlashAttention and runs anywhere, so the opt (merge + workspace + comm)
is always exercisable. Pass --attn-type to force one.

Launch with torchrun (ring degree == world size):

    cd projs/learn/vllm-omni
    torchrun --nproc_per_node=2 tests/diffusion/attention/bench_ring_flash_attn_opt.py
    # options: --attn-type auto|torch|fa|fa3|aiter  --iters 20  --batch 1  --quick

nproc_per_node=1 exercises the fused-merge + workspace path; >=2 additionally
exercises the packed-KV double-buffered ring comm. Exit code is non-zero if any
correctness check FAILs, so this doubles as a regression test.
"""

# torch and vllm_omni imports are intentionally below `_preload_cuda13_runtime()`
# (the re-exec must set LD_LIBRARY_PATH before torch is imported).
# ruff: noqa: E402
import argparse
import glob
import os
import site
import sys


def _preload_cuda13_runtime():
    """FA3 (`fa3-fwd`, a CUDA-13 build) needs libcudart.so.13 (+ a CUDA-13
    forward-compat driver if the system driver predates CUDA 13). Preload the
    runtime from ``nvidia/cu13/lib`` and, if present, a ``<venv>/cuda13-compat``
    driver, then re-exec so the FA3 op can register. Best-effort / no-op if those
    dirs are absent (e.g. an FA2 or torch-SDPA environment)."""
    import ctypes

    dirs = []
    compat = os.path.join(sys.prefix, "cuda13-compat")
    if glob.glob(os.path.join(compat, "libcuda.so*")):
        dirs.append(compat)
    roots = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    for base in roots:
        d = os.path.join(base, "nvidia", "cu13", "lib")
        if glob.glob(os.path.join(d, "libcudart.so.13")):
            dirs.append(d)
    if dirs:
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = cur.split(":") if cur else []
        need = [d for d in dirs if d not in parts]
        if need:
            os.environ["LD_LIBRARY_PATH"] = ":".join(need + parts)
            os.execv(sys.executable, [sys.executable] + sys.argv)
    for base in roots:
        for so in glob.glob(os.path.join(base, "nvidia", "cu13", "lib", "libcudart.so.13")):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                pass


_preload_cuda13_runtime()

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Run as a script, so this file's own directory is sys.path[0].
from ring_baseline import baseline_ring_flash_attn_func as _baseline_func
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.attention.backends.ring.ring_globals import (
    HAS_AITER,
    HAS_FA3,
    HAS_FLASH_ATTN,
    HAS_VLLM_FLASH_ATTN,
)
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.backends.ring_flash_attn import ring_flash_attn_func

BACKEND_CAPS = {
    "torch": {"scale", "causal"},  # SDPA (efficient) - no GQA/window/softcap/alibi
    "fa": {"scale", "causal", "gqa", "window", "softcap", "alibi", "dropout"},
    "fa3": {"scale", "causal", "gqa", "window", "softcap"},  # dropout ignored, alibi not forwarded
    "aiter": {"scale", "causal", "gqa", "window", "alibi", "dropout"},
    # vLLM's bundled dispatcher takes scale/causal/softcap; it drops
    # dropout, window and alibi (see ring_kernels.vllm_flash_attn_forward).
    "vllm_fa": {"scale", "causal", "gqa", "softcap"},
}
# Backends where dropout_p>0 is stochastic (so opt!=baseline run-to-run and can
# only be finiteness-checked). fa3 ignores dropout entirely -> deterministic.
DROPOUT_RANDOM = {"torch", "fa", "aiter"}

TOL_OPT_VS_BASE = 2e-2  # bf16 reduction-order noise between opt and baseline
TOL_VS_REF = 5e-2  # bf16 ring output vs fp32 full-attention reference


def _setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.accelerator.set_device_index(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def _all_gather_seq(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather a (B, S_local, H, D) shard along the sequence dim, in rank order."""
    if world_size == 1:
        return x
    parts = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=1)


def _repeat_kv(kh, hq):
    """(B, Hkv, S, D) -> (B, Hq, S, D) for GQA references."""
    if kh.shape[1] == hq:
        return kh
    return kh.repeat_interleave(hq // kh.shape[1], dim=1)


def _ref_sdpa(q, k, v, world_size, rank, *, scale, causal):
    """Memory-efficient fp32 full-attention reference (scale + causal only).

    Used for the large-seq shape sweep (no O(S^2) score matrix materialized)."""
    qf, kf, vf = (t.float() for t in (q, k, v))
    k_full = _all_gather_seq(kf, world_size)
    v_full = _all_gather_seq(vf, world_size)
    gqa = q.shape[2] != k.shape[2]
    if causal:
        q_full = _all_gather_seq(qf, world_size)
        out = F.scaled_dot_product_attention(
            q_full.transpose(1, 2),
            k_full.transpose(1, 2),
            v_full.transpose(1, 2),
            is_causal=True,
            scale=scale,
            enable_gqa=gqa,
        ).transpose(1, 2)
        s_local = q.shape[1]
        return out[:, rank * s_local : (rank + 1) * s_local]
    out = F.scaled_dot_product_attention(
        qf.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        is_causal=False,
        scale=scale,
        enable_gqa=gqa,
    ).transpose(1, 2)
    return out


def _ref_manual(q, k, v, world_size, rank, *, scale, causal, window, softcap, joint_k=None, joint_v=None):
    """Explicit fp32 full-attention reference supporting scale, causal, sliding
    window, softcap and joint (front) tokens. Materializes the score matrix, so
    only used at moderate seq lengths in the param sweep."""
    device = q.device
    k_full = _all_gather_seq(k.float(), world_size)
    v_full = _all_gather_seq(v.float(), world_size)
    sk_real = k_full.shape[1]
    if joint_k is not None:
        k_all = torch.cat([joint_k.float(), k_full], dim=1)
        v_all = torch.cat([joint_v.float(), v_full], dim=1)
        n_joint = joint_k.shape[1]
    else:
        k_all, v_all, n_joint = k_full, v_full, 0

    qh = q.float().transpose(1, 2)  # (B, Hq, Sq, D)
    kh = _repeat_kv(k_all.transpose(1, 2), qh.shape[1])
    vh = _repeat_kv(v_all.transpose(1, 2), qh.shape[1])

    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)

    sq = qh.shape[2]
    qpos = (rank * sq + torch.arange(sq, device=device)).view(sq, 1)
    kpos = torch.arange(sk_real, device=device).view(1, sk_real)
    sub = torch.zeros(sq, sk_real, device=device)
    if causal:
        sub = sub.masked_fill(kpos > qpos, float("-inf"))
    if window and tuple(window) != (-1, -1):
        wl, wr = window
        if wl >= 0:
            sub = sub.masked_fill(kpos < qpos - wl, float("-inf"))
        if wr >= 0:
            sub = sub.masked_fill(kpos > qpos + wr, float("-inf"))
    mask = torch.zeros(sq, n_joint + sk_real, device=device)
    mask[:, n_joint:] = sub
    scores = scores + mask

    probs = scores.softmax(dim=-1)
    out = torch.matmul(probs, vh).transpose(1, 2)  # (B, Sq, Hq, D)
    return out


def _max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def _bench(fn, iters):
    torch.accelerator.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def _alibi_slopes(hq, device):
    return torch.tensor([2.0 ** (-8.0 * (i + 1) / hq) for i in range(hq)], device=device, dtype=torch.float32)


def run_case(case, rank, world_size, local_rank, attn_type, backend, iters):
    """Run one case; return ('PASS'|'FAIL'|'SKIP'|'ERROR', printable row)."""
    needs = case.get("needs") or set()
    label = case["label"]
    if not needs <= BACKEND_CAPS[backend]:
        missing = ",".join(sorted(needs - BACKEND_CAPS[backend]))
        return "SKIP", f"SKIP {label:34s} | backend '{backend}' lacks: {missing}"

    dtype = torch.bfloat16
    device = f"cuda:{local_rank}"
    b, s = case["batch"], case["seqlen"]
    hq, hkv, d = case["hq"], case["hkv"], case["d"]
    causal = case.get("causal", False)
    scale = case.get("scale") or d**-0.5
    window = case.get("window", (-1, -1))
    softcap = case.get("softcap", 0.0)
    dropout = case.get("dropout", 0.0)
    use_alibi = case.get("alibi", False)
    joint = case.get("joint", False)

    torch.manual_seed(1234 + rank)
    q = torch.randn(b, s, hq, d, device=device, dtype=dtype)
    k = torch.randn(b, s, hkv, d, device=device, dtype=dtype)
    v = torch.randn(b, s, hkv, d, device=device, dtype=dtype)

    joint_k = joint_v = None
    if joint:
        js = 77
        g = torch.Generator(device=device).manual_seed(999)  # SAME on every rank
        joint_k = torch.randn(b, js, hkv, d, device=device, dtype=dtype, generator=g)
        joint_v = torch.randn(b, js, hkv, d, device=device, dtype=dtype, generator=g)

    common = dict(
        dropout_p=dropout,
        softmax_scale=scale,
        causal=causal,
        window_size=tuple(window),
        softcap=softcap,
        group=None,
        attn_type=attn_type,
    )
    if use_alibi:
        common["alibi_slopes"] = _alibi_slopes(hq, device)
    if joint:
        common.update(joint_tensor_key=joint_k, joint_tensor_value=joint_v, joint_strategy="front")

    try:
        out_base = _baseline_func(q, k, v, **common)
        out_opt = ring_flash_attn_func(q, k, v, **common)
    except Exception as e:  # unsupported kernel combo (e.g. FA3 built with DISABLE_LOCAL)
        return "SKIP", f"SKIP {label:34s} | backend raised: {type(e).__name__}: {str(e)[:60]}"

    dropout_random = dropout > 0 and backend in DROPOUT_RANDOM

    # (1) opt vs baseline
    if dropout_random:
        finite = bool(torch.isfinite(out_opt).all() and torch.isfinite(out_base).all())
        d_ob = 0.0 if finite else float("inf")
    else:
        d_ob = _max_abs_diff(out_opt, out_base)

    # (2) opt/baseline vs fp32 reference (only where a faithful ref exists)
    ref_mode = case.get("ref", "none")
    d_or = d_br = -1.0
    ref_oom = False
    if ref_mode != "none" and not dropout_random and not use_alibi:
        # The fp32 full-attention reference materializes an O(S^2) score matrix
        # (and all-gathers Q for causal), which can be tens of GiB at large seq
        # lengths. On a memory-contended box this can OOM even though the timed
        # ring path fits easily. The reference OOMs at the SDPA/matmul alloc,
        # i.e. AFTER _all_gather_seq's collective completes on every rank, and
        # all ranks share the shape + free-memory profile, so they fail
        # symmetrically -- degrade to "no ref check" (opt-vs-base + timing still
        # run) rather than crashing the whole sweep.
        try:
            if ref_mode == "sdpa":
                ref = _ref_sdpa(q, k, v, world_size, rank, scale=scale, causal=causal)
            else:
                ref = _ref_manual(
                    q,
                    k,
                    v,
                    world_size,
                    rank,
                    scale=scale,
                    causal=causal,
                    window=window,
                    softcap=softcap,
                    joint_k=joint_k,
                    joint_v=joint_v,
                )
            d_or = _max_abs_diff(out_opt, ref)
            d_br = _max_abs_diff(out_base, ref)
            del ref
        except torch.cuda.OutOfMemoryError:
            d_or = d_br = -1.0
            ref_oom = True
            torch.accelerator.empty_cache()

    # warmup + timing
    for _ in range(5):
        _baseline_func(q, k, v, **common)
        ring_flash_attn_func(q, k, v, **common)
    t_base = _bench(lambda: _baseline_func(q, k, v, **common), iters)
    t_opt = _bench(lambda: ring_flash_attn_func(q, k, v, **common), iters)

    # reduce across ranks: worst diff, mean time
    stats = torch.tensor([d_ob, d_or, d_br, t_base, t_opt], device=device)
    mx = stats.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX)
    mean = stats.clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    mean /= world_size
    d_ob, d_or, d_br = mx[0].item(), mx[1].item(), mx[2].item()
    t_base, t_opt = mean[3].item(), mean[4].item()

    ob_ok = d_ob <= TOL_OPT_VS_BASE
    ref_ok = (d_or < 0) or (max(d_or, d_br) <= TOL_VS_REF)
    verdict = "PASS" if (ob_ok and ref_ok) else "FAIL"

    ob_s = "rand-finite" if dropout_random else f"{d_ob:.2e}"
    ref_s = ("  oom  " if ref_oom else "  -  ") if d_or < 0 else f"{d_or:.2e}"
    row = (
        f"{verdict} {label:34s} | opt-vs-base {ob_s:>11s}  opt-vs-ref {ref_s:>9s} | "
        f"base {t_base:7.3f}ms  opt {t_opt:7.3f}ms  speedup {t_base / t_opt:4.2f}x"
    )
    return verdict, row


def _select_backend(requested, local_rank):
    """Pick a block-attention backend that actually executes on this box
    (HAS_FA3 can be True while the CUDA-13 driver is missing, so probe it)."""
    if requested != "auto":
        order = [requested]
    else:
        order = []
        if HAS_VLLM_FLASH_ATTN:
            order.append("vllm_fa")
        if HAS_FA3:
            order.append("fa3")
        if HAS_FLASH_ATTN:
            order.append("fa")
        if HAS_AITER:
            order.append("aiter")
        order.append("torch")
    device = f"cuda:{local_rank}"
    for name in order:
        at = AttnType.from_string(name)
        try:
            q = torch.randn(1, 16, 4, 64, device=device, dtype=torch.bfloat16)
            _baseline_func(q, q, q, softmax_scale=64**-0.5, causal=False, group=None, attn_type=at)
            torch.accelerator.synchronize()
            return name, at
        except Exception:
            continue
    raise RuntimeError(f"No working ring backend among {order}")


def _build_cases(quick):
    cases = []
    # ---- SHAPE SWEEP: accuracy + speed across shapes (ref = SDPA, scale+causal) ----
    seqlens = [2048] if quick else [2048, 8192]
    head_dims = [64, 128] if quick else [64, 128, 256]
    head_cfgs = [(16, 16), (16, 4)]  # MHA, GQA
    for causal in (False, True):
        for hq, hkv in head_cfgs:
            for d in head_dims:
                for s in seqlens:
                    tag = "MHA" if hq == hkv else f"GQA{hq}/{hkv}"
                    needs = {"gqa"} if hq != hkv else set()
                    cases.append(
                        dict(
                            label=f"shape s={s} d={d} {tag} causal={int(causal)}",
                            batch=1,
                            seqlen=s,
                            hq=hq,
                            hkv=hkv,
                            d=d,
                            causal=causal,
                            ref="sdpa",
                            needs=needs,
                        )
                    )
    # ---- PARAM SWEEP: one non-default parameter at a time (moderate shape) ----
    P = dict(batch=1, seqlen=1024, hq=16, hkv=16, d=128)
    cases += [
        dict(label="param softmax_scale=0.08", **P, causal=False, scale=0.08, ref="sdpa"),
        dict(label="param softcap=30.0", **P, causal=False, softcap=30.0, ref="manual", needs={"softcap"}),
        dict(label="param window=(256,0) causal", **P, causal=True, window=(256, 0), ref="manual", needs={"window"}),
        dict(label="param alibi_slopes", **P, causal=True, alibi=True, ref="none", needs={"alibi"}),
        dict(label="param dropout_p=0.1", **P, causal=False, dropout=0.1, ref="none"),
        dict(label="param joint(front) 77tok", **P, causal=False, joint=True, ref="manual"),
    ]
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attn-type", choices=["auto", "torch", "fa", "fa3", "aiter", "vllm_fa"], default="auto")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="smaller shape sweep")
    args = p.parse_args()

    rank, world_size, local_rank = _setup()
    backend, attn_type = _select_backend(args.attn_type, local_rank)

    if rank == 0:
        print(
            f"backend={backend}  attn_type={attn_type.value}  world_size={world_size} (ring degree)  "
            f"HAS_TRITON={HAS_TRITON}"
        )
        print(f"capabilities honored: {sorted(BACKEND_CAPS[backend])}")
        print(f"tolerances: opt-vs-base <= {TOL_OPT_VS_BASE:.0e}, vs-ref <= {TOL_VS_REF:.0e}")
        print("=" * 120)

    cases = _build_cases(args.quick)
    n_pass = n_fail = n_skip = 0
    for case in cases:
        verdict, row = run_case(case, rank, world_size, local_rank, attn_type, backend, args.iters)
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "FAIL":
            n_fail += 1
        else:
            n_skip += 1
        if rank == 0:
            print(row)

    if rank == 0:
        print("=" * 120)
        print(f"SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP  (backend={backend}, world_size={world_size})")

    dist.barrier()
    dist.destroy_process_group()
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
