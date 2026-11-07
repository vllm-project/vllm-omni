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
     A reference can also be inapplicable at a given ring degree rather than
     unsupported; such a case still runs check 1 and prints `n/a` for check 2.

Coverage:
  * SHAPE SWEEP  -- accuracy across seq_len x head_dim x {MHA, GQA} x
                    {non-causal, causal}. Each row also prints a per-case
                    `ratio~`, which is a sanity signal and NOT speed evidence:
                    the two legs enter through different wrappers and are timed
                    once each in a fixed order. Read speed from
                    bench_ring_attention.py instead.
  * PARAM SWEEP  -- softmax_scale, softcap, window_size (sliding), alibi_slopes,
                    dropout_p, and joint (text-conditioning) tokens.

Backend is auto-probed at runtime and prefers vLLM's bundled FlashAttention,
which is NOT one of the packing backends. Pass --attn-type to force one.

Launch with torchrun (ring degree == world size):

    cd projs/learn/vllm-omni
    torchrun --nproc_per_node=4 tests/diffusion/attention/bench_ring_flash_attn_opt.py \
        --attn-type fa3 --quick
    # options: --attn-type auto|torch|fa|fa3|aiter|vllm_fa  --iters 20  --quick

Only FA and FA3 at ring degree 3 or above circulate K/V packed, so only such a
run is evidence for packing: anywhere else the candidate executes the same code
as the baseline copy in ring_baseline.py and `opt-vs-base` is 0.0 by
construction. The header and the summary say which case applies, and the exit
status enforces it: 0 = validated with packing exercised, 1 = a case failed,
2 = every case skipped, 3 = cases ran but none exercised packing. So this
doubles as a regression test only when it exits 0.
"""

# torch and vllm_omni imports are intentionally below `_preload_cuda13_runtime()`
# (the re-exec must set LD_LIBRARY_PATH before torch is imported).
# ruff: noqa: E402
import argparse
import contextlib
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
from vllm_omni.diffusion.distributed.comm import RingComm

BACKEND_CAPS = {
    "torch": {"scale", "causal"},  # SDPA (efficient) - no GQA/window/softcap/alibi
    # "alibi" is deliberately absent everywhere: the shipped entry point
    # ring_flash_attn_func asserts alibi_slopes is None, so the ring path does
    # not support ALiBi at all, whatever the block kernel could do. Declaring
    # it would make the case run, the baseline (which calls the forward
    # directly and never sees that assert) succeed, and the candidate raise --
    # a false regression report rather than a real one.
    "fa": {"scale", "causal", "gqa", "window", "softcap", "dropout"},
    "fa3": {"scale", "causal", "gqa", "window", "softcap"},  # dropout ignored
    "aiter": {"scale", "causal", "gqa", "window", "dropout"},
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


def _agree_fail(failed_here, device):
    """Return whether *any* rank failed the current case.

    Keeps the ranks in lockstep: every rank must take the same branch, both so
    one rank's failure fails the run and so no rank returns early while the
    others wait in a later collective.
    """
    flag = torch.tensor([1 if failed_here else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


# What each exit status means, for the greppable status line main() prints.
_EXIT_MEANING = {
    0: "validated, packed-KV path exercised",
    1: "a case FAILed",
    2: "every case skipped; nothing validated",
    3: "cases ran but none exercised packed-KV",
}


def summary_exit_code(n_pass, n_fail, n_skip, n_packed):
    """Process exit status for a completed sweep.

    0  validated, and the packed-KV path was actually exercised.
    1  any case failed.
    2  no case was validated: a sweep that skipped everything compared no
       outputs, so it is not correctness evidence and must not exit like a pass.
    3  cases ran, but none of them exercised the packed-KV path. The shipped
       forward only packs for FA/FA3 at ring degree > 1 (see the gate in
       ring_flash_attn.py); outside that it runs the same code as the baseline
       copy in ring_baseline.py, so `opt-vs-base` is 0.0 by construction and
       every case passes whatever packing does. Such a run is still evidence
       for the fp32-reference property, but not for this change, and must not
       be reported as if it were.
    """
    if n_fail:
        return 1
    if n_pass == 0:
        return 2
    if n_packed == 0:
        return 3
    return 0


def _packing_active(attn_type, world_size):
    """Whether the shipped forward's packing gate is open for this run.

    The production gate has three clauses::

        attn_type in (AttnType.FA, AttnType.FA3) and comm.world_size > 2 and k.shape == v.shape

    Only the first two are decidable before the cases run, so only those are
    mirrored here. Every case in this file builds k and v with the same shape,
    so the third clause always holds; ``run_case`` asserts that rather than
    leave it implicit, because a future case with differing k/v shapes would
    otherwise be counted as PASS-with-packing without having packed -- exactly
    the overcounting the exit-3 gate exists to prevent.

    The degree floor is ``> 2``, not ``> 1``: at degree 2 the single hop does
    not pay for the packing copy (measured 0.98x on the full ring call), so
    the shipped forward does not pack there. A degree-2 sweep is therefore a
    gate-closed control and must report status 3, exactly like degree 1.

    The first two clauses are duplicated rather than imported on purpose, so a
    change to the production gate shows up here as a disagreement instead of
    being silently tracked. What makes that true is ``_observe_packing``, which
    watches what the forward actually circulates: this function only predicts,
    and a prediction nothing checks is just a second opinion free to drift.
    """
    return attn_type in (AttnType.FA, AttnType.FA3) and world_size > 2


@contextlib.contextmanager
def _observe_packing(unpacked_ndim):
    """Record whether the forward really circulated K and V as one tensor.

    Yields a dict that fills in while the block runs: ``sends`` counts
    ``RingComm.send_recv`` calls and ``packed`` says whether any of them
    carried a packed ``[2, ...]`` block.

    Tensor rank is an unambiguous discriminator here. The packed path sends one
    ``[2, B, S, H, D]`` tensor per hop and the split path sends ``[B, S, H, D]``
    twice, and nothing else goes over the wire -- joint tokens are concatenated
    locally with ``torch.cat``, never circulated -- so one extra dimension
    means packed and no other code path can forge it.

    The patch is on the class rather than the instance because the forward
    constructs its own ``RingComm``, and it is restored in a ``finally`` so a
    raising case cannot leak the spy into later cases or into the timing loop.
    """
    seen = {"packed": False, "sends": 0}
    original = RingComm.send_recv

    def spy(self, to_send, recv_tensor=None):
        seen["sends"] += 1
        if to_send.dim() == unpacked_ndim + 1:
            seen["packed"] = True
        return original(self, to_send, recv_tensor)

    RingComm.send_recv = spy
    try:
        yield seen
    finally:
        RingComm.send_recv = original


def _require_packing_agreement(predicted, observed, label, device):
    """Fail the run when the forward and ``_packing_active`` disagree.

    Without this, the duplicated gate is unenforced: lowering the production
    degree floor back to ``world_size > 1`` left every degree-2 sweep still
    reporting ``NOT EXERCISED`` and exit 3 while the forward packed on every
    hop -- the harness describing a path it had not taken, which is the whole
    class of defect this file exists to rule out. Mutation M3 covers it.

    Both directions matter, and the dangerous one is a *raised* production
    floor against a still-open mirror: there the sweep would count cases as
    PASS-with-packing and exit 0 on the strength of code that never ran.

    ``observed`` is ``None`` on a rank whose candidate never ran, which is not
    symmetric: the baseline can raise on one device and not another. So every
    rank must enter the reductions below unconditionally -- gating the call on
    ``observed is not None`` would strand the ranks that skipped it -- and the
    ranks that did run are the only ones whose observations are compared.
    """
    # -1 and 2 are out-of-band sentinels for "this rank has no observation",
    # chosen to fall outside [0, 1] on the low and high side respectively so
    # that one MAX reduce yields both the max and the min over *reporting*
    # ranks. A single reduce keeps the ranks in lockstep by construction.
    high_code = -1 if observed is None else int(observed)
    low_code = 2 if observed is None else int(observed)
    agg = torch.tensor([high_code, -low_code], device=device, dtype=torch.int32)
    dist.all_reduce(agg, op=dist.ReduceOp.MAX)
    high, low = int(agg[0].item()), -int(agg[1].item())

    if high < 0:
        return  # no rank reached the candidate; there is nothing to compare
    if low != high:
        raise RuntimeError(
            f"ranks disagree about what the forward did on case {label!r}: some packed "
            "K/V and others did not. The gate depends only on the backend, the ring "
            "degree and the k/v shapes, all of which are identical across ranks, so "
            "this means the ranks are not running the code they appear to be."
        )
    if bool(high) != predicted:
        raise RuntimeError(
            f"packing gate disagreement on case {label!r}: _packing_active predicted "
            f"{'packed' if predicted else 'not packed'} but the forward actually "
            f"{'packed' if high else 'did not pack'}. The production gate in "
            "ring_flash_attn.py and the mirror in _packing_active have drifted; fix "
            "the mirror (and this file's docstrings) to match the shipped gate."
        )


def _require_backend_agreement(backend, world_size, device):
    """Refuse to report a sweep whose ranks probed different kernels.

    _select_backend probes each rank independently, so an uneven build or a
    per-device driver fault can leave rank 0 on fa3 and rank 1 on torch. The
    ring would then compare a mixed-kernel candidate against a mixed-kernel
    baseline and still pass, while everything derived from `backend` -- which
    cases run, which capabilities are claimed, the printed summary -- describes
    rank 0 alone. Every rank reduces the same two values, so a disagreement
    raises on all of them rather than stranding one in a later collective.
    """
    if world_size == 1:
        return
    order = sorted(BACKEND_CAPS)
    lo = torch.tensor([order.index(backend)], device=device, dtype=torch.int32)
    hi = lo.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if lo.item() != hi.item():
        raise RuntimeError(
            f"ranks probed different backends ({order[lo.item()]} vs {order[hi.item()]}); "
            "this sweep compares mixed kernels and is not evidence -- pin one with --attn-type"
        )


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
        return "SKIP", f"SKIP {label:34s} | backend '{backend}' lacks: {missing}", None

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
    # The third clause of the shipped packing gate, which _packing_active
    # cannot evaluate before the case is built. Holds for every case here;
    # asserted so that adding one with differing k/v shapes fails loudly
    # instead of being counted as PASS-with-packing without packing.
    assert k.shape == v.shape, f"case {label!r} builds k and v with different shapes; n_packed would overcount"

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

    # The capability table consulted at the top of this function is the only
    # place a case may be skipped: it declares, per backend, which optional
    # features the build honors, and it is rank-invariant so every rank skips
    # the same cases. Anything that raises past it is unexpected. In
    # particular a candidate that raises where the baseline succeeded is the
    # regression this harness exists to catch, so it fails rather than skips.
    # If a build genuinely lacks a feature, drop it from BACKEND_CAPS instead
    # of letting the exception be swallowed.
    err = None
    out_base = out_opt = None
    observed_packed = None
    try:
        out_base = _baseline_func(q, k, v, **common)
    except Exception as e:
        err = f"baseline raised {type(e).__name__}: {str(e)[:60]}"
    if err is None:
        try:
            with _observe_packing(k.dim()) as seen:
                out_opt = ring_flash_attn_func(q, k, v, **common)
            observed_packed = seen["packed"]
        except Exception as e:
            err = f"candidate raised {type(e).__name__}: {str(e)[:60]}"

    # Agree on the verdict before anyone returns. Returning early on one rank
    # alone would leave the others waiting in the reductions below, and a
    # candidate that raises on a single rank must fail the run rather than be
    # skipped there. (A rank that hangs *inside* the ring instead of raising
    # cannot be rescued here; that needs a collective timeout.)
    if _agree_fail(err is not None, device):
        return "FAIL", f"FAIL {label:34s} | {err or 'another rank failed this case'}", observed_packed

    dropout_random = dropout > 0 and backend in DROPOUT_RANDOM

    # (1) opt vs baseline
    if dropout_random:
        finite = bool(torch.isfinite(out_opt).all() and torch.isfinite(out_base).all())
        d_ob = 0.0 if finite else float("inf")
    else:
        d_ob = _max_abs_diff(out_opt, out_base)

    # (2) opt/baseline vs fp32 reference (only where a faithful ref exists)
    ref_mode = case.get("ref", "none")
    # Some oracles only describe the ring's semantics at degree 1. The ring
    # passes window_size unchanged to every circulated block (see
    # ring_flash_attn.py), so each block gets a window relative to its own
    # start, while _ref_manual applies one window over global positions. Those
    # coincide only when there is a single block. Above degree 1 the case still
    # runs -- opt-vs-base is the comparison this change needs, and it is exact
    # there -- but the inapplicable oracle is dropped instead of being counted
    # as a failure, and the row prints "n/a" so it is not read as a pass.
    ref_na = case.get("ref_single_rank_only", False) and world_size > 1
    if ref_na:
        ref_mode = "none"
    d_or = d_br = -1.0
    ref_oom = False
    # Every input to this is rank-invariant (the case, world_size, and the
    # now-agreed backend), so all ranks attempt the oracle together and all
    # reach the agreement collective below together.
    ref_attempted = ref_mode != "none" and not dropout_random and not use_alibi
    if ref_attempted:
        # The fp32 full-attention reference materializes an O(S^2) score matrix
        # (and all-gathers Q for causal), which can be tens of GiB at large seq
        # lengths. On a memory-contended box this can OOM even though the timed
        # ring path fits easily, so an OOM here degrades to "no ref check"
        # (opt-vs-base + timing still run) rather than crashing the sweep.
        #
        # Which rank OOMs is not guaranteed to be the same everywhere: free
        # memory differs per device on a shared box. So the ranks agree on the
        # outcome below and all drop the oracle together, instead of some
        # comparing against a reference the others never built. A rank that
        # OOMs *inside* _all_gather_seq, before its collective, still strands
        # the ranks already waiting in it -- that needs a collective timeout,
        # same as the in-ring hang noted above.
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

    if ref_attempted and _agree_fail(ref_oom, device):
        # Any rank short of memory drops the oracle on all of them, so the
        # reduced opt-vs-ref below is not a mix of ranks that built a reference
        # and ranks that reported -1.0 for not having one.
        d_or = d_br = -1.0
        ref_oom = True

    # Warmup, then one timing sample per leg. This ratio is a sanity signal,
    # NOT speed evidence, and is labelled as such in the row: the baseline is
    # called as a plain function while the candidate goes through
    # RingFlashAttnFunc.apply, and t_base is always measured before t_opt, so
    # any drift during the case lands on the candidate. Measured on the
    # non-packing VLLM_FA sweep -- where both legs run identical code -- these
    # ratios still ranged 0.84x-1.01x. Speed evidence comes from
    # bench_ring_attention.py, which calls the forward directly on both arms
    # across separate SHA-pinned process launches.
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
    ref_s = ("  oom  " if ref_oom else "  n/a  " if ref_na else "  -  ") if d_or < 0 else f"{d_or:.2e}"
    row = (
        f"{verdict} {label:34s} | opt-vs-base {ob_s:>11s}  opt-vs-ref {ref_s:>9s} | "
        f"base {t_base:7.3f}ms  opt {t_opt:7.3f}ms  ratio~{t_base / t_opt:4.2f}x"
    )
    return verdict, row, observed_packed


def _select_backend(requested, local_rank):
    """Pick a block-attention backend that actually executes on this box
    (HAS_FA3 can be True while the CUDA-13 driver is missing, so probe it).

    The probe runs a real ring rotation: RingComm over the default group, so
    all ranks take part. Two consequences for `auto`, which is why every
    command in the PR evidence pins --attn-type instead:

    * A candidate that raises partway through a rotation leaves that hop's
      committed irecv in flight, since the ring commits the next hop before
      running attention and only waits afterwards. The next candidate then
      rotates with a stale op outstanding.
    * Each rank walks the order list independently, so ranks that disagree
      about a candidate desync here, inside the probe, before
      _require_backend_agreement can report it.

    Pinning the backend reduces the order list to one entry, so the only
    outcomes are success or a raise that propagates on every rank.
    """
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
        dict(
            label="param window=(256,0) causal",
            **P,
            causal=True,
            window=(256, 0),
            ref="manual",
            ref_single_rank_only=True,
            needs={"window"},
        ),
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
    device = torch.device(f"cuda:{local_rank}")
    backend, attn_type = _select_backend(args.attn_type, local_rank)
    _require_backend_agreement(backend, world_size, device)

    packed = _packing_active(attn_type, world_size)
    if rank == 0:
        print(
            f"backend={backend}  attn_type={attn_type.value}  world_size={world_size} (ring degree)  "
            f"HAS_TRITON={HAS_TRITON}"
        )
        print(f"packed-KV path: {'EXERCISED' if packed else 'NOT EXERCISED'}")
        print(f"capabilities honored: {sorted(BACKEND_CAPS[backend])}")
        print(f"tolerances: opt-vs-base <= {TOL_OPT_VS_BASE:.0e}, vs-ref <= {TOL_VS_REF:.0e}")
        print("per-case ratio~ is a sanity signal, not speed evidence (see bench_ring_attention.py)")
        print("=" * 120)

    cases = _build_cases(args.quick)
    n_pass = n_fail = n_skip = n_packed = 0
    for case in cases:
        verdict, row, observed = run_case(
            case, rank, world_size, local_rank, attn_type, backend, args.iters
        )
        # `packed` above is a prediction made before any case ran; `observed` is
        # what the forward did. Cross-check them on every case, then count from
        # the observation: a case is PASS-with-packing because it packed, not
        # because the gate was expected to be open. This also covers the gate's
        # third clause, which _packing_active cannot evaluate -- a case with
        # differing k/v shapes would not pack, and would be caught here rather
        # than counted. Called unconditionally; it reduces internally.
        _require_packing_agreement(packed, observed, case["label"], device)
        if verdict == "PASS":
            n_pass += 1
            if observed:
                n_packed += 1
        elif verdict == "FAIL":
            n_fail += 1
        else:
            n_skip += 1
        if rank == 0:
            print(row)

    if rank == 0:
        print("=" * 120)
        print(
            f"SUMMARY: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {n_packed} PASS-with-packing  "
            f"(backend={backend}, world_size={world_size})"
        )

    # Every rank agreed on each verdict, so all ranks compute the same code.
    code = summary_exit_code(n_pass, n_fail, n_skip, n_packed)
    if rank == 0 and code == 2:
        print(
            "ERROR: no case was validated (every case was skipped); this run is "
            "not correctness evidence for the ring path."
        )
    if rank == 0 and code == 3:
        print(
            f"ERROR: no case exercised the packed-KV path (backend={backend}, ring degree "
            f"{world_size}); the shipped forward only packs for FA/FA3 at ring degree 3 or "
            "above, so the candidate ran the same code as the baseline and opt-vs-base is 0.0 "
            "by construction. This run is NOT evidence for packed-KV. Re-run with "
            "--attn-type fa3 (or fa) and --nproc-per-node >= 3."
        )
    if rank == 0:
        # torchrun reports only its own status and collapses every nonzero child
        # exit to 1, so the status below cannot be recovered from the launcher's
        # exit code. Print it in one greppable line for CI.
        print(f"HARNESS_STATUS={code} ({_EXIT_MEANING[code]})")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(code)


if __name__ == "__main__":
    main()
