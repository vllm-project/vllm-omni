# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone diagnostic for the diffusion attention backends added in #3079.

Exercises the same synthetic attention shape through each backend and each
torch SDPA sub-kernel, with and without an attn_mask, so we can pinpoint why
CUDNN_ATTN underperforms on some SKUs. Two typical causes:

  (a) cuDNN lacks a tuned kernel for this (SM, head_dim, seq) combination
      and silently falls back to MATH.
  (b) attn_mask is non-None and the cuDNN SDPA dispatch rejects it,
      walking CUDNN -> FLASH -> MATH.

Run:
    python benchmarks/diffusion/bench_attention_backends.py --preset hv15
    python benchmarks/diffusion/bench_attention_backends.py --preset wan22
    python benchmarks/diffusion/bench_attention_backends.py \
        --batch 1 --heads 24 --seq 14336 --head-dim 128

Optional extras:
    pip install --pre flash-attn-4        # FA4 is currently pre-release only
    pip install -U flashinfer              # latest FlashInfer (0.6.9)

The table at the end is the data we want on the PR — surface the row where a
backend is >1.5x the SDPA baseline, that's the one to gate off the auto-route.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Presets mirror the DiT hot-path attention call (single-stream block) at the
# shapes we run in the PR's validation configs.
_PRESETS = {
    # HunyuanVideo-1.5 480p/33f: (30 * 52 * 9) latent + ~256 text tokens.
    "hv15": {"batch": 1, "heads": 24, "seq": 14336, "head_dim": 128},
    # Wan 2.2 480p/33f rough estimate; adjust when we measure real shapes.
    "wan22": {"batch": 1, "heads": 40, "seq": 16384, "head_dim": 128},
    # Smaller image-gen shape for quick smoke tests.
    "flux": {"batch": 1, "heads": 24, "seq": 4096, "head_dim": 128},
}

_SDPA_BACKENDS = [
    ("CUDNN_ATTENTION", [SDPBackend.CUDNN_ATTENTION]),
    ("FLASH_ATTENTION", [SDPBackend.FLASH_ATTENTION]),
    ("EFFICIENT_ATTENTION", [SDPBackend.EFFICIENT_ATTENTION]),
    ("MATH", [SDPBackend.MATH]),
    # The PR's CUDNN_ATTN impl uses this priority chain; keep it here to show
    # what torch actually picks when multiple are allowed.
    ("CUDNN_ATTN_CHAIN", [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]),
]


def _env_report() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this diagnostic needs a GPU.")
    cc = torch.cuda.get_device_capability()
    print("=" * 72)
    print(f"GPU           : {torch.cuda.get_device_name()}")
    print(f"SM capability : sm_{cc[0]}{cc[1]}")
    print(f"torch         : {torch.__version__}")
    print(f"cuDNN version : {torch.backends.cudnn.version()}")
    print("flashinfer    : ", end="")
    try:
        import flashinfer

        print(getattr(flashinfer, "__version__", "present"))
    except Exception as e:
        print(f"not installed ({type(e).__name__})")
    print("=" * 72)


def _make_qkv(batch: int, heads: int, seq: int, head_dim: int, device: str, dtype: torch.dtype):
    # Layout: (B, S, H, D) — what the PR's backends accept as input.
    q = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, heads, head_dim, device=device, dtype=dtype)
    return q, k, v


def _make_mask(batch: int, seq: int, device: str, dtype: torch.dtype, pad_tokens: int = 128) -> torch.Tensor:
    # Mirrors a text-encoder padding mask: last `pad_tokens` positions masked.
    mask = torch.zeros(batch, 1, seq, seq, device=device, dtype=dtype)
    if pad_tokens > 0:
        mask[..., -pad_tokens:] = float("-inf")
    return mask


def _time_call(fn, *args, warmup: int = 3, iters: int = 10) -> tuple[float, str]:
    """Return (median ms, error string). Skips on any exception so that a
    backend that rejects our args (wrong dtype, missing JIT module, unsupported
    kwarg value) doesn't abort the whole sweep."""
    try:
        for _ in range(warmup):
            fn(*args)
        torch.accelerator.synchronize()
        times = []
        for _ in range(iters):
            torch.accelerator.synchronize()
            t0 = time.perf_counter()
            fn(*args)
            torch.accelerator.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return times[len(times) // 2], ""
    except Exception as e:  # noqa: BLE001 — probe script, keep the whole table going
        msg = str(e).split("\n", 1)[0][:60]
        return float("nan"), f"{type(e).__name__}: {msg}" if msg else type(e).__name__


def _run_sdpa_variants(q, k, v, attn_mask, scale: float) -> list[tuple[str, float, str]]:
    # q/k/v come in (B, S, H, D); F.sdpa wants (B, H, S, D).
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    rows: list[tuple[str, float, str]] = []
    for name, backends in _SDPA_BACKENDS:

        def _call(mask=attn_mask):
            with sdpa_kernel(backends):
                return torch.nn.functional.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale
                )

        ms, err = _time_call(_call)
        rows.append((name, ms, err))
    return rows


def _run_flashinfer(q, k, v, scale: float, backend: str | None = None, attn_mask=None) -> tuple[float, str]:
    """Call FlashInfer's dense single-prefill.

    ``backend`` hints at cutlass/fa3/trtllm if the installed version exposes
    that kwarg. ``attn_mask`` is the same additive 4D mask we pass to SDPA —
    we convert it to the 2D boolean ``custom_mask`` form FlashInfer accepts.
    Per FlashInfer docs, ``custom_mask`` only applies when ``causal=False``.
    """
    try:
        import inspect

        from flashinfer.prefill import single_prefill_with_kv_cache
    except Exception as e:
        return float("nan"), f"import-{type(e).__name__}"

    kwargs: dict = {"sm_scale": scale, "causal": False, "return_lse": False}
    if backend is not None:
        sig = inspect.signature(single_prefill_with_kv_cache)
        if "backend" not in sig.parameters:
            return float("nan"), "no-backend-kwarg"
        kwargs["backend"] = backend

    if attn_mask is not None:
        # Collapse (B, 1, S, S) additive float mask to (S, S) boolean.
        # FlashInfer expects True = keep, False = masked out.
        mask_2d = attn_mask[0, 0]
        kwargs["custom_mask"] = mask_2d != float("-inf")

    def _call():
        out = single_prefill_with_kv_cache(q[0], k[0], v[0], **kwargs)
        return out.unsqueeze(0)

    return _time_call(_call)


def _run_fa4(q, k, v, scale: float) -> tuple[float, str]:
    """Call FlashAttention-4 directly (``pip install flash-attn-4``). FA4
    ships a Blackwell-native kernel via CuTe-DSL; on sm_120 it should beat
    cuDNN by ~20%. API lives under ``flash_attn.cute``; the older
    ``flash_attn.flash_attn_func`` path is FA2/FA3 only."""
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as e:
        return float("nan"), f"import-{type(e).__name__}"

    def _call():
        # FA4 accepts (B, S, H, D) directly — same layout the PR's backends use.
        return flash_attn_func(q, k, v, softmax_scale=scale, causal=False)

    return _time_call(_call)


def _run_flashinfer_cudnn_batch(q, k, v, scale: float) -> tuple[float, str]:
    """Call FlashInfer's direct cuDNN wrapper. Bypasses PyTorch SDPA dispatch
    (which has a few hundred ns of overhead per call) and talks to cuDNN FMHA
    straight. Useful as a ceiling for 'pure cuDNN, no SDPA' on Blackwell."""
    try:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache
    except Exception as e:
        return float("nan"), f"import-{type(e).__name__}"

    b, s, h, d = q.shape
    qo_indptr = torch.tensor([0, s], dtype=torch.int32, device=q.device)
    kv_indptr = torch.tensor([0, s], dtype=torch.int32, device=q.device)

    def _call():
        return cudnn_batch_prefill_with_kv_cache(
            q.reshape(b * s, h, d),
            k.reshape(b * s, h, d),
            v.reshape(b * s, h, d),
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            max_qo_len=s,
            max_kv_len=s,
            sm_scale=scale,
            causal=False,
        )

    return _time_call(_call)


def _print_table(title: str, rows: list[tuple[str, float, str]], baseline_name: str | None = None) -> None:
    print(f"\n{title}")
    print("-" * 88)
    header = f"{'backend':<24} {'median (ms)':>14}    {'vs baseline':>12}    status"
    print(header)
    print("-" * 88)
    baseline_ms = None
    if baseline_name is not None:
        for name, ms, _ in rows:
            if name == baseline_name and ms == ms:  # not NaN
                baseline_ms = ms
                break
    for name, ms, err in rows:
        ms_str = f"{ms:>14.3f}" if ms == ms else f"{'n/a':>14}"
        if baseline_ms is not None and ms == ms:
            ratio = baseline_ms / ms
            ratio_str = f"{ratio:>11.2f}x"
        else:
            ratio_str = f"{'—':>12}"
        status = "ok" if not err else f"FAILED ({err})"
        print(f"{name:<24} {ms_str}    {ratio_str}    {status}")


def _pick_winner(rows: list[tuple[str, float, str]]) -> tuple[str, float] | None:
    """Return (backend, ms) of the fastest non-failing row, or None."""
    ok_rows = [(n, m) for n, m, err in rows if not err and m == m]
    if not ok_rows:
        return None
    return min(ok_rows, key=lambda x: x[1])


def _bench_one_shape(shape: dict, dtype: torch.dtype, device: str) -> tuple[list, list]:
    torch.manual_seed(0)
    q, k, v = _make_qkv(**shape, device=device, dtype=dtype)
    scale = 1.0 / (shape["head_dim"] ** 0.5)

    rows_nomask = _run_sdpa_variants(q, k, v, attn_mask=None, scale=scale)
    rows_nomask.append(("FLASHINFER (default)", *_run_flashinfer(q, k, v, scale)))
    # `trtllm-gen` has no sm_120 cubins (NVIDIA/TensorRT-LLM#11799) — skipped.
    for fi_backend in ("fa2", "fa3", "cutlass", "auto"):
        rows_nomask.append((f"FLASHINFER ({fi_backend})", *_run_flashinfer(q, k, v, scale, backend=fi_backend)))
    rows_nomask.append(("FLASHINFER (cudnn-batch)", *_run_flashinfer_cudnn_batch(q, k, v, scale)))
    rows_nomask.append(("FA4 (direct)", *_run_fa4(q, k, v, scale)))

    mask = _make_mask(shape["batch"], shape["seq"], device=device, dtype=dtype, pad_tokens=256)
    rows_mask = _run_sdpa_variants(q, k, v, attn_mask=mask, scale=scale)
    rows_mask.append(("FLASHINFER (dense)", *_run_flashinfer(q, k, v, scale, attn_mask=mask)))
    rows_mask.append(("FA4 (direct)", float("nan"), "mask-not-supported"))

    return rows_nomask, rows_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=list(_PRESETS.keys()), default="hv15")
    parser.add_argument("--sweep", action="store_true", help="Run all presets and print a ranking")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    _env_report()
    print(f"dtype         : {args.dtype}")

    presets = list(_PRESETS.keys()) if args.sweep else [args.preset]
    summary: dict[str, tuple[str, float] | None] = {}

    for preset_name in presets:
        shape = dict(_PRESETS[preset_name])
        if not args.sweep:
            for k in ("batch", "heads", "seq", "head_dim"):
                v = getattr(args, k if k != "head_dim" else "head_dim")
                if v is not None:
                    shape[k] = v

        print("\n" + "=" * 88)
        print(
            f"Preset: {preset_name}  |  batch={shape['batch']} heads={shape['heads']} "
            f"seq={shape['seq']} head_dim={shape['head_dim']}"
        )
        print("=" * 88)

        rows_nomask, rows_mask = _bench_one_shape(shape, dtype, args.device)
        _print_table(f"[{preset_name}] No attention mask", rows_nomask, baseline_name="CUDNN_ATTENTION")
        _print_table(
            f"[{preset_name}] With attention mask (pad 256 tokens)", rows_mask, baseline_name="CUDNN_ATTENTION"
        )

        summary[preset_name] = _pick_winner(rows_nomask)

    if args.sweep or len(presets) > 1:
        print("\n" + "=" * 88)
        print("Winners per preset (no-mask path)")
        print("=" * 88)
        for preset_name, winner in summary.items():
            if winner is None:
                print(f"  {preset_name:<10} — no successful backend")
            else:
                name, ms = winner
                print(f"  {preset_name:<10} {name:<24} {ms:>8.3f} ms")

    print("\nNotes:")
    print("  * Ratios are relative to CUDNN_ATTENTION. >1.0x means faster than cuDNN.")
    print("  * Mask-path winner inherits CUDNN_ATTN's fallback in the PR's backends.")
    print("  * `trtllm-gen` and FA4 4.0.0b10 are known-broken on sm_120 as of Apr 2026.")
    print("  * For e2e timings, run text_to_video.py / text_to_image.py while varying DIFFUSION_ATTENTION_BACKEND.")


if __name__ == "__main__":
    main()
