#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify a ModelOpt FP8 diffusers checkpoint exported by
quantize_wan2_2_modelopt_fp8.py / quantize_hunyuanvideo_15_modelopt_fp8.py.

Three checks:
  A. transformer/config.json has a sane quantization_config block.
  B. transformer/*.safetensors contains FP8 (float8_e4m3fn) quantized tensors.
  C. transformer disk size is materially smaller than a BF16 baseline.

Example:
    python examples/quantization/check_modelopt_fp8_export.py \\
        --output ./hv15-480p-modelopt-fp8

    # Optional: compare disk size against a local or HF BF16 baseline.
    python examples/quantization/check_modelopt_fp8_export.py \\
        --output ./hv15-480p-modelopt-fp8 \\
        --baseline hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SUPPORTED_ALGOS = ("FP8",)


def _check_config(transformer_dir: Path) -> tuple[int, str | None]:
    """Returns (status, quant_algo). status: 0 pass, 1 fail, 2 warn."""
    cfg_path = transformer_dir / "config.json"
    if not cfg_path.exists():
        print(f"[FAIL] {cfg_path} missing.")
        return 1, None

    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print(f"[FAIL] No `quantization_config` block in {cfg_path}.")
        return 1, None

    print(f"[A] quantization_config from {cfg_path}:")
    print(json.dumps(qc, indent=2))

    quant_algo = qc.get("quant_algo")
    issues = []
    if qc.get("quant_method") != "modelopt":
        issues.append(f"quant_method={qc.get('quant_method')!r} (expected 'modelopt')")
    if quant_algo not in SUPPORTED_ALGOS:
        issues.append(
            f"quant_algo={quant_algo!r} (expected one of {SUPPORTED_ALGOS} — vllm-omni adapter may not auto-detect)"
        )

    if issues:
        print("[A] WARN — config looks incomplete:")
        for issue in issues:
            print(f"    - {issue}")
        return 2, quant_algo
    print(f"[A] PASS — config looks correct (quant_algo={quant_algo}).")
    return 0, quant_algo


def _read_safetensors_header(path: Path) -> dict:
    """Read the JSON header of a safetensors file. Bypass-safe — doesn't materialize tensors.

    Returns {tensor_name: {'dtype': 'F8_E4M3', 'shape': [...], 'data_offsets': [...]}}.
    Header dtype strings: F8_E4M3, F8_E5M2, BF16, F16, F32, F64, I8, I16, I32, I64, BOOL, U8, ...
    """
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header


def _classify_weight_scale_granularity(weight_scale_shapes: list[list[int]]) -> str:
    """Infer per-tensor vs per-channel vs per-block from sample weight_scale shapes.

    ModelOpt block-wise produces shapes like `[16, 1, 16, 1]` (broadcasting dims of 1
    interleaved with block-count dims). We count "meaningful" dims — ones with size > 1 —
    and classify: 0 meaningful dims = per-tensor (scalar), 1 = per-channel, 2+ = per-block.
    """
    if not weight_scale_shapes:
        return "no weight_scale tensors found"

    def meaningful_dims(shape: list[int]) -> int:
        return sum(1 for d in shape if d > 1)

    per_tensor = sum(1 for s in weight_scale_shapes if meaningful_dims(s) == 0)
    per_channel = sum(1 for s in weight_scale_shapes if meaningful_dims(s) == 1)
    per_block = sum(1 for s in weight_scale_shapes if meaningful_dims(s) >= 2)
    total = len(weight_scale_shapes)
    if per_tensor == total:
        return "per-tensor (all scalar scales)"
    if per_channel == total:
        return "per-channel (1 meaningful dim)"
    if per_block == total:
        return "per-block (2+ meaningful dims — e.g. [M//bm, 1, N//bn, 1] for tiles)"
    return f"mixed: per-tensor={per_tensor}, per-channel={per_channel}, per-block={per_block} of {total}"


def _check_safetensors(transformer_dir: Path) -> int:
    """Returns 0 on pass, 1 on fail. Reads on-disk dtype from the safetensors header."""
    files = sorted(transformer_dir.glob("*.safetensors"))
    if not files:
        print(f"[FAIL] No *.safetensors in {transformer_dir}.")
        return 1

    header_dtype_counts: Counter[str] = Counter()
    sample_quant_weight_keys: list[str] = []
    sample_scale_keys: list[str] = []
    weight_scale_shapes: list[list[int]] = []
    sample_weight_scale_entries: list[tuple[str, list[int]]] = []

    for f in files:
        try:
            header = _read_safetensors_header(f)
        except Exception as exc:
            print(f"[B] WARN — could not parse header of {f}: {exc}")
            continue
        for k, info in header.items():
            dtype = info.get("dtype", "?")
            header_dtype_counts[dtype] += 1
            # FP8 stores weights as F8_E4M3 directly.
            if dtype.startswith("F8") and k.endswith(".weight") and len(sample_quant_weight_keys) < 5:
                sample_quant_weight_keys.append(k)
            if k.endswith(("_scale", ".weight_scale", ".input_scale", "_scale_inv")) and len(sample_scale_keys) < 5:
                sample_scale_keys.append(k)
            if k.endswith(".weight_scale"):
                weight_scale_shapes.append(info.get("shape", []))
                if len(sample_weight_scale_entries) < 5:
                    sample_weight_scale_entries.append((k, info.get("shape", [])))

    print(f"\n[B] On-disk dtype counts across {len(files)} safetensors file(s) (from header, not get_tensor):")
    for dtype, count in sorted(header_dtype_counts.items(), key=lambda kv: -kv[1]):
        marker = "  <-- FP8" if dtype.startswith("F8") else ""
        print(f"    {dtype:10s} {count:>6d}{marker}")

    quant_count = sum(c for d, c in header_dtype_counts.items() if d.startswith("F8"))
    if quant_count == 0:
        print("[B] FAIL — no FP8 tensors on disk. Calibration likely did not actually quantize the weights.")
        return 1

    print(f"[B] PASS — {quant_count} FP8 tensors stored on disk.")
    if sample_quant_weight_keys:
        print(f"    sample quantized weight tensors: {sample_quant_weight_keys[:3]}")
    if sample_scale_keys:
        print(f"    sample scale tensors:            {sample_scale_keys[:3]}")
    print("    (Note: torch's get_tensor() may return these as bf16 views on some versions —")
    print("     irrelevant; vLLM's loader uses native FP8 ops.)")

    # Weight-scale granularity — per-tensor (scalar) vs per-channel (1-D) vs per-block (N-D).
    print(f"\n    weight_scale granularity: {_classify_weight_scale_granularity(weight_scale_shapes)}")
    for key, shape in sample_weight_scale_entries[:3]:
        print(f"      {key}: shape {shape}")
    return 0


def _disk_size_gib(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024**3)


def _check_size_vs_baseline(transformer_dir: Path, baseline: str | None) -> int:
    """Returns 0 always (informational only)."""
    quant_size = _disk_size_gib(transformer_dir)
    print(f"\n[C] FP8 transformer disk size: {quant_size:.2f} GiB")

    if baseline is None:
        print("[C] SKIP — pass --baseline <path or HF id> to compare against BF16.")
        return 0

    baseline_path = Path(baseline)
    if not baseline_path.exists():
        # Try HF download.
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("[C] SKIP — huggingface_hub not installed and baseline not a local path.")
            return 0
        print(f"    Downloading baseline transformer from HF: {baseline}")
        baseline_path = Path(snapshot_download(baseline, allow_patterns=["transformer/*"]))

    bf16_dir = baseline_path / "transformer" if (baseline_path / "transformer").exists() else baseline_path
    bf16_size = _disk_size_gib(bf16_dir)
    if bf16_size == 0:
        print(f"[C] WARN — baseline transformer dir empty: {bf16_dir}")
        return 0

    # Expected reduction for FP8: ~50%.
    min_reduction = 30
    reduction = (1 - quant_size / bf16_size) * 100
    print(f"[C] BF16 baseline transformer disk size: {bf16_size:.2f} GiB ({bf16_dir})")
    print(f"[C] Disk reduction: {reduction:.1f}%  (FP8 transformer is {quant_size / bf16_size:.0%} of BF16)")
    if reduction < min_reduction:
        print(
            f"[C] WARN — FP8 should typically reduce disk by ~40-50%; <{min_reduction}% suggests partial quantization."
        )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Path to the exported ModelOpt FP8 checkpoint root.")
    p.add_argument(
        "--baseline",
        default=None,
        help="Optional BF16 baseline (local diffusers dir or HF id) for disk-size comparison.",
    )
    args = p.parse_args()

    out_root = Path(args.output).expanduser().resolve()
    transformer_dir = out_root / "transformer"
    if not transformer_dir.exists():
        print(f"[FAIL] {transformer_dir} does not exist.")
        sys.exit(1)

    print(f"Checking: {out_root}\n")

    fail = 0
    config_status, _quant_algo = _check_config(transformer_dir)
    fail |= config_status
    fail |= _check_safetensors(transformer_dir)
    _check_size_vs_baseline(transformer_dir, args.baseline)

    print()
    if fail == 0:
        print("=" * 60)
        print("ALL CHECKS PASSED — checkpoint looks ready for vllm-omni serving.")
    elif fail == 1:
        print("=" * 60)
        print("FAILURES detected — calibration may need to be re-run.")
        sys.exit(1)
    else:
        print("=" * 60)
        print("WARNINGS only — checkpoint may serve but with caveats. See [A] above.")


if __name__ == "__main__":
    main()
