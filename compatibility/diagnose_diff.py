#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""diagnose_diff.py — Image difference diagnostics for compatibility test results.

Compares generated images from one or more feature configs against a reference
directory (default: ``baseline/``) and reports per-image metrics:

  * MeanDiff   — mean absolute pixel difference (range [0, 1])
  * MaxDiff    — maximum absolute pixel difference (range [0, 1])
  * SSIM       — structural similarity index (range [0, 1]; 1 = identical)

Usage:
  # Single config
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel \\
      --config cfg_parallel

  # Multiple configs at once
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel \\
      --config cfg_parallel cfg_parallel+teacache cfg_parallel+cache_dit

  # All non-baseline configs in the results directory
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel --all

  # Use a custom reference directory instead of baseline/
  python diagnose_diff.py --results-dir ./compat_results/cfg_parallel \\
      --config cfg_parallel+teacache --reference cfg_parallel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as _ssim

    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# ── ANSI colour helpers ───────────────────────────────────────────────────────

_RED = "\033[31m"
_YLW = "\033[33m"
_GRN = "\033[32m"
_CYN = "\033[36m"
_RST = "\033[0m"
_BLD = "\033[1m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RST}"


# ── Image helpers ─────────────────────────────────────────────────────────────


def load_image(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def diff_metrics(
    img_ref: Image.Image,
    img_test: Image.Image,
) -> tuple[float, float, float]:
    """Return (mean_abs_diff, max_abs_diff, ssim) all in [0, 1]."""
    if img_ref.size != img_test.size:
        img_test = img_test.resize(img_ref.size, Image.BILINEAR)

    a = np.asarray(img_ref, dtype=np.float32) / 255.0
    b = np.asarray(img_test, dtype=np.float32) / 255.0
    diff = np.abs(a - b)

    mean_d = float(diff.mean())
    max_d = float(diff.max())

    ssim_val = float("nan")
    if _HAS_SKIMAGE:
        ssim_val = float(_ssim(a, b, data_range=1.0, channel_axis=2, win_size=7))

    return mean_d, max_d, ssim_val


def _status(mean_d: float, max_d: float) -> tuple[str, str]:
    """Return (label, colour) for a diff measurement."""
    if max_d > 0.3 or mean_d > 0.05:
        return "LARGE", _RED
    if max_d > 0.1 or mean_d > 0.02:
        return "WARN ", _YLW
    return "OK   ", _GRN


# ── Single-config analysis ────────────────────────────────────────────────────


def analyze_config(
    results_dir: Path,
    config_name: str,
    reference_name: str = "baseline",
) -> dict | None:
    """Analyze one config against *reference_name* and return a result dict."""
    ref_dir = results_dir / reference_name
    cfg_dir = results_dir / config_name

    if not ref_dir.exists():
        print(f"  {_c('ERROR', _RED)} reference directory not found: {ref_dir}")
        return None
    if not cfg_dir.exists():
        print(f"  {_c('ERROR', _RED)} config directory not found: {cfg_dir}")
        return None

    diffs: list[dict] = []
    idx = 0
    while True:
        ref_path = ref_dir / f"prompt_{idx:02d}.png"
        tst_path = cfg_dir / f"prompt_{idx:02d}.png"
        if not ref_path.exists():
            break

        ref_img = load_image(ref_path)
        tst_img = load_image(tst_path)

        if ref_img is None or tst_img is None:
            diffs.append(
                {
                    "idx": idx,
                    "mean_diff": float("nan"),
                    "max_diff": float("nan"),
                    "ssim": float("nan"),
                    "error": "load_failed",
                    "ref_path": str(ref_path),
                    "tst_path": str(tst_path),
                }
            )
            idx += 1
            continue

        mean_d, max_d, ssim_v = diff_metrics(ref_img, tst_img)
        diffs.append(
            {
                "idx": idx,
                "mean_diff": mean_d,
                "max_diff": max_d,
                "ssim": ssim_v,
                "ref_path": str(ref_path),
                "tst_path": str(tst_path),
            }
        )
        idx += 1

    return {"config_name": config_name, "reference": reference_name, "diffs": diffs}


# ── Pretty-print one result ───────────────────────────────────────────────────


def print_result(result: dict, top_n: int = 10) -> None:
    config_name = result["config_name"]
    reference = result["reference"]
    diffs = result["diffs"]

    print(f"\n{'=' * 80}")
    print(_c(f"Config: {config_name}  (vs. {reference})", _BLD) + f"  [{len(diffs)} images]")
    print("=" * 80)

    if not diffs:
        print("  No comparable images found.")
        return

    valid = [d for d in diffs if "error" not in d]
    errors = [d for d in diffs if "error" in d]

    if errors:
        print(f"  {_c(f'{len(errors)} image(s) failed to load', _RED)}")

    if not valid:
        return

    # Per-image table sorted by max_diff
    sorted_by_max = sorted(valid, key=lambda x: x["max_diff"], reverse=True)

    ssim_col = "SSIM  " if _HAS_SKIMAGE else "SSIM  "
    hdr = f"{'#':<4} {'Prompt':<10} {'MeanDiff':<12} {'MaxDiff':<12} {ssim_col:<10} Status"
    print(f"\n  Top {min(top_n, len(sorted_by_max))} by MaxDiff:")
    print("  " + "-" * (len(hdr)))
    print("  " + hdr)
    print("  " + "-" * (len(hdr)))
    for rank, d in enumerate(sorted_by_max[:top_n], 1):
        label, colour = _status(d["mean_diff"], d["max_diff"])
        ssim_str = f"{d['ssim']:.4f}" if not np.isnan(d["ssim"]) else "  n/a  "
        print(
            f"  {rank:<4} prompt_{d['idx']:02d}   "
            f"{d['mean_diff']:<12.6f} {d['max_diff']:<12.6f} {ssim_str:<10} " + _c(label, colour)
        )

    # Statistics
    mean_diffs = [d["mean_diff"] for d in valid]
    max_diffs = [d["max_diff"] for d in valid]
    ssims = [d["ssim"] for d in valid if not np.isnan(d["ssim"])]

    print(f"\n  {'─' * 40}")
    print("  Statistics:")
    print(
        f"    MeanDiff — min={min(mean_diffs):.6f}  max={max(mean_diffs):.6f}"
        f"  avg={np.mean(mean_diffs):.6f}  median={np.median(mean_diffs):.6f}"
    )
    print(
        f"    MaxDiff  — min={min(max_diffs):.6f}  max={max(max_diffs):.6f}"
        f"  avg={np.mean(max_diffs):.6f}  median={np.median(max_diffs):.6f}"
    )
    if ssims:
        print(
            f"    SSIM     — min={min(ssims):.4f}  max={max(ssims):.4f}"
            f"  avg={np.mean(ssims):.4f}  median={np.median(ssims):.4f}"
        )

    n_large = sum(1 for d in valid if d["max_diff"] > 0.3 or d["mean_diff"] > 0.05)
    n_warn = sum(1 for d in valid if 0.1 < d["max_diff"] <= 0.3 or 0.02 < d["mean_diff"] <= 0.05)
    n_ok = len(valid) - n_large - n_warn
    print(
        "\n  Verdict: "
        + _c(f"{n_ok} OK", _GRN)
        + " / "
        + _c(f"{n_warn} WARN", _YLW)
        + " / "
        + _c(f"{n_large} LARGE", _RED)
        + f" (out of {len(valid)} images)"
    )


# ── Summary comparison table (multiple configs) ───────────────────────────────


def print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 80}")
    print(_c("SUMMARY — all configs vs. reference", _BLD))
    print("=" * 80)

    hdr = f"  {'Config':<40} {'Ref':<12} {'AvgMean':<12} {'AvgMax':<12} {'AvgSSIM':<10} {'OK/WARN/LARGE'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        valid = [d for d in r["diffs"] if "error" not in d]
        if not valid:
            print(f"  {r['config_name']:<40} (no valid images)")
            continue
        avg_mean = np.mean([d["mean_diff"] for d in valid])
        avg_max = np.mean([d["max_diff"] for d in valid])
        ssims = [d["ssim"] for d in valid if not np.isnan(d["ssim"])]
        avg_ssim = np.mean(ssims) if ssims else float("nan")

        n_large = sum(1 for d in valid if d["max_diff"] > 0.3 or d["mean_diff"] > 0.05)
        n_warn = sum(1 for d in valid if 0.1 < d["max_diff"] <= 0.3 or 0.02 < d["mean_diff"] <= 0.05)
        n_ok = len(valid) - n_large - n_warn

        ssim_str = f"{avg_ssim:.4f}" if not np.isnan(avg_ssim) else " n/a  "
        verdict = _c(f"{n_ok}✓", _GRN) + " / " + _c(f"{n_warn}⚠", _YLW) + " / " + _c(f"{n_large}✗", _RED)
        print(
            f"  {r['config_name']:<40} {r['reference']:<12} {avg_mean:<12.6f} {avg_max:<12.6f} {ssim_str:<10} {verdict}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnose image differences between feature configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results-dir",
        required=True,
        help="Results directory (contains baseline/ and config subdirectories).",
    )
    p.add_argument(
        "--config",
        nargs="*",
        default=[],
        metavar="CONFIG",
        help="One or more config names to diagnose (e.g. cfg_parallel cfg_parallel+teacache).",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Diagnose ALL non-reference subdirectories found under --results-dir.",
    )
    p.add_argument(
        "--reference",
        default="baseline",
        help="Reference config name to compare against (default: baseline).",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of worst-diff images to show per config (default: 10).",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        help="Save a JSON report for each analyzed config.",
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    # Collect config names to analyze
    configs_to_analyze: list[str] = list(args.config)
    if args.all:
        discovered = sorted(
            d.name
            for d in results_dir.iterdir()
            if d.is_dir() and d.name != args.reference and not d.name.startswith(".") and not d.name.endswith(".json")
        )
        configs_to_analyze = list(dict.fromkeys(configs_to_analyze + discovered))

    if not configs_to_analyze:
        print(
            "[ERROR] No configs specified. Use --config <name> or --all.",
            file=sys.stderr,
        )
        return 1

    if not _HAS_SKIMAGE:
        print("[WARN] scikit-image not found; SSIM metric disabled. Install with: pip install scikit-image")

    all_results: list[dict] = []
    for cfg in configs_to_analyze:
        result = analyze_config(results_dir, cfg, reference_name=args.reference)
        if result is not None:
            all_results.append(result)
            print_result(result, top_n=args.top)

            if args.save_json:
                json_path = results_dir / f"diff_diagnosis_{cfg}.json"
                _save_json(result, json_path)
                print(f"  JSON report → {json_path}")

    if len(all_results) > 1:
        print_summary(all_results)

    return 0


def _save_json(result: dict, path: Path) -> None:
    valid = [d for d in result["diffs"] if "error" not in d]
    mean_diffs = [d["mean_diff"] for d in valid]
    max_diffs = [d["max_diff"] for d in valid]
    ssims = [d["ssim"] for d in valid if not np.isnan(d["ssim"])]

    payload = {
        "config_name": result["config_name"],
        "reference": result["reference"],
        "total_images": len(result["diffs"]),
        "valid_images": len(valid),
        "ssim_available": _HAS_SKIMAGE,
        "statistics": {
            "mean_diff": {
                "min": float(min(mean_diffs)) if mean_diffs else None,
                "max": float(max(mean_diffs)) if mean_diffs else None,
                "avg": float(np.mean(mean_diffs)) if mean_diffs else None,
                "median": float(np.median(mean_diffs)) if mean_diffs else None,
            },
            "max_diff": {
                "min": float(min(max_diffs)) if max_diffs else None,
                "max": float(max(max_diffs)) if max_diffs else None,
                "avg": float(np.mean(max_diffs)) if max_diffs else None,
                "median": float(np.median(max_diffs)) if max_diffs else None,
            },
            "ssim": {
                "min": float(min(ssims)) if ssims else None,
                "max": float(max(ssims)) if ssims else None,
                "avg": float(np.mean(ssims)) if ssims else None,
                "median": float(np.median(ssims)) if ssims else None,
            },
        },
        "high_diff_images": [
            {"idx": d["idx"], "mean_diff": d["mean_diff"], "max_diff": d["max_diff"], "ssim": d["ssim"]}
            for d in sorted(valid, key=lambda x: x["max_diff"], reverse=True)
            if d["max_diff"] > 0.1
        ],
        "all_diffs": result["diffs"],
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
