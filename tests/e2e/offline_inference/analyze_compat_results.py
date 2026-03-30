# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""analyze_compat_results.py — Feature compatibility analysis script.

Reads the directory tree produced by ``run_compat_test.py`` (using batch_text_to_image.py)
and generates:

  • Per-prompt image quality metrics (mean / max pixel diff vs baseline)
  • Per-config performance summary (generation time, speedup)
  • Compatibility status for each config
  • ``report.json``         — full machine-readable report
  • Terminal summary table + suggested compatibility matrix updates
  • ``chart_quality.png``   — MAE vs Baseline 1 for each config  (--charts)
  • ``chart_speedgain.png`` — latency-reduction % vs Baseline 1  (--charts)

Expected directory layout (written by run_compat_test.py):
  <results_dir>/
    manifest.json                 top-level test metadata
    baseline/
      config_info.json            config metadata
      batch_generation.log        batch processing log with timing info
      batch_generation.exitcode   batch exit code
      prompt_00.png               generated image
      prompt_00.log               per-prompt log (placeholder for batch runs)
      prompt_00.exitcode          exit code ("0" or non-zero)
      prompt_01.png / .log / .exitcode
      ...
    cfg_parallel/
      config_info.json
      batch_generation.log
      prompt_00.png / .log / .exitcode
      ...
    cfg_parallel+teacache/
      ...

Usage:
  python analyze_compat_results.py --results-dir ./compat_results/cfg_parallel

  # Custom thresholds:
  python analyze_compat_results.py \\
      --results-dir ./results \\
      --lossless-mean-threshold 0.02 \\
      --lossy-mean-threshold 0.15

  # Generate comparison charts (requires matplotlib):
  python analyze_compat_results.py --results-dir ./results --charts

  # Full report with HTML diff and charts:
  python analyze_compat_results.py --results-dir ./results --html --charts
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ===========================================================================
# Thresholds
# ===========================================================================

LOSSLESS_MEAN_THRESHOLD = 2e-2
LOSSLESS_MAX_THRESHOLD = 2e-1
LOSSY_MEAN_THRESHOLD = 0.15
LOSSY_MAX_THRESHOLD = 0.60

# ===========================================================================
# Log parsing
# ===========================================================================

_GEN_TIME_RE = re.compile(r"Total generation time:\s+([\d.]+)\s+seconds")
_AVG_GEN_TIME_RE = re.compile(r"Average generation time:\s+([\d.]+)\s+seconds")
_PROMPT_GEN_TIME_RE = re.compile(r"Prompt\s+(\d+):\s+([\d.]+)\s+seconds")
_SAVED_RE = re.compile(r"Saved generated image to (.+)")


def parse_log(log_path: Path) -> dict:
    """Extract metrics from one ``prompt_NN.log`` file.

    Returns
    -------
    dict with keys:
        gen_time_ms   float | None    omni.generate() duration
        saved_path    str  | None     path written by text_to_image.py
        has_error     bool            True when ERROR/Traceback found
        error_snippet str             last 5 lines of the log (for diagnosis)
    """
    text = log_path.read_text(errors="replace") if log_path.exists() else ""

    gen_time_ms = None
    m = _GEN_TIME_RE.search(text)
    if m:
        gen_time_ms = float(m.group(1)) * 1_000.0

    saved_path = None
    sm = _SAVED_RE.search(text)
    if sm:
        saved_path = sm.group(1).strip()

    lines = text.splitlines()
    has_error = any(kw in ln for ln in lines for kw in ("Traceback (most recent call last)", "Error:", "CUDA error"))
    error_snippet = "\n".join(lines[-5:]) if lines else ""

    return {
        "gen_time_ms": gen_time_ms,
        "saved_path": saved_path,
        "has_error": has_error,
        "error_snippet": error_snippet,
    }


def parse_batch_log(batch_log_path: Path, num_prompts: int) -> dict[int, float]:
    """Extract per-prompt generation times from batch_generation.log.

    Returns
    -------
    dict[int, float]
        Mapping from prompt index to generation time in milliseconds.
    """
    if not batch_log_path.exists():
        return {}
    
    text = batch_log_path.read_text(errors="replace")
    
    # Try to extract per-prompt generation times
    prompt_times: dict[int, float] = {}
    for match in _PROMPT_GEN_TIME_RE.finditer(text):
        prompt_idx = int(match.group(1)) - 1  # Convert to 0-based index
        gen_time_sec = float(match.group(2))
        prompt_times[prompt_idx] = gen_time_sec * 1_000.0
    
    # If per-prompt times not found, try to use average generation time
    if not prompt_times:
        avg_match = _AVG_GEN_TIME_RE.search(text)
        if avg_match:
            avg_time_ms = float(avg_match.group(1)) * 1_000.0
            # Assign average time to all prompts
            for idx in range(num_prompts):
                prompt_times[idx] = avg_time_ms
    
    return prompt_times


def read_exitcode(rc_path: Path) -> int:
    """Read the integer exit code stored by the bash script."""
    try:
        return int(rc_path.read_text().strip())
    except Exception:
        return -1


# ===========================================================================
# Image metrics
# ===========================================================================


def load_image(path: Path) -> "Image.Image | None":
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def diff_metrics(img_ref: "Image.Image", img_test: "Image.Image") -> tuple[float, float]:
    """Return ``(mean_abs_diff, max_abs_diff)`` in the [0, 1] range."""
    a = np.asarray(img_ref, dtype=np.float32) / 255.0
    b_img = img_test
    if img_ref.size != img_test.size:
        b_img = img_test.resize(img_ref.size, Image.BILINEAR)
    b = np.asarray(b_img, dtype=np.float32) / 255.0
    diff = np.abs(a - b)
    return float(diff.mean()), float(diff.max())


# ===========================================================================
# Config scanning
# ===========================================================================


def scan_config_dir(cfg_dir: Path) -> dict:
    """Scan one config directory and return a per-prompt result list.

    Returns
    -------
    dict
        config_info:   dict from config_info.json
        prompts:       list of per-prompt dicts
        all_success:   bool
    """
    meta_path = cfg_dir / "config_info.json"
    config_info = {}
    if meta_path.exists():
        config_info = json.loads(meta_path.read_text())

    prompt_results = []
    idx = 0
    while True:
        tag = f"prompt_{idx:02d}"
        img_path = cfg_dir / f"{tag}.png"
        log_path = cfg_dir / f"{tag}.log"
        rc_path = cfg_dir / f"{tag}.exitcode"

        if not log_path.exists() and not img_path.exists() and not rc_path.exists():
            break

        rc = read_exitcode(rc_path) if rc_path.exists() else -1
        log_data = parse_log(log_path)
        img = load_image(img_path) if img_path.exists() else None

        success = rc == 0 and img is not None
        if not success:
            print(f"  Prompt {idx:02d} failed: exitcode={rc}, image={img is not None}")

        prompt_results.append(
            {
                "idx": idx,
                "exitcode": rc,
                "success": success,
                "image": img,
                "img_path": str(img_path),
                "log_path": str(log_path),
                **log_data,
            }
        )
        idx += 1

    # Check for batch generation log and extract timing information
    batch_log_path = cfg_dir / "batch_generation.log"
    if batch_log_path.exists() and prompt_results:
        batch_times = parse_batch_log(batch_log_path, len(prompt_results))
        # Update prompt results with batch generation times
        for p in prompt_results:
            if p["idx"] in batch_times:
                p["gen_time_ms"] = batch_times[p["idx"]]

    all_success = bool(prompt_results) and all(p["success"] for p in prompt_results)

    return {
        "config_info": config_info,
        "prompts": prompt_results,
        "all_success": all_success,
    }


# ===========================================================================
# Compatibility status
# ===========================================================================


def compat_status(entry: dict) -> str:
    if entry.get("skipped"):
        return "SKIP"
    if not entry["all_success"]:
        return "ERROR"
    mean_diff = entry.get("mean_diff_avg")
    max_diff = entry.get("max_diff_avg")
    if mean_diff is None:
        return "PASS"  # baseline — no diff to compute
    is_lossy = entry.get("is_lossy", False)
    if is_lossy:
        return "PASS" if mean_diff <= LOSSY_MEAN_THRESHOLD else "WARN"
    return "PASS" if (mean_diff <= LOSSLESS_MEAN_THRESHOLD and max_diff <= LOSSLESS_MAX_THRESHOLD) else "FAIL"


# ===========================================================================
# Main analysis function
# ===========================================================================


def analyze(
    results_dir: Path,
    lossless_mean_threshold: float = LOSSLESS_MEAN_THRESHOLD,
    lossless_max_threshold: float = LOSSLESS_MAX_THRESHOLD,
    lossy_mean_threshold: float = LOSSY_MEAN_THRESHOLD,
    generate_html: bool = False,
    generate_charts: bool = False,
) -> dict:
    """Analyze a result directory produced by run_compat_test.sh.

    Parameters
    ----------
    results_dir:
        Path to the row output directory (contains manifest.json + config subdirs).
    lossless_mean_threshold / lossless_max_threshold:
        Quality thresholds for lossless features (CFG-Parallel, SP, TP).
    lossy_mean_threshold:
        Quality threshold for lossy features (TeaCache, Cache-DiT).
    generate_html:
        If True, write a ``diff_report.html`` with side-by-side image grids.
    generate_charts:
        If True, write ``chart_quality.png`` and ``chart_speedgain.png``
        (requires matplotlib).

    Returns
    -------
    dict
        Full report, also written to ``results_dir/report.json``.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # ── Load manifest ────────────────────────────────────────────────────────
    manifest_path = results_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    row = manifest.get("row", results_dir.name)
    print(f"\n{'=' * 72}")
    print(f"  Analyzing: {results_dir}")
    print(f"  Row: {row}  |  Model: {manifest.get('model', 'unknown')}")
    print(f"{'=' * 72}")

    # ── Scan all config subdirectories ───────────────────────────────────────
    # Order: baseline first, then alphabetically by name
    config_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: ("" if d.name == "baseline" else d.name),
    )

    print(f"\nFound {len(config_dirs)} config directories: {[d.name for d in config_dirs]}")

    scanned: dict[str, dict] = {}
    for cfg_dir in config_dirs:
        print(f"  Scanning {cfg_dir.name}/ …", end="", flush=True)
        data = scan_config_dir(cfg_dir)
        n_prompts = len(data["prompts"])
        n_ok = sum(1 for p in data["prompts"] if p["success"])
        print(f" {n_ok}/{n_prompts} OK")
        scanned[cfg_dir.name] = data

    # ── Load baseline images ─────────────────────────────────────────────────
    baseline_data = scanned.get("baseline", {})
    baseline_prompts = baseline_data.get("prompts", [])
    baseline_images: list[Image.Image | None] = [p.get("image") for p in baseline_prompts]
    baseline_gen_times = [p["gen_time_ms"] for p in baseline_prompts if p.get("gen_time_ms") is not None]
    baseline_mean_ms = sum(baseline_gen_times) / len(baseline_gen_times) if baseline_gen_times else None

    # ── Compute per-config metrics ───────────────────────────────────────────
    report_entries: list[dict] = []

    for cfg_name, data in scanned.items():
        cfg_info = data["config_info"]
        prompt_results = data["prompts"]
        is_lossy = cfg_info.get("is_lossy", False)

        # ── Timing ──────────────────────────────────────────────────────────
        gen_times = [p["gen_time_ms"] for p in prompt_results if p.get("gen_time_ms") is not None]

        entry: dict = {
            "name": cfg_name,
            "description": cfg_info.get("extra_args", ""),
            "total_gpus": cfg_info.get("total_gpus", "?"),
            "is_lossy": is_lossy,
            "cli_args": cfg_info.get("extra_args", ""),
            "all_success": data["all_success"],
            "num_success": sum(1 for p in prompt_results if p["success"]),
            "num_prompts": len(prompt_results),
            "skipped": len(prompt_results) == 0 and cfg_name != "baseline",
        }

        if gen_times:
            entry["mean_gen_time_ms"] = sum(gen_times) / len(gen_times)
            entry["min_gen_time_ms"] = min(gen_times)
            entry["max_gen_time_ms"] = max(gen_times)
            entry["p50_gen_time_ms"] = float(np.percentile(gen_times, 50))
            entry["p95_gen_time_ms"] = float(np.percentile(gen_times, 95))
            entry["per_prompt_gen_times_ms"] = gen_times
            entry["speedup"] = (
                (baseline_mean_ms / entry["mean_gen_time_ms"])
                if baseline_mean_ms and cfg_name != "baseline"
                else (1.0 if cfg_name == "baseline" else None)
            )
        else:
            entry["mean_gen_time_ms"] = None
            entry["speedup"] = None

        # ── Image quality vs baseline ────────────────────────────────────────
        if cfg_name != "baseline" and baseline_images:
            mean_diffs, max_diffs = [], []
            failed_diffs = []

            for p in prompt_results:
                if not p["success"]:
                    continue
                ref = baseline_images[p["idx"]] if p["idx"] < len(baseline_images) else None
                test_img = p.get("image")

                if ref is None:
                    print(f"    [WARN] No baseline image for prompt {p['idx']:02d} — skipping diff")
                    continue
                if test_img is None:
                    failed_diffs.append(p["idx"])
                    continue

                md, xd = diff_metrics(ref, test_img)
                mean_diffs.append(md)
                max_diffs.append(xd)

            entry["mean_diff_avg"] = sum(mean_diffs) / len(mean_diffs) if mean_diffs else None
            entry["max_diff_avg"] = sum(max_diffs) / len(max_diffs) if max_diffs else None
            entry["mean_diff_per_prompt"] = mean_diffs
            entry["max_diff_per_prompt"] = max_diffs
            entry["failed_diff_prompts"] = failed_diffs

            if mean_diffs:
                entry["mean_diff_p50"] = float(np.percentile(mean_diffs, 50))
                entry["mean_diff_p95"] = float(np.percentile(mean_diffs, 95))
        else:
            entry["mean_diff_avg"] = None
            entry["max_diff_avg"] = None

        # ── Error details ────────────────────────────────────────────────────
        failed_prompts = [p for p in prompt_results if not p["success"]]
        if failed_prompts:
            entry["errors"] = [
                {
                    "prompt_idx": p["idx"],
                    "exitcode": p["exitcode"],
                    "snippet": p.get("error_snippet", ""),
                }
                for p in failed_prompts[:5]  # at most 5
            ]

        entry["compat_status"] = compat_status(entry)
        report_entries.append(entry)

    # ── Assemble report ──────────────────────────────────────────────────────
    report = {
        "row": row,
        "results_dir": str(results_dir),
        "manifest": manifest,
        "thresholds": {
            "lossless_mean": lossless_mean_threshold,
            "lossless_max": lossless_max_threshold,
            "lossy_mean": lossy_mean_threshold,
        },
        "results": report_entries,
    }

    report_path = results_dir / "report.json"
    # Strip non-serializable PIL images before saving
    serializable = json.loads(json.dumps(report, default=str))
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull report written → {report_path}")

    # ── Print summary ────────────────────────────────────────────────────────
    _print_summary(report_entries, row)

    # ── Optional HTML diff report ────────────────────────────────────────────
    if generate_html:
        html_path = results_dir / "diff_report.html"
        _write_html_report(html_path, report_entries, scanned, baseline_images)
        print(f"HTML diff report   → {html_path}")

    # ── Optional matplotlib charts ───────────────────────────────────────────
    if generate_charts:
        _generate_charts(report, results_dir)

    return report


# ===========================================================================
# Pretty-print summary table
# ===========================================================================

_STATUS_SYM: dict[str, str] = {
    "PASS": "✅",
    "WARN": "⚠ ",
    "FAIL": "❌",
    "ERROR": "💥",
    "SKIP": "⏭ ",
}

_COMPAT_SYM = {
    "PASS": "✅",
    "WARN": "✅",
    "FAIL": "❌",
    "ERROR": "❌",
    "SKIP": "❓",
}


def _print_summary(entries: list[dict], row: str) -> None:
    col = [36, 6, 10, 11, 10, 11, 11]
    total_w = sum(col)
    header = (
        f"{'Config':<{col[0]}} {'GPUs':<{col[1]}} {'Status':<{col[2]}}"
        f" {'Gen ms':<{col[3]}} {'Speedup':<{col[4]}}"
        f" {'MeanDiff':<{col[5]}} {'MaxDiff':<{col[6]}}"
    )

    print(f"\n{'=' * total_w}")
    print(f"  COMPATIBILITY RESULTS — row: {row}")
    print(f"{'=' * total_w}")
    print(header)
    print(f"{'─' * total_w}")

    for e in entries:
        sym = _STATUS_SYM.get(e["compat_status"], "?")
        status_str = f"{sym}{e['compat_status']}"
        gpus = str(e.get("total_gpus", "?"))
        gen_ms = f"{e['mean_gen_time_ms']:.0f}" if e.get("mean_gen_time_ms") is not None else "N/A"
        speedup = f"{e['speedup']:.2f}x" if e.get("speedup") is not None and not e.get("skipped") else "N/A"
        mean_diff = f"{e['mean_diff_avg']:.4f}" if e.get("mean_diff_avg") is not None else "—"
        max_diff = f"{e['max_diff_avg']:.4f}" if e.get("max_diff_avg") is not None else "—"
        print(
            f"{e['name']:<{col[0]}} {gpus:<{col[1]}} {status_str:<{col[2]}}"
            f" {gen_ms:<{col[3]}} {speedup:<{col[4]}}"
            f" {mean_diff:<{col[5]}} {max_diff:<{col[6]}}"
        )

    print(f"{'=' * total_w}")
    print()
    print("Legend: ✅ PASS  ⚠ WARN  ❌ FAIL  💥 ERROR  ⏭ SKIP")
    print()

    # ── Suggested matrix updates ─────────────────────────────────────────────
    print(f"Suggested diffusion_features.md updates — row '{row}':")
    for e in entries:
        if e["name"] == "baseline" or e["name"] == row:
            continue
        col_feat = e["name"].replace(f"{row}+", "").replace(f"+{row}", "")
        sym = _COMPAT_SYM.get(e["compat_status"], "❓")
        print(f"  {row:<22} × {col_feat:<22} → {sym}  ({e['compat_status']})")
    print()

    # ── Per-config timing detail ─────────────────────────────────────────────
    print("Performance detail (generation time only, excludes model loading):")
    print(f"  {'Config':<36} {'Mean ms':>9}  {'Min ms':>8}  {'Max ms':>8}  {'P95 ms':>8}  {'Speedup':>9}")
    print(f"  {'─' * 36} {'─' * 9}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 9}")
    for e in entries:
        if e.get("mean_gen_time_ms") is None:
            continue
        sp_str = f"{e['speedup']:.2f}x" if e.get("speedup") is not None else "N/A"
        print(
            f"  {e['name']:<36}"
            f" {e['mean_gen_time_ms']:>9.0f}"
            f"  {e.get('min_gen_time_ms', 0):>8.0f}"
            f"  {e.get('max_gen_time_ms', 0):>8.0f}"
            f"  {e.get('p95_gen_time_ms', 0):>8.0f}"
            f"  {sp_str:>9}"
        )
    print()


# ===========================================================================
# Optional HTML diff report
# ===========================================================================


def _write_html_report(
    html_path: Path,
    entries: list[dict],
    scanned: dict[str, dict],
    baseline_images: list,
) -> None:
    """Write a minimal HTML page showing baseline vs each config side by side."""
    import base64
    from io import BytesIO

    def img_to_data_uri(img: "Image.Image") -> str:
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    rows_html = ""
    non_baseline = [e for e in entries if e["name"] != "baseline"]

    for entry in non_baseline:
        cfg_name = entry["name"]
        cfg_data = scanned.get(cfg_name, {})
        prompt_results = cfg_data.get("prompts", [])

        rows_html += f"<h2>{cfg_name} — {entry.get('compat_status', '?')}</h2>\n"
        rows_html += f"<p>GPUs: {entry.get('total_gpus', '?')} | Lossy: {entry.get('is_lossy', False)} | "
        rows_html += f"MeanDiff: {entry.get('mean_diff_avg', 'N/A')} | Speedup: {entry.get('speedup', 'N/A')}</p>\n"
        rows_html += "<div style='display:flex;flex-wrap:wrap;gap:8px;'>\n"

        for p in prompt_results[:10]:  # show first 10 prompts
            b_img = baseline_images[p["idx"]] if p["idx"] < len(baseline_images) else None
            t_img = p.get("image")
            prompt_short = str(p.get("prompt", f"prompt {p['idx']}"))[:60]

            rows_html += "<div style='border:1px solid #ccc;padding:4px;font-size:11px;'>\n"
            rows_html += f"<div style='max-width:420px;word-wrap:break-word;'>{prompt_short}</div>\n"
            rows_html += "<div style='display:flex;gap:4px;margin-top:4px;'>\n"
            if b_img:
                rows_html += f"<div><div>Baseline</div><img src='{img_to_data_uri(b_img.resize((200, 200)))}' width='200'/></div>\n"
            if t_img:
                md_str = f"{entry.get('mean_diff_per_prompt', [None])[p['idx']] or 0:.3f}"
                rows_html += f"<div><div>{cfg_name} (diff={md_str})</div><img src='{img_to_data_uri(t_img.resize((200, 200)))}' width='200'/></div>\n"
            rows_html += "</div></div>\n"

        rows_html += "</div>\n<hr/>\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Compatibility Diff Report</title>
  <style>body{{font-family:sans-serif;padding:20px;}} h2{{color:#333;}} img{{border:1px solid #ddd;}}</style>
</head>
<body>
<h1>Feature Compatibility Diff Report</h1>
{rows_html}
</body>
</html>"""
    html_path.write_text(html)


# ===========================================================================
# Matplotlib charts
# ===========================================================================


def _generate_charts(report: dict, output_dir: Path) -> list[Path]:
    """Write ``chart_quality.png`` and ``chart_speedgain.png`` to *output_dir*.

    Chart 1 — Image quality (MAE vs Baseline 1):
        Horizontal bar chart showing the mean absolute pixel error of every
        non-baseline config relative to the pure baseline (no-feature) run.
        Lower is better.  Threshold lines mark the acceptable limits for
        lossless and lossy features.

    Chart 2 — Speed gain (latency reduction % vs Baseline 1):
        Horizontal bar chart showing how much end-to-end latency each config
        reduces compared to the pure baseline.
        ``reduction = (baseline_ms − config_ms) / baseline_ms × 100 %``
        Positive = faster; negative = slower than baseline.

    Parameters
    ----------
    report:
        The dict returned by :func:`analyze` (or loaded from ``report.json``).
    output_dir:
        Directory where chart ONGs are written.

    Returns
    -------
    list[Path]  paths of written chart files (may be empty on import failure).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; safe on headless nodes
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping chart generation.")
        print("       Install with:  pip install matplotlib")
        return []

    results = report.get("results", [])
    manifest = report.get("manifest", {})
    baseline_feature = manifest.get("baseline_feature", None)
    thresholds = report.get("thresholds", {})

    # Configs to chart: everything except Baseline 1 ("baseline")
    non_baseline = [e for e in results if e["name"] != "baseline"]
    if not non_baseline:
        print("[WARN] No non-baseline configs found — skipping charts.")
        return []

    # ── Color palette ─────────────────────────────────────────────────────────
    # Baseline 2 (feature alone) → blue
    # Addon lossless             → green
    # Addon lossy (cache)        → orange
    _C_BASELINE2 = "#4472C4"
    _C_LOSSLESS = "#70AD47"
    _C_LOSSY = "#ED7D31"
    _C_SLOWER = "#D44D3B"
    _C_MISSING = "#BBBBBB"

    def _bar_color(entry: dict) -> str:
        if entry["name"] == baseline_feature:
            return _C_BASELINE2
        return _C_LOSSY if entry.get("is_lossy") else _C_LOSSLESS

    # ── Shared style helpers ─────────────────────────────────────────────────

    def _annotate_bar(ax, bar, text: str, x_range: float, align: str = "right") -> None:
        """Place a value label just outside the end of *bar*."""
        offset = x_range * 0.012
        bx = bar.get_width()
        if align == "right":
            ax.text(bx + offset, bar.get_y() + bar.get_height() / 2, text, va="center", ha="left", fontsize=8.5)
        else:
            ax.text(bx - offset, bar.get_y() + bar.get_height() / 2, text, va="center", ha="right", fontsize=8.5)

    chart_paths: list[Path] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # Chart 1 — Image quality (MAE)
    # ═══════════════════════════════════════════════════════════════════════════
    names_q = [e["name"] for e in non_baseline]
    values_q = [e.get("mean_diff_avg") for e in non_baseline]
    colors_q = [_bar_color(e) for e in non_baseline]
    bar_q = [v if v is not None else 0.0 for v in values_q]

    fig_h = max(4.0, len(names_q) * 0.65 + 2.5)
    fig1, ax1 = plt.subplots(figsize=(11, fig_h))

    bars1 = ax1.barh(names_q, bar_q, color=colors_q, alpha=0.87, height=0.60, edgecolor="white", linewidth=0.6)

    x_range_q = max(bar_q) if max(bar_q) > 0 else 0.01
    for bar, val in zip(bars1, values_q):
        if val is not None:
            _annotate_bar(ax1, bar, f"{val:.4f}", x_range_q)
        else:
            ax1.text(
                x_range_q * 0.012,
                bar.get_y() + bar.get_height() / 2,
                "N/A",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#888888",
                style="italic",
            )

    # Threshold reference lines
    ll_thresh = thresholds.get("lossless_mean", LOSSLESS_MEAN_THRESHOLD)
    lo_thresh = thresholds.get("lossy_mean", LOSSY_MEAN_THRESHOLD)
    ax1.axvline(
        ll_thresh,
        color=_C_LOSSLESS,
        linestyle="--",
        linewidth=1.4,
        alpha=0.80,
        label=f"Lossless threshold ({ll_thresh:.3f})",
    )
    ax1.axvline(
        lo_thresh, color=_C_LOSSY, linestyle="--", linewidth=1.4, alpha=0.80, label=f"Lossy threshold ({lo_thresh:.3f})"
    )

    ax1.set_xlabel(
        "Mean Absolute Pixel Error vs Baseline 1 — lower is better",
        fontsize=10,
    )
    ax1.set_title(
        "Image Quality Comparison vs Baseline 1 (no features)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xlim(0, x_range_q * 1.25)
    ax1.invert_yaxis()
    ax1.grid(axis="x", linestyle="--", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    legend_patches_q = [
        mpatches.Patch(color=_C_BASELINE2, label="Baseline 2 (feature alone)"),
        mpatches.Patch(color=_C_LOSSLESS, label="Addon — lossless"),
        mpatches.Patch(color=_C_LOSSY, label="Addon — lossy (cache)"),
    ]
    ax1.legend(
        handles=legend_patches_q
        + [
            plt.Line2D(
                [0],
                [0],
                color=_C_LOSSLESS,
                linestyle="--",
                linewidth=1.4,
                label=f"Lossless threshold ({ll_thresh:.3f})",
            ),
            plt.Line2D(
                [0], [0], color=_C_LOSSY, linestyle="--", linewidth=1.4, label=f"Lossy threshold ({lo_thresh:.3f})"
            ),
        ],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.85,
    )

    plt.tight_layout()
    chart1_path = output_dir / "chart_quality.png"
    fig1.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Quality chart      → {chart1_path}")
    chart_paths.append(chart1_path)

    # ═══════════════════════════════════════════════════════════════════════════
    # Chart 2 — Speed gain (latency reduction %)
    # ═══════════════════════════════════════════════════════════════════════════
    # reduction = (baseline_ms - config_ms) / baseline_ms
    #           = 1 - 1/speedup   (speedup = baseline_ms / config_ms)
    def _latency_reduction(entry: dict) -> "float | None":
        sp = entry.get("speedup")
        if sp is None or sp <= 0:
            return None
        return (1.0 - 1.0 / sp) * 100.0

    names_s = [e["name"] for e in non_baseline]
    values_s = [_latency_reduction(e) for e in non_baseline]
    bar_s = [v if v is not None else 0.0 for v in values_s]

    colors_s = []
    for entry, v in zip(non_baseline, values_s):
        if v is None:
            colors_s.append(_C_MISSING)
        elif v < 0:
            colors_s.append(_C_SLOWER)
        else:
            colors_s.append(_bar_color(entry))

    x_abs_max = max((abs(v) for v in bar_s), default=1.0) or 1.0

    fig2, ax2 = plt.subplots(figsize=(11, fig_h))
    bars2 = ax2.barh(names_s, bar_s, color=colors_s, alpha=0.87, height=0.60, edgecolor="white", linewidth=0.6)

    for bar, val in zip(bars2, values_s):
        if val is not None:
            sign = "right" if val >= 0 else "left"
            _annotate_bar(ax2, bar, f"{val:+.1f}%", x_abs_max, align=sign)
        else:
            ax2.text(
                x_abs_max * 0.012,
                bar.get_y() + bar.get_height() / 2,
                "N/A",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#888888",
                style="italic",
            )

    ax2.axvline(0, color="black", linewidth=1.0, alpha=0.7)
    ax2.set_xlabel(
        "End-to-End Latency Reduction vs Baseline 1 (%)  —  higher is better (faster)",
        fontsize=10,
    )
    ax2.set_title(
        "Speed Gain Comparison vs Baseline 1 (no features)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax2.invert_yaxis()
    ax2.grid(axis="x", linestyle="--", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    legend_patches_s = [
        mpatches.Patch(color=_C_BASELINE2, label="Baseline 2 (feature alone)"),
        mpatches.Patch(color=_C_LOSSLESS, label="Addon — lossless"),
        mpatches.Patch(color=_C_LOSSY, label="Addon — lossy (cache)"),
        mpatches.Patch(color=_C_SLOWER, label="Slower than Baseline 1"),
    ]
    ax2.legend(handles=legend_patches_s, loc="lower right", fontsize=8.5, framealpha=0.85)

    plt.tight_layout()
    chart2_path = output_dir / "chart_speedgain.png"
    fig2.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Speed gain chart   → {chart2_path}")
    chart_paths.append(chart2_path)

    return chart_paths


# ===========================================================================
# CLI entry point
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Analyze compatibility test results from run_compat_test.py (or the legacy run_compat_test.sh)."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir",
        required=True,
        metavar="PATH",
        help=("Row output directory produced by run_compat_test.py (contains manifest.json and config subdirs)."),
    )
    p.add_argument(
        "--lossless-mean-threshold",
        type=float,
        default=LOSSLESS_MEAN_THRESHOLD,
        help="Max acceptable mean pixel diff for lossless features.",
    )
    p.add_argument(
        "--lossless-max-threshold",
        type=float,
        default=LOSSLESS_MAX_THRESHOLD,
        help="Max acceptable per-pixel diff for lossless features.",
    )
    p.add_argument(
        "--lossy-mean-threshold",
        type=float,
        default=LOSSY_MEAN_THRESHOLD,
        help="Max acceptable mean pixel diff for lossy features (TeaCache, Cache-DiT).",
    )
    p.add_argument(
        "--html",
        action="store_true",
        help="Also generate an HTML side-by-side diff report (diff_report.html).",
    )
    p.add_argument(
        "--charts",
        action="store_true",
        help=(
            "Generate comparison charts (requires matplotlib): "
            "chart_quality.png (MAE vs Baseline 1) and "
            "chart_speedgain.png (latency reduction %% vs Baseline 1)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = analyze(
            results_dir=Path(args.results_dir),
            lossless_mean_threshold=args.lossless_mean_threshold,
            lossless_max_threshold=args.lossless_max_threshold,
            lossy_mean_threshold=args.lossy_mean_threshold,
            generate_html=args.html,
            generate_charts=args.charts,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    failed = [e for e in report["results"] if e["compat_status"] in ("FAIL", "ERROR")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
