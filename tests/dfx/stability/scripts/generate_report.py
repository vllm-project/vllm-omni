#!/usr/bin/env python3
"""
Generate a GPU memory monitoring report (HTML with charts and simple anomaly markers)
from the CSV produced by `resource_monitor.sh`.

This is used to generate an archivable report in CI after a long-running stability
test so that the report remains available even after environment cleanup.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def load_csv(csv_path: str) -> list[dict]:
    """Load and parse rows from a GPU monitoring CSV file.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A list of parsed row dicts. Each row contains keys such as
        `timestamp_epoch`, `gpu_index`, and `memory_*`.
        Invalid rows are skipped and logged.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["timestamp_epoch"] = int(float(r["timestamp_epoch"]))
                r["gpu_index"] = int(r["gpu_index"])
                r["memory_used_mb"] = int(r["memory_used_mb"])
                r["memory_total_mb"] = int(r["memory_total_mb"])
                r["memory_util_pct"] = int(r["memory_util_pct"])
                rows.append(r)
            except (KeyError, ValueError) as e:
                logger.debug("Skip invalid CSV row: %s", e)
                continue
    return rows


def compute_stats(rows: list[dict]) -> dict:
    """Compute per-GPU min/max/avg/P50/P95 memory utilization and sample counts.

    Args:
        rows: Rows returned by `load_csv`.

    Returns:
        A stats dict keyed by `gpu_index`, with values containing
        `min`/`max`/`avg`/`p50`/`p95`/`samples`.
    """
    by_gpu = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu_index"]].append(r["memory_util_pct"])
    stats = {}
    for gpu, pcts in by_gpu.items():
        if not pcts:
            continue
        pcts_sorted = sorted(pcts)
        n = len(pcts_sorted)
        stats[gpu] = {
            "min": min(pcts),
            "max": max(pcts),
            "avg": round(sum(pcts) / n, 1),
            "p50": pcts_sorted[n // 2] if n else 0,
            "p95": pcts_sorted[int(n * 0.95)] if n > 1 else pcts_sorted[0],
            "samples": n,
        }
    return stats


def find_anomalies(rows: list[dict], high_pct: int = 95, low_pct: int = 5) -> list[dict]:
    """Find simple anomalies where memory utilization exceeds `high_pct` or drops below `low_pct`."""
    anomalies = []
    for r in rows:
        pct = r["memory_util_pct"]
        ts_iso = r.get("timestamp_iso") or datetime.fromtimestamp(r["timestamp_epoch"]).strftime("%Y-%m-%d %H:%M:%S")
        extra = {"timestamp_iso": ts_iso}
        if pct >= high_pct:
            anomalies.append({**r, **extra, "type": "high", "threshold": high_pct})
        elif pct <= low_pct:
            anomalies.append({**r, **extra, "type": "low", "threshold": low_pct})
    return anomalies


def _anomaly_sort_key(a: dict) -> tuple[int, str, int]:
    return (a["gpu_index"], a["type"], a["timestamp_epoch"])


def merge_anomalies_into_periods(anomalies: list[dict], max_gap_seconds: int = 120) -> list[dict]:
    """Merge consecutive anomalies (same GPU, same type) into time periods for display.

    This avoids truncation: instead of showing only the first N raw points, we show
    each continuous period once, so all below-threshold (or above-threshold) periods
    are visible in the report.
    """
    if not anomalies:
        return []
    # Sort by GPU, type, then time so we can merge consecutive same-GPU same-type.
    sorted_anomalies = sorted(anomalies, key=_anomaly_sort_key)
    periods = []
    for a in sorted_anomalies:
        ts = a["timestamp_epoch"]
        ts_iso = a.get("timestamp_iso") or datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        if not periods:
            periods.append(
                {
                    "gpu_index": a["gpu_index"],
                    "type": a["type"],
                    "threshold": a["threshold"],
                    "start_epoch": ts,
                    "end_epoch": ts,
                    "start_iso": ts_iso,
                    "end_iso": ts_iso,
                    "min_pct": a["memory_util_pct"],
                    "max_pct": a["memory_util_pct"],
                    "samples": 1,
                }
            )
            continue
        last = periods[-1]
        if (
            last["gpu_index"] == a["gpu_index"]
            and last["type"] == a["type"]
            and (ts - last["end_epoch"]) <= max_gap_seconds
        ):
            last["end_epoch"] = ts
            last["end_iso"] = ts_iso
            last["min_pct"] = min(last["min_pct"], a["memory_util_pct"])
            last["max_pct"] = max(last["max_pct"], a["memory_util_pct"])
            last["samples"] += 1
        else:
            periods.append(
                {
                    "gpu_index": a["gpu_index"],
                    "type": a["type"],
                    "threshold": a["threshold"],
                    "start_epoch": ts,
                    "end_epoch": ts,
                    "start_iso": ts_iso,
                    "end_iso": ts_iso,
                    "min_pct": a["memory_util_pct"],
                    "max_pct": a["memory_util_pct"],
                    "samples": 1,
                }
            )
    return periods


def build_series_by_gpu(rows: list[dict]) -> tuple[list[float], dict[int, list[float]]]:
    """Deduplicate timestamps in time order and build a memory usage series (GB) for each GPU."""
    times = []
    by_ts_gpu = defaultdict(dict)
    for r in rows:
        t = r["timestamp_epoch"]
        g = r["gpu_index"]
        # Convert MB to GB for chart display.
        by_ts_gpu[t][g] = r["memory_used_mb"] / 1024.0
    for t in sorted(by_ts_gpu.keys()):
        times.append(t)
    gpu_series = defaultdict(list)
    for t in times:
        gpus = by_ts_gpu[t]
        for g in sorted(gpus.keys()):
            gpu_series[g].append(gpus[g])
    return times, dict(gpu_series)


def render_html(
    run_id: str,
    csv_path: str,
    rows: list[dict],
    stats: dict,
    anomalies: list[dict],
    out_path: str,
) -> None:
    """Generate a single-file HTML report and write it to `out_path`.

    Args:
        run_id: Run identifier used in the title.
        csv_path: Source CSV path, used only for displaying the file name.
        rows: Raw data rows.
        stats: Statistics returned by `compute_stats`.
        anomalies: Anomaly list returned by `find_anomalies`.
        out_path: Output HTML path; parent directories are created if needed.
    """
    times, gpu_series = build_series_by_gpu(rows)
    # X-axis time, e.g. 02-27 11:38:07 (local timezone), which is easier to read for long stability runs.
    labels_js = [f'"{datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S")}"' for t in times]
    datasets_js = []
    colors = ["#e94560", "#0f3460", "#533483", "#16c79a"]
    for i, (gpu, series) in enumerate(sorted(gpu_series.items())):
        color = colors[i % len(colors)]
        data_str = ",".join(str(v) for v in series)
        datasets_js.append(
            f'{{ label: "GPU {gpu}", data: [{data_str}], borderColor: "{color}", '
            f'backgroundColor: "{color}20", fill: true, tension: 0.2 }}'
        )

    stats_rows = []
    for gpu in sorted(stats.keys()):
        s = stats[gpu]
        stats_rows.append(
            f"<tr><td>GPU {gpu}</td><td>{s['min']}%</td><td>{s['max']}%</td>"
            f"<td>{s['avg']}%</td><td>{s['p50']}</td><td>{s['p95']}</td><td>{s['samples']}</td></tr>"
        )
    stats_table = "\n".join(stats_rows)

    # Use merged periods so every below/above-threshold time period is shown (no truncation).
    periods = merge_anomalies_into_periods(anomalies)
    anomaly_cells = []
    for p in periods:
        time_range = f"{p['start_iso']} — {p['end_iso']}" if p["start_iso"] != p["end_iso"] else p["start_iso"]
        pct_str = f"{p['min_pct']}–{p['max_pct']}%" if p["min_pct"] != p["max_pct"] else f"{p['min_pct']}%"
        anomaly_cells.append(
            f"<tr><td>{time_range}</td><td>GPU {p['gpu_index']}</td>"
            f"<td>{pct_str}</td><td>{p['type']} (threshold {p['threshold']}%)</td><td>{p['samples']}</td></tr>"
        )
    anomaly_table = "\n".join(anomaly_cells) if anomaly_cells else "<tr><td colspan='5'>None</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GPU memory monitor report - {run_id}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    body {{ font-family: system-ui,sans-serif; margin: 1rem; background: #1a1a2e; color: #eee; }}
    h1 {{ font-size: 1.25rem; }} .meta {{ color: #888; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; margin: 1rem 0; }} th, td {{ border: 1px solid #333; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #16213e; }} canvas {{ max-height: 400px; }}
  </style>
</head>
<body>
  <h1>Stability GPU memory monitor report</h1>
  <p class="meta">Run: {run_id} | Data file: {os.path.basename(csv_path)} | Samples: {len(rows)}</p>
  <h2>Statistics</h2>
  <table>
    <tr><th>GPU</th><th>Min %</th><th>Max %</th><th>Avg %</th><th>P50</th><th>P95</th><th>Samples</th></tr>
    {stats_table}
  </table>
  <h2>Memory utilization over time</h2>
  <button id="toggleMode">Split per GPU</button>
  <div id="chartsContainer">
    <canvas id="chart"></canvas>
  </div>
  <h2>Anomalies (high/low threshold)</h2>
  <p class="meta">Consecutive samples merged into periods; all periods listed.</p>
  <table>
    <tr><th>Period</th><th>GPU</th><th>Util %</th><th>Type</th><th>Samples</th></tr>
    {anomaly_table}
  </table>
  <script>
    const labels = [{",".join(labels_js)}];
    const gpuDatasets = [{",".join(datasets_js)}];

    let mode = "combined";
    let charts = [];

    const commonOptions = {{
      responsive: true,
      scales: {{
        y: {{
          title: {{
            display: true,
            text: "Memory used (GB)"
          }}
        }}
      }}
    }};

    function destroyCharts() {{
      charts.forEach(c => c.destroy());
      charts = [];
    }}

    function renderCombined() {{
      const container = document.getElementById("chartsContainer");
      container.innerHTML = '<canvas id="chart"></canvas>';
      const ctx = document.getElementById("chart");
      charts = [new Chart(ctx, {{
        type: "line",
        data: {{ labels, datasets: gpuDatasets }},
        options: commonOptions
      }})];
    }}

    function renderSplit() {{
      const container = document.getElementById("chartsContainer");
      container.innerHTML = "";
      charts = [];
      gpuDatasets.forEach((ds, idx) => {{
        const canvas = document.createElement("canvas");
        canvas.id = "chart-gpu-" + idx;
        container.appendChild(canvas);
        charts.push(new Chart(canvas, {{
          type: "line",
          data: {{ labels, datasets: [ds] }},
          options: commonOptions
        }}));
      }});
    }}

    document.getElementById("toggleMode").addEventListener("click", () => {{
      destroyCharts();
      if (mode === "combined") {{
        mode = "split";
        document.getElementById("toggleMode").textContent = "Combine all GPUs";
        renderSplit();
      }} else {{
        mode = "combined";
        document.getElementById("toggleMode").textContent = "Split per GPU";
        renderCombined();
      }}
    }});

    // Default: combined chart
    renderCombined();
  </script>
</body>
</html>
"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    """Entry point: read the CSV path and optional output path from the command line and generate `report.html`.

    Returns:
        0 on success, 1 on invalid arguments or invalid data.
    """
    if len(sys.argv) < 2:
        logger.error("Usage: generate_report.py <gpu_metrics.csv> [output.html]")
        return 1
    csv_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace(".csv", "_report.html")
    if not os.path.isfile(csv_path):
        logger.error("File not found: %s", csv_path)
        return 1
    run_id = Path(csv_path).parent.name
    rows = load_csv(csv_path)
    if not rows:
        logger.error("CSV has no valid data")
        return 1
    stats = compute_stats(rows)
    anomalies = find_anomalies(rows)
    render_html(run_id, csv_path, rows, stats, anomalies, out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
