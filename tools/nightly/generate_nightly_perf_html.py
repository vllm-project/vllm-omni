#!/usr/bin/env python3
"""
Generate a nightly HTML performance report from JSON results.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

LOGGER = logging.getLogger(__name__)

_RESULT_JSON_PREFIX = "result_test_"
_DIFFUSION_JSON_PREFIX = "diffusion_perf_"
# Fallback to 'tests' when env vars are not set, to match CI_nightly_perf.md defaults.
DEFAULT_INPUT_DIR = os.getenv("DEFAULT_INPUT_DIR") if os.getenv("DEFAULT_INPUT_DIR") else "tests"
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR") if os.getenv("DEFAULT_OUTPUT_DIR") else "tests"
DEFAULT_DIFFUSION_INPUT_DIR = os.getenv("DIFFUSION_BENCHMARK_DIR")


def _vllm_omni_root() -> str:
    path = os.path.dirname(os.path.abspath(__file__))
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "tests")):
            return path
        path = os.path.dirname(path)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _default_input_dir() -> str:
    root = _vllm_omni_root()
    return os.path.join(root, DEFAULT_INPUT_DIR)


def _default_output_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(_vllm_omni_root(), DEFAULT_OUTPUT_DIR, f"nightly_perf_{ts}.html")


def _default_diffusion_input_dir(input_dir: str) -> str:
    return DEFAULT_DIFFUSION_INPUT_DIR if DEFAULT_DIFFUSION_INPUT_DIR else input_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read performance JSON files from vllm-omni/tests/ and generate an HTML report."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=_default_input_dir(),
        help="Directory containing performance JSON files; default is <vllm-omni-root>/DEFAULT_INPUT_DIR.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=_default_output_file(),
        help="Output path of the HTML report; \
            default is <vllm-omni-root>/DEFAULT_OUTPUT_DIR/nightly_perf_<timestamp>.html.",
    )
    parser.add_argument(
        "--diffusion-input-dir",
        type=str,
        default=None,
        help=(
            "Directory containing diffusion_perf_*.json files; default is "
            "DIFFUSION_BENCHMARK_DIR, fallback to --input-dir."
        ),
    )
    return parser.parse_args()


def _load_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("failed to load json '%s': %s", path, exc)
        return None

    if not isinstance(data, dict):
        LOGGER.warning("json root in '%s' is not an object, skip", path)
        return None

    return data


def _parse_from_filename(filename: str) -> dict[str, Any]:
    name, ext = os.path.splitext(filename)
    if ext != ".json" or not name.startswith(_RESULT_JSON_PREFIX):
        return {}

    core = name[len(_RESULT_JSON_PREFIX) :]
    parts = core.split("_")
    if len(parts) < 5:
        LOGGER.warning("filename '%s' does not match expected pattern, skip parsing test metadata", filename)
        return {}

    timestamp = parts[-1]
    num_prompts_str = parts[-2]
    max_concurrency_str = parts[-3]
    dataset_name = parts[-4]
    test_name = "_".join(parts[:-4]) if parts[:-4] else ""

    parsed: dict[str, Any] = {}

    if len(timestamp) >= 15:
        parsed["date"] = timestamp

    if dataset_name in ("random", "random-mm"):
        parsed["dataset_name"] = dataset_name

    try:
        parsed["num_prompts"] = int(num_prompts_str)
    except (TypeError, ValueError):
        pass

    try:
        parsed["max_concurrency"] = int(max_concurrency_str)
    except (TypeError, ValueError):
        pass

    if test_name:
        parsed["test_name"] = test_name

    return parsed


def _iter_omni_json_records(input_dir: str) -> Iterable[dict[str, Any]]:
    if not os.path.isdir(input_dir):
        LOGGER.warning("input dir '%s' does not exist or is not a directory", input_dir)
        return

    for entry in sorted(os.listdir(input_dir)):
        if not entry.endswith(".json"):
            continue
        if not entry.startswith(_RESULT_JSON_PREFIX):
            continue
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            continue

        data = _load_json_file(full_path)
        if data is None:
            continue

        record: dict[str, Any] = dict(data)
        filename_meta = _parse_from_filename(os.path.basename(full_path))

        if "date" not in record or not record["date"]:
            if "date" in filename_meta:
                record["date"] = filename_meta["date"]
            else:
                record["date"] = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        if "num_prompts" not in record or record["num_prompts"] is None:
            if "num_prompts" in filename_meta:
                record["num_prompts"] = filename_meta["num_prompts"]

        if "max_concurrency" not in record or record["max_concurrency"] is None:
            if "max_concurrency" in filename_meta:
                record["max_concurrency"] = filename_meta["max_concurrency"]

        if "test_name" not in record or not record.get("test_name"):
            if "test_name" in filename_meta:
                record["test_name"] = filename_meta["test_name"]

        if "dataset_name" not in record or not record.get("dataset_name"):
            if "dataset_name" in filename_meta:
                record["dataset_name"] = filename_meta["dataset_name"]

        record["source_file"] = os.path.basename(full_path)
        yield record


def _parse_diffusion_from_filename(filename: str) -> dict[str, Any]:
    name, ext = os.path.splitext(filename)
    if ext != ".json" or not name.startswith(_DIFFUSION_JSON_PREFIX):
        return {}
    core = name[len(_DIFFUSION_JSON_PREFIX) :]
    parts = core.split("_")
    if len(parts) < 2:
        return {}
    timestamp = parts[-1]
    test_name = "_".join(parts[:-1]) if parts[:-1] else ""
    parsed: dict[str, Any] = {}
    if len(timestamp) >= 15:
        parsed["date"] = timestamp
    if test_name:
        parsed["test_name"] = test_name
    return parsed


def _iter_diffusion_json_records(input_dir: str) -> Iterable[dict[str, Any]]:
    if not os.path.isdir(input_dir):
        LOGGER.warning("diffusion input dir '%s' does not exist or is not a directory", input_dir)
        return

    for entry in sorted(os.listdir(input_dir)):
        if not entry.endswith(".json"):
            continue
        if not entry.startswith(_DIFFUSION_JSON_PREFIX):
            continue
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            continue

        data = _load_json_file(full_path)
        if data is None:
            continue

        record: dict[str, Any] = dict(data)
        filename_meta = _parse_diffusion_from_filename(os.path.basename(full_path))
        if "date" not in record or not record.get("date"):
            record["date"] = filename_meta.get("date") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        if "test_name" not in record or not record.get("test_name"):
            if "test_name" in filename_meta:
                record["test_name"] = filename_meta["test_name"]
        record["source_file"] = os.path.basename(full_path)
        yield record


def _collect_omni_records(input_dir: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_omni_json_records(input_dir):
        records.append(record)
    return records


def _collect_diffusion_records(input_dir: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_diffusion_json_records(input_dir):
        records.append(record)
    return records


def _sort_omni_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date_desc = sorted(records, key=lambda r: (r.get("date") or ""), reverse=True)
    return sorted(
        by_date_desc,
        key=lambda r: (
            r.get("model_id") or "",
            r.get("test_name") or "",
            r.get("dataset_name") or "",
            r.get("max_concurrency") or 0,
            r.get("num_prompts") or 0,
        ),
    )


def _sort_diffusion_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_date_desc = sorted(records, key=lambda r: (r.get("date") or ""), reverse=True)
    return sorted(by_date_desc, key=lambda r: (r.get("test_name") or ""))


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if not parent:
        return
    os.makedirs(parent, exist_ok=True)


def _html_escape(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    )


def _build_html_document(
    omni_columns: Sequence[str],
    omni_records: Sequence[dict[str, Any]],
    diffusion_columns: Sequence[str],
    diffusion_records: Sequence[dict[str, Any]],
) -> str:
    # Styling is aligned with vllm-omni/tests/buildkite_testcase_statistics_preview.html.
    styles = """
:root {
  --bg: #0f1419;
  --card: #1a2332;
  --border: #2d3a4f;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  margin: 0;
  padding: 2rem;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 { font-size: 1.75rem; font-weight: 600; margin: 0 0 0.25rem 0; }
.meta { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.25rem; }
section {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.5rem;
}
section h2 {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 0 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
  color: var(--accent);
}
.filter-bar {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}
.filter-bar label { color: var(--muted); font-size: 0.9rem; }
input[type="text"] {
  min-width: 220px;
  padding: 0.55rem 0.75rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  color: var(--text);
  outline: none;
}
input[type="text"]:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.18);
}
.hint { color: var(--muted); font-size: 0.85rem; }
.grid {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr;
  gap: 1rem;
  align-items: start;
}
@media (max-width: 980px) {
  .grid { grid-template-columns: 1fr; }
}
.chart-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.75rem;
  background: rgba(0,0,0,0.12);
}
.chart-title { font-size: 0.95rem; color: var(--muted); margin: 0 0 0.5rem 0; }
canvas { width: 100%; height: 260px; display: block; }
.chart-wrap { position: relative; }
.chart-tooltip {
  position: absolute;
  pointer-events: none;
  display: none;
  background: rgba(15, 20, 25, 0.92);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.5rem 0.6rem;
  color: var(--text);
  font-size: 0.85rem;
  max-width: 320px;
  white-space: normal;
}
.chart-tooltip .muted { color: var(--muted); }
.chart-tooltip code { color: var(--text); }
details.more-charts > summary {
  cursor: pointer;
  color: var(--accent);
  font-weight: 600;
  margin: 0.25rem 0 0.75rem 0;
}
details.more-charts[open] > summary { margin-bottom: 0.75rem; }
.info {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.75rem 0.85rem;
  background: rgba(0,0,0,0.12);
  color: var(--muted);
  font-size: 0.9rem;
}
.info code { color: var(--text); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th {
  text-align: left;
  padding: 0.65rem 0.85rem;
  background: rgba(88, 166, 255, 0.12);
  color: var(--accent);
  font-weight: 600;
  border: 1px solid var(--border);
  white-space: nowrap;
}
td {
  padding: 0.6rem 0.85rem;
  border: 1px solid var(--border);
  vertical-align: top;
  white-space: nowrap;
}
tr.odd td { background: rgba(255,255,255,0.02); }
tr.even td { background: transparent; }
tr:hover td { background: rgba(88, 166, 255, 0.06); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
"""

    omni_data_json = json.dumps(list(omni_records), ensure_ascii=False)
    diffusion_data_json = json.dumps(list(diffusion_records), ensure_ascii=False)
    omni_cols_json = json.dumps(list(omni_columns), ensure_ascii=False)
    diffusion_cols_json = json.dumps(list(diffusion_columns), ensure_ascii=False)

    script = f"""
const OMNI_COLUMNS = {omni_cols_json};
const DIFF_COLUMNS = {diffusion_cols_json};
const OMNI_DATA = {omni_data_json};
const DIFF_DATA = {diffusion_data_json};

function uniqSorted(arr) {{
  const s = new Set(arr.filter(v => v !== null && v !== undefined && String(v).trim() !== ""));
  return Array.from(s).sort((a,b) => String(a).localeCompare(String(b)));
}}

function toNumber(v) {{
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  const s = String(v).trim();
  if (s === "" || s.toLowerCase() === "inf") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}}

function fmt(v) {{
  const n = toNumber(v);
  if (n === null) return "";
  return n.toFixed(4);
}}

function fillDatalist(datalistEl, values) {{
  datalistEl.innerHTML = "";
  values.forEach(v => {{
    const opt = document.createElement("option");
    opt.value = String(v);
    datalistEl.appendChild(opt);
  }});
}}

function renderTable(containerId, columns, rows, numericCols) {{
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const c of columns) {{
    const th = document.createElement("th");
    th.textContent = c;
    trh.appendChild(th);
  }}
  thead.appendChild(trh);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  rows.forEach((r, idx) => {{
    const tr = document.createElement("tr");
    tr.className = (idx % 2 === 0) ? "even" : "odd";
    for (const c of columns) {{
      const td = document.createElement("td");
      if (numericCols.has(c)) {{
        td.className = "num";
        td.textContent = fmt(r[c]);
      }} else {{
        td.textContent = (r[c] === null || r[c] === undefined) ? "" : String(r[c]);
      }}
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }});
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}}

function buildSeries(rows, metric, metaKeys) {{
  // rows should be sorted by date asc for plotting.
  const points = [];
  for (const r of rows) {{
    const x = String(r["date"] || "");
    const y = toNumber(r[metric]);
    if (!x || y === null) continue;
    const meta = {{}};
    for (const k of metaKeys) {{
      meta[k] = r[k];
    }}
    points.push({{x, y, meta}});
  }}
  return points;
}}

function drawMultiLineChart(canvas, tooltipEl, seriesList, labels) {{
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const padLeft = 18;
  const padRight = 10;
  const padTop = 10;
  const padBottom = 18;
  const innerW = w - padLeft - padRight;
  const innerH = h - padTop - padBottom;

  // Collect y range.
  let yMin = Infinity, yMax = -Infinity;
  let xLabels = [];
  for (const s of seriesList) {{
    for (const p of s) {{
      yMin = Math.min(yMin, p.y);
      yMax = Math.max(yMax, p.y);
      xLabels.push(p.x);
    }}
  }}
  xLabels = uniqSorted(xLabels);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || xLabels.length < 2) {{
    ctx.fillStyle = "rgba(139,148,158,0.9)";
    ctx.font = "13px -apple-system, BlinkMacSystemFont, \\"Segoe UI\\", sans-serif";
    ctx.fillText("No data to plot.", 12, 24);
    return;
  }}
  if (yMin === yMax) {{
    yMin -= 1;
    yMax += 1;
  }}

  function xScale(x) {{
    const idx = xLabels.indexOf(x);
    return padLeft + (idx / (xLabels.length - 1)) * innerW;
  }}
  function yScale(y) {{
    return padTop + (1 - (y - yMin) / (yMax - yMin)) * innerH;
  }}

  // Grid.
  ctx.strokeStyle = "rgba(45,58,79,0.9)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const yy = padTop + (i / 4) * innerH;
    ctx.beginPath();
    ctx.moveTo(padLeft, yy);
    ctx.lineTo(padLeft + innerW, yy);
    ctx.stroke();
  }}

  const colors = ["#58a6ff", "#3fb950", "#f0883e"];
  seriesList.forEach((series, idx) => {{
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((p, i) => {{
      const xx = xScale(p.x);
      const yy = yScale(p.y);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    }});
    ctx.stroke();
  }});

  // Legend.
  ctx.font = "12px -apple-system, BlinkMacSystemFont, \\"Segoe UI\\", sans-serif";
  labels.forEach((lab, idx) => {{
    ctx.fillStyle = colors[idx % colors.length];
    ctx.fillRect(padLeft + idx * 180, h - 18, 10, 10);
    ctx.fillStyle = "rgba(230,237,243,0.95)";
    ctx.fillText(lab, padLeft + idx * 180 + 14, h - 9);
  }});

  // Hover tooltip (nearest x index, then nearest series).
  function hideTooltip() {{
    tooltipEl.style.display = "none";
  }}
  function showTooltip(clientX, clientY, content) {{
    tooltipEl.innerHTML = content;
    tooltipEl.style.display = "block";
    const parentRect = canvas.parentElement.getBoundingClientRect();
    const x = clientX - parentRect.left + 10;
    const y = clientY - parentRect.top + 10;
    tooltipEl.style.left = `${{Math.min(x, parentRect.width - 320)}}px`;
    tooltipEl.style.top = `${{Math.min(y, parentRect.height - 120)}}px`;
  }}

  function onMove(ev) {{
    const cRect = canvas.getBoundingClientRect();
    const x = ev.clientX - cRect.left;
    const y = ev.clientY - cRect.top;
    if (x < padLeft || x > padLeft + innerW || y < padTop || y > padTop + innerH) {{
      hideTooltip();
      return;
    }}
    const rel = (x - padLeft) / innerW;
    const idx = Math.round(rel * (xLabels.length - 1));
    const xVal = xLabels[Math.max(0, Math.min(xLabels.length - 1, idx))];
    let best = null;
    seriesList.forEach((s, si) => {{
      const p = s.find(pp => pp.x === xVal);
      if (!p) return;
      const yy = yScale(p.y);
      const dist = Math.abs(yy - y);
      if (best === null || dist < best.dist) {{
        best = {{ dist, p, si }};
      }}
    }});
    if (!best) {{
      hideTooltip();
      return;
    }}
    const label = labels[best.si] || "";
    const meta = best.p.meta || {{}};
    const metaLines = Object.keys(meta)
      .filter(k => meta[k] !== null && meta[k] !== undefined && String(meta[k]).trim() !== "")
      .map(k => `<div class="muted">${{k}}: <code>${{String(meta[k])}}</code></div>`)
      .join("");
    showTooltip(ev.clientX, ev.clientY, `
      <div><strong>${{label}}</strong></div>
      <div class="muted">date: <code>${{String(best.p.x)}}</code></div>
      <div class="muted">value: <code>${{best.p.y.toFixed(4)}}</code></div>
      ${{metaLines}}
    `);
  }}

  canvas.onmousemove = onMove;
  canvas.onmouseleave = hideTooltip;
}}

function renderChartGroups(containerEl, rowsAsc, groups, metaKeys, configFields, labelFields, maxVisible) {{
  containerEl.innerHTML = "";
  const visible = [];
  const extra = [];
  groups.forEach((g, idx) => {{
    (idx < maxVisible ? visible : extra).push(g);
  }});

  function renderGroupList(groupList, parentEl) {{
    for (const g of groupList) {{
      const seriesByKey = new Map();
      for (const r of rowsAsc) {{
        const dateStr = String(r["date"] || "");
        if (!dateStr) continue;
        const cfgParts = configFields
          .map(f => r[f])
          .filter(v => v !== null && v !== undefined && String(v).trim() !== "")
          .map(v => String(v));
        const cfgKey = cfgParts.length ? cfgParts.join("||") : "config";
        for (const metric of g.metrics) {{
          const y = toNumber(r[metric]);
          if (y === null) continue;
          const key = `${{metric}}||${{cfgKey}}`;
          let entry = seriesByKey.get(key);
          if (!entry) {{
            const labelParts = labelFields
              .map(f => r[f])
              .filter(v => v !== null && v !== undefined && String(v).trim() !== "")
              .map(v => String(v));
            const shortLabel = labelParts.length ? labelParts.join(" | ") : cfgKey.replaceAll("||", " | ");
            entry = {{ label: `${{metric}} | ${{shortLabel}}`, points: [] }};
            seriesByKey.set(key, entry);
          }}
          const meta = {{}};
          metaKeys.forEach(k => {{ meta[k] = r[k]; }});
          entry.points.push({{ x: dateStr, y, meta }});
        }}
      }}
      const seriesEntries = Array.from(seriesByKey.values());
      const seriesList = seriesEntries.map(e => e.points).filter(s => s.length > 0);
      const labels = seriesEntries.filter(e => e.points.length > 0).map(e => e.label);
      if (seriesList.length === 0) {{
        continue;
      }}
      const totalPoints = seriesList.reduce((acc, s) => acc + s.length, 0);
      if (totalPoints < 2) {{
        // Snapshot card for 0/1 points: show latest value(s) instead of a line chart.
        const latest = rowsAsc[rowsAsc.length - 1];
        const card = document.createElement("div");
        card.className = "chart-card";
        const title = document.createElement("div");
        title.className = "chart-title";
        title.textContent = g.title + " (snapshot)";
        const body = document.createElement("div");
        body.className = "hint";
        const date = latest ? String(latest["date"] || "") : "";
        let html = "";
        if (date) {{
          html += `<div>date: <code>${{date}}</code></div>`;
        }}
        const metaSummary = metaKeys
          .filter(k => latest && latest[k] !== null && latest[k] !== undefined && String(latest[k]).trim() !== "")
          .map(k => `<div>${{k}}: <code>${{String(latest[k])}}</code></div>`)
          .join("");
        if (metaSummary) {{
          html += metaSummary;
        }}
        const metricLines = g.metrics
          .map(m => {{
            const val = latest ? latest[m] : null;
            const n = toNumber(val);
            if (n === null) return "";
            return `<div>${{m}}: <code>${{n.toFixed(4)}} </code></div>`;
          }})
          .filter(Boolean)
          .join("");
        if (metricLines) {{
          html += metricLines;
        }}
        body.innerHTML = html || "No numeric data for snapshot.";
        card.appendChild(title);
        card.appendChild(body);
        parentEl.appendChild(card);
        continue;
      }}
      const card = document.createElement("div");
      card.className = "chart-card";
      const title = document.createElement("div");
      title.className = "chart-title";
      title.textContent = g.title;
      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      const canvas = document.createElement("canvas");
      const tooltip = document.createElement("div");
      tooltip.className = "chart-tooltip";
      wrap.appendChild(canvas);
      wrap.appendChild(tooltip);
      card.appendChild(title);
      card.appendChild(wrap);
      parentEl.appendChild(card);
      canvas.__draw = () => {{
        const r = canvas.getBoundingClientRect();
        if (!r || r.width < 10 || r.height < 10) return false;
        drawMultiLineChart(canvas, tooltip, seriesList, labels);
        return true;
      }};
      // Draw only when layout has real size (prevents black charts in collapsed <details>).
      requestAnimationFrame(() => {{ canvas.__draw(); }});
    }}
  }}

  renderGroupList(visible, containerEl);

  if (extra.length) {{
    const details = document.createElement("details");
    details.className = "more-charts";
    const summary = document.createElement("summary");
    summary.textContent = `More charts (${{extra.length}})`;
    details.appendChild(summary);
    const inner = document.createElement("div");
    details.appendChild(inner);
    renderGroupList(extra, inner);
    containerEl.appendChild(details);
    details.addEventListener("toggle", () => {{
      if (!details.open) return;
      requestAnimationFrame(() => {{
        details.querySelectorAll("canvas").forEach((c) => {{
          if (typeof c.__draw === "function") c.__draw();
        }});
      }});
    }});
  }}

  if (!containerEl.querySelector(".chart-card")) {{
    const card = document.createElement("div");
    card.className = "chart-card";
    const title = document.createElement("div");
    title.className = "chart-title";
    title.textContent = "Trend";
    const msg = document.createElement("div");
    msg.className = "hint";
    msg.textContent = "No data to plot for current filters.";
    card.appendChild(title);
    card.appendChild(msg);
    containerEl.appendChild(card);
  }}
}}

function filterRows(rows, filters) {{
  return rows.filter(r => {{
    if (filters.model && String(r[filters.modelKey] || "") !== filters.model) return false;
    if (filters.testName && String(r["test_name"] || "") !== filters.testName) return false;
    if (filters.datasetName && String(r[filters.datasetKey] || "") !== filters.datasetName) return false;
    return true;
  }});
}}

function sortByDateAsc(rows) {{
  return [...rows].sort((a,b) => String(a["date"] || "").localeCompare(String(b["date"] || "")));
}}

function initSection(prefix, columns, data, numericCols, groups) {{
  const modelInput = document.getElementById(prefix + "-model");
  const modelList = document.getElementById(prefix + "-model-list");
  const testInput = document.getElementById(prefix + "-test");
  const testList = document.getElementById(prefix + "-test-list");
  const datasetInput = document.getElementById(prefix + "-dataset");
  const datasetList = document.getElementById(prefix + "-dataset-list");
  const infoEl = document.getElementById(prefix + "-info");
  const chartsEl = document.getElementById(prefix + "-charts");

  const modelKey = (prefix === "diff") ? "model" : "model_id";
  const datasetKey = (prefix === "diff") ? "dataset" : "dataset_name";
  const metaKeys = (prefix === "diff") ? ["test_name"] : ["test_name","max_concurrency","num_prompts"];
  const configFields = (prefix === "diff")
    ? ["test_name","model","backend","dataset"]
    : ["endpoint_type","backend","model_id","tokenizer_id","test_name","dataset_name","max_concurrency","num_prompts"];
  const labelFields = (prefix === "diff")
    ? ["test_name","dataset"]
    : ["test_name","dataset_name"];

  function refreshOptions() {{
    const models = uniqSorted(data.map(r => r[modelKey]));
    const tests = uniqSorted(data.map(r => r["test_name"]));
    const datasets = uniqSorted(data.map(r => r[datasetKey]));
    fillDatalist(modelList, models);
    fillDatalist(testList, tests);
    fillDatalist(datasetList, datasets);
  }}

  function render() {{
    const filters = {{
      model: modelInput.value.trim(),
      testName: testInput.value.trim(),
      datasetName: datasetInput.value.trim(),
      modelKey,
      datasetKey,
    }};
    const filtered = filterRows(data, filters);
    const filteredAsc = sortByDateAsc(filtered);

    infoEl.innerHTML = `
      <div>Records: <code>${{filtered.length}}</code></div>
      <div>Model: <code>${{filters.model || "All"}}</code></div>
      <div>Test: <code>${{filters.testName || "All"}}</code></div>
      <div>Dataset: <code>${{filters.datasetName || "All"}}</code></div>
    `;

    if (prefix === "omni" && !filters.model) {{
      chartsEl.innerHTML = "<div class='hint'>请选择 model_id 后查看趋势曲线。</div>";
      renderTable(prefix + "-table", columns, filteredAsc.slice().reverse(), numericCols);
      return;
    }}

    renderChartGroups(chartsEl, filteredAsc, groups, metaKeys, configFields, labelFields, 3);
    renderTable(prefix + "-table", columns, filteredAsc.slice().reverse(), numericCols);
  }}

  refreshOptions();
  modelInput.addEventListener("input", render);
  testInput.addEventListener("input", render);
  datasetInput.addEventListener("input", render);
  render();
}}

window.addEventListener("load", () => {{
  const omniNumeric = new Set([
    "num_prompts","burstiness","max_concurrency","duration","completed","failed",
    "request_throughput","output_throughput","total_token_throughput",
    "mean_ttft_ms","p99_ttft_ms","mean_tpot_ms","p99_tpot_ms","mean_itl_ms","p99_itl_ms",
    "mean_e2el_ms","p99_e2el_ms","mean_audio_rtf","p99_audio_rtf","mean_audio_duration_s","p99_audio_duration_s",
  ]);
  const diffNumeric = new Set([
    "duration","completed_requests","failed_requests","throughput_qps",
    "latency_mean","latency_median","latency_p50","latency_p99",
    "peak_memory_mb_max","peak_memory_mb_mean","peak_memory_mb_median","slo_attainment_rate",
  ]);

  const omniGroups = [
    {{ title: "throughput", metrics: ["output_throughput","total_token_throughput"] }},
    {{ title: "ttft", metrics: ["mean_ttft_ms","median_ttft_ms","p99_ttft_ms"] }},
    {{
      title: "tpot + itl",
      metrics: [
        "mean_tpot_ms","median_tpot_ms","p99_tpot_ms",
        "mean_itl_ms","median_itl_ms","p99_itl_ms",
      ],
    }},
    {{ title: "e2el", metrics: ["mean_e2el_ms","median_e2el_ms","p99_e2el_ms"] }},
    {{ title: "audio rtf", metrics: ["mean_audio_rtf","median_audio_rtf","p99_audio_rtf"] }},
    {{ title: "audio ttfp", metrics: ["mean_audio_ttfp_ms","median_audio_ttfp_ms","p99_audio_ttfp_ms"] }},
    {{ title: "audio duration", metrics: ["mean_audio_duration_s","median_audio_duration_s","p99_audio_duration_s"] }},
  ];
  const diffGroups = [
    {{ title: "throughput", metrics: ["throughput_qps"] }},
    {{
      title: "latency",
      metrics: ["latency_mean", "latency_median", "latency_p99", "latency_p50"],
    }},
  ];

  initSection("omni", OMNI_COLUMNS, OMNI_DATA, omniNumeric, omniGroups);
  initSection("diff", DIFF_COLUMNS, DIFF_DATA, diffNumeric, diffGroups);
}});
"""

    html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8" />',
        '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
        "  <title>Nightly Performance Report</title>",
        f"  <style>{styles}</style>",
        "</head>",
        "<body>",
        '  <div class="container">',
        "    <h1>Nightly Performance Report</h1>",
        (
            '    <div class="meta">Interactive view: filter by model/test/dataset, '
            "numeric formatting (4 decimals), and trend charts.</div>"
        ),
        '    <section id="omni-section">',
        "      <h2>Omni</h2>",
        '      <div class="filter-bar">',
        (
            '        <label>model_id</label><input type="text" id="omni-model" '
            'list="omni-model-list" placeholder="All" /><datalist '
            'id="omni-model-list"></datalist>'
        ),
        (
            '        <label>test_name</label><input type="text" id="omni-test" '
            'list="omni-test-list" placeholder="All" /><datalist '
            'id="omni-test-list"></datalist>'
        ),
        (
            '        <label>dataset_name</label><input type="text" id="omni-dataset" '
            'list="omni-dataset-list" placeholder="All" /><datalist '
            'id="omni-dataset-list"></datalist>'
        ),
        '        <span class="hint">按指标分组多图展示，缺失指标自动跳过</span>',
        "      </div>",
        '      <div class="grid">',
        '        <div id="omni-charts"></div>',
        '        <div class="info" id="omni-info"></div>',
        "      </div>",
        '      <div id="omni-table"></div>',
        "    </section>",
        '    <section id="diff-section">',
        "      <h2>Diffusion</h2>",
        '      <div class="filter-bar">',
        (
            '        <label>model</label><input type="text" id="diff-model" '
            'list="diff-model-list" placeholder="All" /><datalist '
            'id="diff-model-list"></datalist>'
        ),
        (
            '        <label>test_name</label><input type="text" id="diff-test" '
            'list="diff-test-list" placeholder="All" /><datalist '
            'id="diff-test-list"></datalist>'
        ),
        (
            '        <label>dataset</label><input type="text" id="diff-dataset" '
            'list="diff-dataset-list" placeholder="All" /><datalist '
            'id="diff-dataset-list"></datalist>'
        ),
        '        <span class="hint">throughput / latency 分图展示，缺失指标自动跳过</span>',
        "      </div>",
        '      <div class="grid">',
        '        <div id="diff-charts"></div>',
        '        <div class="info" id="diff-info"></div>',
        "      </div>",
        '      <div id="diff-table"></div>',
        "    </section>",
        "  </div>",
        f"  <script>{script}</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(html)


def generate_html_report(input_dir: str, diffusion_input_dir: str, output_file: str) -> None:
    omni_records = _collect_omni_records(input_dir)
    diffusion_records = _collect_diffusion_records(diffusion_input_dir)
    if not omni_records:
        LOGGER.warning("no valid omni json records found under '%s'", input_dir)
    if not diffusion_records:
        LOGGER.warning("no valid diffusion json records found under '%s'", diffusion_input_dir)

    omni_sorted = _sort_omni_records(omni_records)
    diffusion_sorted = _sort_diffusion_records(diffusion_records)

    omni_columns: list[str] = [
        "date",
        "endpoint_type",
        "backend",
        "model_id",
        "tokenizer_id",
        "test_name",
        "dataset_name",
        "max_concurrency",
        "num_prompts",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_e2el_ms",
        "p99_e2el_ms",
        "mean_audio_rtf",
        "p99_audio_rtf",
        "duration",
        "completed",
        "failed",
        "source_file",
    ]
    diffusion_columns: list[str] = [
        "date",
        "test_name",
        "model",
        "backend",
        "dataset",
        "task",
        "duration",
        "throughput_qps",
        "latency_mean",
        "latency_median",
        "latency_p50",
        "latency_p99",
        "completed_requests",
        "failed_requests",
        "peak_memory_mb_max",
        "peak_memory_mb_mean",
        "peak_memory_mb_median",
        "slo_attainment_rate",
        "source_file",
    ]

    html_content = _build_html_document(
        omni_columns=omni_columns,
        omni_records=omni_sorted,
        diffusion_columns=diffusion_columns,
        diffusion_records=diffusion_sorted,
    )
    _ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    LOGGER.info("html report saved to '%s'", output_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    diffusion_input_dir = args.diffusion_input_dir or _default_diffusion_input_dir(args.input_dir)
    generate_html_report(
        input_dir=args.input_dir, diffusion_input_dir=diffusion_input_dir, output_file=args.output_file
    )


if __name__ == "__main__":
    main()
