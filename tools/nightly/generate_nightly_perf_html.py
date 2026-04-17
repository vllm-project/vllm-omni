#!/usr/bin/env python3
"""
Generate an improved nightly HTML performance dashboard without modifying the
existing generator.
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
DEFAULT_INPUT_DIR = os.getenv("DEFAULT_INPUT_DIR") or "tests"
DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR") or "tests"
DEFAULT_DIFFUSION_INPUT_DIR = os.getenv("DIFFUSION_BENCHMARK_DIR")


def _vllm_omni_root() -> str:
    path = os.path.dirname(os.path.abspath(__file__))
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "tests")):
            return path
        path = os.path.dirname(path)
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
    )


def _default_input_dir() -> str:
    return os.path.join(_vllm_omni_root(), DEFAULT_INPUT_DIR)


def _default_output_file() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return os.path.join(
        _vllm_omni_root(),
        DEFAULT_OUTPUT_DIR,
        f"nightly_perf_{ts}.html",
    )


def _default_diffusion_input_dir(input_dir: str) -> str:
    return DEFAULT_DIFFUSION_INPUT_DIR if DEFAULT_DIFFUSION_INPUT_DIR else input_dir


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
    """Parse ``result_test_*.json`` filenames; same rules as ``generate_nightly_perf_excel``."""
    name, ext = os.path.splitext(filename)
    if ext != ".json" or not name.startswith(_RESULT_JSON_PREFIX):
        return {}

    core = name[len(_RESULT_JSON_PREFIX) :]
    parts = core.split("_")
    if len(parts) < 5:
        LOGGER.warning(
            "filename '%s' does not match expected pattern (need >= 5 segments), skip parsing",
            filename,
        )
        return {}

    idx = len(parts) - 1
    timestamp = parts[idx]
    idx -= 1

    parsed: dict[str, Any] = {}
    if len(timestamp) >= 15:
        parsed["date"] = timestamp

    if idx >= 0 and parts[idx].startswith("out"):
        parsed["random_output_len"] = parts[idx][3:]
        idx -= 1
    if idx >= 0 and parts[idx].startswith("in"):
        parsed["random_input_len"] = parts[idx][2:]
        idx -= 1

    if idx < 3:
        LOGGER.warning(
            "filename '%s' has too few segments after timestamp / optional in-out (idx=%s)",
            filename,
            idx,
        )
        return parsed

    num_prompts_str = parts[idx]
    idx -= 1
    flow_str = parts[idx]
    idx -= 1
    dataset_name = parts[idx]
    idx -= 1
    test_name = "_".join(parts[: idx + 1]) if idx >= 0 else ""

    try:
        parsed["num_prompts"] = int(num_prompts_str)
    except (TypeError, ValueError):
        pass

    try:
        parsed["max_concurrency"] = int(flow_str)
    except (TypeError, ValueError):
        pass

    if test_name:
        parsed["test_name"] = test_name

    if dataset_name in ("random", "random-mm"):
        parsed["dataset_name"] = dataset_name

    return parsed


def _iter_omni_json_records(input_dir: str) -> Iterable[dict[str, Any]]:
    if not os.path.isdir(input_dir):
        LOGGER.warning("input dir '%s' does not exist or is not a directory", input_dir)
        return

    for entry in sorted(os.listdir(input_dir)):
        if not entry.endswith(".json") or not entry.startswith(_RESULT_JSON_PREFIX):
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
            record["date"] = filename_meta.get("date") or datetime.now(
                timezone.utc,
            ).strftime("%Y%m%d-%H%M%S")
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
        LOGGER.warning(
            "diffusion input dir '%s' does not exist or is not a directory",
            input_dir,
        )
        return

    for entry in sorted(os.listdir(input_dir)):
        if not entry.endswith(".json") or not entry.startswith(_DIFFUSION_JSON_PREFIX):
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
            record["date"] = filename_meta.get("date") or datetime.now(
                timezone.utc,
            ).strftime("%Y%m%d-%H%M%S")
        if "test_name" not in record or not record.get("test_name"):
            if "test_name" in filename_meta:
                record["test_name"] = filename_meta["test_name"]
        record["source_file"] = os.path.basename(full_path)
        yield record


def _collect_omni_records(input_dir: str) -> list[dict[str, Any]]:
    return list(_iter_omni_json_records(input_dir))


def _collect_diffusion_records(input_dir: str) -> list[dict[str, Any]]:
    return list(_iter_diffusion_json_records(input_dir))


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
    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an improved nightly HTML dashboard from performance JSON files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=_default_input_dir(),
        help="Directory containing result_test_*.json files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=_default_output_file().replace(".html", "_v2.html"),
        help="Output path of the HTML report.",
    )
    parser.add_argument(
        "--diffusion-input-dir",
        type=str,
        default=None,
        help="Directory containing diffusion_perf_*.json files; default falls back to --input-dir.",
    )
    return parser.parse_args()


def _build_html_document(
    omni_columns: Sequence[str],
    omni_records: Sequence[dict[str, object]],
    diffusion_columns: Sequence[str],
    diffusion_records: Sequence[dict[str, object]],
) -> str:
    styles = """
:root {
  --bg: #0b1018;
  --surface: rgba(26, 35, 50, 0.9);
  --surface-2: rgba(17, 24, 39, 0.96);
  --border: #2d3a4f;
  --text: #e6edf3;
  --muted: #8b949e;
  --accent: #58a6ff;
  --accent-2: #3fb950;
  --accent-3: #f0883e;
  --shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
}}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(88, 166, 255, 0.18), transparent 26%),
    radial-gradient(circle at top right, rgba(63, 185, 80, 0.12), transparent 22%),
    linear-gradient(180deg, #111827, #0b1018 64%);
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.12;
  background:
    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 24px 24px;
}
.container { max-width: 1500px; margin: 0 auto; padding: 24px 20px 64px; }
.hero, section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(8px);
}
.hero {
  padding: 28px 30px;
  margin-bottom: 22px;
  background: linear-gradient(145deg, rgba(17, 24, 39, 0.96), rgba(26, 35, 50, 0.88));
}
h1 { margin: 0 0 8px 0; font-size: 2rem; line-height: 1.08; }
.meta { color: var(--muted); font-size: 0.96rem; max-width: 820px; }
.hero-badges, .section-pills {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 16px;
}
.badge, .pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 0.4rem 0.74rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  font-size: 0.84rem;
}
.badge strong, .pill strong { color: var(--text); }
section { padding: 18px; margin-bottom: 22px; }
.section-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
  align-items: flex-start;
  margin-bottom: 14px;
}
.section-title { margin: 0; font-size: 1.24rem; }
.section-subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 4px; }
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}
.stat-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
}
.stat-label {
  color: var(--muted);
  font-size: 0.78rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.stat-value { font-size: 1.3rem; font-weight: 700; line-height: 1.12; }
.stat-sub { color: var(--muted); font-size: 0.82rem; margin-top: 6px; }
.toolbar {
  display: grid;
  gap: 12px;
  margin-bottom: 12px;
}
.toolbar-row {
  display: grid;
  gap: 12px;
  min-width: 0;
}
.toolbar-row.primary {
  grid-template-columns: minmax(0, 2fr) repeat(3, minmax(0, 1fr));
}
.toolbar-row.secondary {
  grid-template-columns: repeat(5, minmax(0, 1fr));
}
.filter-field { display: flex; flex-direction: column; gap: 6px; }
.filter-field label { color: var(--muted); font-size: 0.82rem; padding-left: 2px; }
.filter-field.compact label { font-size: 0.78rem; }
.filter-field { min-width: 0; }
.filter-input, .ghost {
  width: 100%;
  min-width: 0;
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 0.72rem 0.88rem;
  background: rgba(255,255,255,0.035);
  color: var(--text);
  outline: none;
}
.filter-input {
  overflow: hidden;
  text-overflow: ellipsis;
}
.filter-select {
  width: 100%;
  min-width: 0;
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 0.72rem 2.1rem 0.72rem 0.88rem;
  background: rgba(255,255,255,0.035);
  color: var(--text);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  background-image:
    linear-gradient(45deg, transparent 50%, var(--muted) 50%),
    linear-gradient(135deg, var(--muted) 50%, transparent 50%);
  background-position:
    calc(100% - 18px) calc(50% - 2px),
    calc(100% - 12px) calc(50% - 2px);
  background-size: 6px 6px, 6px 6px;
  background-repeat: no-repeat;
}
.filter-select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.16);
}
.filter-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.16);
}
.ghost {
  cursor: pointer;
  background: rgba(88, 166, 255, 0.08);
}
.ghost:hover { background: rgba(88, 166, 255, 0.16); }
.hint { color: var(--muted); font-size: 0.84rem; }
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.9fr);
  gap: 16px;
  align-items: start;
  margin: 14px 0 16px;
}
.charts-stack { display: grid; gap: 14px; }
.chart-card, .panel, .table-shell {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 18px;
}
.chart-card { padding: 14px; }
.chart-card.empty {
  min-height: 180px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
}
.chart-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}
.chart-title { font-size: 0.96rem; font-weight: 600; }
.chart-caption { color: var(--muted); font-size: 0.8rem; }
.chart-wrap { position: relative; }
canvas { width: 100%; height: 320px; display: block; }
.chart-tooltip {
  position: absolute;
  pointer-events: none;
  display: none;
  max-width: 340px;
  padding: 0.55rem 0.7rem;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(11, 16, 24, 0.97);
  box-shadow: 0 12px 30px rgba(0,0,0,0.4);
  font-size: 0.84rem;
}
.panel {
  position: sticky;
  top: 16px;
  padding: 14px;
}
.panel h3 { margin: 0 0 12px 0; font-size: 1rem; }
.kv { display: grid; gap: 10px; }
.kv-row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  padding-bottom: 8px;
}
.kv-row:last-child { border-bottom: 0; padding-bottom: 0; }
.kv-label { color: var(--muted); font-size: 0.84rem; }
.kv-value { text-align: right; font-size: 0.88rem; word-break: break-word; }
.snapshot-list { display: grid; gap: 10px; margin-top: 14px; }
.snapshot-item {
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px;
  background: rgba(255,255,255,0.02);
}
.snapshot-item strong { display: block; margin-bottom: 4px; font-size: 0.9rem; }
details.more-charts {
  border: 1px dashed var(--border);
  border-radius: 16px;
  padding: 0 14px 14px;
  background: rgba(255,255,255,0.015);
}
details.more-charts > summary {
  cursor: pointer;
  color: var(--accent);
  font-weight: 600;
  padding: 14px 0;
}
.table-shell { overflow: hidden; }
.table-headline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
}
.table-title { font-size: 0.95rem; font-weight: 600; }
.table-note { color: var(--muted); font-size: 0.82rem; }
.table-wrap { overflow: auto; max-height: 720px; }
table { width: 100%; border-collapse: collapse; font-size: 0.89rem; }
th {
  position: sticky;
  top: 0;
  z-index: 2;
  text-align: left;
  padding: 0.72rem 0.85rem;
  background: rgba(17, 24, 39, 0.98);
  color: var(--accent);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  cursor: pointer;
}
th.sort-active { background: rgba(88, 166, 255, 0.16); color: var(--text); }
td {
  padding: 0.68rem 0.85rem;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  white-space: nowrap;
  vertical-align: top;
}
td.truncate-cell {
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
}
tr.odd td { background: rgba(255,255,255,0.022); }
tr.even td { background: transparent; }
tr:hover td { background: rgba(88, 166, 255, 0.07); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code {
  color: var(--text);
  background: rgba(255,255,255,0.06);
  padding: 0.14rem 0.36rem;
  border-radius: 6px;
}
.back-to-top {
  position: fixed;
  right: 22px;
  bottom: 22px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(17, 24, 39, 0.86);
  color: var(--text);
  text-decoration: none;
  padding: 0.75rem 0.95rem;
}
@media (max-width: 1280px) {
  .stats-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .toolbar-row.primary,
  .toolbar-row.secondary {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 980px) {
  .layout { grid-template-columns: 1fr; }
  .panel { position: static; }
}
@media (max-width: 760px) {
  .container { padding: 16px 12px 56px; }
  .hero { padding: 22px 18px; }
  .toolbar-row.primary,
  .toolbar-row.secondary,
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
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

function escapeHtml(value) {{
  if (value === null || value === undefined) return "";
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}}

function uniqSorted(arr) {{
  const items = new Set(arr.filter(v => v !== null && v !== undefined && String(v).trim() !== ""));
  return Array.from(items).sort((a, b) => String(a).localeCompare(String(b)));
}}

function toNumber(v) {{
  if (v === null || v === undefined) return null;
  if (typeof v === "number") return Number.isFinite(v) ? v : null;
  const s = String(v).trim();
  if (!s || s.toLowerCase() === "inf") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}}

function fmt(v) {{
  const n = toNumber(v);
  return n === null ? "" : n.toFixed(4);
}}

function formatDate(v) {{
  const s = String(v || "");
  const m = s.match(/^(\\d{{4}})(\\d{{2}})(\\d{{2}})-(\\d{{2}})(\\d{{2}})(\\d{{2}})$/);
  return m ? `${{m[1]}}-${{m[2]}}-${{m[3]}} ${{m[4]}}:${{m[5]}}:${{m[6]}}` : s;
}}

function shortenModelName(value) {{
  const s = String(value || "");
  if (!s) return "";
  const tail = s.includes("/") ? s.split("/").pop() : s;
  return tail
    .replace("-Instruct", "")
    .replace("-Chat", "")
    .replace("-Preview", "");
}}

function shortenDatasetName(value) {{
  const s = String(value || "");
  if (s === "random") return "rnd";
  if (s === "random-mm") return "rnd-mm";
  return s;
}}

function compactSeriesLabel(prefix, row, metric, fallback) {{
  if (prefix === "omni") {{
    const parts = [
      row.test_name || "",
      shortenDatasetName(row.dataset_name),
      row.max_concurrency !== undefined && row.max_concurrency !== null ? `c${{row.max_concurrency}}` : "",
      row.num_prompts !== undefined && row.num_prompts !== null ? `p${{row.num_prompts}}` : "",
    ].filter(Boolean);
    return `${{metric}} | ${{parts.join(" | ") || fallback}}`;
  }}
  const parts = [
    row.test_name || "",
    shortenModelName(row.model),
    shortenDatasetName(row.dataset),
  ].filter(Boolean);
  return `${{metric}} | ${{parts.join(" | ") || fallback}}`;
}}

function compactLegendLabel(prefix, row, metric, fallback) {{
  if (prefix === "omni") {{
    const parts = [
      row.max_concurrency !== undefined && row.max_concurrency !== null ? `c${{row.max_concurrency}}` : "",
      row.num_prompts !== undefined && row.num_prompts !== null ? `p${{row.num_prompts}}` : "",
    ].filter(Boolean);
    return `${{metric}} | ${{parts.join(" | ") || fallback}}`;
  }}
  const parts = [
    (row.test_name || "").replace(/^test_/, ""),
    shortenModelName(row.model),
  ].filter(Boolean);
  return parts.join(" | ") || fallback;
}}

function truncateCellText(column, value) {{
  const raw = value === null || value === undefined ? "" : String(value);
  if (!raw) return {{ text: "", title: "" }};
  if (column === "model_id" || column === "tokenizer_id" || column === "model") {{
    const short = shortenModelName(raw);
    return {{
      text: short.length > 28 ? `${{short.slice(0, 28)}}...` : short,
      title: raw,
    }};
  }}
  if (column === "source_file") {{
    return {{
      text: raw.length > 38 ? `${{raw.slice(0, 38)}}...` : raw,
      title: raw,
    }};
  }}
  if (column === "test_name") {{
    return {{
      text: raw.length > 32 ? `${{raw.slice(0, 32)}}...` : raw,
      title: raw,
    }};
  }}
  if (raw.length > 42) {{
    return {{
      text: `${{raw.slice(0, 42)}}...`,
      title: raw,
    }};
  }}
  return {{ text: raw, title: raw }};
}}

function fillDatalist(el, values) {{
  el.innerHTML = "";
  values.forEach((v) => {{
    const opt = document.createElement("option");
    opt.value = String(v);
    el.appendChild(opt);
  }});
}}

function fillSelect(el, values, selectedValue) {{
  if (!el) return;
  const previous = selectedValue ?? el.value;
  el.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = "All";
  el.appendChild(allOption);
  values.forEach((v) => {{
    const opt = document.createElement("option");
    opt.value = String(v);
    opt.textContent = String(v);
    el.appendChild(opt);
  }});
  if (previous && values.map((v) => String(v)).includes(String(previous))) {{
    el.value = String(previous);
  }} else {{
    el.value = "";
  }}
}}

function sortRows(rows, column, desc, numericCols) {{
  const next = [...rows];
  next.sort((a, b) => {{
    let cmp = 0;
    if (numericCols.has(column)) {{
      cmp = (toNumber(a[column]) ?? -Infinity) - (toNumber(b[column]) ?? -Infinity);
    }} else {{
      cmp = String(a[column] || "").localeCompare(String(b[column] || ""));
    }}
    if (cmp === 0) {{
      cmp = String(a.date || "").localeCompare(String(b.date || ""));
    }}
    return desc ? -cmp : cmp;
  }});
  return next;
}}

function renderStats(container, cards) {{
  container.innerHTML = "";
  cards.forEach((card) => {{
    const el = document.createElement("div");
    el.className = "stat-card";
    el.innerHTML = `
      <div class="stat-label">${{escapeHtml(card.label)}}</div>
      <div class="stat-value">${{escapeHtml(card.value)}}</div>
      <div class="stat-sub">${{escapeHtml(card.sub || "")}}</div>
    `;
    container.appendChild(el);
  }});
}}

function renderTable(containerId, columns, rows, numericCols, sortState) {{
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const shell = document.createElement("div");
  shell.className = "table-shell";
  const headline = document.createElement("div");
  headline.className = "table-headline";
  headline.innerHTML = `
    <div class="table-title">Detailed records</div>
    <div class="table-note">${{rows.length}} row(s) shown. Click a header to sort.</div>
  `;
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const col of columns) {{
    const th = document.createElement("th");
    th.textContent = col + (sortState.column === col ? (sortState.desc ? " ↓" : " ↑") : "");
    if (sortState.column === col) th.className = "sort-active";
    th.addEventListener("click", () => {{
      if (sortState.column === col) {{
        sortState.desc = !sortState.desc;
      }} else {{
        sortState.column = col;
        sortState.desc = false;
      }}
      sortState.onChange();
    }});
    trh.appendChild(th);
  }}
  thead.appendChild(trh);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  rows.forEach((row, idx) => {{
    const tr = document.createElement("tr");
    tr.className = idx % 2 === 0 ? "even" : "odd";
    columns.forEach((col) => {{
      const td = document.createElement("td");
      if (numericCols.has(col)) {{
        td.className = "num";
        td.textContent = fmt(row[col]);
      }} else {{
        const display = col === "date"
          ? {{ text: formatDate(row[col]), title: formatDate(row[col]) }}
          : truncateCellText(col, row[col]);
        td.textContent = display.text;
        if (display.title && display.title !== display.text) {{
          td.title = display.title;
          td.classList.add("truncate-cell");
        }}
      }}
      tr.appendChild(td);
    }});
    tbody.appendChild(tr);
  }});
  table.appendChild(tbody);
  wrap.appendChild(table);
  shell.appendChild(headline);
  shell.appendChild(wrap);
  container.appendChild(shell);
}}

function drawChart(canvas, tooltip, seriesList, labels, legendLabels) {{
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width;
  const h = rect.height;
  const padLeft = 18;
  const padRight = 10;
  const padTop = 14;
  const padBottom = 64;
  const innerW = w - padLeft - padRight;
  const innerH = h - padTop - padBottom;
  ctx.clearRect(0, 0, w, h);

  let yMin = Infinity;
  let yMax = -Infinity;
  let xLabels = [];
  seriesList.forEach((series) => {{
    series.forEach((p) => {{
      yMin = Math.min(yMin, p.y);
      yMax = Math.max(yMax, p.y);
      xLabels.push(p.x);
    }});
  }});
  xLabels = uniqSorted(xLabels);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || xLabels.length < 2) {{
    ctx.fillStyle = "rgba(139,148,158,0.95)";
    ctx.font = "13px -apple-system, BlinkMacSystemFont, \\"Segoe UI\\", sans-serif";
    ctx.fillText("No data to plot.", 12, 24);
    return;
  }}
  if (yMin === yMax) {{
    yMin -= 1;
    yMax += 1;
  }}

  function xScale(x) {{
    return padLeft + (xLabels.indexOf(x) / (xLabels.length - 1)) * innerW;
  }}
  function yScale(y) {{
    return padTop + (1 - (y - yMin) / (yMax - yMin)) * innerH;
  }}

  ctx.strokeStyle = "rgba(45,58,79,0.9)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const yy = padTop + (i / 4) * innerH;
    ctx.beginPath();
    ctx.moveTo(padLeft, yy);
    ctx.lineTo(padLeft + innerW, yy);
    ctx.stroke();
  }}

  ctx.font = "11px -apple-system, BlinkMacSystemFont, \\"Segoe UI\\", sans-serif";
  ctx.fillStyle = "rgba(139,148,158,0.95)";
  for (let i = 0; i <= 4; i++) {{
    const yy = padTop + (i / 4) * innerH;
    const val = (yMax - ((i / 4) * (yMax - yMin))).toFixed(2);
    ctx.fillText(val, padLeft + 4, yy - 4);
  }}
  const tickLabels = xLabels.length <= 3
    ? xLabels
    : [xLabels[0], xLabels[Math.floor((xLabels.length - 1) / 2)], xLabels[xLabels.length - 1]];
  tickLabels.forEach((tick) => {{
    const xx = xScale(tick);
    ctx.fillText(formatDate(tick).slice(5, 16), Math.max(0, xx - 28), h - 26);
  }});

  const colors = ["#58a6ff", "#3fb950", "#f0883e", "#d2a8ff", "#ff7b72", "#79c0ff"];
  seriesList.forEach((series, idx) => {{
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    series.forEach((p, pointIdx) => {{
      const xx = xScale(p.x);
      const yy = yScale(p.y);
      if (pointIdx === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    }});
    ctx.stroke();
    series.forEach((p) => {{
      const xx = xScale(p.x);
      const yy = yScale(p.y);
      ctx.fillStyle = colors[idx % colors.length];
      ctx.beginPath();
      ctx.arc(xx, yy, 3, 0, Math.PI * 2);
      ctx.fill();
    }});
  }});

  legendLabels.slice(0, 4).forEach((label, idx) => {{
    ctx.fillStyle = colors[idx % colors.length];
    ctx.fillRect(padLeft + idx * 180, h - 14, 10, 10);
    ctx.fillStyle = "rgba(230,237,243,0.95)";
    const shortLabel = label.length > 22 ? `${{label.slice(0, 22)}}...` : label;
    ctx.fillText(shortLabel, padLeft + idx * 180 + 14, h - 5);
  }});

  function hideTooltip() {{
    tooltip.style.display = "none";
  }}

  function showTooltip(clientX, clientY, html) {{
    tooltip.innerHTML = html;
    tooltip.style.display = "block";
    const parentRect = canvas.parentElement.getBoundingClientRect();
    tooltip.style.left = `${{Math.min(clientX - parentRect.left + 10, parentRect.width - 320)}}px`;
    tooltip.style.top = `${{Math.min(clientY - parentRect.top + 10, parentRect.height - 120)}}px`;
  }}

  canvas.onmousemove = (ev) => {{
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
    seriesList.forEach((series, seriesIdx) => {{
      const point = series.find((p) => p.x === xVal);
      if (!point) return;
      const yy = yScale(point.y);
      const dist = Math.abs(yy - y);
      if (!best || dist < best.dist) {{
        best = {{ dist, point, seriesIdx }};
      }}
    }});
    if (!best) {{
      hideTooltip();
      return;
    }}
    const metaLines = Object.entries(best.point.meta || {{}})
      .filter(([, value]) => value !== null && value !== undefined && String(value).trim() !== "")
      .map(
        ([key, value]) =>
          `<div class="muted">${{escapeHtml(key)}}: ` +
          `<code>${{escapeHtml(String(value))}}</code></div>`,
      )
      .join("");
    showTooltip(ev.clientX, ev.clientY, `
      <div><strong>${{escapeHtml(labels[best.seriesIdx] || "")}}</strong></div>
      <div class="muted">date: <code>${{escapeHtml(formatDate(best.point.x))}}</code></div>
      <div class="muted">value: <code>${{best.point.y.toFixed(4)}}</code></div>
      ${{metaLines}}
    `);
  }};
  canvas.onmouseleave = hideTooltip;
}}

function renderChartGroups(container, rowsAsc, groups, metaKeys, configFields, labelFields, maxVisible, prefix) {{
  container.innerHTML = "";
  const visible = groups.slice(0, maxVisible);
  const extra = groups.slice(maxVisible);

  function renderInto(groupList, parent) {{
    groupList.forEach((group) => {{
      const seriesByKey = new Map();
      rowsAsc.forEach((row) => {{
        const dateStr = String(row.date || "");
        if (!dateStr) return;
        const cfgParts = configFields
          .map((field) => row[field])
          .filter((value) => value !== null && value !== undefined && String(value).trim() !== "")
          .map((value) => String(value));
        const cfgKey = cfgParts.length ? cfgParts.join("||") : "config";
        group.metrics.forEach((metric) => {{
          const y = toNumber(row[metric]);
          if (y === null) return;
          const key = `${{metric}}||${{cfgKey}}`;
          if (!seriesByKey.has(key)) {{
            const fallback = cfgKey.replaceAll("||", " | ");
            seriesByKey.set(key, {{
              label: compactSeriesLabel(prefix, row, metric, fallback),
              legend: compactLegendLabel(prefix, row, metric, fallback),
              points: [],
            }});
          }}
          const meta = {{}};
          metaKeys.forEach((keyName) => {{ meta[keyName] = row[keyName]; }});
          seriesByKey.get(key).points.push({{ x: dateStr, y, meta }});
        }});
      }});
      const seriesEntries = Array.from(seriesByKey.values()).filter(
        (entry) => entry.points.length > 0,
      );
      if (!seriesEntries.length) return;
      const seriesList = seriesEntries.map((entry) => entry.points);
      const labels = seriesEntries.map((entry) => entry.label);
      const legendLabels = seriesEntries.map((entry) => entry.legend);
      const totalPoints = seriesList.reduce((acc, series) => acc + series.length, 0);
      const card = document.createElement("div");
      card.className = "chart-card";
      const head = document.createElement("div");
      head.className = "chart-head";
      head.innerHTML =
        `<div class="chart-title">${{escapeHtml(group.title)}}` +
        `${{totalPoints < 2 ? " (snapshot)" : ""}}</div>` +
        `<div class="chart-caption">${{seriesList.length}} series · ` +
        `${{totalPoints}} point(s)</div>`;
      card.appendChild(head);
      if (totalPoints < 2) {{
        const latest = rowsAsc[rowsAsc.length - 1];
        const body = document.createElement("div");
        body.className = "hint";
        const lines = [];
        if (latest && latest.date) {{
          lines.push(`date: <code>${{escapeHtml(formatDate(latest.date))}}</code>`);
        }}
        metaKeys.forEach((keyName) => {{
          if (
            !latest ||
            latest[keyName] === null ||
            latest[keyName] === undefined ||
            String(latest[keyName]).trim() === ""
          ) {{
            return;
          }}
          lines.push(`${{escapeHtml(keyName)}}: <code>${{escapeHtml(String(latest[keyName]))}}</code>`);
        }});
        group.metrics.forEach((metric) => {{
          const value = latest ? fmt(latest[metric]) : "";
          if (value) lines.push(`${{escapeHtml(metric)}}: <code>${{value}}</code>`);
        }});
        body.innerHTML = lines.join("<br>") || "No numeric data for snapshot.";
        card.appendChild(body);
        parent.appendChild(card);
        return;
      }}
      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      const canvas = document.createElement("canvas");
      const tooltip = document.createElement("div");
      tooltip.className = "chart-tooltip";
      wrap.appendChild(canvas);
      wrap.appendChild(tooltip);
      card.appendChild(wrap);
      parent.appendChild(card);
      canvas.__draw = () => {{
        const r = canvas.getBoundingClientRect();
        if (!r || r.width < 10 || r.height < 10) return;
        drawChart(canvas, tooltip, seriesList, labels, legendLabels);
      }};
      requestAnimationFrame(() => canvas.__draw());
    }});
  }}

  renderInto(visible, container);
  if (extra.length) {{
    const details = document.createElement("details");
    details.className = "more-charts";
    const summary = document.createElement("summary");
    summary.textContent = `More charts (${{extra.length}})`;
    details.appendChild(summary);
    const inner = document.createElement("div");
    details.appendChild(inner);
    renderInto(extra, inner);
    details.addEventListener("toggle", () => {{
      if (!details.open) return;
      requestAnimationFrame(() => {{
        details.querySelectorAll("canvas").forEach((canvas) => {{
          if (typeof canvas.__draw === "function") canvas.__draw();
        }});
      }});
    }});
    container.appendChild(details);
  }}
  if (!container.querySelector(".chart-card")) {{
    const empty = document.createElement("div");
    empty.className = "chart-card empty";
    empty.textContent = "No data to plot for current filters.";
    container.appendChild(empty);
  }}
}}

function filterRows(rows, filters) {{
  return rows.filter((row) => {{
    if (
      filters.modelSearch &&
      !String(row[filters.modelKey] || "")
        .toLowerCase()
        .includes(filters.modelSearch.toLowerCase())
    ) {{
      return false;
    }}
    if (filters.model && String(row[filters.modelKey] || "") !== filters.model) return false;
    if (filters.testName && String(row.test_name || "") !== filters.testName) return false;
    if (filters.datasetName && String(row[filters.datasetKey] || "") !== filters.datasetName) return false;
    if (
      filters.maxConcurrency &&
      String(row.max_concurrency || "") !== filters.maxConcurrency
    ) {{
      return false;
    }}
    if (
      filters.numPrompts &&
      String(row.num_prompts || "") !== filters.numPrompts
    ) {{
      return false;
    }}
    if (filters.backend && String(row.backend || "") !== filters.backend) return false;
    if (
      filters.extraKey &&
      filters.extraValue &&
      String(row[filters.extraKey] || "") !== filters.extraValue
    ) {{
      return false;
    }}
    return true;
  }});
}}

function sortByDateAsc(rows) {{
  return [...rows].sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")));
}}

function averageMetric(rows, key) {{
  const values = rows.map((row) => toNumber(row[key])).filter((v) => v !== null);
  if (!values.length) return "n/a";
  const avg = values.reduce((acc, value) => acc + value, 0) / values.length;
  return avg.toFixed(4);
}}

function initSection(prefix, columns, data, numericCols, groups) {{
  const modelInput = document.getElementById(`${{prefix}}-model`);
  const modelSearchInput = document.getElementById(`${{prefix}}-model-search`);
  const testInput = document.getElementById(`${{prefix}}-test`);
  const datasetInput = document.getElementById(`${{prefix}}-dataset`);
  const concurrencyInput = document.getElementById(`${{prefix}}-concurrency`);
  const promptsInput = document.getElementById(`${{prefix}}-prompts`);
  const backendInput = document.getElementById(`${{prefix}}-backend`);
  const extraInput = document.getElementById(`${{prefix}}-extra`);
  const statsEl = document.getElementById(`${{prefix}}-stats`);
  const panelEl = document.getElementById(`${{prefix}}-panel`);
  const chartsEl = document.getElementById(`${{prefix}}-charts`);
  const clearBtn = document.getElementById(`${{prefix}}-clear`);
  const sortState = {{ column: "date", desc: true, onChange: render }};

  const modelKey = prefix === "diff" ? "model" : "model_id";
  const datasetKey = prefix === "diff" ? "dataset" : "dataset_name";
  const extraKey = prefix === "diff" ? "" : "tokenizer_id";
  const metaKeys = prefix === "diff"
    ? ["test_name"]
    : ["test_name", "max_concurrency", "num_prompts"];
  const configFields = prefix === "diff"
    ? ["test_name", "model", "backend", "dataset"]
    : [
        "endpoint_type",
        "backend",
        "model_id",
        "tokenizer_id",
        "test_name",
        "dataset_name",
        "max_concurrency",
        "num_prompts",
      ];
  const labelFields = prefix === "diff"
    ? ["test_name", "dataset"]
    : ["test_name", "dataset_name"];
  const latestMetric = prefix === "diff" ? "throughput_qps" : "output_throughput";
  const secondaryMetric = prefix === "diff" ? "latency_mean" : "mean_e2el_ms";
  const modelList = document.getElementById(`${{prefix}}-model-list`);
  const testList = document.getElementById(`${{prefix}}-test`);
  const datasetList = document.getElementById(`${{prefix}}-dataset`);
  const concurrencyList = document.getElementById(`${{prefix}}-concurrency`);
  const promptsList = document.getElementById(`${{prefix}}-prompts`);
  const backendList = document.getElementById(`${{prefix}}-backend`);
  const extraList = document.getElementById(`${{prefix}}-extra`);

  function currentFilters() {{
    return {{
      model: modelInput.value.trim(),
      modelSearch: modelSearchInput ? modelSearchInput.value.trim() : "",
      testName: testInput.value.trim(),
      datasetName: datasetInput.value.trim(),
      maxConcurrency: concurrencyInput ? concurrencyInput.value.trim() : "",
      numPrompts: promptsInput ? promptsInput.value.trim() : "",
      backend: backendInput.value.trim(),
      extraValue: extraInput ? extraInput.value.trim() : "",
      extraKey,
      modelKey,
      datasetKey,
    }};
  }}

  function refreshOptionLists(baseRows) {{
    fillDatalist(modelList, uniqSorted(baseRows.map((row) => row[modelKey])));
    fillSelect(
      testList,
      uniqSorted(baseRows.map((row) => row.test_name)),
      testInput.value,
    );
    fillSelect(
      datasetList,
      uniqSorted(baseRows.map((row) => row[datasetKey])),
      datasetInput.value,
    );
    if (concurrencyList) {{
      fillSelect(
        concurrencyList,
        uniqSorted(baseRows.map((row) => row.max_concurrency)),
        concurrencyInput.value,
      );
    }}
    if (promptsList) {{
      fillSelect(
        promptsList,
        uniqSorted(baseRows.map((row) => row.num_prompts)),
        promptsInput.value,
      );
    }}
    fillSelect(
      backendList,
      uniqSorted(baseRows.map((row) => row.backend)),
      backendInput.value,
    );
    if (extraList) {{
      fillSelect(
        extraList,
        uniqSorted(baseRows.map((row) => row[extraKey])),
        extraInput.value,
      );
    }}
  }}

  function firstValue(rows, key) {{
    const values = uniqSorted(rows.map((row) => row[key]));
    return values.length ? String(values[0]) : "";
  }}

  function applyDefaultSelections() {{
    if (!data.length) return;
    let rows = data;

    if (!modelInput.value.trim()) {{
      modelInput.value = firstValue(rows, modelKey);
    }}
    rows = filterRows(data, {{
      ...currentFilters(),
      testName: "",
      datasetName: "",
      maxConcurrency: "",
      numPrompts: "",
      backend: "",
      extraValue: "",
      modelSearch: "",
    }});

    if (!testInput.value.trim()) {{
      testInput.value = firstValue(rows, "test_name");
    }}
    rows = filterRows(data, {{
      ...currentFilters(),
      datasetName: "",
      maxConcurrency: "",
      numPrompts: "",
      backend: "",
      extraValue: "",
      modelSearch: "",
    }});

    if (!datasetInput.value.trim()) {{
      datasetInput.value = firstValue(rows, datasetKey);
    }}
    rows = filterRows(data, {{
      ...currentFilters(),
      maxConcurrency: "",
      numPrompts: "",
      backend: "",
      extraValue: "",
      modelSearch: "",
    }});

    if (concurrencyInput && !concurrencyInput.value.trim()) {{
      concurrencyInput.value = firstValue(rows, "max_concurrency");
    }}
    rows = filterRows(data, {{
      ...currentFilters(),
      numPrompts: "",
      backend: "",
      extraValue: "",
      modelSearch: "",
    }});

    if (promptsInput && !promptsInput.value.trim()) {{
      promptsInput.value = firstValue(rows, "num_prompts");
    }}
    rows = filterRows(data, {{
      ...currentFilters(),
      backend: "",
      extraValue: "",
      modelSearch: "",
    }});

    if (!backendInput.value.trim()) {{
      backendInput.value = firstValue(rows, "backend");
    }}
    rows = filterRows(data, {{
      ...currentFilters(),
      extraValue: "",
      modelSearch: "",
    }});

    if (extraKey && extraInput && !extraInput.value.trim()) {{
      extraInput.value = firstValue(rows, extraKey);
    }}
  }}

  refreshOptionLists(data);
  applyDefaultSelections();

  function render() {{
    const filters = currentFilters();
    const filtered = filterRows(data, filters);
    const filteredAsc = sortByDateAsc(filtered);
    const latest = filteredAsc[filteredAsc.length - 1] || null;
    const configCount = new Set(
      filtered.map((row) =>
        configFields.map((field) => String(row[field] || "")).join("||"),
      ),
    ).size;
    refreshOptionLists(filterRows(data, {{
      ...filters,
      testName: "",
      datasetName: "",
      maxConcurrency: "",
      numPrompts: "",
      backend: "",
      extraValue: "",
      modelSearch: "",
    }}));

    renderStats(statsEl, [
      {{ label: "filtered records", value: String(filtered.length), sub: `${{configCount}} unique config key(s)` }},
      {{
        label: "latest run",
        value: latest ? formatDate(latest.date) : "n/a",
        sub: "newest visible point",
      }},
      {{
        label: `latest ${{latestMetric}}`,
        value: latest ? (fmt(latest[latestMetric]) || "n/a") : "n/a",
        sub: latestMetric,
      }},
      {{
        label: `avg ${{secondaryMetric}}`,
        value: averageMetric(filtered, secondaryMetric),
        sub: secondaryMetric,
      }},
    ]);

    panelEl.innerHTML = `
      <h3>Selection details</h3>
      <div class="kv">
        <div class="kv-row"><div class="kv-label">Model</div><div class="kv-value"><code>${{escapeHtml(
          filters.model || "All",
        )}}</code></div></div>
        <div class="kv-row"><div class="kv-label">Model search</div><div class="kv-value"><code>${{escapeHtml(
          filters.modelSearch || "All",
        )}}</code></div></div>
        <div class="kv-row"><div class="kv-label">Test</div><div class="kv-value"><code>${{escapeHtml(
          filters.testName || "All",
        )}}</code></div></div>
        <div class="kv-row"><div class="kv-label">Dataset</div><div class="kv-value"><code>${{escapeHtml(
          filters.datasetName || "All",
        )}}</code></div></div>
        ${{
          prefix === "diff"
            ? ""
            : (
              `<div class="kv-row"><div class="kv-label">Concurrency</div>` +
              `<div class="kv-value"><code>${{escapeHtml(filters.maxConcurrency || "All")}}</code></div></div>`
            )
        }}
        ${{
          prefix === "diff"
            ? ""
            : (
              `<div class="kv-row"><div class="kv-label">Num prompts</div>` +
              `<div class="kv-value"><code>${{escapeHtml(filters.numPrompts || "All")}}</code></div></div>`
            )
        }}
        <div class="kv-row"><div class="kv-label">Backend</div><div class="kv-value"><code>${{escapeHtml(
          filters.backend || "All",
        )}}</code></div></div>
        ${{
          prefix === "diff"
            ? ""
            : (
              `<div class="kv-row"><div class="kv-label">Tokenizer</div>` +
              `<div class="kv-value"><code>${{escapeHtml(filters.extraValue || "All")}}</code></div></div>`
            )
        }}
        <div class="kv-row"><div class="kv-label">Latest source</div><div class="kv-value"><code>${{
          escapeHtml(latest ? String(latest.source_file || "") : "n/a")
        }}</code></div></div>
      </div>
      <div class="snapshot-list">
        ${{
          latest ? [
            latest[latestMetric] !== undefined
              ? `<div class="snapshot-item"><strong>${{escapeHtml(
                  latestMetric,
                )}}</strong><div>${{fmt(latest[latestMetric]) || "n/a"}}</div></div>`
              : "",
            latest[secondaryMetric] !== undefined
              ? `<div class="snapshot-item"><strong>${{escapeHtml(
                  secondaryMetric,
                )}}</strong><div>${{fmt(latest[secondaryMetric]) || "n/a"}}</div></div>`
              : "",
            latest.date
              ? `<div class="snapshot-item"><strong>date</strong><div>${{
                  escapeHtml(formatDate(latest.date))
                }}</div></div>`
              : "",
          ].join("") : '<div class="hint">No visible snapshot.</div>'
        }}
      </div>
    `;

    if (prefix === "omni" && !filters.model) {{
      chartsEl.innerHTML =
        "<div class='chart-card empty'>Select a <code>model_id</code> " +
        "to focus the Omni trend view.</div>";
    }} else {{
      renderChartGroups(chartsEl, filteredAsc, groups, metaKeys, configFields, labelFields, 3, prefix);
    }}
    renderTable(
      `${{prefix}}-table`,
      columns,
      sortRows(filteredAsc, sortState.column, sortState.desc, numericCols),
      numericCols,
      sortState,
    );
  }}

  [
    modelInput,
    modelSearchInput,
    testInput,
    datasetInput,
    concurrencyInput,
    promptsInput,
    backendInput,
    extraInput,
  ].filter(Boolean).forEach((el) => {{
    el.addEventListener("input", render);
    el.addEventListener("change", render);
  }});
  clearBtn.addEventListener("click", () => {{
    modelInput.value = "";
    if (modelSearchInput) modelSearchInput.value = "";
    testInput.value = "";
    datasetInput.value = "";
    if (concurrencyInput) concurrencyInput.value = "";
    if (promptsInput) promptsInput.value = "";
    backendInput.value = "";
    if (extraInput) extraInput.value = "";
    applyDefaultSelections();
    render();
  }});
  render();
}}

window.addEventListener("load", () => {{
  const omniNumeric = new Set([
    "num_prompts", "burstiness", "max_concurrency", "duration", "completed", "failed",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms",
    "mean_audio_rtf", "median_audio_rtf", "p99_audio_rtf",
    "mean_audio_ttfp_ms", "median_audio_ttfp_ms", "p99_audio_ttfp_ms",
    "mean_audio_duration_s", "median_audio_duration_s", "p99_audio_duration_s",
  ]);
  const diffNumeric = new Set([
    "duration", "completed_requests", "failed_requests", "throughput_qps",
    "latency_mean", "latency_median", "latency_p50", "latency_p99",
    "peak_memory_mb_max", "peak_memory_mb_mean", "peak_memory_mb_median", "slo_attainment_rate",
  ]);
  const omniGroups = [
    {{ title: "throughput", metrics: ["output_throughput", "total_token_throughput"] }},
    {{ title: "ttft", metrics: ["mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms"] }},
    {{
      title: "tpot + itl",
      metrics: [
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
      ],
    }},
    {{ title: "e2el", metrics: ["mean_e2el_ms", "median_e2el_ms", "p99_e2el_ms"] }},
    {{ title: "audio rtf", metrics: ["mean_audio_rtf", "median_audio_rtf", "p99_audio_rtf"] }},
    {{ title: "audio ttfp", metrics: ["mean_audio_ttfp_ms", "median_audio_ttfp_ms", "p99_audio_ttfp_ms"] }},
    {{
      title: "audio duration",
      metrics: [
        "mean_audio_duration_s",
        "median_audio_duration_s",
        "p99_audio_duration_s",
      ],
    }},
  ];
  const diffGroups = [
    {{ title: "throughput", metrics: ["throughput_qps"] }},
    {{
      title: "latency",
      metrics: [
        "latency_mean",
        "latency_median",
        "latency_p99",
        "latency_p50",
      ],
    }},
  ];
  initSection("omni", OMNI_COLUMNS, OMNI_DATA, omniNumeric, omniGroups);
  initSection("diff", DIFF_COLUMNS, DIFF_DATA, diffNumeric, diffGroups);
}});
"""

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8" />',
            '  <meta name="viewport" content="width=device-width, initial-scale=1" />',
            "  <title>Nightly Performance Report V2</title>",
            f"  <style>{styles}</style>",
            "</head>",
            "<body>",
            '  <div class="container" id="top">',
            '    <div class="hero">',
            "      <h1>Nightly Performance Report</h1>",
            (
                '      <div class="meta">Improved dashboard view for nightly Omni '
                "and Diffusion benchmarks. The page keeps the current dark color "
                "scheme, but adds stronger filtering, summary cards, sticky "
                "tables, and more readable trend blocks.</div>"
            ),
            '      <div class="hero-badges">',
            f'        <div class="badge">Omni records <strong>{len(omni_records)}</strong></div>',
            f'        <div class="badge">Diffusion records <strong>{len(diffusion_records)}</strong></div>',
            '        <div class="badge">Numeric precision <strong>4 decimals</strong></div>',
            "      </div>",
            "    </div>",
            '    <section id="omni-section">',
            (
                '      <div class="section-head"><div><h2 class="section-title">Omni</h2>'
                '<div class="section-subtitle">Chat/audio benchmark history with '
                "trend charts grouped by throughput, latency, and audio metrics."
                '</div></div><div class="section-pills"><div class="pill"><strong>'
                "Grouping</strong> test_name + dataset + concurrency + prompts</div>"
                '<div class="pill"><strong>Snapshots</strong> single-point series '
                "stay readable</div></div></div>"
            ),
            '      <div class="stats-grid" id="omni-stats"></div>',
            '      <div class="toolbar">',
            '        <div class="toolbar-row primary">',
            (
                '        <div class="filter-field"><label>model_id</label><input '
                'class="filter-input" type="text" id="omni-model" '
                'list="omni-model-list" placeholder="Select model_id" /><datalist '
                'id="omni-model-list"></datalist></div>'
            ),
            (
                '        <div class="filter-field compact"><label>search</label><input '
                'class="filter-input" type="text" id="omni-model-search" '
                'placeholder="Fuzzy search model_id" /></div>'
            ),
            (
                '        <div class="filter-field compact"><label>test</label>'
                '<select class="filter-select" id="omni-test"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>dataset</label>'
                '<select class="filter-select" id="omni-dataset"></select></div>'
            ),
            "        </div>",
            '        <div class="toolbar-row secondary">',
            (
                '        <div class="filter-field compact"><label>max_conc</label>'
                '<select class="filter-select" id="omni-concurrency"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>prompts</label>'
                '<select class="filter-select" id="omni-prompts"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>backend</label>'
                '<select class="filter-select" id="omni-backend"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>tokenizer</label>'
                '<select class="filter-select" id="omni-extra"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>&nbsp;</label><button '
                'class="ghost" type="button" id="omni-clear">Clear filters</button>'
                "</div>"
            ),
            "        </div>",
            "      </div>",
            (
                '      <div class="hint">Searchable inputs use browser datalist '
                "suggestions for quick keyboard selection.</div>"
            ),
            (
                '      <div class="layout"><div class="charts-stack" '
                'id="omni-charts"></div><div class="panel" id="omni-panel"></div>'
                "</div>"
            ),
            '      <div id="omni-table"></div>',
            "    </section>",
            '    <section id="diff-section">',
            (
                '      <div class="section-head"><div><h2 class="section-title">'
                'Diffusion</h2><div class="section-subtitle">Image generation '
                "benchmark history with throughput and latency trend views.</div>"
                '</div><div class="section-pills"><div class="pill"><strong>'
                "Grouping</strong> test_name + model + backend + dataset</div>"
                '<div class="pill"><strong>Snapshots</strong> single-point series '
                "stay readable</div></div></div>"
            ),
            '      <div class="stats-grid" id="diff-stats"></div>',
            '      <div class="toolbar">',
            '        <div class="toolbar-row primary">',
            (
                '        <div class="filter-field"><label>model</label><input '
                'class="filter-input" type="text" id="diff-model" '
                'list="diff-model-list" placeholder="Select model" /><datalist '
                'id="diff-model-list"></datalist></div>'
            ),
            (
                '        <div class="filter-field compact"><label>search</label><input '
                'class="filter-input" type="text" id="diff-model-search" '
                'placeholder="Fuzzy search model" /></div>'
            ),
            (
                '        <div class="filter-field compact"><label>test</label>'
                '<select class="filter-select" id="diff-test"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>dataset</label>'
                '<select class="filter-select" id="diff-dataset"></select></div>'
            ),
            "        </div>",
            '        <div class="toolbar-row secondary">',
            (
                '        <div class="filter-field compact"><label>backend</label>'
                '<select class="filter-select" id="diff-backend"></select></div>'
            ),
            (
                '        <div class="filter-field compact"><label>&nbsp;</label><button '
                'class="ghost" type="button" id="diff-clear">Clear filters</button>'
                "</div>"
            ),
            "        </div>",
            "      </div>",
            (
                '      <div class="hint">Charts automatically skip empty metric '
                "groups and collapse anything beyond the first three groups.</div>"
            ),
            (
                '      <div class="layout"><div class="charts-stack" '
                'id="diff-charts"></div><div class="panel" id="diff-panel"></div>'
                "</div>"
            ),
            '      <div id="diff-table"></div>',
            "    </section>",
            "  </div>",
            '  <a class="back-to-top" href="#top">Back to top</a>',
            f"  <script>{script}</script>",
            "</body>",
            "</html>",
        ]
    )


def generate_html_report(input_dir: str, diffusion_input_dir: str, output_file: str) -> None:
    omni_records = _sort_omni_records(_collect_omni_records(input_dir))
    diffusion_records = _sort_diffusion_records(_collect_diffusion_records(diffusion_input_dir))

    if not omni_records:
        LOGGER.warning("no valid omni json records found under '%s'", input_dir)
    if not diffusion_records:
        LOGGER.warning("no valid diffusion json records found under '%s'", diffusion_input_dir)

    omni_columns = [
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
    diffusion_columns = [
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

    html = _build_html_document(
        omni_columns=omni_columns,
        omni_records=omni_records,
        diffusion_columns=diffusion_columns,
        diffusion_records=diffusion_records,
    )
    _ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    LOGGER.info("improved html report saved to '%s'", output_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    diffusion_input_dir = args.diffusion_input_dir or _default_diffusion_input_dir(args.input_dir)
    generate_html_report(
        input_dir=args.input_dir,
        diffusion_input_dir=diffusion_input_dir,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
