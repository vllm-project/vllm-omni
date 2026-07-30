#!/usr/bin/env python3
"""
Fetch vllm-omni and vllm-omni-npu-ci builds from the Buildkite REST API for a
date range (default: yesterday in CST / UTC+8) and compute per-resource-pool
statistics:

  - Queue wait time: started_at - scheduled_at per job, aggregated by pool
    (avg, max, p50, p90)
  - Job duration: finished_at - started_at per job, aggregated by pool
    (avg, total occupancy)
  - Job count per pool
  - Hourly time-series: avg wait & avg duration per hour per pool (inline SVG chart)

The date window is interpreted in **Beijing Time (CST, UTC+8)** — i.e. each
``--from`` / ``--to`` date denotes a full 00:00-23:59 CST calendar day,
which maps to the UTC span ``(date-1) 16:00 UTC`` → ``date 15:59:59 UTC``.

Resource pool identification uses each job's ``agent_query_rules`` array.
The convention is ``queue=<pool-name>`` entries. Jobs without an explicit
queue rule are grouped into the ``default`` pool.

Usage:

  Set BUILDKITE_API_TOKEN (or BUILDKITE_TOKEN).
  pip install requests  # if missing
  python scripts/resource_pool_stats.py [--from YYYY-MM-DD --to YYYY-MM-DD] \
    [--pipeline vllm-omni,vllm-omni-npu-ci] \
    [--format html|markdown|json] \
    [--output PATH] [--verbose]

Default output is **HTML** written to ``pool-stats-YYYY-MM-DD.html`` in the
current directory. ``--format markdown`` or ``--format json`` prints to stdout.

If ``--from`` / ``--to`` are both omitted, the window is **yesterday CST**
(00:00 to 23:59:59 CST). If you pass one, pass both (CST calendar dates,
inclusive).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

# ── Buildkite API constants ──────────────────────────────────────────────

BUILDKITE_API_BASE = "https://api.buildkite.com/v2"
ORG_SLUG = "vllm"
DEFAULT_PIPELINES = ["vllm-omni", "vllm-omni-npu-ci"]

# ── Timezone handling ──────────────────────────────────────────────────

# All date windows are interpreted in Beijing Time (CST, UTC+8).
# Buildkite's API expects ISO-8601 UTC timestamps, so each user-supplied
# CST date is mapped to a (start_utc, end_utc) pair covering the full CST
# calendar day.
CST = timezone(timedelta(hours=8))

# ── Pool color palette for charts ────────────────────────────────────────

POOL_COLORS = [
    "#7c3aed",  # purple (ci)
    "#3b82f6",  # blue (accent)
    "#1f9d63",  # green (healthy)
    "#d97706",  # amber (warning)
    "#ef4444",  # red (alert)
    "#06b6d4",  # cyan
    "#f472b6",  # pink
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#f59e0b",  # orange
]


# ── Editorial CSS (aligned with report_html_theme.py palette) ───────────

POOL_STATS_CSS = """
:root {
  --dashboard-bg: #f5f8fb;
  --dashboard-panel-bg: #ffffff;
  --dashboard-panel-strong: #f1f5f9;
  --dashboard-border: #d9e2ec;
  --dashboard-border-strong: #d6dde6;
  --dashboard-text: #26323f;
  --dashboard-muted: #607080;
  --dashboard-soft-text: #52606d;
  --dashboard-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
  --dashboard-badge-bg: #edf3f8;
  --dashboard-badge-text: #435466;
  --dashboard-chart-text: #5b6775;
  --dashboard-chart-grid: rgba(148, 163, 184, 0.2);
  --dashboard-tooltip-bg: rgba(15, 23, 42, 0.92);
  --dashboard-tooltip-text: #f8fafc;
  --dashboard-healthy: #1f9d63;
  --dashboard-alert: #d14343;
  --dashboard-warning: #d97706;
  --dashboard-violet: #4f46e5;
  --dashboard-violet-bg: rgba(79, 70, 229, 0.1);
  --bg: var(--dashboard-bg);
  --surface: var(--dashboard-panel-bg);
  --surface-muted: #edf3f8;
  --text: var(--dashboard-text);
  --muted: var(--dashboard-muted);
  --border: var(--dashboard-border);
  --shadow: var(--dashboard-shadow);
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --accent-soft: rgba(59, 130, 246, 0.22);
  --accent-tint: rgba(59, 130, 246, 0.08);
  --ci: #7c3aed;
  --ci-soft: rgba(124, 58, 237, 0.18);
  --ci-tint: rgba(124, 58, 237, 0.08);
  --danger: var(--dashboard-alert);
  --danger-strong: #b91c1c;
  --danger-bg: rgba(209, 67, 67, 0.08);
  --ok: var(--dashboard-healthy);
  --ok-bg: var(--dashboard-healthy-bg);
  --ok-edge: var(--dashboard-healthy);
  --fail-bg: var(--dashboard-alert-bg);
  --fail-edge: var(--dashboard-alert);
  --radius: 12px;
  --radius-sm: 8px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --dashboard-bg: #111827;
    --dashboard-panel-bg: #162130;
    --dashboard-panel-strong: #131d2b;
    --dashboard-border: rgba(148, 163, 184, 0.18);
    --dashboard-border-strong: rgba(148, 163, 184, 0.22);
    --dashboard-text: #edf3fb;
    --dashboard-muted: #b6c4d5;
    --dashboard-soft-text: #c6d3e1;
    --dashboard-shadow: 0 20px 40px rgba(2, 6, 23, 0.35);
    --dashboard-badge-bg: rgba(148, 163, 184, 0.12);
    --dashboard-badge-text: #d5dfeb;
    --dashboard-chart-text: #c8d4e3;
    --dashboard-chart-grid: rgba(148, 163, 184, 0.16);
    --dashboard-tooltip-bg: rgba(15, 23, 42, 0.96);
    --dashboard-tooltip-text: #f8fafc;
    --dashboard-healthy-bg: rgba(31, 157, 99, 0.16);
    --dashboard-alert-bg: rgba(209, 67, 67, 0.18);
    --dashboard-warning: #fbbf24;
    --dashboard-warning-bg: rgba(251, 191, 36, 0.14);
    --dashboard-violet: #818cf8;
    --dashboard-violet-bg: rgba(129, 140, 248, 0.14);
    --surface-muted: rgba(148, 163, 184, 0.1);
    --accent: #60a5fa;
    --accent-hover: #3b82f6;
    --accent-soft: rgba(96, 165, 250, 0.28);
    --accent-tint: rgba(96, 165, 250, 0.12);
    --ci: #a78bfa;
    --ci-soft: rgba(167, 139, 250, 0.22);
    --ci-tint: rgba(167, 139, 250, 0.1);
    --danger-strong: #fecaca;
    --danger-bg: rgba(209, 67, 67, 0.14);
    --unknown-bg: rgba(148, 163, 184, 0.08);
    --unknown-edge: #94a3b8;
  }
}
* { box-sizing: border-box; }
body {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  font-size: 15px;
}
.top-bar {
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  padding: 0;
  box-shadow: 0 1px 0 rgba(148, 163, 184, 0.18);
  border-bottom: 3px solid var(--ci);
}
.top-bar-inner {
  padding: 1.4rem 1.25rem 1.55rem;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
}
.brand {
  display: flex;
  align-items: center;
  gap: 1.1rem;
}
.brand-mark {
  width: 3.2rem;
  height: 3.2rem;
  border-radius: var(--radius-sm);
  background: var(--dashboard-badge-bg);
  border: 1px solid var(--dashboard-border);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}
.brand-mark .ico { stroke: var(--ci); }
.brand-copy h1 {
  margin: 0;
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: var(--dashboard-text);
}
.tagline {
  margin: 0.35rem 0 0;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--dashboard-muted);
}
.shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 1.5rem 1.25rem 3rem;
}
.panel {
  background: var(--dashboard-panel-bg);
  border-radius: var(--radius);
  border: 1px solid var(--dashboard-border);
  box-shadow: var(--dashboard-shadow);
  padding: 1.15rem 1.3rem;
  margin-bottom: 1.35rem;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.panel:hover {
  box-shadow: 0 16px 28px rgba(15, 23, 42, 0.1);
  border-color: rgba(124, 58, 237, 0.28);
}
.panel-bk {
  border-top: 4px solid var(--ci);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--ci-soft) 35%, var(--dashboard-panel-bg)) 100%
  );
}
.panel h2 {
  margin: 0 0 1rem;
  font-size: 1.12rem;
  font-weight: 800;
  color: var(--dashboard-text);
  letter-spacing: -0.02em;
  border-bottom: 2px solid var(--dashboard-border);
  padding-bottom: 0.55rem;
}
.panel-bk h2 {
  border-bottom-color: var(--ci-soft);
}
.heading-row {
  display: inline-flex;
  align-items: flex-start;
  gap: 0.65rem;
}
.heading-ico {
  display: flex;
  margin-top: 0.12rem;
  color: var(--ci);
}
.meta {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 0.4rem 0;
}
.meta strong {
  color: var(--dashboard-badge-text);
  font-weight: 650;
}
.table-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  border-radius: var(--radius-sm);
  margin: 0.65rem 0 0;
  border: 1px solid var(--border);
  background: var(--surface-muted);
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--dashboard-panel-bg) 65%, transparent);
}
.table-scroll > table {
  width: 100%;
  min-width: 860px;
}
table.pool-stats {
  border-collapse: collapse;
  font-size: 0.92rem;
}
table.pool-stats th, table.pool-stats td {
  border: 1px solid var(--border);
  padding: 0.65rem 0.8rem;
  text-align: left;
  vertical-align: top;
}
table.pool-stats th {
  background: var(--dashboard-panel-strong);
  font-weight: 650;
  color: var(--dashboard-chart-text);
  white-space: nowrap;
}
table.pool-stats tbody tr:nth-child(even) td {
  background: color-mix(in srgb, var(--dashboard-badge-bg) 45%, var(--dashboard-panel-bg));
}
table.pool-stats td.num {
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
table.pool-stats td.pool-name {
  font-weight: 650;
  color: var(--dashboard-text);
}
table.pool-stats td.pipeline-cell {
  border-left: 3px solid var(--ci);
  padding-left: calc(0.8rem - 2px);
  font-weight: 650;
  color: var(--dashboard-violet);
}
table.pool-stats td.na {
  color: var(--dashboard-muted);
  font-style: italic;
}
.legend {
  margin: 1rem 0 0;
  padding: 0.85rem 1rem;
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--dashboard-badge-bg) 72%, var(--dashboard-panel-bg));
  border: 1px solid var(--dashboard-border);
  font-size: 0.88rem;
  color: var(--dashboard-soft-text);
}
.legend dt {
  font-weight: 650;
  color: var(--dashboard-text);
  margin-top: 0.3rem;
}
.legend dd {
  margin: 0 0 0.3rem 0.3rem;
}
.legend dl {
  margin: 0;
  padding: 0;
}
.focus-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
  gap: 0.85rem;
  margin: 0.85rem 0 1rem;
}
.focus-card {
  border-radius: var(--radius-sm);
  border: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-bg);
  padding: 0.85rem 0.95rem;
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--dashboard-panel-bg) 65%, transparent);
}
.focus-card--ci {
  border-left: 4px solid var(--ci);
  background: color-mix(in srgb, var(--ci-soft) 76%, var(--dashboard-panel-bg));
}
.focus-card-title {
  color: var(--dashboard-muted);
  font-size: 0.78rem;
  font-weight: 760;
  text-transform: uppercase;
  letter-spacing: 0.035em;
}
.focus-card-value {
  margin-top: 0.2rem;
  font-size: 1.18rem;
  font-weight: 820;
  color: var(--dashboard-text);
}
.focus-card-detail {
  margin-top: 0.15rem;
  color: var(--dashboard-muted);
  font-size: 0.84rem;
}
.ico { display: block; }
.chart-container {
  margin: 0.65rem 0 0;
  overflow-x: auto;
}
.chart-container svg {
  display: block;
  width: 100%;
  max-width: 860px;
  height: auto;
}
.chart-legend-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 1rem;
  margin: 0.55rem 0 0;
  font-size: 0.82rem;
  color: var(--dashboard-chart-text);
}
.chart-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.chart-legend-swatch {
  width: 1rem;
  height: 0.35rem;
  border-radius: 2px;
  flex-shrink: 0;
}
.chart-group-title {
  margin: 1.2rem 0 0.55rem;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--dashboard-soft-text);
}
.chart-group-title:first-child {
  margin-top: 0;
}
.cat-stats {
  margin: 0 0 1.4rem;
}
.cat-stats-title {
  margin: 0 0 0.75rem;
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--dashboard-soft-text);
  letter-spacing: -0.01em;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}
.cat-stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.85rem;
}
@media (max-width: 640px) {
  .cat-stats-grid {
    grid-template-columns: 1fr;
  }
}
.cat-card {
  background: var(--surface-muted);
  border: 1px solid var(--dashboard-border);
  border-radius: var(--radius-sm);
  padding: 0.85rem 1rem 0.8rem;
  border-left: 4px solid var(--ci);
}
.cat-card--ready   { border-left-color: #3b82f6; }
.cat-card--merge   { border-left-color: #1f9d63; }
.cat-card--nightly { border-left-color: #d97706; }
.cat-card--weekly  { border-left-color: #ef4444; }
.cat-card-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin: 0 0 0.7rem;
  padding-bottom: 0.55rem;
  border-bottom: 1px solid var(--dashboard-border);
}
.cat-card-label {
  font-size: 0.86rem;
  font-weight: 750;
  color: var(--dashboard-text);
  letter-spacing: 0.01em;
}
.cat-card-sub {
  font-size: 0.74rem;
  color: var(--dashboard-muted);
  font-weight: 500;
  margin-left: 0.45rem;
}
.cat-card-count {
  font-size: 0.74rem;
  color: var(--dashboard-muted);
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  white-space: nowrap;
}
.latest-subcards {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
}
.latest-subcard {
  background: var(--dashboard-panel-bg);
  border: 1px solid var(--dashboard-border);
  border-radius: 6px;
  padding: 0.55rem 0.7rem;
}
.latest-subcard-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.3rem;
}
.latest-subcard-pipeline {
  font-weight: 750;
  color: var(--dashboard-violet);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.84rem;
}
.latest-subcard-num {
  color: var(--dashboard-muted);
  font-variant-numeric: tabular-nums;
  font-weight: 650;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.84rem;
}
.latest-subcard-meta {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  margin-bottom: 0.4rem;
  font-size: 0.74rem;
}
.latest-subcard-branch {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--dashboard-soft-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}
.latest-subcard-state {
  flex-shrink: 0;
  padding: 0.05rem 0.45rem;
  border-radius: 4px;
  font-weight: 700;
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  background: color-mix(in srgb, var(--dashboard-badge-bg) 60%, transparent);
  color: var(--dashboard-muted);
  border: 1px solid var(--dashboard-border);
}
.latest-subcard-msg {
  font-size: 0.76rem;
  color: var(--dashboard-soft-text);
  margin-bottom: 0.45rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-style: italic;
}
.latest-pool-list {
  list-style: none;
  padding: 0;
  margin: 0;
  font-size: 0.8rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.18rem 0.8rem;
}
.latest-pool-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.18rem 0;
  border-top: 1px dashed color-mix(in srgb, var(--dashboard-border) 55%, transparent);
  gap: 0.4rem;
}
.latest-pool-row:first-child {
  border-top: none;
}
.latest-pool-name {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--dashboard-text);
  font-weight: 600;
  font-size: 0.78rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.latest-pool-count {
  color: var(--dashboard-violet);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  font-size: 0.84rem;
  flex-shrink: 0;
}
.latest-pool-empty {
  color: var(--dashboard-muted);
  font-style: italic;
  justify-content: flex-start;
}
"""

# ── SVG icons ────────────────────────────────────────────────────────────

ICON_SERVER = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="2" y="2" width="20" height="8" rx="2"/>'
    '<rect x="2" y="14" width="20" height="8" rx="2"/>'
    '<circle cx="6" cy="6" r="1"/>'
    '<circle cx="6" cy="18" r="1"/>'
    "</svg>"
)

ICON_CLOCK = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/>'
    '<polyline points="12 6 12 12 16 14"/>'
    "</svg>"
)

ICON_CHART = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="18" y1="20" x2="18" y2="10"/>'
    '<line x1="12" y1="20" x2="12" y2="4"/>'
    '<line x1="6" y1="20" x2="6" y2="14"/>'
    "</svg>"
)

ICON_TREND = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
    "</svg>"
)


# ── Utility functions ────────────────────────────────────────────────────


def get_api_token() -> str | None:
    token = os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN")
    return token.strip() if token else None


def parse_buildkite_time(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    text = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def yesterday_range_cst() -> tuple[str, str]:
    """Yesterday CST as YYYY-MM-DD strings. Default reporting window."""
    yesterday_cst = (datetime.now(CST) - timedelta(days=1)).date()
    return yesterday_cst.isoformat(), yesterday_cst.isoformat()


def cst_day_to_utc_window(date_str: str) -> tuple[str, str]:
    """Convert a CST calendar date (YYYY-MM-DD) to the (start_utc, end_utc)
    ISO-8601 timestamps covering that full CST day.

    CST 00:00:00 == UTC (date-1) 16:00:00
    CST 23:59:59 == UTC date      15:59:59
    """
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_utc = datetime.combine(d - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).replace(hour=16)
    end_utc = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).replace(hour=15, minute=59, second=59)
    return (
        start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m}m{s}s"


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    n = len(sorted_values)
    k = pct / 100.0 * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def _hour_key(dt: datetime) -> int:
    """Return the CST hour (0-23) for a datetime."""
    return dt.astimezone(CST).hour


# ── Buildkite API helpers ────────────────────────────────────────────────


def parse_link_header(link: str | None) -> dict[str, str]:
    if not link:
        return {}
    out = {}
    for part in link.split(","):
        part = part.strip()
        m = re.match(r'<([^>]+)>;\s*rel="([^"]+)"', part)
        if m:
            out[m.group(2).strip().lower()] = m.group(1).strip()
    return out


def _bk_get_with_retries(
    url: str,
    token: str,
    *,
    params: dict[str, str | int] | None = None,
    max_attempts: int = 10,
) -> requests.Response:
    last_exc: Exception | None = None
    r: requests.Response | None = None
    for attempt in range(max_attempts):
        try:
            r = requests.get(
                url,
                params=params or {},
                headers={"Authorization": f"Bearer {token}"},
                timeout=180,
            )
            if r.status_code == 429:
                ra = r.headers.get("Retry-After", "60")
                try:
                    wait_s = int(float(ra)) + 1
                except ValueError:
                    wait_s = 61
                time.sleep(min(180, max(1, wait_s)))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(min(8, 2 ** min(attempt, 3)))
    assert last_exc is not None
    raise last_exc


def fetch_builds(
    token: str,
    pipeline_slug: str,
    created_from: str,
    created_to: str,
    *,
    branch: str | None = None,
    per_page: int = 100,
) -> list[dict]:
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{pipeline_slug}/builds"
    # created_from / created_to are CST calendar dates (YYYY-MM-DD).
    # Map them to UTC timestamps covering the full CST day each.
    from_utc, _ = cst_day_to_utc_window(created_from)
    _, to_utc = cst_day_to_utc_window(created_to)
    params: dict[str, str | int] = {
        "created_from": from_utc,
        "created_to": to_utc,
        "per_page": per_page,
    }
    if branch:
        params["branch"] = branch
    all_builds: list[dict] = []
    while True:
        r = _bk_get_with_retries(url, token, params=params)
        data = r.json()
        page = data if isinstance(data, list) else [data]
        all_builds.extend(page)
        link = r.headers.get("Link") or r.headers.get("link")
        links = parse_link_header(link)
        next_url = links.get("next")
        if not next_url:
            break
        url = next_url
        params = {}
        time.sleep(max(0.0, float(os.environ.get("BUILDKITE_BUILDS_PAGE_SLEEP", "0.12"))))
    return all_builds


def fetch_build_with_jobs(token: str, pipeline_slug: str, build_number: int | str) -> dict:
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{pipeline_slug}/builds/{build_number}"
    r = _bk_get_with_retries(url, token)
    out = r.json()
    if not isinstance(out, dict):
        raise ValueError("unexpected Buildkite JSON for single build")
    return out


def ensure_build_with_jobs(token: str, pipeline_slug: str, build: dict) -> dict:
    if build.get("jobs"):
        return build
    num = build.get("number")
    if num is None:
        return build
    return fetch_build_with_jobs(token, pipeline_slug, num)


# ── Resource pool / queue extraction ─────────────────────────────────────


def extract_queue_from_job(job: dict) -> str:
    rules = job.get("agent_query_rules") or []
    for rule in rules:
        if isinstance(rule, dict):
            query = (rule.get("query") or "").strip()
            if (rule.get("rule") or "").lower() == "include" and query.startswith("queue="):
                return query[len("queue=") :]
        elif isinstance(rule, str):
            m = re.match(r"^queue=(.+)$", rule.strip(), re.IGNORECASE)
            if m:
                return m.group(1)
    q = (job.get("queue") or "").strip()
    if q:
        return q
    return "default"


# ── Statistics accumulation ──────────────────────────────────────────────


@dataclass
class HourBucket:
    """Accumulator for a single hour bucket within a pool."""

    job_count: int = 0
    wait_seconds: list[float] = field(default_factory=list)
    duration_seconds: list[float] = field(default_factory=list)


@dataclass
class PoolStats:
    """Accumulator for per-pool statistics."""

    pipeline: str = ""
    pool_name: str = ""
    job_count: int = 0
    wait_seconds: list[float] = field(default_factory=list)
    duration_seconds: list[float] = field(default_factory=list)
    # Hourly time-series: hour (0-23) -> HourBucket
    hourly: dict[int, HourBucket] = field(default_factory=lambda: defaultdict(HourBucket))


@dataclass
class BuildEntry:
    """Per-build record: a single build's pool usage breakdown.

    Stores the per-pool job count for the build plus identifying metadata
    (branch, message, state) so the "latest build" view can render
    everything from this object without re-fetching.
    """

    pipeline: str
    number: int
    branch: str
    message: str
    state: str
    pool_job_counts: dict[str, int] = field(default_factory=dict)

    @property
    def distinct_pool_count(self) -> int:
        return len(self.pool_job_counts)

    @property
    def distinct_pools(self) -> set[str]:
        return set(self.pool_job_counts.keys())


@dataclass
class CategoryStats:
    """Aggregated stats for a build category (ready / merge / nightly / weekly)."""

    category: str
    label: str
    builds: list[BuildEntry] = field(default_factory=list)

    @property
    def build_count(self) -> int:
        return len(self.builds)

    @property
    def distinct_pool_counts(self) -> list[int]:
        return [b.distinct_pool_count for b in self.builds]


def classify_build(branch: str, message: str) -> str | None:
    """Classify a Buildkite build by its branch and trigger message.

    - "ready":   non-main branch (PR / fork runs)
    - "merge":   main branch, not a scheduled nightly/weekly build
    - "nightly": main branch, scheduled nightly build
    - "weekly":  main branch, scheduled weekly build
    """
    branch = (branch or "").strip()
    msg_lower = (message or "").lower()
    if branch != "main":
        return "ready"
    if "weekly" in msg_lower and "schedule" in msg_lower:
        return "weekly"
    if "nightly" in msg_lower and "schedule" in msg_lower:
        return "nightly"
    return "merge"


# ── Build/job state filtering ────────────────────────────────────────────

# Build states considered "finished" (terminal). Anything in this set is
# a build that has stopped progressing (regardless of pass/fail).
FINISHED_BUILD_STATES = frozenset(
    {
        "passed",
        "failed",
        "canceled",
        "blocked",
        "skipped",
        "failing",
        "not_run",
        "broken",
        "timed_out",
    }
)

# Job states considered "ran" (the job actually executed on an agent).
# Excludes scheduled/assigned/running/skipped/not_run/blocked.
RAN_JOB_STATES = frozenset(
    {
        "passed",
        "failed",
        "canceled",
        "broken",
        "timed_out",
    }
)


def is_finished_build(state: str | None) -> bool:
    return (state or "").strip().lower() in FINISHED_BUILD_STATES


def is_ran_job(state: str | None) -> bool:
    return (state or "").strip().lower() in RAN_JOB_STATES


def compute_pool_stats(
    token: str,
    pipeline_slug: str,
    created_from: str,
    created_to: str,
    *,
    verbose: bool = False,
) -> tuple[dict[str, PoolStats], dict[str, CategoryStats]]:
    """Fetch builds for a pipeline in the date range, extract job timing data,
    and return:
      - pools:          dict of pool_name -> PoolStats (per-pool aggregates + hourly)
      - category_stats: dict of category -> CategoryStats (per-build distinct pool counts)
    """
    from_utc, to_utc = cst_day_to_utc_window(created_from)
    _, to_utc_full = cst_day_to_utc_window(created_to)
    print(
        f"Fetching {ORG_SLUG}/{pipeline_slug} builds {created_from} ~ {created_to} CST "
        f"(UTC {from_utc} ~ {to_utc_full})..."
    )
    builds = fetch_builds(token, pipeline_slug, created_from, created_to)
    print(f"Fetched {len(builds)} build(s) for {pipeline_slug}.")

    pools: dict[str, PoolStats] = {}
    category_stats: dict[str, CategoryStats] = {
        "ready": CategoryStats(category="ready", label="ready CI"),
        "merge": CategoryStats(category="merge", label="merge"),
        "nightly": CategoryStats(category="nightly", label="nightly"),
        "weekly": CategoryStats(category="weekly", label="weekly"),
    }

    for b in builds:
        b = ensure_build_with_jobs(token, pipeline_slug, b)
        jobs = b.get("jobs") or []
        if verbose:
            bnum = b.get("number", "?")
            bstate = (b.get("state") or "").strip()
            print(f"  Build #{bnum} state={bstate} jobs={len(jobs)}")

        # Track distinct pools + per-pool job counts for this build
        # (used by the "latest build by category" view).
        distinct_pools: set[str] = set()
        pool_job_counts: dict[str, int] = {}
        script_job_count = 0

        for j in jobs:
            jtype = (j.get("type") or "").strip().lower()
            if jtype not in ("script", "command"):
                continue

            # Only count jobs that actually ran on an agent
            # (exclude scheduled/assigned/running/skipped/not_run/blocked).
            jstate = (j.get("state") or "").strip().lower()
            if not is_ran_job(jstate):
                continue

            script_job_count += 1
            scheduled_at = parse_buildkite_time(j.get("scheduled_at"))
            started_at = parse_buildkite_time(j.get("started_at"))
            finished_at = parse_buildkite_time(j.get("finished_at"))

            pool_name = extract_queue_from_job(j)
            distinct_pools.add(pool_name)
            pool_job_counts[pool_name] = pool_job_counts.get(pool_name, 0) + 1

            if pool_name not in pools:
                pools[pool_name] = PoolStats(pipeline=pipeline_slug, pool_name=pool_name)
            ps = pools[pool_name]
            ps.job_count += 1

            # Determine the hour bucket from scheduled_at (the time the job entered the queue)
            hour: int | None = None
            if scheduled_at is not None:
                hour = _hour_key(scheduled_at)

            # Queue wait time
            if scheduled_at is not None and started_at is not None:
                wait = (started_at - scheduled_at).total_seconds()
                if wait >= 0:
                    ps.wait_seconds.append(wait)
                    if hour is not None:
                        ps.hourly[hour].wait_seconds.append(wait)

            # Job duration
            if started_at is not None and finished_at is not None:
                dur = (finished_at - started_at).total_seconds()
                if dur >= 0:
                    ps.duration_seconds.append(dur)
                    if hour is not None:
                        ps.hourly[hour].duration_seconds.append(dur)

            if hour is not None:
                ps.hourly[hour].job_count += 1

        # Classify this build for the category view: only finished builds
        # with at least one ran script job qualify.
        if script_job_count > 0:
            branch = (b.get("branch") or "").strip()
            message = (b.get("message") or "").strip()
            state = (b.get("state") or "").strip()
            if is_finished_build(state):
                cat = classify_build(branch, message)
                if cat is not None:
                    try:
                        bnum_int = int(b.get("number") or 0)
                    except (TypeError, ValueError):
                        bnum_int = 0
                    category_stats[cat].builds.append(
                        BuildEntry(
                            pipeline=pipeline_slug,
                            number=bnum_int,
                            branch=branch,
                            message=message,
                            state=state,
                            pool_job_counts=dict(pool_job_counts),
                        )
                    )

    return pools, category_stats


# ── Compute aggregate summary cards ─────────────────────────────────────


def _compute_summary_cards(all_pools: dict[str, dict[str, PoolStats]]) -> list[dict]:
    total_jobs = 0
    total_pools = 0
    all_waits: list[float] = []
    total_occ = 0.0

    for pipeline_slug, pools in all_pools.items():
        total_pools += len(pools)
        for ps in pools.values():
            total_jobs += ps.job_count
            all_waits.extend(ps.wait_seconds)
            total_occ += sum(ps.duration_seconds)

    avg_wait = (sum(all_waits) / len(all_waits)) if all_waits else 0.0

    return [
        {
            "title": "Total Jobs",
            "value": str(total_jobs),
            "detail": f"across {total_pools} pool(s)",
            "icon": ICON_CHART,
        },
        {
            "title": "Avg Queue Wait",
            "value": format_duration(avg_wait) if all_waits else "N/A",
            "detail": f"{len(all_waits)} job(s) with wait data",
            "icon": ICON_CLOCK,
        },
        {
            "title": "Total Occupancy",
            "value": format_duration(total_occ) if total_occ else "N/A",
            "detail": "sum of all job runtimes",
            "icon": ICON_SERVER,
        },
        {
            "title": "Resource Pools",
            "value": str(total_pools),
            "detail": f"{', '.join(all_pools.keys())}",
            "icon": ICON_SERVER,
        },
    ]


# ── Inline SVG chart generation ─────────────────────────────────────────


def _render_trend_svg(
    title: str,
    y_label: str,
    series: list[tuple[str, str, list[float | None]]],
    hours: list[int],
    y_unit: str = "s",
    width: int = 860,
    height: int = 280,
    pad_left: int = 60,
    pad_right: int = 20,
    pad_top: int = 30,
    pad_bottom: int = 45,
) -> str:
    """
    Render an inline SVG line chart.

    series: list of (pool_name, color, values_per_hour). values_per_hour[h]
            may be None for hours with no data.
    hours:  list of hour indices (0-23) shown on the x-axis.
    y_unit: "s" for seconds, "" for count.
    """
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom

    # Find y range across all series
    all_vals = [v for _, _, vals in series for v in vals if v is not None]
    if not all_vals:
        return f'<p class="na">No data for {html.escape(title)}.</p>'
    y_max = max(all_vals) * 1.15  # 15% headroom
    y_min = 0.0
    if y_max == y_min:
        y_max = 1.0

    n_hours = len(hours)
    if n_hours == 0:
        return f'<p class="na">No data for {html.escape(title)}.</p>'

    def x_pos(i: int) -> float:
        return pad_left + (i / max(n_hours - 1, 1)) * chart_w

    def y_pos(val: float) -> float:
        if y_max == y_min:
            return pad_top + chart_h / 2
        return pad_top + chart_h - ((val - y_min) / (y_max - y_min)) * chart_h

    # Grid lines (5 horizontal)
    grid_lines = ""
    n_grid = 5
    for gi in range(n_grid + 1):
        gy_val = y_min + (y_max - y_min) * gi / n_grid
        gy = y_pos(gy_val)
        label = f"{gy_val:.0f}{y_unit}" if gy_val < 3600 else format_duration(gy_val)
        grid_lines += (
            f'<line x1="{pad_left:.0f}" y1="{gy:.0f}" '
            f'x2="{width - pad_right:.0f}" y2="{gy:.0f}" '
            f'stroke="var(--dashboard-chart-grid)" stroke-width="1"/>\n'
            f'<text x="{pad_left - 5:.0f}" y="{gy + 4:.0f}" '
            f'text-anchor="end" font-size="11" fill="var(--dashboard-chart-text)">'
            f"{html.escape(label)}</text>\n"
        )

    # X-axis hour labels
    x_labels = ""
    for i, h in enumerate(hours):
        xp = x_pos(i)
        x_labels += (
            f'<text x="{xp:.0f}" y="{height - pad_bottom + 18:.0f}" '
            f'text-anchor="middle" font-size="11" fill="var(--dashboard-chart-text)">'
            f"{h:02d}:00</text>\n"
        )
    # Tick marks on x axis
    x_ticks = ""
    for i, h in enumerate(hours):
        xp = x_pos(i)
        x_ticks += (
            f'<line x1="{xp:.0f}" y1="{pad_top + chart_h:.0f}" '
            f'x2="{xp:.0f}" y2="{pad_top + chart_h + 6:.0f}" '
            f'stroke="var(--dashboard-chart-text)" stroke-width="1"/>\n'
        )

    # Series lines
    series_svg = ""
    for pool_name, color, vals in series:
        # Build polyline points
        points_parts: list[str] = []
        has_any = False
        for i, h in enumerate(hours):
            v = vals[i] if i < len(vals) else None
            if v is not None:
                xp = x_pos(i)
                yp = y_pos(v)
                points_parts.append(f"{xp:.1f},{yp:.1f}")
                has_any = True
        if not has_any or len(points_parts) < 2:
            continue

        polyline_pts = " ".join(points_parts)

        # Area fill (semi-transparent)
        area_pts = polyline_pts
        # Close the area polygon down to the x-axis
        first_x = x_pos(hours.index(next(h for h, idx in zip(hours, range(len(hours))) if vals[idx] is not None)))
        last_x = x_pos(
            hours.index(next(h for h, idx in reversed(list(zip(hours, range(len(hours))))) if vals[idx] is not None))
        )
        area_pts += f" {last_x:.1f},{y_pos(0):.1f} {first_x:.1f},{y_pos(0):.1f}"

        series_svg += (
            f'<polygon points="{area_pts}" '
            f'fill="{color}" fill-opacity="0.12" stroke="none"/>\n'
            f'<polyline points="{polyline_pts}" '
            f'fill="none" stroke="{color}" stroke-width="2.5" '
            f'stroke-linejoin="round" stroke-linecap="round"/>\n'
        )
        # Data point dots
        for i, h in enumerate(hours):
            v = vals[i] if i < len(vals) else None
            if v is not None:
                xp = x_pos(i)
                yp = y_pos(v)
                series_svg += (
                    f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="3.5" fill="{color}" stroke="{color}" stroke-width="1"/>\n'
                )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{html.escape(title)}">\n'
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="var(--dashboard-panel-bg)" rx="8"/>\n'
        # Title
        f'<text x="{pad_left:.0f}" y="{18:.0f}" '
        f'font-size="13" font-weight="700" fill="var(--dashboard-text)">'
        f"{html.escape(title)}</text>\n"
        # Y-axis label
        f'<text x="{4:.0f}" y="{pad_top + chart_h / 2:.0f}" '
        f'font-size="11" fill="var(--dashboard-chart-text)">'
        f"{html.escape(y_label)}</text>\n"
        # Grid
        + grid_lines
        # X axis baseline
        + f'<line x1="{pad_left:.0f}" y1="{pad_top + chart_h:.0f}" '
        f'x2="{width - pad_right:.0f}" y2="{pad_top + chart_h:.0f}" '
        f'stroke="var(--dashboard-chart-text)" stroke-width="1.5"/>\n'
        # Y axis line
         + f'<line x1="{pad_left:.0f}" y1="{pad_top:.0f}" '
        f'x2="{pad_left:.0f}" y2="{pad_top + chart_h:.0f}" '
        f'stroke="var(--dashboard-chart-text)" stroke-width="1.5"/>\n' + x_ticks + x_labels + series_svg + "</svg>"
    )
    return svg


def _build_hourly_series(
    all_pools: dict[str, dict[str, PoolStats]],
    metric: str,  # "avg_wait" or "avg_duration" or "job_count"
    hours: list[int],
) -> list[tuple[str, str, list[float | None]]]:
    """Build per-pool time series for a given metric over the 24 hours."""
    # Assign colors to pool names across all pipelines
    pool_color_map: dict[str, str] = {}
    color_idx = 0
    for pipeline_slug in sorted(all_pools.keys()):
        for pool_name in sorted(all_pools[pipeline_slug].keys()):
            key = f"{pipeline_slug}/{pool_name}"
            pool_color_map[key] = POOL_COLORS[color_idx % len(POOL_COLORS)]
            color_idx += 1

    series: list[tuple[str, str, list[float | None]]] = []
    for pipeline_slug in sorted(all_pools.keys()):
        pools = all_pools[pipeline_slug]
        for pool_name in sorted(pools.keys()):
            ps = pools[pool_name]
            key = f"{pipeline_slug}/{pool_name}"
            color = pool_color_map[key]

            vals: list[float | None] = []
            for h in hours:
                hb = ps.hourly.get(h)
                if hb is None or hb.job_count == 0:
                    vals.append(None)
                elif metric == "avg_wait":
                    if hb.wait_seconds:
                        vals.append(sum(hb.wait_seconds) / len(hb.wait_seconds))
                    else:
                        vals.append(None)
                elif metric == "avg_duration":
                    if hb.duration_seconds:
                        vals.append(sum(hb.duration_seconds) / len(hb.duration_seconds))
                    else:
                        vals.append(None)
                elif metric == "job_count":
                    vals.append(float(hb.job_count))
                else:
                    vals.append(None)

            series.append((key, color, vals))

    return series


def _charts_html(
    all_pools: dict[str, dict[str, PoolStats]],
    date_from: str,
) -> str:
    """Render the trend charts section: one SVG per metric, shared legend."""
    hours = list(range(24))  # 0..23

    wait_series = _build_hourly_series(all_pools, "avg_wait", hours)
    count_series = _build_hourly_series(all_pools, "job_count", hours)

    # Check if any series has data
    has_wait = any(v is not None for _, _, vals in wait_series for v in vals)
    has_count = any(v is not None for _, _, vals in count_series for v in vals)

    if not (has_wait or has_count):
        return '<p class="na">No hourly data available for trend charts.</p>'

    # Pool color legend
    pool_color_map: dict[str, str] = {}
    color_idx = 0
    for pipeline_slug in sorted(all_pools.keys()):
        for pool_name in sorted(all_pools[pipeline_slug].keys()):
            key = f"{pipeline_slug}/{pool_name}"
            pool_color_map[key] = POOL_COLORS[color_idx % len(POOL_COLORS)]
            color_idx += 1

    legend_items = ""
    for key, color in pool_color_map.items():
        legend_items += (
            f'<span class="chart-legend-item">'
            f'<span class="chart-legend-swatch" style="background:{color}"></span>'
            f"{html.escape(key)}</span>"
        )

    parts: list[str] = []

    if has_wait:
        parts.append(
            '<div class="chart-group-title">Avg Queue Wait per Hour (CST, UTC+8)</div>\n'
            '<div class="chart-container">\n'
            + _render_trend_svg(
                title=f"Avg Queue Wait — {date_from}",
                y_label="Wait (s)",
                series=wait_series,
                hours=hours,
                y_unit="s",
            )
            + "\n</div>"
        )

    if has_count:
        parts.append(
            '<div class="chart-group-title">Job Count per Hour (CST, UTC+8)</div>\n'
            '<div class="chart-container">\n'
            + _render_trend_svg(
                title=f"Job Count — {date_from}",
                y_label="Jobs",
                series=count_series,
                hours=hours,
                y_unit="",
            )
            + "\n</div>"
        )

    return "\n".join(parts) + f'\n<div class="chart-legend-row">{legend_items}</div>'


# ── HTML output ──────────────────────────────────────────────────────────


def _pool_row_html(ps: PoolStats) -> str:
    if ps.wait_seconds:
        sorted_w = sorted(ps.wait_seconds)
        avg_w = sum(ps.wait_seconds) / len(ps.wait_seconds)
        max_w = sorted_w[-1]
        p50_w = percentile(sorted_w, 50)
        p90_w = percentile(sorted_w, 90)
        avg_wait = format_duration(avg_w)
        max_wait = format_duration(max_w)
        p50_wait = format_duration(p50_w) if p50_w is not None else "N/A"
        p90_wait = format_duration(p90_w) if p90_w is not None else "N/A"
        total_wait = format_duration(sum(ps.wait_seconds))
    else:
        avg_wait = max_wait = p50_wait = p90_wait = total_wait = "N/A"

    if ps.duration_seconds:
        avg_d = sum(ps.duration_seconds) / len(ps.duration_seconds)
        avg_dur = format_duration(avg_d)
        total_occ = format_duration(sum(ps.duration_seconds))
    else:
        avg_dur = total_occ = "N/A"

    def _td(val: str, cls: str = "num") -> str:
        if val == "N/A":
            return f'<td class="{cls} na">{html.escape(val)}</td>'
        return f'<td class="{cls}">{html.escape(val)}</td>'

    return (
        f"<tr>"
        f'<td class="pipeline-cell">{html.escape(ps.pipeline)}</td>'
        f'<td class="pool-name">{html.escape(ps.pool_name)}</td>'
        f"{_td(str(ps.job_count))}"
        f"{_td(avg_wait)}"
        f"{_td(max_wait)}"
        f"{_td(p50_wait)}"
        f"{_td(p90_wait)}"
        f"{_td(avg_dur)}"
        f"{_td(total_occ)}"
        f"{_td(total_wait)}"
        f"</tr>"
    )


def _render_latest_builds_by_category_html(
    all_categories: dict[str, CategoryStats],
    pipeline_order: list[str],
) -> str:
    """Render one card per CI category (ready / merge / nightly / weekly).

    Inside each card, list sub-cards (one per pipeline that has any build in
    the category) showing the **most recent** build in that category, with
    its per-pool job counts.
    """
    order = [
        ("ready", "ready CI", "non-main branch", "cat-card--ready", "#3b82f6"),
        ("merge", "merge", "main · not scheduled", "cat-card--merge", "#1f9d63"),
        ("nightly", "nightly", "main · scheduled nightly", "cat-card--nightly", "#d97706"),
        ("weekly", "weekly", "main · scheduled weekly", "cat-card--weekly", "#ef4444"),
    ]

    cards_parts: list[str] = []
    for key, label, sub, color_class, _ in order:
        cs = all_categories.get(key)
        if cs is None or cs.build_count == 0:
            continue

        # Pick the latest build (by build number) per pipeline
        latest_per_pipeline: dict[str, BuildEntry] = {}
        for b in cs.builds:
            cur = latest_per_pipeline.get(b.pipeline)
            if cur is None or b.number > cur.number:
                latest_per_pipeline[b.pipeline] = b

        # Render sub-cards in pipeline_order (skip pipelines with no build)
        subcard_parts: list[str] = []
        for pipeline in pipeline_order:
            b = latest_per_pipeline.get(pipeline)
            if b is None:
                continue

            # Truncate branch for display
            branch_full = b.branch
            branch_disp = branch_full if len(branch_full) <= 34 else branch_full[:32] + "…"

            # Sort pools by job count desc, name asc for ties
            sorted_pools = sorted(
                b.pool_job_counts.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
            pool_rows = "".join(
                f'<li class="latest-pool-row">'
                f'<span class="latest-pool-name">{html.escape(name)}</span>'
                f'<span class="latest-pool-count">{count}</span>'
                f"</li>"
                for name, count in sorted_pools
            )
            if not pool_rows:
                pool_rows = '<li class="latest-pool-row latest-pool-empty"><span>no script jobs</span></li>'

            # First line of message, truncated (commit messages can be long)
            msg_first_line = (b.message or "").splitlines()[0] if b.message else ""
            if len(msg_first_line) > 60:
                msg_first_line = msg_first_line[:58] + "…"

            subcard_parts.append(
                f'<div class="latest-subcard">'
                f'<div class="latest-subcard-head">'
                f'<span class="latest-subcard-pipeline">{html.escape(b.pipeline)}</span>'
                f'<span class="latest-subcard-num">#{b.number}</span>'
                f"</div>"
                f'<div class="latest-subcard-meta">'
                f'<span class="latest-subcard-branch" '
                f'title="{html.escape(branch_full)}">{html.escape(branch_disp)}</span>'
                f'<span class="latest-subcard-state">{html.escape(b.state or "?")}</span>'
                f"</div>"
                + (
                    f'<div class="latest-subcard-msg" '
                    f'title="{html.escape(b.message)}">{html.escape(msg_first_line)}</div>'
                    if msg_first_line
                    else ""
                )
                + f'<ul class="latest-pool-list">{pool_rows}</ul>'
                f"</div>"
            )

        if not subcard_parts:
            continue

        cards_parts.append(
            f'<div class="cat-card {color_class}">'
            f'<div class="cat-card-head">'
            f'<span class="cat-card-label">{html.escape(label)}'
            f'<span class="cat-card-sub">— {html.escape(sub)}</span></span>'
            f'<span class="cat-card-count">'
            f"{cs.build_count} total build{'s' if cs.build_count != 1 else ''}</span>"
            f"</div>"
            f'<div class="latest-subcards">{"".join(subcard_parts)}</div>'
            f"</div>"
        )

    if not cards_parts:
        return ""

    return (
        '<div class="cat-stats">'
        '<h3 class="cat-stats-title">Latest Build by CI Category — Pool Job Counts</h3>'
        f'<div class="cat-stats-grid">{"".join(cards_parts)}</div>'
        "</div>"
    )


def _summary_cards_html(cards: list[dict]) -> str:
    parts = []
    for c in cards:
        parts.append(
            f'<div class="focus-card focus-card--ci">\n'
            f'  <div class="focus-card-title">{c["icon"]} {html.escape(c["title"])}</div>\n'
            f'  <div class="focus-card-value">{html.escape(c["value"])}</div>\n'
            f'  <div class="focus-card-detail">{html.escape(c["detail"])}</div>\n'
            f"</div>"
        )
    return '<div class="focus-card-grid">\n' + "\n".join(parts) + "\n</div>"


def format_stats_html(
    all_pools: dict[str, dict[str, PoolStats]],
    all_categories: dict[str, CategoryStats],
    date_from: str,
    date_to: str,
) -> str:
    """Render all pool stats as a self-contained HTML page with trend charts."""

    from_utc, _ = cst_day_to_utc_window(date_from)
    _, to_utc = cst_day_to_utc_window(date_to)

    cards = _compute_summary_cards(all_pools)

    rows_parts = []
    for pipeline_slug in sorted(all_pools.keys()):
        pools = all_pools[pipeline_slug]
        for pool_name in sorted(pools.keys()):
            rows_parts.append(_pool_row_html(pools[pool_name]))

    rows_html = (
        "\n".join(rows_parts)
        if rows_parts
        else ('<tr><td colspan="10" class="na">No builds found in the specified date range.</td></tr>')
    )

    table_html = (
        '<div class="table-scroll">\n'
        '<table class="pool-stats">\n'
        "<thead>\n<tr>\n"
        "  <th>Pipeline</th>\n"
        "  <th>Resource Pool</th>\n"
        "  <th>Jobs</th>\n"
        "  <th>Avg Wait</th>\n"
        "  <th>Max Wait</th>\n"
        "  <th>P50 Wait</th>\n"
        "  <th>P90 Wait</th>\n"
        "  <th>Avg Duration</th>\n"
        "  <th>Total Occupancy</th>\n"
        "  <th>Total Wait</th>\n"
        "</tr>\n</thead>\n"
        "<tbody>\n" + rows_html + "\n"
        "</tbody>\n</table>\n</div>"
    )

    legend_html = (
        '<div class="legend">\n'
        "<dl>\n"
        "  <dt>Wait</dt>\n"
        "  <dd>Time a job spent in the queue before an agent picked it up "
        "(<code>started_at − scheduled_at</code>).</dd>\n"
        "  <dt>Duration</dt>\n"
        "  <dd>Time a job spent running on the agent "
        "(<code>finished_at − started_at</code>).</dd>\n"
        "  <dt>Occupancy</dt>\n"
        "  <dd>Total running time across all jobs in the pool (sum of durations).</dd>\n"
        "  <dt>Total Wait</dt>\n"
        "  <dd>Total queue time across all jobs in the pool (sum of wait times).</dd>\n"
        "  <dt>Resource Pool</dt>\n"
        "  <dd>Derived from each job's <code>agent_query_rules</code> "
        "(<code>queue=…</code>); jobs without an explicit queue go into "
        "<code>default</code>.</dd>\n"
        "</dl>\n</div>"
    )

    # Trend charts
    charts_html = _charts_html(all_pools, date_from)

    page = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>CI Resource Pool Statistics — {html.escape(date_from)} ~ {html.escape(date_to)} CST</title>\n"
        "<style>\n" + POOL_STATS_CSS + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        '<header class="top-bar">\n'
        '<div class="top-bar-inner">\n'
        '<div class="brand">\n'
        f'  <div class="brand-mark">{ICON_SERVER}</div>\n'
        '  <div class="brand-copy">\n'
        f"    <h1>CI Resource Pool Statistics</h1>\n"
        f'    <p class="tagline">{html.escape(date_from)} — {html.escape(date_to)} CST (UTC+8)</p>\n'
        "  </div>\n"
        "</div>\n"
        "</div>\n"
        "</header>\n"
        '<div class="shell">\n' + _summary_cards_html(cards) + "\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_CHART}</span>'
        f" Per-Pool Detail</span></h2>\n"
        + _render_latest_builds_by_category_html(all_categories, list(all_pools.keys()))
        + "\n"
        + table_html
        + "\n"
        + legend_html
        + "\n"
        "</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_TREND}</span>'
        f" Hourly Trends (CST, UTC+8)</span></h2>\n" + charts_html + "\n"
        "</div>\n"
        f'<p class="meta">Source: <code>scripts/resource_pool_stats.py</code>; '
        f"pipelines: {html.escape(', '.join(all_pools.keys()))}; "
        f"window: <code>{html.escape(date_from)}</code> — "
        f"<code>{html.escape(date_to)}</code> CST (UTC+8; "
        f"maps to <code>{from_utc}</code> — <code>{to_utc}</code> UTC).</p>\n"
        "</div>\n"
        "</body>\n</html>"
    )

    return page


# ── Markdown & JSON output ──────────────────────────────────────────────


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    def sep() -> str:
        return "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    lines = [fmt(headers), sep()]
    for row in rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def format_stats_markdown(all_pools: dict[str, dict[str, PoolStats]], date_from: str, date_to: str) -> str:
    lines: list[str] = []
    lines.append(f"# CI Resource Pool Statistics ({date_from} ~ {date_to} CST, UTC+8)")
    lines.append("")
    lines.append(
        f"Source: `scripts/resource_pool_stats.py`; "
        f"pipelines: {', '.join(all_pools.keys())}; "
        f"window: `{date_from}` — `{date_to}` CST (UTC+8)."
    )
    lines.append("")
    headers = [
        "Pipeline",
        "Resource Pool",
        "Jobs",
        "Avg Wait",
        "Max Wait",
        "P50 Wait",
        "P90 Wait",
        "Avg Duration",
        "Total Occupancy",
        "Total Wait",
    ]
    rows: list[list[str]] = []
    for pipeline_slug, pools in all_pools.items():
        for pool_name in sorted(pools.keys()):
            ps = pools[pool_name]
            if ps.wait_seconds:
                sorted_w = sorted(ps.wait_seconds)
                avg_wait_str = format_duration(sum(ps.wait_seconds) / len(ps.wait_seconds))
                max_wait_str = format_duration(sorted_w[-1])
                p50_wait_str = format_duration(percentile(sorted_w, 50)) if percentile(sorted_w, 50) else "N/A"
                p90_wait_str = format_duration(percentile(sorted_w, 90)) if percentile(sorted_w, 90) else "N/A"
                total_wait_str = format_duration(sum(ps.wait_seconds))
            else:
                avg_wait_str = max_wait_str = p50_wait_str = p90_wait_str = total_wait_str = "N/A"
            if ps.duration_seconds:
                avg_dur_str = format_duration(sum(ps.duration_seconds) / len(ps.duration_seconds))
                total_occ_str = format_duration(sum(ps.duration_seconds))
            else:
                avg_dur_str = total_occ_str = "N/A"
            rows.append(
                [
                    pipeline_slug,
                    pool_name,
                    str(ps.job_count),
                    avg_wait_str,
                    max_wait_str,
                    p50_wait_str,
                    p90_wait_str,
                    avg_dur_str,
                    total_occ_str,
                    total_wait_str,
                ]
            )
    if not rows:
        lines.append("*No builds found in the specified date range.*")
    else:
        lines.append(_render_markdown_table(headers, rows))
    lines.append("")
    lines.append("**Legend:**")
    lines.append("- **Wait**: queue time (`started_at - scheduled_at`).")
    lines.append("- **Duration**: runtime (`finished_at - started_at`).")
    lines.append("- **Occupancy**: sum of durations per pool.")
    lines.append("- **Total Wait**: sum of wait times per pool.")
    return "\n".join(lines)


def format_stats_json(all_pools: dict[str, dict[str, PoolStats]], date_from: str, date_to: str) -> str:
    output: dict = {"date_range": {"from": date_from, "to": date_to}, "pipelines": {}}
    for pipeline_slug, pools in all_pools.items():
        pipeline_data: dict = {}
        for pool_name in sorted(pools.keys()):
            ps = pools[pool_name]
            pool_data: dict = {
                "pool_name": pool_name,
                "job_count": ps.job_count,
                "wait_time": {},
                "duration": {},
                "hourly": {},
            }
            if ps.wait_seconds:
                sorted_w = sorted(ps.wait_seconds)
                pool_data["wait_time"] = {
                    "avg_seconds": round(sum(ps.wait_seconds) / len(ps.wait_seconds), 2),
                    "max_seconds": round(sorted_w[-1], 2),
                    "p50_seconds": round(percentile(sorted_w, 50) or 0, 2),
                    "p90_seconds": round(percentile(sorted_w, 90) or 0, 2),
                    "total_seconds": round(sum(ps.wait_seconds), 2),
                    "count": len(ps.wait_seconds),
                }
            if ps.duration_seconds:
                pool_data["duration"] = {
                    "avg_seconds": round(sum(ps.duration_seconds) / len(ps.duration_seconds), 2),
                    "total_seconds": round(sum(ps.duration_seconds), 2),
                    "count": len(ps.duration_seconds),
                }
            # Hourly time series
            hourly_data: dict = {}
            for h in range(24):
                hb = ps.hourly.get(h)
                if hb and hb.job_count > 0:
                    hourly_data[str(h)] = {
                        "job_count": hb.job_count,
                        "avg_wait_seconds": round(sum(hb.wait_seconds) / len(hb.wait_seconds), 2)
                        if hb.wait_seconds
                        else None,
                        "avg_duration_seconds": round(sum(hb.duration_seconds) / len(hb.duration_seconds), 2)
                        if hb.duration_seconds
                        else None,
                    }
            if hourly_data:
                pool_data["hourly"] = hourly_data
            pipeline_data[pool_name] = pool_data
        output["pipelines"][pipeline_slug] = pipeline_data
    return json.dumps(output, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Buildkite builds for vllm-omni pipelines and compute "
        "per-resource-pool queue wait time and occupancy statistics. "
        "Default output is an HTML file with trend charts."
    )
    parser.add_argument(
        "--from",
        dest="created_from",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date (CST calendar date, inclusive). Omit both --from and --to to use yesterday CST.",
    )
    parser.add_argument(
        "--to",
        dest="created_to",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (CST calendar date, inclusive). Omit both --from and --to to use yesterday CST.",
    )
    parser.add_argument(
        "--pipeline",
        dest="pipelines",
        default=None,
        metavar="SLUG1,SLUG2",
        help=f"Comma-separated pipeline slugs. Default: {','.join(DEFAULT_PIPELINES)}",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="html",
        choices=["html", "markdown", "json"],
        help="Output format. Default: html (writes a file with trend charts).",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        metavar="PATH",
        help="Output file path for HTML. Default: pool-stats-YYYY-MM-DD.html.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each build's job count and state.")
    parser.add_argument(
        "--category-lookback-days",
        dest="category_lookback_days",
        type=int,
        default=7,
        help="Lookback window (in days) for the 'latest build by CI category' view. "
        "Default: 7. The per-pool table and hourly trends always use --from/--to.",
    )
    args = parser.parse_args()

    if args.created_from is None and args.created_to is None:
        args.created_from, args.created_to = yesterday_range_cst()
    elif args.created_from is None or args.created_to is None:
        print(
            "resource_pool_stats.py: pass both --from and --to, or omit both (defaults to yesterday CST).",
            file=sys.stderr,
        )
        return 2

    pipeline_slugs: list[str]
    if args.pipelines:
        pipeline_slugs = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    else:
        pipeline_slugs = DEFAULT_PIPELINES

    token = get_api_token()
    if not token:
        print("BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is not set; cannot call the Buildkite API.", file=sys.stderr)
        print("Set one in the environment and retry.", file=sys.stderr)
        return 1

    all_pools: dict[str, dict[str, PoolStats]] = {}
    all_categories: dict[str, CategoryStats] = {
        "ready": CategoryStats(category="ready", label="ready CI"),
        "merge": CategoryStats(category="merge", label="merge"),
        "nightly": CategoryStats(category="nightly", label="nightly"),
        "weekly": CategoryStats(category="weekly", label="weekly"),
    }
    for pipeline_slug in pipeline_slugs:
        try:
            # Pass 1: today's data only — for the per-pool table and hourly trends.
            pools, _ = compute_pool_stats(
                token,
                pipeline_slug,
                args.created_from,
                args.created_to,
                verbose=args.verbose,
            )
            all_pools[pipeline_slug] = pools

            # Pass 2: wider lookback — for the "latest build by category" view
            # (so weekly/nightly that didn't run today still show up).
            lookback_days = max(1, args.category_lookback_days)
            lookback_from = (
                datetime.strptime(args.created_from, "%Y-%m-%d").date() - timedelta(days=lookback_days - 1)
            ).isoformat()
            if lookback_from != args.created_from:
                _, cats = compute_pool_stats(
                    token,
                    pipeline_slug,
                    lookback_from,
                    args.created_to,
                    verbose=False,
                )
                for cat_key, cat in cats.items():
                    all_categories[cat_key].builds.extend(cat.builds)
        except requests.RequestException as e:
            print(f"API request failed for {pipeline_slug}: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                print(f"HTTP status: {e.response.status_code}", file=sys.stderr)
                print(e.response.text[:500], file=sys.stderr)
            all_pools[pipeline_slug] = {}

    if args.output_format == "html":
        html_content = format_stats_html(all_pools, all_categories, args.created_from, args.created_to)
        if args.output_path:
            out_path = Path(args.output_path)
        else:
            out_path = Path(f"pool-stats-{args.created_from}.html")
        out_path.write_text(html_content, encoding="utf-8")
        print(f"HTML report written to {out_path}")
    elif args.output_format == "markdown":
        print(format_stats_markdown(all_pools, args.created_from, args.created_to))
    elif args.output_format == "json":
        print(format_stats_json(all_pools, args.created_from, args.created_to))

    return 0


if __name__ == "__main__":
    sys.exit(main())
