#!/usr/bin/env python3
"""
Fetch vllm-omni and vllm-omni-npu-ci builds from the Buildkite REST API for a
date range (default: today in CST / UTC+8) and compute per-resource-pool
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

If ``--from`` / ``--to`` are both omitted, the window is **today CST**
(00:00 to 23:59:59 CST). Scheduled runs (e.g. cron jobs) should pass both
explicitly to avoid pulling a partial day. If you pass one, pass both
(CST calendar dates, inclusive).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import yaml  # PyYAML — used for static .buildkite/test-*.yml parsing
except ImportError:
    yaml = None  # lazy-install handled inside _ensure_pyyaml()

# ── Buildkite API constants ──────────────────────────────────────────────

BUILDKITE_API_BASE = "https://api.buildkite.com/v2"
ORG_SLUG = "vllm"
DEFAULT_PIPELINES = ["vllm-omni", "vllm-omni-npu-ci"]

# ── Static YAML source (used by Per-Pool Detail section) ────────────────
#
# Per-Pool Detail is computed by `git pull`-ing the local vllm-omni git repo
# and statically parsing `.buildkite/test-*.yml`.  This avoids hitting the
# Buildkite API for that section — the YAML is the source of truth for which
# resource pools each pipeline category intends to use.

DEFAULT_LOCAL_REPO_PATH = "/home/wy/vllm-omni"
MIRROR_HARDWARES_REL_PATH = ".buildkite/common/ci_mirror_hardwares.yml"

# Mapping from Buildkite pipeline slug → list of (category, relative-yaml-path).
# One YAML file per category for the supported pipelines.  The same set of
# files is what `upload_pipeline.py --upload` consumes for that pipeline.
PIPELINE_YAML_MAP: dict[str, list[tuple[str, str]]] = {
    "vllm-omni": [
        ("ready", ".buildkite/cuda/test-ready.yml"),
        ("merge", ".buildkite/cuda/test-merge.yml"),
        ("nightly", ".buildkite/cuda/test-nightly.yml"),
        ("weekly", ".buildkite/cuda/test-weekly.yml"),
    ],
    "vllm-omni-npu-ci": [
        ("ready", ".buildkite/npu/test-npu-ready.yml"),
        ("nightly", ".buildkite/npu/test-npu-nightly.yml"),
    ],
}

CATEGORY_ORDER: list[tuple[str, str, str]] = [
    ("ready", "ready CI", "non-main branch (test-ready.yml)"),
    ("merge", "merge", "main · not scheduled (test-merge.yml)"),
    ("nightly", "nightly", "main · scheduled nightly (test-nightly.yml)"),
    ("weekly", "weekly", "main · scheduled weekly (test-weekly.yml)"),
]

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
table.pool-stats tr.summary-row td {
  font-weight: 760;
  border-top: 2px solid var(--dashboard-border);
  background: color-mix(in srgb, var(--surface-muted) 60%, var(--dashboard-panel-bg));
}
table.pool-stats tr.summary-row--h100 td {
  background: color-mix(in srgb, #ef4444 8%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
}
table.pool-stats tr.summary-row--gpu td {
  background: color-mix(in srgb, #3b82f6 8%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
}
table.pool-stats tr.summary-row td.pool-name {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.86rem;
}
.pool-queue {
  color: var(--dashboard-muted);
  font-weight: 500;
  font-size: 0.85em;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
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
.latest-pool-row--total {
  margin-top: 0.3rem;
  padding-top: 0.4rem;
  border-top: 1px solid var(--dashboard-border);
  border-top-style: solid;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.latest-total-name {
  flex: 0 0 auto;
}
.latest-pool-counts {
  display: inline-flex;
  gap: 0.3rem;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.latest-total-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.2rem;
  padding: 0.05rem 0.45rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  border: 1px solid transparent;
  white-space: nowrap;
}
.latest-total-chip--h100 {
  background: color-mix(in srgb, #ef4444 10%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
  border-color: color-mix(in srgb, #ef4444 25%, transparent);
}
.latest-total-chip--l4 {
  background: color-mix(in srgb, #3b82f6 10%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
  border-color: color-mix(in srgb, #3b82f6 25%, transparent);
}
.latest-total-chip--gpu {
  background: color-mix(in srgb, #3b82f6 10%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
  border-color: color-mix(in srgb, #3b82f6 25%, transparent);
}
.latest-total-chip--a2 {
  background: color-mix(in srgb, #d97706 10%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
  border-color: color-mix(in srgb, #d97706 25%, transparent);
}
.latest-total-chip--a3 {
  background: color-mix(in srgb, #1f9d63 10%, var(--dashboard-panel-bg));
  color: var(--dashboard-text);
  border-color: color-mix(in srgb, #1f9d63 25%, transparent);
}
.latest-total-chip strong {
  font-weight: 800;
  color: var(--dashboard-text);
}
.job-name-cell {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.82rem;
  max-width: 22rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.job-state-cell {
  font-weight: 700;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.18rem 0.5rem;
  border-radius: 4px;
  text-align: center;
  background: color-mix(in srgb, var(--dashboard-badge-bg) 60%, transparent);
  color: var(--dashboard-muted);
  border: 1px solid var(--dashboard-border);
  white-space: nowrap;
}
.job-state--passed   { color: #1f9d63; border-color: color-mix(in srgb, #1f9d63 35%, transparent); }
.job-state--failed   {
  color: #d14343;
  border-color: color-mix(in srgb, #d14343 35%, transparent);
  background: color-mix(in srgb, #d14343 8%, var(--dashboard-panel-bg));
}
.job-state--broken   {
  color: #d14343;
  border-color: color-mix(in srgb, #d14343 35%, transparent);
  background: color-mix(in srgb, #d14343 8%, var(--dashboard-panel-bg));
}
.job-state--timed_out{
  color: #d97706;
  border-color: color-mix(in srgb, #d97706 35%, transparent);
  background: color-mix(in srgb, #d97706 8%, var(--dashboard-panel-bg));
}
.job-state--canceled { color: var(--dashboard-muted); }
tr.job-row--failed td {
  background: color-mix(in srgb, #d14343 4%, var(--dashboard-panel-bg));
}

/* Job-Level Detail — per-pipeline sub-cards */
.job-level-cards {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  margin: 0.65rem 0 0;
}
.cat-card--pipeline {
  /* Inherits --ci border-left from .cat-card; kept as a semantic anchor
     so future pipeline-specific tweaks have a hook. */
}
.cat-card--pipeline > .cat-card-head {
  align-items: center;
  flex-wrap: wrap;
  gap: 0.55rem;
}
.job-level-filter {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin: 0.65rem 0 0.85rem;
  padding: 0.55rem 0.8rem;
  background: var(--surface-muted);
  border: 1px solid var(--dashboard-border);
  border-radius: var(--radius-sm);
}
/* Compact filter inside each pipeline sub-card (sits in the card head,
   right side, scoped to that card). */
.cat-card--pipeline .job-level-filter {
  margin: 0;
  padding: 0.22rem 0.5rem;
  background: transparent;
  border: 1px solid var(--dashboard-border);
  gap: 0.3rem;
}
.cat-card--pipeline .job-level-filter .filter-label {
  font-size: 0.68rem;
  margin-right: 0.05rem;
}
.cat-card--pipeline .filter-chip {
  padding: 0.1rem 0.45rem 0.1rem 0.35rem;
  font-size: 0.7rem;
}
.cat-card--pipeline .filter-chip input[type="checkbox"] {
  width: 0.75rem;
  height: 0.75rem;
}
.filter-label {
  font-size: 0.74rem;
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--dashboard-muted);
  margin-right: 0.25rem;
}
.filter-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.18rem 0.6rem 0.18rem 0.45rem;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 600;
  border: 1px solid transparent;
  cursor: pointer;
  user-select: none;
  transition: opacity 0.15s ease;
}
.filter-chip input[type="checkbox"] {
  margin: 0;
  width: 0.85rem;
  height: 0.85rem;
  accent-color: currentColor;
  cursor: pointer;
}
.filter-chip--ready   {
  background: color-mix(in srgb, #3b82f6 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #3b82f6 35%, transparent);
  color: var(--dashboard-text);
}
.filter-chip--merge   {
  background: color-mix(in srgb, #1f9d63 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #1f9d63 35%, transparent);
  color: var(--dashboard-text);
}
.filter-chip--nightly {
  background: color-mix(in srgb, #d97706 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #d97706 35%, transparent);
  color: var(--dashboard-text);
}
.filter-chip--weekly  {
  background: color-mix(in srgb, #ef4444 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #ef4444 35%, transparent);
  color: var(--dashboard-text);
}
.filter-chip:not(:has(input:checked)) {
  opacity: 0.5;
}
.filter-empty {
  padding: 0.6rem 0.9rem;
  font-size: 0.85rem;
  color: var(--dashboard-muted);
  font-style: italic;
  text-align: center;
}

/* Device-Hours by Preset — horizontal distribution bar */
table.device-hours-table th,
table.device-hours-table td {
  vertical-align: middle;
}
.device-hours-bar {
  position: relative;
  width: 100%;
  min-width: 9rem;
  height: 0.7rem;
  background: color-mix(in srgb, var(--dashboard-badge-bg) 70%, transparent);
  border-radius: 999px;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--dashboard-border) 70%, transparent);
}
.device-hours-bar-fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    var(--ci) 0%,
    color-mix(in srgb, var(--ci) 55%, var(--accent)) 100%
  );
  border-radius: 999px;
  transition: width 0.18s ease;
}
table.job-level-table tr.job-group-row {
  cursor: pointer;
  transition: background 0.12s ease;
}
table.job-level-table tr.job-group-row:hover td {
  background: color-mix(in srgb, var(--ci-tint) 70%, var(--dashboard-panel-bg));
}
table.job-level-table tr.job-group-row td.expand-cell {
  width: 1.6rem;
  padding-left: 0.85rem;
  text-align: left;
  color: var(--dashboard-muted);
  font-size: 0.78rem;
  user-select: none;
}
table.job-level-table tr.job-group-row td.expand-cell .expand-icon {
  display: inline-block;
  transition: transform 0.12s ease;
}
table.job-level-table tr.job-detail-row > td.job-detail-cell {
  padding: 0.65rem 0.9rem 0.9rem 1.6rem;
  background: var(--surface-muted);
  border-top: 0;
}
table.inner-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
  margin: 0;
}
table.inner-table th,
table.inner-table td {
  border: 1px solid var(--dashboard-border);
  padding: 0.4rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
table.inner-table thead th {
  background: var(--dashboard-panel-strong);
  font-weight: 650;
  color: var(--dashboard-chart-text);
  font-size: 0.78rem;
  white-space: nowrap;
}
table.inner-table tbody tr:nth-child(even) td {
  background: color-mix(in srgb, var(--dashboard-badge-bg) 45%, var(--dashboard-panel-bg));
}
.ci-cell {
  white-space: nowrap;
}
.ci-chip {
  display: inline-flex;
  align-items: center;
  padding: 0.05rem 0.45rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  border: 1px solid transparent;
  background: var(--surface-muted);
  color: var(--dashboard-text);
  margin-right: 0.2rem;
  white-space: nowrap;
}
.ci-chip--ready   {
  background: color-mix(in srgb, #3b82f6 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #3b82f6 35%, transparent);
}
.ci-chip--merge   {
  background: color-mix(in srgb, #1f9d63 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #1f9d63 35%, transparent);
}
.ci-chip--nightly {
  background: color-mix(in srgb, #d97706 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #d97706 35%, transparent);
}
.ci-chip--weekly  {
  background: color-mix(in srgb, #ef4444 14%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #ef4444 35%, transparent);
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


def _ensure_pyyaml() -> None:
    """Lazy-install PyYAML if missing (mirrors upload_pipeline.py's pattern)."""
    global yaml
    if yaml is not None:
        return
    import subprocess

    print("Installing PyYAML (one-time)…", file=sys.stderr)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)
    import yaml as _yaml  # noqa: F401

    yaml = _yaml


def git_pull_local_repo(repo_path: Path) -> tuple[bool, str]:
    """Run ``git pull`` in the local vllm-omni repo.  Returns (ok, head_sha).

    On any failure (missing repo, no git, dirty tree, network error) returns
    (False, "") — the caller falls back to whatever's already on disk.  The
    caller decides whether to hard-fail; the Per-Pool Detail section is
    best-effort and should never abort the whole report.
    """
    if not repo_path.exists():
        print(f"local repo not found at {repo_path}; skipping git pull", file=sys.stderr)
        return False, ""
    try:
        # Capture HEAD before pulling so we have a stable SHA to display even
        # if the network pull fails.
        head_before = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(
                f"git pull failed in {repo_path} (rc={result.returncode}): {result.stderr.strip()[:300]}",
                file=sys.stderr,
            )
            return True, head_before  # we know the pre-pull SHA, use it

        head_after = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        return True, head_after
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"git pull raised {type(e).__name__}: {e}", file=sys.stderr)
        return False, ""


def _load_mirror_hardwares_registry(repo_path: Path) -> dict[str, dict]:
    """Load `.buildkite/common/ci_mirror_hardwares.yml` from the local repo.

    Returns the ``mirror_hardwares`` mapping: preset name → preset dict
    (which contains ``agents.queue`` etc.).  Empty dict on missing/invalid
    file — callers should handle that gracefully.
    """
    _ensure_pyyaml()
    assert yaml is not None
    path = repo_path / MIRROR_HARDWARES_REL_PATH
    if not path.is_file():
        print(f"missing {path}; mirror_hardwares preset resolution disabled", file=sys.stderr)
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError) as e:
        print(f"failed to parse {path}: {e}", file=sys.stderr)
        return {}
    if not isinstance(doc, dict):
        return {}
    presets = doc.get("mirror_hardwares")
    if not isinstance(presets, dict):
        return {}
    return presets


# Path (relative to repo root) of the uploader that expands ``mirror_hardwares``
# (and infers it from pytest ``-m`` SKU markers) into ``agents.queue``.  When a
# category YAML omits ``mirror_hardwares`` and instead composes hardware from
# ``-m "H100 and cards_2"`` (the format introduced by PR #7028 for
# ``test-nightly.yml``), static parsing against the raw YAML misses every GPU
# step.  Running the uploader in render mode (``--all``) reproduces exactly
# what Buildkite receives, so the Per-Pool Detail section reflects the real
# queue wiring.
UPLOAD_PIPELINE_REL_PATH = ".buildkite/common/scripts/upload_pipeline.py"


def _render_yaml_with_upload_pipeline(repo_path: Path, rel_yaml_path: str) -> str | None:
    """Render a category YAML via ``upload_pipeline.py --all`` and return the text.

    The uploader expands ``mirror_hardwares`` (explicit preset or inferred from
    pytest ``-m`` SKU + ``cards_n`` markers) into ``agents.queue``, which is
    what Buildkite actually runs.  Returns the rendered YAML text, or ``None``
    on any failure — the caller falls back to the raw YAML so the section is
    never hard-aborted by an uploader issue.

    ``--all`` disables diff-aware step filtering so every wired step survives
    (pool-stats reports intended wiring, not per-PR filtering).  Best-effort:
    a missing uploader, an import error in its dependencies, or a non-zero
    exit all trigger the raw-YAML fallback.
    """
    uploader = repo_path / UPLOAD_PIPELINE_REL_PATH
    if not uploader.is_file():
        return None
    target = repo_path / rel_yaml_path
    if not target.is_file():
        return None
    env = os.environ.copy()
    # upload_pipeline.py reads MIRROR_HW; default (unset) matches H100 then L4
    # in ``-m``, which is the behaviour nightly/ready/merge/weekly run with on
    # a normal (non-B200) scheduled build.  Leave it unset.
    try:
        result = subprocess.run(
            [sys.executable, str(uploader), "--all", rel_yaml_path],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        print(f"upload_pipeline.py render failed for {rel_yaml_path}: {e}", file=sys.stderr)
        return None
    if result.returncode != 0:
        print(
            f"upload_pipeline.py render for {rel_yaml_path} exited {result.returncode}; "
            f"falling back to raw YAML. stderr: {result.stderr.strip()[:300]}",
            file=sys.stderr,
        )
        return None
    return result.stdout


def _extract_queue_for_step(
    step: dict,
    mirror_registry: dict[str, dict],
) -> tuple[str, str, str] | None:
    """Resolve a single step's queue.

    Returns ``(queue_name, mirror_hw_or_empty, source_label)`` or ``None``
    if the step has no queue info we can resolve (e.g. AMD list-style
    ``mirror_hardwares: [amdproduction]`` whose queue is derived elsewhere
    via ``agent_pool`` — we don't have that template here).

    Precedence:
      1. ``agents.queue`` directly on the step (used for upload steps,
         custom-pipeline tests with inline `agents:`).
      2. ``mirror_hardwares: <preset>`` → look up the preset and pull
         ``agents.queue`` from it.
    """
    agents = step.get("agents")
    if isinstance(agents, dict):
        q = (agents.get("queue") or "").strip()
        if q:
            return q, "", "agents.queue"

    mh = step.get("mirror_hardwares")
    if isinstance(mh, str):
        preset = mirror_registry.get(mh)
        if isinstance(preset, dict):
            pa = preset.get("agents") or {}
            q = (pa.get("queue") or "").strip()
            if q:
                return q, mh, "mirror_hardwares"
        # mirror_hardwares present but unresolvable — skip the step
        return None
    if isinstance(mh, list):
        # AMD-style list (e.g. ``[amdproduction]``); the queue is derived
        # from ``agent_pool`` via a Jinja template we don't have here.
        # Skip rather than guess.
        return None
    return None


def _walk_steps_for_queues(
    steps: list,
    mirror_registry: dict[str, dict],
) -> list[tuple[str, str, str, str]]:
    """Recursively walk a steps list (including nested ``group`` blocks) and
    return one ``(queue, mirror_hw, source, label)`` tuple per leaf step that
    has a resolvable queue.  ``label`` is the step's ``label`` field — the
    same string Buildkite renders as the job name on the API, so callers can
    use it to recover per-step metadata that's stripped post-upload.
    """
    out: list[tuple[str, str, str, str]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        result = _extract_queue_for_step(step, mirror_registry)
        if result is not None:
            queue, mh, src = result
            label = (step.get("label") or "").strip()
            out.append((queue, mh, src, label))
        # Recurse into nested `steps:` (groups) — same uploader semantics
        # as ``upload_pipeline.py`` which treats them as expandable.
        nested = step.get("steps")
        if isinstance(nested, list):
            out.extend(_walk_steps_for_queues(nested, mirror_registry))
    return out


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


def today_range_cst() -> tuple[str, str]:
    """Today CST as YYYY-MM-DD strings. Default reporting window.

    Note: cron jobs and other scheduled runs should pass ``--from`` / ``--to``
    explicitly rather than relying on this default, since "today" is only a
    partial day and may yield incomplete Buildkite coverage.
    """
    today_cst = datetime.now(CST).date()
    return today_cst.isoformat(), today_cst.isoformat()


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


# ── Accelerator count extraction ────────────────────────────────────────
#
# Each Buildkite job reports its `resource_class` (or, for k8s plugins,
# `step.agents.resource_class`) — a string like:
#   • "nvidia.com/gpu=4"  → 4 GPUs
#   • "nvidia.com/gpu-2"  → 2 GPUs  (rare plugin variant)
#   • "npu-2"             → 2 NPUs
# We parse this to get the per-job accelerator count, which lets us
# compute total accelerator-hours (Σ duration × count / 3600) — the
# metric that actually answers "did CI resource consumption drop?".

_NVIDIA_GPU_RE = re.compile(r"^nvidia\.com/gpu[=\-](\d+)$")
_NPU_RE = re.compile(r"^npu[=\-]?(\d+)$")
_PRESET_ACCEL_RE = re.compile(r"^(?:h100|l4|a2b3_npu|a3_npu)_(\d+)$")


def _parse_accel_count_from_rc(rc: str | None) -> int:
    """Parse a resource_class string into an accelerator count. Returns 0
    when the value is missing or doesn't match a known shape so callers
    can fall back gracefully."""
    if not rc:
        return 0
    s = rc.strip().lower()
    if not s:
        return 0
    m = _NVIDIA_GPU_RE.match(s)
    if m:
        return int(m.group(1))
    m = _NPU_RE.match(s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return 0


def _accel_count_for_job(
    job: dict,
    *,
    pipeline_slug: str | None = None,
    ci_category: str | None = None,
) -> int:
    """Extract GPU/NPU count from a Buildkite job.

    Precedence:
      1. ``step.agents.resource_class`` (post-upload_pipeline.py value)
      2. ``job.resource_class`` (some API versions)
      3. ``step.mirror_hardwares`` (unexpanded YAML)
      4. YAML label lookup: find the step's ``label`` (== Buildkite job
         name) in the precomputed ``label_to_accel`` table built from
         the local vllm-omni ``test-*.yml``. This is the **only**
         fallback that recovers the true card count for pools like
         ``mithril-h100-pool`` where multiple card counts (h100_1 ..
         h100_4) target the same queue — the API can't disambiguate
         them, but the YAML can.
      5. Default 1 (single-accelerator fallback)

    Returns 1 (not 0) on miss so the job still contributes one
    accelerator-hour to the total rather than vanishing from the count.
    The YAML lookup may legitimately return 0 (CPU-only steps like
    ``cpu_queue_premerge``) — in that case 0 is the correct value and
    we honor it; only the API+default path returns 1.
    """
    # 1. step.agents.resource_class (most reliable post-upload)
    step = job.get("step") or {}
    agents = step.get("agents") or {}
    n = _parse_accel_count_from_rc(agents.get("resource_class") or "")
    if n > 0:
        return n

    # 2. job.resource_class
    n = _parse_accel_count_from_rc(job.get("resource_class") or "")
    if n > 0:
        return n

    # 3. step.mirror_hardwares (if YAML reached Buildkite unexpanded)
    mh = step.get("mirror_hardwares")
    if isinstance(mh, str):
        m = _PRESET_ACCEL_RE.match(mh.strip())
        if m:
            return int(m.group(1))

    # 4. YAML lookup by job name (== step label).
    # Buildkite job name == leaf step's ``label`` field (no group prefix).
    # Only attempt when both pipeline_slug and ci_category are provided
    # so the wrong-CI-category YAML is never consulted.
    job_name = (job.get("name") or "").strip()
    n = _accel_count_from_yaml_lookup(pipeline_slug, ci_category, job_name)
    if n is not None:
        return n

    return 1


def _infer_preset_for_job(queue: str, accel_count: int) -> str:
    """Synthesize the preset name from queue + accelerator count.

    The Buildkite job object doesn't preserve the original
    ``mirror_hardwares`` preset after upload_pipeline.py expansion
    (which replaces it with ``agents.queue`` + ``agents.resource_class``).
    We reconstruct the preset name from the queue + count combo so the
    per-preset Device-hours panel still has labels to group by.

    Returns the raw queue name when no preset inference applies.
    """
    # l4_N → gpu_N_queue
    m = re.match(r"^gpu_(\d+)_queue$", queue)
    if m:
        return f"l4_{accel_count}"
    if queue == "mithril-h100-pool":
        return f"h100_{accel_count}"
    if queue == "ascend-a2b3":
        return f"a2b3_npu_{accel_count}"
    if queue == "ascend-a3":
        return f"a3_npu_{accel_count}"
    return queue


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
    # build_number (str) → job count in that build, for "avg cards per build".
    # Keyed by string build number so it JSON-serializes cleanly.
    build_jobs: dict[str, int] = field(default_factory=dict)
    # Individual job records (for the Job-Level Detail table).  Populated
    # in compute_pool_stats; capped at MAX_JOB_RECORDS to keep memory bounded
    # for very busy days.
    jobs: list[JobRecord] = field(default_factory=list)
    # Hourly time-series: hour (0-23) -> HourBucket
    hourly: dict[int, HourBucket] = field(default_factory=lambda: defaultdict(HourBucket))

    @property
    def build_count(self) -> int:
        return len(self.build_jobs)


# Cap on per-job records retained per (pipeline, pool).  The Job-Level
# Detail table sorts by duration and slices to JOB_LEVEL_TABLE_LIMIT, so
# keeping a generous buffer beyond the cap ensures we don't lose a long
# outlier when the same pool runs hundreds of short jobs.
JOB_RECORD_PER_POOL_CAP = 2000


def _determine_ci_category(branch: str | None, source: str | None, created_at: datetime | None) -> str:
    """Map a Buildkite build to one of ``ready`` / ``merge`` / ``nightly`` / ``weekly``.

    Heuristic:
      • non-main branch → ``ready`` (PR / pre-merge testing)
      • main + non-scheduled source → ``merge`` (push to main, not on a schedule)
      • main + scheduled:
          - created on a Sunday (CST) → ``weekly``
          - otherwise → ``nightly``

    The weekly-day assumption is encoded in ``_WEEKLY_WEEKDAY`` (Sunday by
    default, matching the vllm-omni pipeline).  If the pipeline schedule
    changes, this is the one constant to update.
    """
    if branch and branch.strip() and branch.strip() != "main":
        return "ready"
    if source and source.strip().lower() == "schedule":
        if created_at is not None:
            cst_weekday = created_at.astimezone(CST).weekday()
            # Python: Monday=0 … Sunday=6.  Weekly runs on Sunday by default.
            if cst_weekday == _WEEKLY_WEEKDAY:
                return "weekly"
        return "nightly"
    return "merge"


# Sunday (Python weekday 6) — vllm-omni weekly schedule lands here.
_WEEKLY_WEEKDAY = 6


@dataclass
class JobRecord:
    """One execution of one Buildkite job — feeds the Job-Level Detail table.

    All timestamps are timezone-aware UTC.  ``wait_seconds`` is the queue
    time, ``duration_seconds`` is the on-agent runtime.  Either can be 0
    when the corresponding API fields are missing.  ``ci_category`` is
    the inferred CI label (``ready`` / ``merge`` / ``nightly`` / ``weekly``)
    derived from the parent build's branch and source.  ``accel_count``
    is the per-job GPU/NPU count parsed from ``resource_class`` and is
    used to compute total accelerator-hours for the Device-Hours panel.
    """

    pipeline: str
    pool_name: str
    build_number: str
    job_id: str
    job_name: str
    state: str
    ci_category: str
    scheduled_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    wait_seconds: float
    duration_seconds: float
    accel_count: int = 1


# ── Static-YAML pool data (Per-Pool Detail) ──────────────────────────────


@dataclass
class StaticPoolEntry:
    """Per-(queue, preset) static usage derived from `.buildkite/test-*.yml`.

    One entry per (pipeline × preset).  When a step sets ``agents.queue``
    directly (no ``mirror_hardwares``), ``preset_name`` is empty and
    ``queue`` carries the queue name.  When a step uses a preset like
    ``h100_1`` (which maps to queue ``mithril-h100-pool``), ``preset_name``
    is ``h100_1`` and ``queue`` is ``mithril-h100-pool``.

    Multiple H100 presets (h100_1..h100_4) all map to the same queue, so
    breaking the table out by preset is what lets the user see the GPU
    count per step.
    """

    pipeline: str
    queue: str
    preset_name: str = ""
    gpus_per_unit: int = 0
    total_steps: int = 0
    # category → step count (only categories that actually use this pool)
    categories: dict[str, int] = field(default_factory=dict)
    # test file basename → step count (only files that touch this pool)
    files: dict[str, int] = field(default_factory=dict)
    # mirror_hardwares preset name → step count (kept for legacy display;
    # always {preset_name: total_steps} when preset_name is set, {} otherwise)
    mirror_hardwares: dict[str, int] = field(default_factory=dict)


# ── Preset/queue → accelerator-count helpers (drive the totals) ─────────
#
# Each preset name encodes its per-unit accelerator count: ``h100_4`` → 4
# GPUs, ``a3_npu_8`` → 8 NPUs, etc.  The same is true of the ``gpu_*_queue``
# queue names.  NPU presets contribute to A2/A3 totals (not GPU totals)
# via the ``_is_a2_preset`` / ``_is_a3_preset`` filters below.
_H100_PRESET_RE = re.compile(r"^h100_\d+$")
_L4_PRESET_RE = re.compile(r"^l4_\d+$")
_NPU_PRESET_RE = re.compile(r"^(?:a2b3|a3)_npu_\d+$")
_GPU_QUEUE_RE = re.compile(r"^gpu_(\d+)_queue$")
_PRESET_COUNT_RE = re.compile(r"^(?:h100|l4|a2b3_npu|a3_npu)_(\d+)$")
_ASCEND_QUEUE = ("ascend-a2b3", "ascend-a3")


def _is_npu_preset(preset: str, queue: str) -> bool:
    if preset and _NPU_PRESET_RE.match(preset):
        return True
    return any(queue.startswith(q) for q in _ASCEND_QUEUE)


def _is_h100_preset(preset: str, queue: str) -> bool:
    if preset and _H100_PRESET_RE.match(preset):
        return True
    return queue == "mithril-h100-pool"


def _is_l4_preset(preset: str, queue: str) -> bool:
    """L4 covers both the l4_N preset family and any direct gpu_*_queue step
    (l4_N maps to those queues, so a direct queue reference is still L4)."""
    if preset and _L4_PRESET_RE.match(preset):
        return True
    return bool(_GPU_QUEUE_RE.match(queue))


def _is_a2_preset(preset: str, queue: str) -> bool:
    """A2 covers any preset starting with ``a2`` (currently ``a2b3_npu_*``)
    and any direct ``ascend-a2b3`` queue reference."""
    if preset and preset.startswith("a2"):
        return True
    return queue.startswith("ascend-a2b3")


def _is_a3_preset(preset: str, queue: str) -> bool:
    if preset and preset.startswith("a3_"):
        return True
    return queue.startswith("ascend-a3")


def _is_gpu_preset(preset: str, queue: str) -> bool:
    """True for any GPU-based preset/queue (H100 or L4)."""
    if _is_npu_preset(preset, queue):
        return False
    if preset and (_H100_PRESET_RE.match(preset) or _L4_PRESET_RE.match(preset)):
        return True
    if _GPU_QUEUE_RE.match(queue):
        return True
    return False


# ── Per-pipeline Total chips ─────────────────────────────────────────────
#
# Each pipeline renders two Total chips at the bottom of every category
# subcard.  The mapping below pairs a chip label with a filter that picks
# out the relevant presets/queues for that chip.  The chip value is
# Σ (steps × gpus_per_unit) over the filtered entries, scoped to the
# pipeline × category being shown.
_PIPELINE_TOTAL_CHIPS: dict[str, list[tuple[str, Callable]]] = {
    "vllm-omni": [
        ("h100", _is_h100_preset),
        ("l4", _is_l4_preset),
    ],
    "vllm-omni-npu-ci": [
        ("A2", _is_a2_preset),
        ("A3", _is_a3_preset),
    ],
}


def _gpus_for_unit(preset: str, queue: str) -> int:
    """Return the per-unit accelerator count for a step.

    For H100/L4 presets (``h100_4``, ``l4_1``, …) this is the GPU count;
    for NPU presets (``a2b3_npu_8``, ``a3_npu_2``, …) this is the NPU
    count.  Falls back to the ``gpu_N_queue`` queue name when no preset
    is set.  Returns 0 for unknown entries (they're filtered out by the
    chip filters in ``_render_latest_builds_by_category_html``).
    """
    if preset:
        m = _PRESET_COUNT_RE.match(preset)
        if m:
            return int(m.group(1))
    m = _GPU_QUEUE_RE.match(queue)
    if m:
        return int(m.group(1))
    return 0


# Cache populated by ``main()`` from the per-pipeline StaticPipelineData so
# the per-job aggregation loop can recover each job's accelerator count
# from its Buildkite job name (== YAML step label).  Shape:
#   (pipeline_slug, ci_category) → {label: accel_count}
# Without this, every job in a pool like ``mithril-h100-pool`` would be
# assigned ``accel_count=1`` (the default fallback in
# ``_accel_count_for_job``) because ``mirror_hardwares`` and
# ``agents.resource_class`` are stripped by upload_pipeline.py before
# Buildkite ever sees the step.  See the YAML lookup fallback below.
_LABEL_TO_ACCEL_BY_PIPELINE: dict[str, dict[str, dict[str, int]]] = {}


def _accel_count_from_yaml_lookup(
    pipeline_slug: str | None,
    ci_category: str | None,
    job_name: str,
) -> int | None:
    """Look up the accelerator count for a job from the precomputed
    ``label_to_accel`` table.  Returns ``None`` when no mapping applies
    (missing pipeline / category, or label not in the YAML) so callers
    fall back to the Buildkite-API path.
    """
    if not pipeline_slug or not ci_category or not job_name:
        return None
    by_cat = _LABEL_TO_ACCEL_BY_PIPELINE.get(pipeline_slug)
    if not by_cat:
        return None
    by_label = by_cat.get(ci_category)
    if not by_label:
        return None
    if job_name not in by_label:
        return None
    return by_label[job_name]


def _display_key_for(preset: str, queue: str) -> str:
    """Return the bucket key for grouping static entries.

    Prefer the preset name (more specific — distinguishes h100_1 from h100_4
    which both map to ``mithril-h100-pool``).  Fall back to queue when no
    preset was set.
    """
    return preset if preset else queue


@dataclass
class StaticPipelineData:
    """Per-pipeline static data — drives the Per-Pool Detail panel."""

    pipeline: str
    # category → {pool_name → step_count}
    categories: dict[str, dict[str, int]] = field(default_factory=dict)
    # pool_name → StaticPoolEntry (aggregated across categories)
    pools: dict[str, StaticPoolEntry] = field(default_factory=dict)
    # which YAML files were parsed for this pipeline
    files: list[str] = field(default_factory=list)
    # git HEAD SHA used for the YAML (post-pull, or pre-pull if pull failed)
    git_head: str = ""
    repo_path: str = ""
    # Reverse lookup used to recover the per-job accelerator count from
    # a Buildkite job name (which equals the step's ``label`` post-upload,
    # since ``mirror_hardwares`` and ``agents.resource_class`` are stripped).
    # Shape: category → {label: accel_count}.  ``accel_count`` is 0 for
    # CPU-only steps (e.g. ``cpu_queue_premerge``) and ≥1 for GPU/NPU
    # steps.  A label missing from this dict falls back to default-1
    # behavior in ``_accel_count_for_job`` to preserve prior behavior
    # for jobs we couldn't classify.
    label_to_accel: dict[str, dict[str, int]] = field(default_factory=dict)


# ── Build/job state filtering ────────────────────────────────────────────

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


def is_ran_job(state: str | None) -> bool:
    return (state or "").strip().lower() in RAN_JOB_STATES


def compute_static_pool_data(
    repo_path: Path,
    pipeline_slugs: list[str],
    *,
    skip_git_pull: bool = False,
) -> dict[str, StaticPipelineData]:
    """Compute pool-usage data by statically parsing the local vllm-omni repo.

    Steps:
      1. ``git pull`` the repo (best-effort).
      2. Load ``.buildkite/common/ci_mirror_hardwares.yml`` to resolve
         ``mirror_hardwares: <preset>`` → ``agents.queue``.
      3. For each pipeline in ``PIPELINE_YAML_MAP``, parse its category
         YAML files and walk the steps tree, collecting per-pool counts.

    Returns ``{pipeline_slug: StaticPipelineData}``.  Pipelines without a
    YAML mapping (e.g. AMD-only) are skipped silently.  A failed git pull
    does NOT abort the analysis — we read whatever's currently on disk.
    """
    _ensure_pyyaml()
    assert yaml is not None

    repo_path = Path(repo_path).expanduser()

    if skip_git_pull:
        head_sha = ""
        try:
            head_sha = subprocess.run(  # type: ignore[name-defined]
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            head_sha = ""
    else:
        _, head_sha = git_pull_local_repo(repo_path)

    mirror_registry = _load_mirror_hardwares_registry(repo_path)

    out: dict[str, StaticPipelineData] = {}
    for pipeline_slug in pipeline_slugs:
        yaml_files = PIPELINE_YAML_MAP.get(pipeline_slug)
        if not yaml_files:
            # No static definition for this pipeline — skip silently.  AMD
            # uses an AMD-specific template; Intel has its own pipeline
            # file.  We only handle vllm-omni and vllm-omni-npu-ci.
            continue
        data = StaticPipelineData(
            pipeline=pipeline_slug,
            git_head=head_sha,
            repo_path=str(repo_path),
        )
        for category, rel_path in yaml_files:
            path = repo_path / rel_path
            if not path.is_file():
                print(f"missing {path}; skipping {pipeline_slug}/{category}", file=sys.stderr)
                continue
            # Render via upload_pipeline.py --all first so that steps which
            # omit ``mirror_hardwares`` and compose hardware from pytest ``-m``
            # SKU + ``cards_n`` markers (PR #7028 format) get their
            # ``agents.queue`` expanded — the raw YAML would otherwise leave
            # every such GPU step unresolvable.  Fall back to the raw YAML
            # when the uploader is unavailable or fails so the section is
            # never hard-aborted.
            rendered = _render_yaml_with_upload_pipeline(repo_path, rel_path)
            raw_text = path.read_text(encoding="utf-8")
            if rendered:
                doc_text = rendered
            else:
                doc_text = raw_text
            try:
                doc = yaml.safe_load(doc_text)
            except (yaml.YAMLError, OSError) as e:
                print(f"failed to parse {path}: {e}", file=sys.stderr)
                continue
            if not isinstance(doc, dict):
                continue
            steps = doc.get("steps") or []
            if not isinstance(steps, list):
                continue

            file_basename = Path(rel_path).name
            data.files.append(file_basename)
            cat_buckets: dict[str, int] = {}
            label_to_accel_cat = data.label_to_accel.setdefault(category, {})
            for queue, mh, _src, label in _walk_steps_for_queues(steps, mirror_registry):
                # Bucket by preset when available, else by queue.  This is
                # what lets mithril-h100-pool split into h100_1..h100_4
                # instead of aggregating as a single row.
                key = _display_key_for(mh, queue)
                cat_buckets[key] = cat_buckets.get(key, 0) + 1
                pe = data.pools.get(key)
                if pe is None:
                    pe = StaticPoolEntry(
                        pipeline=pipeline_slug,
                        queue=queue,
                        preset_name=mh,
                        gpus_per_unit=_gpus_for_unit(mh, queue),
                    )
                    data.pools[key] = pe
                pe.total_steps += 1
                pe.categories[category] = pe.categories.get(category, 0) + 1
                pe.files[file_basename] = pe.files.get(file_basename, 0) + 1
                if mh:
                    pe.mirror_hardwares[mh] = pe.mirror_hardwares.get(mh, 0) + 1
                # Reverse lookup: Buildkite job name == step label, so we
                # record each leaf step's accelerator count by label here.
                # Mirrors the precedence _accel_count_for_job uses, but
                # sourced from YAML so it works post-upload_pipeline.py.
                # Skip entries without a label — these can't be matched
                # to Buildkite jobs anyway.
                if label and label not in label_to_accel_cat:
                    accel = _gpus_for_unit(mh, queue)
                    label_to_accel_cat[label] = accel
            data.categories[category] = cat_buckets

        out[pipeline_slug] = data

    return out


def compute_pool_stats(
    token: str,
    pipeline_slug: str,
    created_from: str,
    created_to: str,
    *,
    verbose: bool = False,
) -> dict[str, PoolStats]:
    """Fetch builds for a pipeline in the date range, extract job timing data,
    and return a dict of pool_name -> PoolStats (per-pool aggregates + hourly).

    Note: the Per-Pool Detail section no longer consumes category stats from
    here; that section is sourced from the local vllm-omni YAML repo via
    ``compute_static_pool_data()``.
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

    for b in builds:
        b = ensure_build_with_jobs(token, pipeline_slug, b)
        jobs = b.get("jobs") or []
        bnum_raw = b.get("number")
        bnum_key = str(bnum_raw) if bnum_raw is not None else ""
        b_branch = (b.get("branch") or "").strip()
        b_source = (b.get("source") or "").strip()
        b_created_at = parse_buildkite_time(b.get("created_at"))
        ci_category = _determine_ci_category(b_branch, b_source, b_created_at)
        if verbose:
            bnum = b.get("number", "?")
            bstate = (b.get("state") or "").strip()
            print(
                f"  Build #{bnum} state={bstate} branch={b_branch!r} "
                f"source={b_source!r} ci={ci_category} jobs={len(jobs)}"
            )

        for j in jobs:
            jtype = (j.get("type") or "").strip().lower()
            if jtype not in ("script", "command"):
                continue

            # Only count jobs that actually ran on an agent
            # (exclude scheduled/assigned/running/skipped/not_run/blocked).
            jstate = (j.get("state") or "").strip().lower()
            if not is_ran_job(jstate):
                continue

            scheduled_at = parse_buildkite_time(j.get("scheduled_at"))
            started_at = parse_buildkite_time(j.get("started_at"))
            finished_at = parse_buildkite_time(j.get("finished_at"))

            pool_name = extract_queue_from_job(j)
            accel_count = _accel_count_for_job(
                j,
                pipeline_slug=pipeline_slug,
                ci_category=ci_category,
            )

            if pool_name not in pools:
                pools[pool_name] = PoolStats(pipeline=pipeline_slug, pool_name=pool_name)
            ps = pools[pool_name]
            ps.job_count += 1
            if bnum_key:
                ps.build_jobs[bnum_key] = ps.build_jobs.get(bnum_key, 0) + 1

            # Determine the hour bucket from scheduled_at (the time the job entered the queue)
            hour: int | None = None
            if scheduled_at is not None:
                hour = _hour_key(scheduled_at)

            # Queue wait time
            wait_s = 0.0
            if scheduled_at is not None and started_at is not None:
                wait = (started_at - scheduled_at).total_seconds()
                if wait >= 0:
                    ps.wait_seconds.append(wait)
                    wait_s = wait
                    if hour is not None:
                        ps.hourly[hour].wait_seconds.append(wait)

            # Job duration
            dur_s = 0.0
            if started_at is not None and finished_at is not None:
                dur = (finished_at - started_at).total_seconds()
                if dur >= 0:
                    ps.duration_seconds.append(dur)
                    dur_s = dur
                    if hour is not None:
                        ps.hourly[hour].duration_seconds.append(dur)

            if hour is not None:
                ps.hourly[hour].job_count += 1

            # Record per-job details (capped per pool) for the Job-Level
            # Detail table.  Include even jobs with 0 wait/duration so
            # the row count matches the totals — we'll display "—" in
            # the missing-time columns.
            if len(ps.jobs) < JOB_RECORD_PER_POOL_CAP:
                ps.jobs.append(
                    JobRecord(
                        pipeline=pipeline_slug,
                        pool_name=pool_name,
                        build_number=bnum_key,
                        job_id=str(j.get("id") or ""),
                        job_name=(j.get("name") or j.get("label") or "").strip() or f"job-{bnum_key}",
                        state=jstate,
                        ci_category=ci_category,
                        scheduled_at=scheduled_at,
                        started_at=started_at,
                        finished_at=finished_at,
                        wait_seconds=wait_s,
                        duration_seconds=dur_s,
                        accel_count=accel_count,
                    )
                )

    return pools


# ── Compute aggregate summary cards ─────────────────────────────────────


def _compute_summary_cards(all_pools: dict[str, dict[str, PoolStats]]) -> list[dict]:
    total_jobs = 0
    total_pools = 0
    all_waits: list[float] = []
    total_occ = 0.0
    device_hours_by_chip: dict[str, float] = {"h100": 0.0, "l4": 0.0, "npu": 0.0, "cpu": 0.0}

    for pipeline_slug, pools in all_pools.items():
        total_pools += len(pools)
        for ps in pools.values():
            total_jobs += ps.job_count
            all_waits.extend(ps.wait_seconds)
            total_occ += sum(ps.duration_seconds)
            chip = _chip_family_for_pool(ps.pool_name)
            device_hours_by_chip[chip] = device_hours_by_chip.get(chip, 0.0) + _pool_accelerator_hours(ps)

    avg_wait = (sum(all_waits) / len(all_waits)) if all_waits else 0.0
    total_device_hours = sum(device_hours_by_chip.values())

    # Order GPU types first (they're the headline), then NPU, then CPU —
    # keeps the existing card layout stable when CPU is non-zero, and
    # hides the CPU segment entirely when no CPU jobs ran.
    breakdown = " · ".join(
        f"{chip} {device_hours_by_chip[chip]:.1f}h"
        for chip in ("h100", "l4", "npu", "cpu")
        if device_hours_by_chip[chip] > 0
    )

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
            "title": "Device-Hours",
            "value": f"{total_device_hours:.1f}h" if total_device_hours else "N/A",
            "detail": breakdown or "no accelerator usage",
            "icon": ICON_TREND,
        },
        {
            "title": "Resource Pools",
            "value": str(total_pools),
            "detail": f"{', '.join(all_pools.keys())}",
            "icon": ICON_SERVER,
        },
    ]


def _chip_family_for_pool(pool_name: str) -> str:
    """Bucket a pool name into an accelerator chip family for grouping.

    Returns one of ``"h100"`` / ``"l4"`` / ``"npu"`` / ``"cpu"``.
    CPU-only queues (``cpu_queue_premerge`` and anything else matching
    the ``cpu_*`` prefix) are bucketed into ``"cpu"`` so their wall-clock
    runtime doesn't pollute the GPU device-hours — a CPU job has no
    cards to weight, so it gets its own chip family and its own row in
    the Device-Hours by Preset table.

    Unknown non-CPU pool names bucket into ``"h100"`` as a conservative
    default since the mithril H100 pool is the highest-traffic fallback
    we see today; if we ever see a new GPU pool (e.g. a B200 or H200
    queue) without an explicit mapping, it falls into h100 and would
    need an explicit entry here.
    """
    if pool_name.startswith("cpu_"):
        return "cpu"
    if pool_name == "mithril-h100-pool":
        return "h100"
    if pool_name.startswith("gpu_") and pool_name.endswith("_queue"):
        return "l4"
    if pool_name.startswith("ascend-"):
        return "npu"
    return "h100"


def _pool_accelerator_hours(ps: PoolStats) -> float:
    """Sum of (duration × accel_count) over all jobs in a pool, in hours."""
    return sum(jr.duration_seconds * jr.accel_count for jr in ps.jobs if jr.duration_seconds > 0) / 3600.0


def _aggregate_device_hours_by_preset(
    all_pools: dict[str, dict[str, PoolStats]],
) -> list[dict]:
    """Group jobs by inferred preset, summing accelerator-hours.

    Returns a list of ``{"preset": ..., "queue": ..., "hours": ...,
    "jobs": ..., "cards": ..., "accel_count": ..., "device_type":
    "gpu"|"npu"}`` dicts sorted by hours descending.

    The ``cards`` field sums each job's ``accel_count`` so multi-card
    jobs (h100_4 = 4 cards, a3_npu_8 = 8 cards) contribute their full
    card footprint — useful for distinguishing a preset that's run a
    few times on big machines vs. many times on small ones.

    The ``device_type`` field is used by the renderer to split rows into
    separate GPU / NPU sub-tables (each with its own share denominator)
    so cross-type comparisons don't drown the smaller NPU footprint in
    GPU totals.
    """
    buckets: dict[str, dict] = {}
    for pools in all_pools.values():
        for ps in pools.values():
            for jr in ps.jobs:
                if jr.duration_seconds <= 0:
                    continue
                preset = _infer_preset_for_job(jr.pool_name, jr.accel_count)
                # CPU-only queues (e.g. ``cpu_queue_premerge``) get their
                # own device_type so the renderer splits them into a
                # dedicated CPU sub-table.  ``_infer_preset_for_job``
                # returns the raw queue name for these, which is the
                # signal we use here.
                if jr.pool_name.startswith("cpu_") or preset == jr.pool_name and preset.startswith("cpu_"):
                    device_type = "cpu"
                elif preset.startswith(("a2b3_npu", "a3_npu")):
                    device_type = "npu"
                else:
                    device_type = "gpu"
                b = buckets.setdefault(
                    preset,
                    {
                        "preset": preset,
                        "queue": jr.pool_name,
                        "hours": 0.0,
                        "jobs": 0,
                        "cards": 0,
                        "accel_count": jr.accel_count,
                        "device_type": device_type,
                    },
                )
                b["hours"] += jr.duration_seconds * jr.accel_count / 3600.0
                b["jobs"] += 1
                b["cards"] += jr.accel_count
    return sorted(buckets.values(), key=lambda b: (-b["hours"], b["preset"]))


def _format_hours(hours: float) -> str:
    """Compact hour formatter: 1.5h, 24h, 0.5h."""
    if hours < 0.05:
        return "0h"
    if hours < 10:
        return f"{hours:.1f}h"
    return f"{hours:.0f}h"


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


def _render_latest_builds_by_category_html(
    static_data: dict[str, StaticPipelineData],
) -> str:
    """Render one card per CI category (ready / merge / nightly / weekly).

    Each card shows static YAML-derived pool usage per pipeline, plus a
    ``Total`` row at the bottom of every subcard with the
    pipeline+category-scoped ``h100_total`` and ``gpu_total`` numbers
    (steps × gpus per preset).  Sourced from the local vllm-omni repo
    (after git pull) instead of the Buildkite API — see
    ``compute_static_pool_data()``.
    """
    color_map = {
        "ready": "cat-card--ready",
        "merge": "cat-card--merge",
        "nightly": "cat-card--nightly",
        "weekly": "cat-card--weekly",
    }

    cards_parts: list[str] = []
    for cat_key, label, sub in CATEGORY_ORDER:
        # For each category, collect per-pipeline pool distributions
        subcard_parts: list[str] = []
        for pipeline_slug, pdata in sorted(static_data.items()):
            cat_buckets = pdata.categories.get(cat_key) or {}
            if not cat_buckets:
                continue

            sorted_pools = sorted(
                cat_buckets.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
            total_steps = sum(cat_buckets.values())
            pool_rows = "".join(
                f'<li class="latest-pool-row">'
                f'<span class="latest-pool-name">{html.escape(name)}</span>'
                f'<span class="latest-pool-count">{count}</span>'
                f"</li>"
                for name, count in sorted_pools
            )

            # Per-pipeline × per-category totals — scope each preset's
            # contribution to this category only.  Which chips to show is
            # pipeline-specific: vllm-omni → h100 / l4, vllm-omni-npu-ci →
            # A2 / A3 (see _PIPELINE_TOTAL_CHIPS).
            chip_filters = _PIPELINE_TOTAL_CHIPS.get(pipeline_slug, [])
            chip_totals: list[tuple[str, int, str]] = []
            for chip_label, chip_filter in chip_filters:
                chip_value = 0
                for key, count in cat_buckets.items():
                    pe = pdata.pools.get(key)
                    if pe is None or not pe.gpus_per_unit:
                        continue
                    if chip_filter(pe.preset_name, pe.queue):
                        chip_value += pe.gpus_per_unit * count
                # Cycle through red/blue/amber/green tints by index.
                chip_class = f"latest-total-chip--{chip_label.lower()}"
                chip_title = f"Σ {chip_label} presets × gpus in this category"
                chip_totals.append((chip_label, chip_value, chip_class, chip_title))

            # Find the YAML file for this pipeline + category
            yaml_basename = ""
            for cat, rel in PIPELINE_YAML_MAP.get(pipeline_slug, []):
                if cat == cat_key:
                    yaml_basename = Path(rel).name
                    break

            # Build the Total row — always show both numbers (even when 0)
            # so the layout doesn't shift between pipelines.
            chip_html = "".join(
                f'<span class="latest-total-chip {cls}" title="{html.escape(title)}">'
                f"{html.escape(label)} <strong>{value}</strong></span>"
                for label, value, cls, title in chip_totals
            )
            total_row = (
                f'<li class="latest-pool-row latest-pool-row--total">'
                f'<span class="latest-pool-name latest-total-name">'
                f"<strong>Total</strong></span>"
                f'<span class="latest-pool-counts">{chip_html}</span>'
                f"</li>"
            )

            subcard_parts.append(
                f'<div class="latest-subcard">'
                f'<div class="latest-subcard-head">'
                f'<span class="latest-subcard-pipeline">{html.escape(pipeline_slug)}</span>'
                f'<span class="latest-subcard-num">{total_steps} steps</span>'
                f"</div>"
                f'<div class="latest-subcard-meta">'
                f'<span class="latest-subcard-branch" '
                f'title="{html.escape(yaml_basename)}">{html.escape(yaml_basename or "?")}</span>'
                f'<span class="latest-subcard-state">static yaml</span>'
                f"</div>"
                f'<ul class="latest-pool-list">{pool_rows}{total_row}</ul>'
                f"</div>"
            )

        if not subcard_parts:
            continue

        cards_parts.append(
            f'<div class="cat-card {color_map.get(cat_key, "")}">'
            f'<div class="cat-card-head">'
            f'<span class="cat-card-label">{html.escape(label)}'
            f'<span class="cat-card-sub">— {html.escape(sub)}</span></span>'
            f'<span class="cat-card-count">{len(subcard_parts)} pipeline(s)</span>'
            f"</div>"
            f'<div class="latest-subcards">{"".join(subcard_parts)}</div>'
            f"</div>"
        )

    if not cards_parts:
        return ""

    return (
        '<div class="cat-stats">'
        '<h3 class="cat-stats-title">Pool Usage by CI Category — Static YAML '
        "(from local vllm-omni repo)</h3>"
        f'<div class="cat-stats-grid">{"".join(cards_parts)}</div>'
        "</div>"
    )


def _render_static_pool_table(static_data: dict[str, StaticPipelineData]) -> str:
    """Render one row per (pipeline × preset) plus summary rows for totals.

    The bucket key is the preset name when a step uses ``mirror_hardwares``
    (so h100_1..h100_4 each get their own row even though they all share
    the ``mithril-h100-pool`` queue), or the queue name when no preset
    is set.

    At the bottom of each pipeline's block we add two summary rows:
      • ``h100_total`` — Σ steps×gpus over h100_* presets
      • ``gpu_total``  — Σ steps×gpus over all GPU-based pools (H100 + L4)
    """
    rows_parts: list[str] = []

    def _td(val: object, cls: str = "") -> str:
        if val is None or val == "":
            return f'<td class="{cls} na">—</td>'
        return f'<td class="{cls}">{val}</td>'

    for pipeline_slug in sorted(static_data.keys()):
        pdata = static_data[pipeline_slug]

        # Sort: GPU-based first (by preset name), then NPU / unknown.  Group
        # the h100_* family together so the totals below it make sense.
        def _sort_key(pe: StaticPoolEntry) -> tuple:
            gpu = _is_gpu_preset(pe.preset_name, pe.queue)
            h100 = _is_h100_preset(pe.preset_name, pe.queue)
            label = pe.preset_name or pe.queue
            return (
                0 if h100 else (1 if gpu else 2),  # H100 → GPU → other
                label,
            )

        for pe in sorted(pdata.pools.values(), key=_sort_key):
            cats_sorted = sorted(pe.categories.items())
            cats_disp = ", ".join(f"{html.escape(c)}:{n}" for c, n in cats_sorted)
            files_disp = ", ".join(html.escape(f) for f, _ in sorted(pe.files.items()))

            # Display name: prefer the preset (more specific), then the queue.
            if pe.preset_name:
                display = f"{pe.preset_name} <span class='pool-queue'>→ {html.escape(pe.queue)}</span>"
            else:
                display = html.escape(pe.queue)

            gpu_usage = pe.gpus_per_unit * pe.total_steps
            rows_parts.append(
                f"<tr>"
                f'<td class="pipeline-cell">{html.escape(pe.pipeline)}</td>'
                f'<td class="pool-name">{display}</td>'
                f"{_td(pe.total_steps, 'num')}"
                f"{_td(pe.gpus_per_unit if pe.gpus_per_unit else '', 'num')}"
                f"{_td(gpu_usage if pe.gpus_per_unit else '', 'num')}"
                f"{_td(cats_disp)}"
                f"{_td(files_disp)}"
                f"</tr>"
            )

        # Per-pipeline summary rows
        h100_total = sum(
            pe.gpus_per_unit * pe.total_steps
            for pe in pdata.pools.values()
            if _is_h100_preset(pe.preset_name, pe.queue)
        )
        gpu_total = sum(
            pe.gpus_per_unit * pe.total_steps
            for pe in pdata.pools.values()
            if _is_gpu_preset(pe.preset_name, pe.queue)
        )

        rows_parts.append(
            f'<tr class="summary-row summary-row--h100">'
            f'<td class="pipeline-cell">{html.escape(pipeline_slug)}</td>'
            f'<td class="pool-name">h100_total</td>'
            f'<td class="num">—</td>'
            f'<td class="num">—</td>'
            f'<td class="num">{h100_total}</td>'
            f"<td>Σ h100 presets × gpus</td>"
            f'<td class="na">—</td>'
            f"</tr>"
        )
        rows_parts.append(
            f'<tr class="summary-row summary-row--gpu">'
            f'<td class="pipeline-cell">{html.escape(pipeline_slug)}</td>'
            f'<td class="pool-name">gpu_total</td>'
            f'<td class="num">—</td>'
            f'<td class="num">—</td>'
            f'<td class="num">{gpu_total}</td>'
            f"<td>Σ all GPU pools × gpus</td>"
            f'<td class="na">—</td>'
            f"</tr>"
        )

    if not rows_parts or all(r.startswith('<tr class="summary-row') for r in rows_parts):
        return (
            '<tr><td colspan="7" class="na">'
            "No static pool data found — check the local repo path and git pull status."
            "</td></tr>"
        )

    # The first column is h100_breakdown hidden as a title attribute on hover
    # — keeps the row dense while still surfacing the formula.
    return (
        '<div class="table-scroll">\n'
        '<table class="pool-stats">\n'
        "<thead>\n<tr>\n"
        "  <th>Pipeline</th>\n"
        "  <th>Resource Pool</th>\n"
        "  <th>Total Steps</th>\n"
        "  <th>GPUs / Unit</th>\n"
        "  <th>GPU-Usage</th>\n"
        "  <th>Categories (per-pool count)</th>\n"
        "  <th>Test YAML Files</th>\n"
        "</tr>\n</thead>\n"
        "<tbody>\n" + "\n".join(rows_parts) + "\n"
        "</tbody>\n</table>\n</div>"
    )


def _render_bk_pool_table(all_pools: dict[str, dict[str, PoolStats]]) -> str:
    """Render the daily per-pool table from Buildkite data.

    Restores the original "daily per-pool usage / occupancy / wait" view:
    one row per (pipeline × pool) with the day-aggregated job count,
    distinct build count, avg cards per build, total occupancy (sum of
    runtimes), total wait (sum of queue time), and avg/max/p50/p90 wait.
    Sourced from the Buildkite API — independent of the static YAML detail.

    Adds a `` Device-Hours `` column (Σ duration × accel_count) so users can
    cross-reference per-pool resource consumption against the per-preset
    panel below.  Device-Hours / Build is also surfaced for the "cost per
    build" signal.
    """
    rows_parts: list[str] = []
    for pipeline_slug in sorted(all_pools.keys()):
        pools = all_pools[pipeline_slug]
        for pool_name in sorted(pools.keys()):
            ps = pools[pool_name]
            if ps.wait_seconds:
                sorted_w = sorted(ps.wait_seconds)
                avg_wait_s = sum(ps.wait_seconds) / len(ps.wait_seconds)
                max_wait_s = sorted_w[-1]
                p50_wait_s = percentile(sorted_w, 50) or 0.0
                p90_wait_s = percentile(sorted_w, 90) or 0.0
                total_wait_s = sum(ps.wait_seconds)
            else:
                avg_wait_s = max_wait_s = p50_wait_s = p90_wait_s = total_wait_s = 0.0
            if ps.duration_seconds:
                total_occ_s = sum(ps.duration_seconds)
                avg_dur_s = total_occ_s / len(ps.duration_seconds)
            else:
                total_occ_s = 0.0
                avg_dur_s = 0.0
            build_count = ps.build_count
            # Avg Cards / Build = Σ accel_count across jobs ÷ distinct build
            # numbers. Each multi-card job (h100_4 = 4 cards) is weighted
            # accordingly so a pool running few-but-large jobs reads heavier
            # than a pool running many-but-single-card jobs.
            total_cards = sum(jr.accel_count for jr in ps.jobs)
            avg_cards_per_build = total_cards / build_count if build_count else 0.0
            device_hours = _pool_accelerator_hours(ps)
            device_hours_per_build = device_hours / build_count if build_count else 0.0

            def _td(val: object, cls: str = "") -> str:
                if val is None or val == "":
                    return f'<td class="{cls} na">—</td>'
                return f'<td class="{cls}">{val}</td>'

            rows_parts.append(
                f"<tr>"
                f'<td class="pipeline-cell">{html.escape(pipeline_slug)}</td>'
                f'<td class="pool-name">{html.escape(pool_name)}</td>'
                f"{_td(ps.job_count, 'num')}"
                f"{_td(build_count if build_count else '—', 'num')}"
                f"{_td(f'{avg_cards_per_build:.1f}' if build_count else '—', 'num')}"
                f"{_td(_format_hours(device_hours) if device_hours else '—', 'num')}"
                f"{_td(f'{device_hours_per_build:.1f}h' if build_count else '—', 'num')}"
                f"{_td(format_duration(total_occ_s) if total_occ_s else '—', 'num')}"
                f"{_td(format_duration(avg_dur_s) if avg_dur_s else '—', 'num')}"
                f"{_td(format_duration(total_wait_s) if total_wait_s else '—', 'num')}"
                f"{_td(format_duration(avg_wait_s) if ps.wait_seconds else '—', 'num')}"
                f"{_td(format_duration(max_wait_s) if ps.wait_seconds else '—', 'num')}"
                f"{_td(format_duration(p50_wait_s) if ps.wait_seconds else '—', 'num')}"
                f"{_td(format_duration(p90_wait_s) if ps.wait_seconds else '—', 'num')}"
                f"</tr>"
            )

    if not rows_parts:
        return (
            '<p class="na">No Buildkite build data for the date range — check the token / pipeline / date window.</p>'
        )

    return (
        '<div class="table-scroll">\n'
        '<table class="pool-stats">\n'
        "<thead>\n<tr>\n"
        "  <th>Pipeline</th>\n"
        "  <th>Resource Pool</th>\n"
        "  <th>Jobs</th>\n"
        "  <th>Builds</th>\n"
        "  <th>Avg Cards / Build</th>\n"
        "  <th>Device-Hours</th>\n"
        "  <th>Device-Hours / Build</th>\n"
        "  <th>Total Occupancy</th>\n"
        "  <th>Avg Duration</th>\n"
        "  <th>Total Wait</th>\n"
        "  <th>Avg Wait</th>\n"
        "  <th>Max Wait</th>\n"
        "  <th>P50 Wait</th>\n"
        "  <th>P90 Wait</th>\n"
        "</tr>\n</thead>\n"
        "<tbody>\n" + "\n".join(rows_parts) + "\n"
        "</tbody>\n</table>\n</div>"
    )


# Default cap on rows shown in the Job-Level Detail table.  The full set
# is collected in PoolStats.jobs; the renderer picks the top-N by duration.
JOB_LEVEL_TABLE_LIMIT = 50


def _render_device_hours_by_preset(
    all_pools: dict[str, dict[str, PoolStats]],
) -> str:
    """Render the Device-Hours by Preset panel — split into GPU + NPU sub-tables.

    Groups jobs by inferred preset (h100_1..h100_4, l4_1, l4_4, a2b3_npu_*,
    a3_npu_*) and surfaces, **per device-type sub-table**:
      • Total accelerator-hours (∑ duration × accel_count / 3600)
      • Share within the device type (each preset's % of its own
        type's total — GPU share is computed against GPU subtotal,
        NPU share against NPU subtotal; cross-type comparison is
        intentionally avoided so the smaller NPU footprint doesn't
        get drowned in GPU totals)
      • Job count
      • Card count (∑ accel_count — multi-card jobs contribute their
        full footprint, so a preset running few-but-large jobs reads
        heavier than a preset running many-but-single-card jobs)

    The bar visualizes each preset's intra-type share. Useful for
    spotting drift toward smaller (cheaper) presets within each
    accelerator family.
    """
    rows = _aggregate_device_hours_by_preset(all_pools)
    if not rows:
        return (
            '<p class="na">No accelerator-usage data available — '
            "no jobs reported GPU/NPU counts in the date window.</p>"
        )

    # Split by device type. Within each group, compute share against
    # that group's total so a tiny NPU preset doesn't get a near-zero
    # share just because the GPU pool is much larger overall.  CPU jobs
    # get their own sub-table — they have no "cards" to weight, so the
    # cards column there is intentionally just the job count.
    gpu_rows = [r for r in rows if r["device_type"] == "gpu"]
    npu_rows = [r for r in rows if r["device_type"] == "npu"]
    cpu_rows = [r for r in rows if r["device_type"] == "cpu"]
    gpu_total = sum(r["hours"] for r in gpu_rows)
    npu_total = sum(r["hours"] for r in npu_rows)
    cpu_total = sum(r["hours"] for r in cpu_rows)

    def _render_subtable(
        label: str,
        sub_rows: list[dict],
        sub_total: float,
        type_label: str,
        cards_label: str = "Cards",
    ) -> str:
        if not sub_rows:
            return (
                '<div class="device-hours-subsection">'
                '<h3 class="device-hours-subhead">'
                f"{html.escape(label)}"
                "</h3>"
                f'<p class="na">No {type_label} jobs in this date window.</p>'
                "</div>"
            )
        body_rows: list[str] = []
        for r in sub_rows:
            preset = r["preset"]
            hours = r["hours"]
            jobs = r["jobs"]
            cards = r["cards"]
            share = (hours / sub_total) * 100 if sub_total else 0.0
            bar_width = max(2.0, min(100.0, share))
            body_rows.append(
                "<tr>"
                f'<td class="pool-name">{html.escape(preset)}</td>'
                f'<td class="num">{_format_hours(hours)}</td>'
                f'<td class="num">{share:.1f}%</td>'
                f'<td class="num">{jobs}</td>'
                f'<td class="num">{cards}</td>'
                "<td>"
                f'<div class="device-hours-bar" '
                f'role="img" aria-label="{html.escape(preset)} {share:.1f}% of {type_label} Device-hours">'
                f'<div class="device-hours-bar-fill" style="width: {bar_width:.1f}%"></div>'
                "</div>"
                "</td>"
                "</tr>"
            )
        total_cards = sum(r["cards"] for r in sub_rows)
        # CPU subhead uses "instances" instead of "cards" since a CPU
        # "card" isn't a thing; the count there is just the job count.
        meta_segments = [
            f"{_format_hours(sub_total)} total",
            f"{sum(r['jobs'] for r in sub_rows)} jobs",
        ]
        if type_label == "CPU":
            meta_segments.append(f"{total_cards} instance(s)")
        else:
            meta_segments.append(f"{total_cards} cards")
        meta_joined = " &middot; ".join(meta_segments)
        return (
            '<div class="device-hours-subsection">'
            '<h3 class="device-hours-subhead">'
            f"{html.escape(label)} &middot; "
            f'<span class="device-hours-subhead-meta">{meta_joined}</span></h3>'
            '<div class="table-scroll">'
            '<table class="pool-stats device-hours-table">'
            "<thead><tr>"
            f"  <th>Preset</th>"
            f"  <th>Device-Hours</th>"
            f"  <th>Share</th>"
            f"  <th>Jobs</th>"
            f"  <th>{html.escape(cards_label)}</th>"
            f"  <th>Distribution</th>"
            "</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
            "</div>"
        )

    gpu_section = _render_subtable("GPU", gpu_rows, gpu_total, "GPU")
    npu_section = _render_subtable("NPU", npu_rows, npu_total, "NPU")
    cpu_section = _render_subtable("CPU", cpu_rows, cpu_total, "CPU", cards_label="Instances")

    return gpu_section + "\n" + npu_section + "\n" + cpu_section


# CI category display labels and ordering (for the grouped view's chip list).
_CI_CATEGORY_ORDER = ("ready", "merge", "nightly", "weekly")
_CI_CATEGORY_LABELS = {
    "ready": "ready",
    "merge": "merge",
    "nightly": "nightly",
    "weekly": "weekly",
}
_CI_CATEGORY_CLASS = {
    "ready": "ci-chip--ready",
    "merge": "ci-chip--merge",
    "nightly": "ci-chip--nightly",
    "weekly": "ci-chip--weekly",
}


def _render_job_level_table(
    all_pools: dict[str, dict[str, PoolStats]],
    limit: int = JOB_LEVEL_TABLE_LIMIT,
) -> str:
    """Render the Job-Level Detail table.

    Layout — one sub-card per pipeline:
      • ``vllm-omni`` card and ``vllm-omni-npu-ci`` card, rendered as
        independent ``.cat-card`` panels so each pipeline's busiest jobs
        are visible without crowding the other.
      • Within each card: one row per (job name, resource pool) group,
        sorted by **average** duration descending. Each row shows run
        count, avg/total/max duration, and the set of CI categories
        that ran it. Click the row to expand.
      • Inner (expandable) table — one row per individual job run
        inside the group, with build #, duration, wait, started/finished
        (CST), state, and the CI category for that run.

    Each pipeline is capped at ``limit`` groups independently so a busy
    pipeline doesn't push the other off the page.  The full records
    stay in ``PoolStats.jobs`` for callers that want more.
    """
    # Flatten all jobs across pools/pipelines.
    all_jobs: list[JobRecord] = []
    for pools in all_pools.values():
        for ps in pools.values():
            all_jobs.extend(ps.jobs)

    if not all_jobs:
        return (
            '<p class="na">No individual job records found for the date range — '
            "the Buildkite API returned no ran-state jobs.</p>"
        )

    # Split by pipeline first — each pipeline becomes its own sub-card.
    by_pipeline: dict[str, list[JobRecord]] = defaultdict(list)
    for jr in all_jobs:
        by_pipeline[jr.pipeline].append(jr)

    cards_parts: list[str] = []
    for pipeline_slug in sorted(by_pipeline.keys()):
        cards_parts.append(_render_job_level_card(pipeline_slug, by_pipeline[pipeline_slug], limit))

    summary = (
        f'<p class="meta">Showing top {limit} job-name groups per pipeline '
        f"(aggregated by average occupancy, descending). "
        f"Click a row to expand individual runs; use the per-card CI "
        f"filter chips to narrow each pipeline independently.</p>"
    )

    return summary + '\n<div class="job-level-cards">\n' + "\n".join(cards_parts) + "\n</div>"


def _render_job_level_card(
    pipeline_slug: str,
    pipeline_jobs: list[JobRecord],
    limit: int,
) -> str:
    """Render one pipeline's sub-card for the Job-Level Detail table."""
    # Within a pipeline, group by (job_name, pool_name) — same name in
    # different pools stays separate so the pool attribution is preserved.
    groups: dict[tuple[str, str], list[JobRecord]] = {}
    for jr in pipeline_jobs:
        key = (jr.job_name, jr.pool_name)
        groups.setdefault(key, []).append(jr)

    group_rows: list[dict] = []
    for (name, pool), records in groups.items():
        durations = [r.duration_seconds for r in records if r.duration_seconds > 0]
        ci_cats = {r.ci_category for r in records}
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        total_dur = sum(durations)
        max_dur = max(durations) if durations else 0.0
        # Per-category aggregates so the JS filter can recompute run_count /
        # avg / total / max for whichever subset of CI buckets is currently
        # selected (without re-parsing each inner row's duration string).
        per_cat: dict[str, dict[str, float]] = {}
        for r in records:
            cat = r.ci_category or ""
            dur = r.duration_seconds if r.duration_seconds else 0.0
            slot = per_cat.setdefault(
                cat, {"count": 0, "total": 0.0, "max": 0.0}
            )
            slot["count"] += 1
            if dur > 0:
                slot["total"] += dur
                if dur > slot["max"]:
                    slot["max"] = dur
        group_rows.append(
            {
                "name": name,
                "pool": pool,
                "records": records,
                "run_count": len(records),
                "avg_duration": avg_dur,
                "total_duration": total_dur,
                "max_duration": max_dur,
                "ci_categories": ci_cats,
                "per_cat": per_cat,
            }
        )

    # Sort by average duration descending; ties broken by total, then run count.
    group_rows.sort(key=lambda g: (-g["avg_duration"], -g["total_duration"], -g["run_count"], g["name"]))
    top_groups = group_rows[:limit]
    total_groups = len(group_rows)
    total_jobs = len(pipeline_jobs)
    distinct_pools = len({g["pool"] for g in group_rows})

    def _fmt_cst(dt: datetime | None) -> str:
        if dt is None:
            return "—"
        return dt.astimezone(CST).strftime("%m-%d %H:%M:%S")

    def _ci_chips(cats: set[str]) -> str:
        chips = []
        for cat in _CI_CATEGORY_ORDER:
            if cat in cats:
                label = _CI_CATEGORY_LABELS[cat]
                cls = _CI_CATEGORY_CLASS[cat]
                chips.append(f'<span class="ci-chip {cls}">{html.escape(label)}</span>')
        return "".join(chips) if chips else '<span class="na">—</span>'

    rows_parts: list[str] = []
    for idx, grp in enumerate(top_groups, start=1):
        # Prefix the data-target with the pipeline slug so expand/collapse
        # lookups stay unique across cards.
        group_id = f"{pipeline_slug}-job-grp-{idx}"
        # Sort individual records by duration desc within the group so the
        # longest runs are at the top when expanded.
        sorted_records = sorted(
            grp["records"],
            key=lambda r: (-r.duration_seconds, r.started_at or r.scheduled_at),
        )

        # Comma-separated list of CI categories touched by any run in this
        # group — the JS filter reads this to decide whether to hide the row.
        ci_cats_attr = html.escape(",".join(sorted(grp["ci_categories"])))

        # Per-category aggregates as data-cat-{cat}-{count,avg,total,max}
        # attributes so the JS filter can recompute the group row's metrics
        # across whichever subset of CI buckets is currently selected.
        per_cat_attrs: list[str] = []
        for cat in _CI_CATEGORY_ORDER:
            slot = grp["per_cat"].get(
                cat, {"count": 0, "total": 0.0, "max": 0.0}
            )
            cnt = int(slot["count"])
            tot = float(slot["total"])
            mx = float(slot["max"])
            avg = (tot / cnt) if cnt > 0 else 0.0
            per_cat_attrs.append(f'data-cat-{cat}-count="{cnt}"')
            per_cat_attrs.append(f'data-cat-{cat}-total="{tot:.2f}"')
            per_cat_attrs.append(f'data-cat-{cat}-max="{mx:.2f}"')
            per_cat_attrs.append(f'data-cat-{cat}-avg="{avg:.2f}"')
        per_cat_attr_str = " ".join(per_cat_attrs)

        # Initial "current" values match the all-categories aggregate; the
        # JS filter overwrites these on every checkbox change.
        rows_parts.append(
            f'<tr class="job-group-row" data-target="{group_id}" '
            f'data-ci-categories="{ci_cats_attr}" '
            f'data-current-count="{grp["run_count"]}" '
            f'data-current-avg="{grp["avg_duration"]:.2f}" '
            f'data-current-total="{grp["total_duration"]:.2f}" '
            f'data-current-max="{grp["max_duration"]:.2f}" '
            f'{per_cat_attr_str} '
            f'role="button" tabindex="0" aria-expanded="false">'
            f'<td class="expand-cell"><span class="expand-icon">▶</span></td>'
            f'<td class="pool-name job-name-cell" title="{html.escape(grp["name"])}">'
            f"{html.escape(grp['name'])}</td>"
            f'<td class="pool-name">{html.escape(grp["pool"])}</td>'
            f'<td class="num">{grp["run_count"]}</td>'
            f'<td class="num">{format_duration(grp["avg_duration"]) if grp["avg_duration"] else "—"}</td>'
            f'<td class="num">{format_duration(grp["total_duration"]) if grp["total_duration"] else "—"}</td>'
            f'<td class="num">{format_duration(grp["max_duration"]) if grp["max_duration"] else "—"}</td>'
            f'<td class="ci-cell">{_ci_chips(grp["ci_categories"])}</td>'
            f"</tr>"
        )

        # Inner detail row (hidden until expanded).  Outer table has 8
        # columns since the Pipeline column is gone.
        inner_rows: list[str] = []
        for jrank, jr in enumerate(sorted_records, start=1):
            state_class = ""
            if jr.state in ("failed", "broken", "timed_out"):
                state_class = "job-row--failed"
            elif jr.state == "canceled":
                state_class = "job-row--canceled"
            ci_cls = _CI_CATEGORY_CLASS.get(jr.ci_category, "")
            inner_rows.append(
                f'<tr class="job-detail-inner-row {state_class}" '
                f'data-ci-category="{html.escape(jr.ci_category or "")}">'
                f'<td class="num">{jrank}</td>'
                f'<td class="num">#{html.escape(jr.build_number) or "?"}</td>'
                f'<td class="num">'
                f"{format_duration(jr.duration_seconds) if jr.duration_seconds else '—'}"
                f"</td>"
                f'<td class="num">'
                f"{format_duration(jr.wait_seconds) if jr.wait_seconds else '—'}"
                f"</td>"
                f'<td class="num">{_fmt_cst(jr.started_at)}</td>'
                f'<td class="num">{_fmt_cst(jr.finished_at)}</td>'
                f'<td class="num">'
                f'<span class="ci-chip {ci_cls}">{html.escape(jr.ci_category)}</span>'
                f"</td>"
                f'<td class="num job-state-cell job-state--{html.escape(jr.state)}">'
                f"{html.escape(jr.state.upper())}</td>"
                f"</tr>"
            )
        inner_table = (
            '<table class="inner-table"><thead><tr>'
            "<th>#</th><th>Build</th><th>Duration</th><th>Wait</th>"
            "<th>Started (CST)</th><th>Finished (CST)</th>"
            "<th>CI</th><th>State</th>"
            f"</tr></thead><tbody>{''.join(inner_rows)}</tbody></table>"
        )
        rows_parts.append(
            f'<tr class="job-detail-row" data-group="{group_id}" hidden>'
            f'<td colspan="8" class="job-detail-cell">{inner_table}</td>'
            f"</tr>"
        )

    table_html = (
        '\n<div class="table-scroll">\n'
        '<table class="pool-stats job-level-table">\n'
        "<thead>\n<tr>\n"
        "  <th></th>\n"
        "  <th>Job Name</th>\n"
        "  <th>Resource Pool</th>\n"
        "  <th>Runs</th>\n"
        "  <th>Avg Duration</th>\n"
        "  <th>Total Duration</th>\n"
        "  <th>Max Duration</th>\n"
        "  <th>CI</th>\n"
        "</tr>\n</thead>\n"
        "<tbody>\n" + "\n".join(rows_parts) + "\n"
        "</tbody>\n</table>\n</div>"
    )

    sub = (
        f"top {len(top_groups)} of {total_groups} job-name groups · "
        f"{total_jobs} runs across {distinct_pools} pool(s)"
    )
    # Store the original sub text in a data attribute so the filter JS
    # can rewrite the visible-group count without losing the rest of the
    # text (" X runs across Y pool(s)").
    sub_text = f"— {sub}"

    # Per-pipeline CI-category filter — chips sit inside the card head,
    # right side, scoped to this card only.  All four are checked by
    # default (everything visible).  JS reads the input state and
    # hides any outer group row whose union of CI categories has no
    # intersection with the checked set.
    filter_chips = "".join(
        f'<label class="filter-chip filter-chip--{cat}">'
        f'<input type="checkbox" data-ci-filter="{cat}" checked>'
        f"<span>{html.escape(label)}</span>"
        f"</label>"
        for cat, label in (
            ("ready", "ready"),
            ("merge", "merge"),
            ("nightly", "nightly"),
            ("weekly", "weekly"),
        )
    )
    filter_html = (
        '<div class="job-level-filter" role="group" '
        f'aria-label="Filter {html.escape(pipeline_slug)} by CI category">'
        '<span class="filter-label">Filter</span>'
        f"{filter_chips}"
        "</div>"
    )

    return (
        f'<div class="cat-card cat-card--pipeline">\n'
        f'  <div class="cat-card-head">\n'
        f'    <span class="cat-card-label">{html.escape(pipeline_slug)}'
        f'<span class="cat-card-sub" '
        f'data-original-sub="{html.escape(sub_text, quote=True)}">'
        f"{sub_text}</span></span>\n"
        f"    {filter_html}\n"
        f"  </div>\n"
        f"  {table_html}\n"
        f"</div>"
    )


# Small inline script that wires the expand/collapse behavior for the
# Job-Level Detail table.  Loaded as a string so the HTML page stays
# self-contained.
_JOB_LEVEL_EXPAND_JS = r"""
<script>
(function () {
  function toggleGroup(row) {
    var target = row.getAttribute('data-target');
    if (!target) return;
    var detailRows = document.querySelectorAll(
      'tr.job-detail-row[data-group="' + target + '"]'
    );
    if (!detailRows.length) return;
    var willOpen = detailRows[0].hasAttribute('hidden');
    detailRows.forEach(function (r) {
      if (willOpen) r.removeAttribute('hidden');
      else r.setAttribute('hidden', '');
    });
    row.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    var icon = row.querySelector('.expand-icon');
    if (icon) icon.textContent = willOpen ? '▼' : '▶';
  }

  // CI-category filter — scoped per pipeline sub-card.  Each pipeline
  // has its own filter bar inside its card head; toggling chips only
  // re-filters rows in that card (other pipelines are untouched).
  //
  // For matching groups we also recompute the group row's run_count /
  // avg_duration / total_duration / max_duration across the currently
  // selected CI buckets only (so a group with mixed ready + nightly jobs
  // shows the *nightly* average when only nightly is checked), and then
  // re-sort the group rows by the new avg so the rank reflects the
  // filtered subset — not the unfiltered population.
  function fmtDur(seconds) {
    if (!seconds || seconds <= 0) return '—';
    seconds = Math.round(seconds);
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    if (h > 0) return h + 'h' + m + 'm' + s + 's';
    if (m > 0) return m + 'm' + s + 's';
    return s + 's';
  }

  // Mirror of the Python _CI_CATEGORY_* helpers so the JS filter can
  // rebuild the outer group row's CI chip cell on the fly (column 8).
  var CI_ORDER = ['ready', 'merge', 'nightly', 'weekly'];
  var CI_LABELS = {
    ready: 'ready', merge: 'merge', nightly: 'nightly', weekly: 'weekly'
  };
  var CI_CLASS = {
    ready: 'ci-chip--ready',
    merge: 'ci-chip--merge',
    nightly: 'ci-chip--nightly',
    weekly: 'ci-chip--weekly'
  };

  function renderChipsHtml(cats) {
    var chips = [];
    for (var i = 0; i < CI_ORDER.length; i++) {
      var cat = CI_ORDER[i];
      if (cats.indexOf(cat) >= 0) {
        chips.push(
          '<span class="ci-chip ' + CI_CLASS[cat] + '">' +
          CI_LABELS[cat] +
          '</span>'
        );
      }
    }
    return chips.length ? chips.join('') : '<span class="na">—</span>';
  }

  function recomputeGroupAggregate(row, selected, anyChecked) {
    var cats = anyChecked
      ? Object.keys(selected)
      : ['ready', 'merge', 'nightly', 'weekly'];
    var count = 0, total = 0, max = 0;
    for (var i = 0; i < cats.length; i++) {
      var cat = cats[i];
      count += parseInt(
        row.getAttribute('data-cat-' + cat + '-count') || '0', 10
      );
      total += parseFloat(
        row.getAttribute('data-cat-' + cat + '-total') || '0'
      );
      var catMax = parseFloat(
        row.getAttribute('data-cat-' + cat + '-max') || '0'
      );
      if (catMax > max) max = catMax;
    }
    var avg = count > 0 ? total / count : 0;
    row.setAttribute('data-current-count', String(count));
    row.setAttribute('data-current-avg', avg.toFixed(2));
    row.setAttribute('data-current-total', total.toFixed(2));
    row.setAttribute('data-current-max', max.toFixed(2));
    // Build the visible-CI set: categories that have at least one inner
    // row in this group AND (when a filter is active) belong to the
    // selected set. No filter → show every category present in the group.
    var presentCats = [];
    for (var i = 0; i < CI_ORDER.length; i++) {
      var cat = CI_ORDER[i];
      var catCount = parseInt(
        row.getAttribute('data-cat-' + cat + '-count') || '0', 10
      );
      if (catCount > 0) presentCats.push(cat);
    }
    var visibleCats = anyChecked
      ? presentCats.filter(function (c) { return selected[c]; })
      : presentCats;
    var cells = row.querySelectorAll('td');
    if (cells.length >= 8) {
      cells[3].textContent = String(count);
      cells[4].textContent = fmtDur(avg);
      cells[5].textContent = fmtDur(total);
      cells[6].textContent = fmtDur(max);
      cells[7].innerHTML = renderChipsHtml(visibleCats);
    }
  }

  function sortGroupRowsByAvg(card) {
    var table = card.querySelector('table.job-level-table');
    if (!table) return;
    var tbody = table.querySelector('tbody');
    if (!tbody) return;
    var rows = Array.prototype.slice.call(
      tbody.querySelectorAll('tr.job-group-row')
    );
    var nameOf = function (r) {
      var td = r.querySelectorAll('td')[1];
      return td ? (td.textContent || '') : '';
    };
    rows.sort(function (a, b) {
      var avgA = parseFloat(a.getAttribute('data-current-avg') || '0');
      var avgB = parseFloat(b.getAttribute('data-current-avg') || '0');
      var totA = parseFloat(a.getAttribute('data-current-total') || '0');
      var totB = parseFloat(b.getAttribute('data-current-total') || '0');
      var cntA = parseInt(a.getAttribute('data-current-count') || '0', 10);
      var cntB = parseInt(b.getAttribute('data-current-count') || '0', 10);
      if (avgB !== avgA) return avgB - avgA;
      if (totB !== totA) return totB - totA;
      if (cntB !== cntA) return cntB - cntA;
      return nameOf(a).localeCompare(nameOf(b));
    });
    // Re-attach in sorted order, keeping each group's detail-row wrapper
    // glued to its parent group so the expand/collapse pairing survives.
    var newOrder = [];
    rows.forEach(function (r) {
      newOrder.push(r);
      var target = r.getAttribute('data-target');
      if (target) {
        tbody
          .querySelectorAll('tr.job-detail-row[data-group="' + target + '"]')
          .forEach(function (d) { newOrder.push(d); });
      }
    });
    newOrder.forEach(function (r) { tbody.appendChild(r); });
  }

  function applyCiFilterToCard(card) {
    if (!card) return;
    var table = card.querySelector('table.job-level-table');
    if (!table) return;

    var selected = {};
    card.querySelectorAll('input[data-ci-filter]').forEach(function (cb) {
      if (cb.checked) selected[cb.getAttribute('data-ci-filter')] = true;
    });
    var anyChecked = Object.keys(selected).length > 0;
    var visibleGroups = 0;
    var totalGroups = 0;

    table.querySelectorAll('tr.job-group-row').forEach(function (row) {
      totalGroups++;
      var raw = row.getAttribute('data-ci-categories') || '';
      var cats = raw ? raw.split(',') : [];
      var match = anyChecked && cats.some(function (c) { return selected[c]; });
      var target = row.getAttribute('data-target');
      if (match) {
        row.style.display = '';
        visibleGroups++;
      } else {
        row.style.display = 'none';
      }
      // For matching groups, show the wrapper so the user can expand; for
      // non-matching groups, hide the wrapper entirely. Once expanded, the
      // wrapper's individual ``tr.job-detail-inner-row`` rows are filtered by
      // their own ``data-ci-category`` attribute so a group that mixes
      // (e.g.) ready + nightly jobs only reveals the category currently
      // selected — no leakage from sibling categories.
      if (target) {
        document.querySelectorAll(
          'tr.job-detail-row[data-group="' + target + '"]'
        ).forEach(function (d) {
          d.style.display = match ? '' : 'none';
          if (match) {
            d.querySelectorAll('tr.job-detail-inner-row').forEach(function (ir) {
              var irCat = ir.getAttribute('data-ci-category') || '';
              ir.style.display = selected[irCat] ? '' : 'none';
            });
          }
        });
      }
      // Recompute the group row's aggregate cells (run_count / avg / total
      // / max) so they reflect the currently selected CI subset, not the
      // full population of inner rows.
      recomputeGroupAggregate(row, selected, anyChecked);
    });

    // Re-sort the group rows by the freshly recomputed avg so the table
    // ordering tracks the filter (descending by avg, ties broken by total,
    // count, then job name).
    sortGroupRowsByAvg(card);

    var sub = card.querySelector('.cat-card-sub');
    if (sub && sub.dataset.originalSub) {
      if (visibleGroups === 0) {
        sub.textContent =
          '— no groups match the selected CI categories';
      } else {
        // Replace only the leading "top N of M" while keeping the rest
        // ("X runs across Y pool(s)") intact.
        sub.textContent = sub.dataset.originalSub.replace(
          /top \d+ of (\d+)/,
          'top ' + visibleGroups + ' of $1'
        );
      }
    }
  }

  document.addEventListener('click', function (e) {
    var row = e.target.closest('tr.job-group-row');
    if (row) toggleGroup(row);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    var row = e.target.closest('tr.job-group-row');
    if (row) {
      e.preventDefault();
      toggleGroup(row);
    }
  });
  document.querySelectorAll('input[data-ci-filter]').forEach(function (cb) {
    cb.addEventListener('change', function (e) {
      // Scope to the changed checkbox's containing pipeline card so
      // each pipeline's filter is independent.
      var card = e.target.closest('.cat-card--pipeline');
      applyCiFilterToCard(card);
    });
  });
})();
</script>
"""


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
    static_data: dict[str, StaticPipelineData],
    date_from: str,
    date_to: str,
) -> str:
    """Render all pool stats as a self-contained HTML page with trend charts."""

    from_utc, _ = cst_day_to_utc_window(date_from)
    _, to_utc = cst_day_to_utc_window(date_to)

    cards = _compute_summary_cards(all_pools)

    # Per-Pool Detail section is sourced from local YAML, NOT Buildkite.
    # The detailed per-preset table was removed in favor of inline totals
    # on each Pool Usage by CI Category card (h100_total / gpu_total per
    # pipeline × category), per the user's request to keep the static view
    # compact.
    cat_html = _render_latest_builds_by_category_html(static_data)
    bk_pool_table_html = _render_bk_pool_table(all_pools)
    job_level_table_html = _render_job_level_table(all_pools)
    device_hours_panel_html = _render_device_hours_by_preset(all_pools)

    # Build the source-meta line for the static YAML portion
    static_files = []
    static_heads = []
    for pslug, pdata in static_data.items():
        if pdata.files:
            static_files.append(f"{pslug}: {', '.join(pdata.files)}")
        if pdata.git_head:
            static_heads.append(f"{pslug}@{pdata.git_head[:12]}")
    static_meta = (
        f"Sourced from local vllm-omni repo after git pull "
        f"({' / '.join(static_files) or 'no files'}; "
        f"HEAD {' / '.join(static_heads) or 'unknown'})."
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
        "  <dd>For the Buildkite-driven sections: derived from each job's "
        "<code>agent_query_rules</code> (<code>queue=…</code>); jobs "
        "without an explicit queue go into <code>default</code>.<br>\n"
        "  For the Per-Pool Detail section: parsed from "
        "<code>.buildkite/test-*.yml</code> by resolving "
        "<code>mirror_hardwares</code> presets via "
        "<code>.buildkite/common/ci_mirror_hardwares.yml</code>.</dd>\n"
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
        f" Per-Pool Detail</span></h2>\n" + (cat_html + "\n" if cat_html else "") + "\n" + legend_html + "\n"
        f'<p class="meta">{html.escape(static_meta)}</p>\n'
        "</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_SERVER}</span>'
        f" Daily Resource Pool Usage (Buildkite)</span></h2>\n"
        f'  <p class="meta">Per-pool job count, distinct build count, avg '
        f"cards per build (Σ accel_count across jobs ÷ distinct build "
        f"numbers — multi-card jobs are weighted accordingly), total "
        f"accelerator-hours (Σ duration × accel_count), Device-hours "
        f"per build, total occupancy (sum of runtimes), and queue wait "
        f"metrics for the date window. Sourced from Buildkite.</p>\n" + bk_pool_table_html + "\n</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_TREND}</span>'
        f" Device-Hours by Preset</span></h2>\n"
        f'  <p class="meta">Total accelerator-hours split into separate '
        f"GPU and NPU sub-tables. Within each sub-table, share is computed "
        f"against that device type&apos;s subtotal (not the day-wide total) "
        f"so the smaller NPU footprint is visible next to GPU. Use the "
        f"distribution bars to spot whether CI is drifting toward smaller "
        f"(cheaper) presets within each accelerator family.</p>\n" + device_hours_panel_html + "\n</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_CHART}</span>'
        f" Job-Level Detail</span></h2>\n"
        f'  <p class="meta">One sub-card per pipeline — individual Buildkite '
        f"job runs grouped by job name within each pipeline, sorted by "
        f"average occupancy (duration) descending so the longest jobs surface "
        f"first. Click a row to expand and see every run inside that group. "
        f"Useful for spotting which tests are eating the most GPU/NPU time "
        f"on each pool.</p>\n" + job_level_table_html + "\n</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_TREND}</span>'
        f" Hourly Trends (CST, UTC+8)</span></h2>\n" + charts_html + "\n"
        "</div>\n"
        f'<p class="meta">Source: <code>scripts/resource_pool_stats.py</code>; '
        f"pipelines: {html.escape(', '.join(all_pools.keys()))}; "
        f"window: <code>{html.escape(date_from)}</code> — "
        f"<code>{html.escape(date_to)}</code> CST (UTC+8; "
        f"maps to <code>{from_utc}</code> — <code>{to_utc}</code> UTC).</p>\n"
        "</div>\n" + _JOB_LEVEL_EXPAND_JS + "</body>\n</html>"
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


def format_stats_markdown(
    all_pools: dict[str, dict[str, PoolStats]],
    static_data: dict[str, StaticPipelineData],
    date_from: str,
    date_to: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# CI Resource Pool Statistics ({date_from} ~ {date_to} CST, UTC+8)")
    lines.append("")
    lines.append(
        f"Source: `scripts/resource_pool_stats.py`; "
        f"pipelines: {', '.join(all_pools.keys())}; "
        f"window: `{date_from}` — `{date_to}` CST (UTC+8)."
    )
    lines.append("")

    # Per-Pool Detail: static YAML section
    lines.append("## Per-Pool Detail (Static YAML from local vllm-omni repo)")
    lines.append("")
    if not static_data:
        lines.append("*No static YAML data — local repo not found or no pipelines mapped.*")
    else:
        static_rows: list[list[str]] = []
        for pipeline_slug in sorted(static_data.keys()):
            pdata = static_data[pipeline_slug]
            for pe in sorted(
                pdata.pools.values(),
                key=lambda pe: (
                    0 if _is_h100_preset(pe.preset_name, pe.queue) else 1,
                    pe.preset_name or pe.queue,
                ),
            ):
                cats_disp = ", ".join(f"{c}:{n}" for c, n in sorted(pe.categories.items()))
                files_disp = ", ".join(f for f, _ in sorted(pe.files.items()))
                display = f"{pe.preset_name} → {pe.queue}" if pe.preset_name else pe.queue
                static_rows.append(
                    [
                        pipeline_slug,
                        display,
                        str(pe.total_steps),
                        str(pe.gpus_per_unit) if pe.gpus_per_unit else "—",
                        str(pe.gpus_per_unit * pe.total_steps) if pe.gpus_per_unit else "—",
                        cats_disp,
                        files_disp,
                    ]
                )
            # Summary rows
            h100_total = sum(
                pe.gpus_per_unit * pe.total_steps
                for pe in pdata.pools.values()
                if _is_h100_preset(pe.preset_name, pe.queue)
            )
            gpu_total = sum(
                pe.gpus_per_unit * pe.total_steps
                for pe in pdata.pools.values()
                if _is_gpu_preset(pe.preset_name, pe.queue)
            )
            static_rows.append(
                [pipeline_slug, "**h100_total**", "—", "—", str(h100_total), "Σ h100 presets × gpus", "—"]
            )
            static_rows.append(
                [pipeline_slug, "**gpu_total**", "—", "—", str(gpu_total), "Σ all GPU pools × gpus", "—"]
            )
        lines.append(
            _render_markdown_table(
                [
                    "Pipeline",
                    "Resource Pool",
                    "Total Steps",
                    "GPUs / Unit",
                    "GPU-Usage",
                    "Categories",
                    "Test YAML Files",
                ],
                static_rows,
            )
        )
        # Per-category breakdown
        lines.append("")
        lines.append("### Pool Usage by CI Category")
        lines.append("")
        for pipeline_slug in sorted(static_data.keys()):
            pdata = static_data[pipeline_slug]
            if not pdata.categories:
                continue
            for cat_key, cat_label, cat_sub in CATEGORY_ORDER:
                cat_buckets = pdata.categories.get(cat_key) or {}
                if not cat_buckets:
                    continue
                # Per-pipeline × per-category totals (mirrors the HTML card)
                h100_cat_total = 0
                gpu_cat_total = 0
                for key, count in cat_buckets.items():
                    pe = pdata.pools.get(key)
                    if pe is None or not pe.gpus_per_unit:
                        continue
                    contribution = pe.gpus_per_unit * count
                    if _is_h100_preset(pe.preset_name, pe.queue):
                        h100_cat_total += contribution
                    if _is_gpu_preset(pe.preset_name, pe.queue):
                        gpu_cat_total += contribution

                lines.append(f"- **{pipeline_slug} · {cat_label}** — {cat_sub}")
                lines.append("")

                cat_table_rows: list[list[str]] = []
                for name, n in sorted(cat_buckets.items(), key=lambda kv: (-kv[1], kv[0])):
                    pe = pdata.pools.get(name)
                    gpu_usage = str(pe.gpus_per_unit * n) if pe and pe.gpus_per_unit else "—"
                    cat_table_rows.append([name, str(n), gpu_usage])
                cat_table_rows.append(["**h100_total**", "—", str(h100_cat_total)])
                cat_table_rows.append(["**gpu_total**", "—", str(gpu_cat_total)])

                lines.append(
                    _render_markdown_table(
                        ["Resource Pool", "Steps", "GPU-Usage (steps × gpus)"],
                        cat_table_rows,
                    )
                )
                lines.append("")
        for pslug, pdata in static_data.items():
            if pdata.git_head:
                lines.append(f"_Source: {pdata.repo_path} @ {pdata.git_head[:12]}_")
        lines.append("")

    # Buildkite-driven hourly / pool aggregates
    lines.append("## Buildkite Pool Aggregates")
    lines.append("")
    headers = [
        "Pipeline",
        "Resource Pool",
        "Jobs",
        "Builds",
        "Avg Cards / Build",
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
            build_count = ps.build_count
            avg_jobs_str = f"{ps.job_count / build_count:.1f}" if build_count else "N/A"
            rows.append(
                [
                    pipeline_slug,
                    pool_name,
                    str(ps.job_count),
                    str(build_count) if build_count else "0",
                    avg_jobs_str,
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


def format_stats_json(
    all_pools: dict[str, dict[str, PoolStats]],
    static_data: dict[str, StaticPipelineData],
    date_from: str,
    date_to: str,
) -> str:
    output: dict = {"date_range": {"from": date_from, "to": date_to}, "pipelines": {}}

    # Buildkite-driven pipeline aggregates
    for pipeline_slug, pools in all_pools.items():
        pipeline_data: dict = {}
        for pool_name in sorted(pools.keys()):
            ps = pools[pool_name]
            pool_data: dict = {
                "pool_name": pool_name,
                "job_count": ps.job_count,
                "build_count": ps.build_count,
                "total_cards": sum(jr.accel_count for jr in ps.jobs),
                "avg_cards_per_build": round(sum(jr.accel_count for jr in ps.jobs) / ps.build_count, 2)
                if ps.build_count
                else None,
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

    # Static YAML per-pool data (Per-Pool Detail section)
    output["static_yaml_per_pool_detail"] = {}
    for pipeline_slug, pdata in static_data.items():
        per_pool: dict[str, dict] = {}
        for pe in sorted(
            pdata.pools.values(),
            key=lambda pe: (
                0 if _is_h100_preset(pe.preset_name, pe.queue) else 1,
                pe.preset_name or pe.queue,
            ),
        ):
            key = pe.preset_name if pe.preset_name else pe.queue
            per_pool[key] = {
                "queue": pe.queue,
                "preset_name": pe.preset_name,
                "gpus_per_unit": pe.gpus_per_unit,
                "total_steps": pe.total_steps,
                "gpu_usage": pe.gpus_per_unit * pe.total_steps,
                "categories": dict(pe.categories),
                "files": dict(pe.files),
                "mirror_hardwares": dict(pe.mirror_hardwares),
            }
        # Summary rows
        h100_total = sum(
            pe.gpus_per_unit * pe.total_steps
            for pe in pdata.pools.values()
            if _is_h100_preset(pe.preset_name, pe.queue)
        )
        gpu_total = sum(
            pe.gpus_per_unit * pe.total_steps for pe in pdata.pools.values() if _is_gpu_preset(pe.preset_name, pe.queue)
        )
        per_pool["h100_total"] = {"gpu_usage": h100_total, "note": "Σ h100 presets × gpus"}
        per_pool["gpu_total"] = {"gpu_usage": gpu_total, "note": "Σ all GPU pools × gpus"}
        output["static_yaml_per_pool_detail"][pipeline_slug] = {
            "repo_path": pdata.repo_path,
            "git_head": pdata.git_head,
            "files": list(pdata.files),
            "categories": {cat: dict(buckets) for cat, buckets in pdata.categories.items()},
            "pools": per_pool,
        }

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
        help="Start date (CST calendar date, inclusive). Omit both --from and --to to use today CST.",
    )
    parser.add_argument(
        "--to",
        dest="created_to",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (CST calendar date, inclusive). Omit both --from and --to to use today CST.",
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
        "--repo-path",
        dest="repo_path",
        default=DEFAULT_LOCAL_REPO_PATH,
        metavar="PATH",
        help=(
            "Path to the local vllm-omni git repo used by the Per-Pool Detail "
            "section. The script runs `git pull --ff-only` here and parses "
            "`.buildkite/test-*.yml`. Default: " + DEFAULT_LOCAL_REPO_PATH
        ),
    )
    parser.add_argument(
        "--skip-git-pull",
        dest="skip_git_pull",
        action="store_true",
        help="Skip the `git pull` in the local repo (use whatever's currently on disk).",
    )
    args = parser.parse_args()

    if args.created_from is None and args.created_to is None:
        args.created_from, args.created_to = today_range_cst()
    elif args.created_from is None or args.created_to is None:
        print(
            "resource_pool_stats.py: pass both --from and --to, or omit both (defaults to today CST).",
            file=sys.stderr,
        )
        return 2

    pipeline_slugs: list[str]
    if args.pipelines:
        pipeline_slugs = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    else:
        pipeline_slugs = DEFAULT_PIPELINES

    # Per-Pool Detail section: statically computed from the local vllm-omni
    # git repo (after git pull).  Independent of the Buildkite token; runs
    # even when BUILDKITE_API_TOKEN is missing so we can show *what pools
    # the YAML intends to use* without the API.
    print(
        f"Computing Per-Pool Detail from local repo {args.repo_path} "
        f"(git pull={'off' if args.skip_git_pull else 'on'})…"
    )
    static_data = compute_static_pool_data(
        Path(args.repo_path),
        pipeline_slugs,
        skip_git_pull=args.skip_git_pull,
    )

    # Populate the (pipeline, ci_category) → {label: accel_count} lookup so
    # the per-job aggregation loop can recover each job's accelerator
    # count from its Buildkite job name.  Without this every job's
    # ``accel_count`` falls back to 1, which makes Device-Hours
    # mathematically equal to Total Occupancy.
    _LABEL_TO_ACCEL_BY_PIPELINE.clear()
    for pipeline_slug, pdata in static_data.items():
        _LABEL_TO_ACCEL_BY_PIPELINE[pipeline_slug] = dict(pdata.label_to_accel)
    total_mapped = sum(len(by_label) for by_cat in _LABEL_TO_ACCEL_BY_PIPELINE.values() for by_label in by_cat.values())
    print(
        f"YAML label→accel lookup populated: {total_mapped} (pipeline,category,label) triples "
        f"across {len(_LABEL_TO_ACCEL_BY_PIPELINE)} pipeline(s)."
    )

    token = get_api_token()
    if not token:
        print("BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is not set; cannot call the Buildkite API.", file=sys.stderr)
        print(
            "Set one in the environment and retry (Per-Pool Detail still works, but hourly trends will be empty).",
            file=sys.stderr,
        )
        return 1

    all_pools: dict[str, dict[str, PoolStats]] = {}
    for pipeline_slug in pipeline_slugs:
        try:
            all_pools[pipeline_slug] = compute_pool_stats(
                token,
                pipeline_slug,
                args.created_from,
                args.created_to,
                verbose=args.verbose,
            )
        except requests.RequestException as e:
            print(f"API request failed for {pipeline_slug}: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                print(f"HTTP status: {e.response.status_code}", file=sys.stderr)
                print(e.response.text[:500], file=sys.stderr)
            all_pools[pipeline_slug] = {}

    if args.output_format == "html":
        html_content = format_stats_html(all_pools, static_data, args.created_from, args.created_to)
        if args.output_path:
            out_path = Path(args.output_path)
        else:
            out_path = Path(f"pool-stats-{args.created_from}.html")
        out_path.write_text(html_content, encoding="utf-8")
        print(f"HTML report written to {out_path}")
    elif args.output_format == "markdown":
        print(format_stats_markdown(all_pools, static_data, args.created_from, args.created_to))
    elif args.output_format == "json":
        print(format_stats_json(all_pools, static_data, args.created_from, args.created_to))

    return 0


if __name__ == "__main__":
    sys.exit(main())
