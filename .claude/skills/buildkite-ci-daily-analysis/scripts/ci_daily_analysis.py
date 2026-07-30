#!/usr/bin/env python3
"""
Fetch yesterday's Buildkite builds (in Beijing Time / CST) for vllm-omni
and vllm-omni-npu-ci and analyze per-job and per-build success/failure/duration,
then emit a self-contained HTML report with interactive **Pipeline** /
**Branch** / **CI** / **State** / **Job Name** filter dropdowns plus a CI
Aggregate panel broken down by ``ready`` / ``merge`` / ``nightly`` / ``weekly``.

The default window is the **previous full Beijing Time (CST, UTC+8) calendar
day** — i.e. yesterday 00:00 — 23:59 CST, which maps to the UTC span
``(yesterday-1) 16:00 UTC`` → ``yesterday 15:59:59 UTC``. Pass ``--today``
for the current CST day, or ``--date YYYY-MM-DD`` for an arbitrary CST day.

> **Note on the 18:00 UTC nightly run:** under the old UTC default, the
> nightly job triggered at 18:00 UTC fell into "yesterday UTC". Under the
> CST default it falls at 02:00 CST on the next calendar day and is part of
> that day's window, not yesterday's.

For each ``script``/``command`` job, the script records:

  - pipeline (vllm-omni / vllm-omni-npu-ci)
  - branch, ci_bucket, build number, job name, job state, job URL
  - duration seconds (finished_at - started_at)

For each build (regardless of job count), the script records the build's
own ``state`` and wall-clock duration (``finished_at - started_at``) so the
CI Aggregate panel can show build-level success rate and runtime per bucket.

Usage:

  export BUILDKITE_API_TOKEN=...
  python scripts/ci_daily_analysis.py                  # default = yesterday CST
  python scripts/ci_daily_analysis.py --date 2026-07-22   # explicit CST date
  python scripts/ci_daily_analysis.py --today          # current CST day
  python scripts/ci_daily_analysis.py --pipeline vllm-omni,vllm-omni-npu-ci
  python scripts/ci_daily_analysis.py --output my-report.html

Default output is an HTML file written to ``ci-daily-YYYY-MM-DD.html`` in
the current directory.
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

PIPELINE_DISPLAY = {
    "vllm-omni": "vllm-omni (GPU)",
    "vllm-omni-npu-ci": "vllm-omni-npu-ci (NPU)",
}

# CI bucket names — used for the CI Aggregate panel and as `data-ci-bucket`
# attributes on aggregate cards. The mapping follows the test-report skill:
#   - ready:    non-`main` branches
#   - merge:    `main`, ordinary runs (not scheduled nightly / weekly)
#   - nightly:  `main`, scheduled nightly (or legacy "nightly"/"scheduled"+"build")
#   - weekly:   `main`, "scheduled weekly" build message
CI_BUCKET_READY = "ready"
CI_BUCKET_MERGE = "merge"
CI_BUCKET_NIGHTLY = "nightly"
CI_BUCKET_WEEKLY = "weekly"
CI_BUCKET_ORDER = [CI_BUCKET_READY, CI_BUCKET_MERGE, CI_BUCKET_NIGHTLY, CI_BUCKET_WEEKLY]
CI_BUCKET_DISPLAY = {
    CI_BUCKET_READY: "Ready (non-main branches)",
    CI_BUCKET_MERGE: "Merge (main, ordinary runs)",
    CI_BUCKET_NIGHTLY: "Nightly (main, scheduled)",
    CI_BUCKET_WEEKLY: "Weekly (main, scheduled weekly)",
}

# Job names that are Buildkite orchestration / setup steps rather than real
# CI work. By default these are filtered out of the report so they don't
# pollute success-rate and duration aggregates. Override with
# `--include-infra` (keep them) or `--exclude-jobs "regex1,regex2"`
# (extend the list). Patterns are matched as case-insensitive regexes
# against the job label as reported by Buildkite.
DEFAULT_JOB_SKIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^:pipeline:\s*init\s*$", re.IGNORECASE),
    re.compile(r"^:docker:\s*Build image\s*$", re.IGNORECASE),
    re.compile(r"^:buildkit:\s*Build and Push\b", re.IGNORECASE),
    re.compile(r"resolve\s+skip-ci", re.IGNORECASE),
    re.compile(r"upload[^\n]*\bpipeline\b", re.IGNORECASE),
    re.compile(r"collect[\s_\-]*results?", re.IGNORECASE),
]


def should_skip_job(job_name: str, extra_patterns: list[re.Pattern[str]] | None = None) -> bool:
    """True if the job label matches any of the default or extra skip patterns."""
    patterns = DEFAULT_JOB_SKIP_PATTERNS + (extra_patterns or [])
    return any(p.search(job_name or "") for p in patterns)


def compile_extra_patterns(raw: str | None) -> list[re.Pattern[str]]:
    """Parse comma-separated regex strings from `--exclude-jobs`."""
    if not raw:
        return []
    out: list[re.Pattern[str]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(re.compile(piece, re.IGNORECASE))
    return out


_WEEKLY_MSG = re.compile(r"scheduled\s+weekly", re.IGNORECASE)


def _is_scheduled_weekly(build: dict) -> bool:
    """True if the build's message indicates a Buildkite *Scheduled weekly* run."""
    return bool(_WEEKLY_MSG.search(build.get("message") or ""))


def _is_nightly_bucket(build: dict) -> bool:
    """
    True if a `main`-branch build counts as **nightly**.

    Excludes scheduled weekly (handled separately). Includes other scheduled
    `main` runs (`source == "schedule"`) and legacy message heuristics
    (`"nightly"` substring, or `"scheduled"` + `"build"`).
    """
    if _is_scheduled_weekly(build):
        return False
    source = (build.get("source") or "").strip().lower()
    if source == "schedule":
        return True
    msg = (build.get("message") or "").lower()
    if "nightly" in msg:
        return True
    if "scheduled" in msg and "build" in msg:
        return True
    return False


def classify_build(build: dict) -> str:
    """Map a Buildkite build object to one of the four CI buckets."""
    branch = (build.get("branch") or "").strip()
    if branch != "main":
        return CI_BUCKET_READY
    if _is_scheduled_weekly(build):
        return CI_BUCKET_WEEKLY
    if _is_nightly_bucket(build):
        return CI_BUCKET_NIGHTLY
    return CI_BUCKET_MERGE


# State buckets used for analytics
STATE_PASSED = "passed"
STATE_FAILED = "failed"
STATE_CANCELED = "canceled"
STATE_RUNNING = "running"
STATE_OTHER = "other"  # scheduled / blocked / skipped / not_run / broken / unknown

# Job state → bucket
STATE_BUCKET = {
    "passed": STATE_PASSED,
    "failed": STATE_FAILED,
    "canceled": STATE_CANCELED,
    "running": STATE_RUNNING,
    "scheduled": STATE_OTHER,
    "blocked": STATE_OTHER,
    "skipped": STATE_OTHER,
    "not_run": STATE_OTHER,
    "broken": STATE_OTHER,
}


# ── Editorial CSS (aligned with the dashboard palette) ──────────────────

ANALYSIS_CSS = """
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
  --ok-bg: rgba(31, 157, 99, 0.1);
  --ok-edge: var(--dashboard-healthy);
  --fail-bg: var(--dashboard-alert-bg);
  --fail-edge: var(--dashboard-alert);
  --neutral-bg: rgba(148, 163, 184, 0.12);
  --neutral-edge: #94a3b8;
  --warn-bg: rgba(217, 119, 6, 0.12);
  --warn-edge: #d97706;
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
    --neutral-bg: rgba(148, 163, 184, 0.16);
    --neutral-edge: #94a3b8;
    --warn-bg: rgba(251, 191, 36, 0.16);
    --warn-edge: #fbbf24;
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
  max-width: 1400px;
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
  display: flex;
  align-items: center;
  gap: 0.4rem;
}
.focus-card-value {
  margin-top: 0.2rem;
  font-size: 1.4rem;
  font-weight: 820;
  color: var(--dashboard-text);
  font-variant-numeric: tabular-nums;
}
.focus-card-detail {
  margin-top: 0.15rem;
  color: var(--dashboard-muted);
  font-size: 0.84rem;
}
.ico { display: block; }

/* Filter row */
.filter-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.85rem;
  align-items: flex-end;
  margin: 0.85rem 0 1rem;
  padding: 0.95rem 1rem;
  background: color-mix(in srgb, var(--ci-tint) 60%, var(--dashboard-panel-bg));
  border: 1px solid var(--ci-soft);
  border-radius: var(--radius-sm);
}
.filter-field {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  min-width: 11rem;
  flex: 1 1 13rem;
}
.filter-field label {
  font-size: 0.74rem;
  font-weight: 760;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--dashboard-muted);
}
.filter-field select {
  appearance: none;
  -webkit-appearance: none;
  background: var(--dashboard-panel-bg);
  border: 1px solid var(--dashboard-border-strong);
  border-radius: var(--radius-sm);
  padding: 0.55rem 2.2rem 0.55rem 0.7rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--dashboard-text);
  background-image: linear-gradient(45deg, transparent 50%, var(--dashboard-muted) 50%),
                    linear-gradient(135deg, var(--dashboard-muted) 50%, transparent 50%);
  background-position: right 1rem center, right 0.65rem center;
  background-size: 6px 6px;
  background-repeat: no-repeat;
  cursor: pointer;
}
.filter-field select:focus {
  outline: 2px solid var(--ci-soft);
  outline-offset: 1px;
  border-color: var(--ci);
}
.filter-reset {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.55rem 0.9rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--dashboard-border-strong);
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  font-size: 0.88rem;
  font-weight: 650;
  cursor: pointer;
  transition: background 0.12s ease, border-color 0.12s ease;
}
.filter-reset:hover {
  background: var(--accent-tint);
  border-color: var(--accent);
}
.filter-status {
  margin-left: auto;
  color: var(--dashboard-muted);
  font-size: 0.88rem;
  align-self: center;
  font-variant-numeric: tabular-nums;
}
.filter-status strong {
  color: var(--dashboard-text);
}

/* Tables */
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
  min-width: 760px;
}
table.job-analysis {
  border-collapse: collapse;
  font-size: 0.92rem;
}
table.job-analysis th, table.job-analysis td {
  border: 1px solid var(--border);
  padding: 0.55rem 0.7rem;
  text-align: left;
  vertical-align: top;
}
table.job-analysis th {
  background: var(--dashboard-panel-strong);
  font-weight: 650;
  color: var(--dashboard-chart-text);
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 1;
}
table.job-analysis tbody tr:nth-child(even) td {
  background: color-mix(in srgb, var(--dashboard-badge-bg) 45%, var(--dashboard-panel-bg));
}
table.job-analysis td.num {
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
table.job-analysis td.pipeline-cell {
  border-left: 3px solid var(--ci);
  padding-left: calc(0.7rem - 2px);
  font-weight: 650;
  color: var(--dashboard-violet);
}
table.job-analysis td.branch-cell {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 0.86rem;
  color: var(--dashboard-soft-text);
}
table.job-analysis td.name-cell {
  font-weight: 650;
  color: var(--dashboard-text);
}
table.job-analysis td.url-cell a {
  color: var(--ci);
  text-decoration: none;
  font-weight: 650;
}
table.job-analysis td.url-cell a:hover {
  text-decoration: underline;
}

/* State pills */
.state-pill {
  display: inline-block;
  padding: 0.18rem 0.6rem;
  border-radius: 999px;
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  border: 1px solid transparent;
}
.state-pill.passed {
  background: var(--ok-bg);
  border-color: var(--ok-edge);
  color: var(--ok-edge);
}
.state-pill.failed {
  background: var(--fail-bg);
  border-color: var(--fail-edge);
  color: var(--fail-edge);
}
.state-pill.canceled {
  background: var(--warn-bg);
  border-color: var(--warn-edge);
  color: var(--warn-edge);
}
.state-pill.running,
.state-pill.other {
  background: var(--neutral-bg);
  border-color: var(--neutral-edge);
  color: var(--neutral-edge);
}
.state-pill.failed-text {
  color: var(--dashboard-alert);
  font-weight: 760;
}
.state-pill.passed-text {
  color: var(--ok-edge);
  font-weight: 760;
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
.no-data {
  color: var(--dashboard-muted);
  font-style: italic;
  text-align: center;
  padding: 1.2rem 1rem;
}

/* Aggregate section */
.aggregate-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
  gap: 0.85rem;
  margin: 0.85rem 0 0;
}
.aggregate-card {
  border-radius: var(--radius-sm);
  border: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-bg);
  padding: 0.85rem 0.95rem;
}
.aggregate-card h3 {
  margin: 0 0 0.6rem;
  font-size: 0.92rem;
  font-weight: 750;
  color: var(--dashboard-violet);
  border-bottom: 1px solid var(--dashboard-border);
  padding-bottom: 0.4rem;
}
.aggregate-card .row {
  display: flex;
  justify-content: space-between;
  font-size: 0.88rem;
  margin: 0.18rem 0;
}
.aggregate-card .row span:first-child {
  color: var(--dashboard-muted);
}
.aggregate-card .row span:last-child {
  font-weight: 650;
  color: var(--dashboard-text);
  font-variant-numeric: tabular-nums;
}
.aggregate-card .agg-section-title {
  font-size: 0.72rem;
  font-weight: 760;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--dashboard-muted);
  margin-bottom: 0.4rem;
}

/* Pill-button toggle (Job level | Build level) on CI Aggregate cards */
.aggregate-card .agg-toggle {
  display: inline-flex;
  gap: 0;
  padding: 3px;
  border: 1px solid var(--dashboard-border-strong);
  border-radius: 999px;
  background: var(--surface-muted);
  margin-bottom: 0.7rem;
  box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
}
.aggregate-card .agg-toggle input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
  width: 0;
  height: 0;
}
.aggregate-card .agg-toggle label {
  padding: 0.32rem 0.85rem;
  font-size: 0.74rem;
  font-weight: 760;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  border-radius: 999px;
  cursor: pointer;
  user-select: none;
  transition: background 0.12s ease, color 0.12s ease;
}
.aggregate-card .agg-toggle label:hover {
  color: var(--dashboard-text);
}
.aggregate-card .agg-toggle input:checked + label {
  background: var(--dashboard-panel-bg);
  color: var(--ci);
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12);
}

/* CI bucket cell in the Job-level Detail table */
table.job-analysis td.ci-cell {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: lowercase;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  display: inline-block;
  white-space: nowrap;
  margin: 0.15rem 0;
}
table.job-analysis td.ci-cell--ready {
  background: var(--accent-tint);
  color: var(--accent);
  border: 1px solid var(--accent-soft);
}
table.job-analysis td.ci-cell--merge {
  background: var(--ok-bg);
  color: var(--ok);
  border: 1px solid var(--ok-edge);
}
table.job-analysis td.ci-cell--nightly {
  background: var(--warn-bg);
  color: var(--warn-edge);
  border: 1px solid var(--warn-edge);
}
table.job-analysis td.ci-cell--weekly {
  background: var(--ci-tint);
  color: var(--ci);
  border: 1px solid var(--ci-soft);
}
"""

# ── SVG icons ────────────────────────────────────────────────────────────

ICON_JOBS = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<path d="M9 9h6M9 13h6M9 17h4"/>'
    "</svg>"
)

ICON_CHECK = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/>'
    '<polyline points="9 12 11 14 15 10"/>'
    "</svg>"
)

ICON_X = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/>'
    '<line x1="9" y1="9" x2="15" y2="15"/>'
    '<line x1="15" y1="9" x2="9" y2="15"/>'
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

ICON_TREND = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
    "</svg>"
)

ICON_FILTER = (
    '<svg class="ico" width="24" height="24" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>'
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


def today_range_cst(now: datetime | None = None) -> tuple[str, str]:
    """Today CST as YYYY-MM-DD strings."""
    ref = now or datetime.now(CST)
    today = ref.date()
    return today.isoformat(), today.isoformat()


def yesterday_range_cst(now: datetime | None = None) -> tuple[str, str]:
    """Yesterday CST as YYYY-MM-DD strings. This is the default reporting
    window — the daily report covers the previous day's full 00:00–23:59
    CST (which maps to (yesterday-1) 16:00 UTC → yesterday 15:59:59 UTC)."""
    ref = now or datetime.now(CST)
    yesterday = (ref - timedelta(days=1)).date()
    return yesterday.isoformat(), yesterday.isoformat()


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


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


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


def success_rate(passed: int, failed: int) -> float | None:
    denom = passed + failed
    if denom == 0:
        return None
    return passed / denom * 100.0


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


# ── Buildkite API helpers ────────────────────────────────────────────────


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


# ── Job record dataclass ─────────────────────────────────────────────────


@dataclass
class JobRecord:
    pipeline: str
    branch: str
    build_number: int
    build_url: str
    commit: str
    job_id: str
    job_name: str
    state: str
    bucket: str
    ci_bucket: str
    build_message: str
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    job_url: str
    exit_status: int | None


@dataclass
class BuildRecord:
    """One Buildkite build, for the CI Aggregate build-level stats.

    Build-level stats answer "did the build itself succeed and how long
    did it take?", independent of which jobs were filtered as infra.
    """

    pipeline: str
    branch: str
    build_number: int
    state: str
    bucket: str
    ci_bucket: str
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None
    build_url: str
    message: str


@dataclass
class AggregateStats:
    """Bucket of jobs (or builds, when used at the build level) for one
    (pipeline, ci_bucket, ...) aggregation."""

    job_count: int = 0
    passed: int = 0
    failed: int = 0
    canceled: int = 0
    running: int = 0
    other: int = 0
    durations: list[float] = field(default_factory=list)

    def add(self, rec) -> None:
        """``rec`` is any object with ``.bucket`` (one of STATE_*) and
        ``.duration_seconds`` (Optional[float]). Both JobRecord and
        BuildRecord satisfy this."""
        self.job_count += 1
        if rec.bucket == STATE_PASSED:
            self.passed += 1
        elif rec.bucket == STATE_FAILED:
            self.failed += 1
        elif rec.bucket == STATE_CANCELED:
            self.canceled += 1
        elif rec.bucket == STATE_RUNNING:
            self.running += 1
        else:
            self.other += 1
        if rec.duration_seconds is not None and rec.duration_seconds >= 0:
            self.durations.append(rec.duration_seconds)

    def avg_duration(self) -> float | None:
        return (sum(self.durations) / len(self.durations)) if self.durations else None

    def p50_duration(self) -> float | None:
        return percentile(sorted(self.durations), 50) if self.durations else None

    def p90_duration(self) -> float | None:
        return percentile(sorted(self.durations), 90) if self.durations else None

    def max_duration(self) -> float | None:
        return max(self.durations) if self.durations else None

    def sr(self) -> float | None:
        return success_rate(self.passed, self.failed)


# ── Data collection ──────────────────────────────────────────────────────


def collect_jobs(
    token: str,
    pipeline_slug: str,
    date_str: str,
    *,
    verbose: bool = False,
    include_infra: bool = False,
    extra_skip_patterns: list[re.Pattern[str]] | None = None,
) -> tuple[list[JobRecord], list[BuildRecord], list[tuple[str, str]]]:
    """
    Fetch all builds for a pipeline on a given CST calendar date and return:

      - a flat list of per-job records (script/command jobs only);
      - a flat list of per-build records (one per build, regardless of
        how many jobs ran) for the build-level CI Aggregate stats;
      - a list of (job_name, build_number) tuples for jobs that were
        filtered out as infrastructure.

    Pass ``include_infra=True`` to keep orchestration/setup jobs in the
    output (matches against ``DEFAULT_JOB_SKIP_PATTERNS`` are skipped).
    Pass ``extra_skip_patterns`` to extend the skip list at runtime.
    """
    from_utc, to_utc = cst_day_to_utc_window(date_str)
    print(f"Fetching {ORG_SLUG}/{pipeline_slug} builds for {date_str} CST (UTC {from_utc} — {to_utc})...")
    builds = fetch_builds(token, pipeline_slug, date_str, date_str)
    print(f"Fetched {len(builds)} build(s) for {pipeline_slug}.")

    records: list[JobRecord] = []
    build_records: list[BuildRecord] = []
    skipped: list[tuple[str, str]] = []

    for b in builds:
        b = ensure_build_with_jobs(token, pipeline_slug, b)
        jobs = b.get("jobs") or []
        if verbose:
            bnum = b.get("number", "?")
            bstate = (b.get("state") or "").strip()
            print(f"  Build #{bnum} state={bstate} jobs={len(jobs)}")

        branch = (b.get("branch") or "").strip() or "(unknown)"
        commit = (b.get("commit") or "").strip()
        bnum = b.get("number")
        build_url = (b.get("web_url") or "").strip()
        build_message = (b.get("message") or "").strip()
        ci_bucket = classify_build(b)

        # ── Build-level record (independent of job-level infra filter)
        build_state_raw = (b.get("state") or "").strip().lower() or "unknown"
        build_bucket = STATE_BUCKET.get(build_state_raw, STATE_OTHER)
        build_started = parse_buildkite_time(b.get("started_at"))
        build_finished = parse_buildkite_time(b.get("finished_at"))
        build_duration: float | None = None
        if build_started is not None and build_finished is not None:
            build_duration = (build_finished - build_started).total_seconds()
            if build_duration < 0:
                build_duration = None
        build_records.append(
            BuildRecord(
                pipeline=pipeline_slug,
                branch=branch,
                build_number=bnum if bnum is not None else 0,
                state=build_state_raw,
                bucket=build_bucket,
                ci_bucket=ci_bucket,
                started_at=build_started,
                finished_at=build_finished,
                duration_seconds=build_duration,
                build_url=build_url,
                message=build_message,
            )
        )

        for j in jobs:
            jtype = (j.get("type") or "").strip().lower()
            if jtype not in ("script", "command"):
                continue

            job_name = (j.get("name") or "").strip() or "(unnamed)"

            if not include_infra and should_skip_job(job_name, extra_skip_patterns):
                skipped.append((job_name, str(bnum) if bnum is not None else "?"))
                continue

            raw_state = (j.get("state") or "").strip().lower() or "unknown"
            bucket = STATE_BUCKET.get(raw_state, STATE_OTHER)

            started_at = parse_buildkite_time(j.get("started_at"))
            finished_at = parse_buildkite_time(j.get("finished_at"))
            duration: float | None = None
            if started_at is not None and finished_at is not None:
                duration = (finished_at - started_at).total_seconds()
                if duration < 0:
                    duration = None

            exit_status = j.get("exit_status")
            if exit_status is not None:
                try:
                    exit_status = int(exit_status)
                except (ValueError, TypeError):
                    exit_status = None

            records.append(
                JobRecord(
                    pipeline=pipeline_slug,
                    branch=branch,
                    build_number=bnum if bnum is not None else 0,
                    build_url=build_url,
                    commit=commit[:7] if commit else "",
                    job_id=str(j.get("id") or ""),
                    job_name=job_name,
                    state=raw_state,
                    bucket=bucket,
                    ci_bucket=ci_bucket,
                    build_message=build_message,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=duration,
                    job_url=(j.get("web_url") or "").strip(),
                    exit_status=exit_status,
                )
            )

    return records, build_records, skipped


# ── HTML rendering ───────────────────────────────────────────────────────


def _summary_cards(records: list[JobRecord]) -> list[dict]:
    n = len(records)
    passed = sum(1 for r in records if r.bucket == STATE_PASSED)
    failed = sum(1 for r in records if r.bucket == STATE_FAILED)
    sr = success_rate(passed, failed)
    durations = [r.duration_seconds for r in records if r.duration_seconds is not None]
    avg_dur = (sum(durations) / len(durations)) if durations else None

    sr_str = f"{sr:.1f}%" if sr is not None else "N/A"
    avg_dur_str = format_duration(avg_dur) if avg_dur is not None else "N/A"
    return [
        {"title": "Total Jobs", "value": str(n), "detail": "all pipelines · today CST", "icon": ICON_JOBS},
        {
            "title": "Passed",
            "value": str(passed),
            "detail": f"{passed / n * 100:.1f}% of all jobs" if n else "—",
            "icon": ICON_CHECK,
        },
        {
            "title": "Failed",
            "value": str(failed),
            "detail": f"{failed / n * 100:.1f}% of all jobs" if n else "—",
            "icon": ICON_X,
        },
        {"title": "Success Rate", "value": sr_str, "detail": "passed / (passed + failed)", "icon": ICON_TREND},
        {"title": "Avg Duration", "value": avg_dur_str, "detail": f"{len(durations)} job(s) timed", "icon": ICON_CLOCK},
    ]


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


def _aggregate_card_html(
    title: str,
    agg: AggregateStats,
    *,
    data_ci_bucket: str | None = None,
    build_stats: AggregateStats | None = None,
    toggle_id_suffix: str | None = None,
) -> str:
    """Render a single aggregate card.

    Without ``build_stats``, the card shows a single "Job level" section.

    With ``build_stats``, the card renders a pill-button toggle
    (Job level | Build level). Default visible section is Job level; the
    toggle is wired up by the inline script in ``render_html`` so clicking
    the other pill swaps which section is visible.
    """
    data_attr = f' data-ci-bucket="{html.escape(data_ci_bucket)}"' if data_ci_bucket is not None else ""

    def _section(label: str, a: AggregateStats, *, hidden: bool = False) -> str:
        sr_local = a.sr()
        sr_local_str = f"{sr_local:.1f}%" if sr_local is not None else "N/A"
        noun = "jobs" if label == "Job" else "builds"
        hidden_attr = ' style="display:none"' if hidden else ""
        return (
            f'<div class="agg-section" data-section="{html.escape(label.lower())}"{hidden_attr}>\n'
            f'  <div class="row"><span>Total {noun}</span><span>{a.job_count}</span></div>\n'
            f'  <div class="row"><span>Passed</span><span>{a.passed}</span></div>\n'
            f'  <div class="row"><span>Failed</span><span>{a.failed}</span></div>\n'
            f'  <div class="row"><span>Canceled</span><span>{a.canceled}</span></div>\n'
            f'  <div class="row"><span>Running</span><span>{a.running}</span></div>\n'
            f'  <div class="row"><span>Other</span><span>{a.other}</span></div>\n'
            f'  <div class="row"><span>Success rate</span><span>{sr_local_str}</span></div>\n'
            f'  <div class="row"><span>Avg duration</span><span>{format_duration(a.avg_duration())}</span></div>\n'
            f'  <div class="row"><span>P50 duration</span><span>{format_duration(a.p50_duration())}</span></div>\n'
            f'  <div class="row"><span>P90 duration</span><span>{format_duration(a.p90_duration())}</span></div>\n'
            f'  <div class="row"><span>Max duration</span><span>{format_duration(a.max_duration())}</span></div>\n'
            f"</div>"
        )

    if build_stats is None:
        body = f'<h3>{html.escape(title)}</h3>\n<div class="agg-section-title">Job level</div>\n' + _section("Job", agg)
    else:
        toggle_id = (
            f"agg-toggle-{html.escape(toggle_id_suffix)}"
            if toggle_id_suffix
            else f"agg-toggle-{html.escape(data_ci_bucket or 'card')}"
        )
        body = (
            f"<h3>{html.escape(title)}</h3>\n"
            f'<div class="agg-toggle" id="{toggle_id}">\n'
            f'  <input type="radio" id="{toggle_id}-job" name="{toggle_id}" value="job" checked>\n'
            f'  <label for="{toggle_id}-job">Job level</label>\n'
            f'  <input type="radio" id="{toggle_id}-build" name="{toggle_id}" value="build">\n'
            f'  <label for="{toggle_id}-build">Build level</label>\n'
            f"</div>\n" + _section("Job", agg) + _section("Build", build_stats, hidden=True)
        )

    return f'<div class="aggregate-card"{data_attr}>\n{body}\n</div>'


def _row_html(rec: JobRecord, idx: int) -> str:
    state_pill = f'<span class="state-pill {html.escape(rec.bucket)}">{html.escape(rec.state)}</span>'
    dur_str = format_duration(rec.duration_seconds)
    # Timestamps are stored in UTC; render them in Beijing Time (CST, UTC+8).
    started = rec.started_at.astimezone(CST).strftime("%H:%M:%S") if rec.started_at else "—"
    finished = rec.finished_at.astimezone(CST).strftime("%H:%M:%S") if rec.finished_at else "—"

    pipeline_label = PIPELINE_DISPLAY.get(rec.pipeline, rec.pipeline)
    branch = rec.branch or "(unknown)"

    job_link = f'<a href="{html.escape(rec.job_url)}" target="_blank" rel="noopener">open ↗</a>' if rec.job_url else "—"

    ci_label = rec.ci_bucket
    return (
        f'<tr data-pipeline="{html.escape(rec.pipeline)}" '
        f'data-branch="{html.escape(branch)}" '
        f'data-ci-bucket="{html.escape(rec.ci_bucket)}" '
        f'data-job-name="{html.escape(rec.job_name)}" '
        f'data-state="{html.escape(rec.state)}">\n'
        f'  <td class="pipeline-cell">{html.escape(pipeline_label)}</td>\n'
        f'  <td class="branch-cell">{html.escape(branch)}</td>\n'
        f'  <td class="ci-cell ci-cell--{html.escape(ci_label)}">{html.escape(ci_label)}</td>\n'
        f'  <td class="num">#{rec.build_number}</td>\n'
        f'  <td class="name-cell">{html.escape(rec.job_name)}</td>\n'
        f"  <td>{state_pill}</td>\n"
        f'  <td class="num">{html.escape(dur_str)}</td>\n'
        f'  <td class="num">{html.escape(started)}</td>\n'
        f'  <td class="num">{html.escape(finished)}</td>\n'
        f'  <td class="url-cell">{job_link}</td>\n'
        f"</tr>"
    )


def _state_color_class(bucket: str) -> str:
    return bucket  # bucket name matches CSS class


def _filter_options_html(
    pipelines: list[str],
    branches: list[str],
    job_name_counts: dict[str, int],
    state_counts: dict[str, int],
    ci_buckets: list[str] = CI_BUCKET_ORDER,
) -> str:
    """Render the filter row.

    ``pipelines``     — list of pipeline slugs (used to populate Pipeline).
    ``branches``      — list of branch names (used to populate Branch).
    ``job_name_counts``— {job_name: total_count} (all rows, no filter).
    ``state_counts``  — {state: total_count} (all rows, no filter).
    ``ci_buckets``    — list of CI bucket names (fixed order).

    The HTML is populated server-side so the dropdowns work even if JS
    hasn't run yet; JS still rebuilds on filter changes to only show
    values that survive the upstream selections.
    """
    pipeline_options = ['<option value="__ALL__">All pipelines</option>']
    for p in pipelines:
        label = PIPELINE_DISPLAY.get(p, p)
        pipeline_options.append(f'<option value="{html.escape(p)}">{html.escape(label)}</option>')

    branch_options = ['<option value="__ALL__">All branches</option>']
    for b in branches:
        branch_options.append(f'<option value="{html.escape(b)}">{html.escape(b)}</option>')

    ci_options = ['<option value="__ALL__">All CI buckets</option>']
    for b in ci_buckets:
        ci_options.append(f'<option value="{html.escape(b)}">{html.escape(b)}</option>')

    # Job Name dropdown — server-side: list all job names with their counts.
    job_options = ['<option value="__ALL__">All job names</option>']
    for name in sorted(job_name_counts):
        job_options.append(
            f'<option value="{html.escape(name)}">{html.escape(name)}  ({job_name_counts[name]})</option>'
        )

    # State dropdown — server-side: list all states with their counts.
    state_options = ['<option value="__ALL__">All states</option>']
    for st in sorted(state_counts):
        state_options.append(f'<option value="{html.escape(st)}">{html.escape(st)}  ({state_counts[st]})</option>')

    return (
        f'<div class="filter-row">\n'
        f'  <div class="filter-field">\n'
        f'    <label for="filter-pipeline">Pipeline</label>\n'
        f'    <select id="filter-pipeline">\n'
        f"      {''.join(pipeline_options)}\n"
        f"    </select>\n"
        f"  </div>\n"
        f'  <div class="filter-field">\n'
        f'    <label for="filter-branch">Branch</label>\n'
        f'    <select id="filter-branch">\n'
        f"      {''.join(branch_options)}\n"
        f"    </select>\n"
        f"  </div>\n"
        f'  <div class="filter-field">\n'
        f'    <label for="filter-ci">CI</label>\n'
        f'    <select id="filter-ci">\n'
        f"      {''.join(ci_options)}\n"
        f"    </select>\n"
        f"  </div>\n"
        f'  <div class="filter-field">\n'
        f'    <label for="filter-state">State</label>\n'
        f'    <select id="filter-state">\n'
        f"      {''.join(state_options)}\n"
        f"    </select>\n"
        f"  </div>\n"
        f'  <div class="filter-field">\n'
        f'    <label for="filter-job-name">Job Name</label>\n'
        f'    <select id="filter-job-name">\n'
        f"      {''.join(job_options)}\n"
        f"    </select>\n"
        f"  </div>\n"
        f'  <button type="button" class="filter-reset" id="filter-reset">↺ Reset</button>\n'
        f'  <span class="filter-status">Showing '
        f'<strong id="visible-count">—</strong> of '
        f'<strong id="total-count">—</strong> jobs</span>\n'
        f"</div>"
    )


def _filter_script(total: int) -> str:
    # Stable state order: passed first, then failed, then running/canceled,
    # then everything else (scheduled/blocked/skipped/not_run/broken/unknown).
    # Used to alphabetize the State dropdown after a rebuild.
    return f"""
<script>
(function() {{
  var pipelineSel = document.getElementById('filter-pipeline');
  var branchSel = document.getElementById('filter-branch');
  var ciSel = document.getElementById('filter-ci');
  var stateSel = document.getElementById('filter-state');
  var jobSel = document.getElementById('filter-job-name');
  var resetBtn = document.getElementById('filter-reset');
  var visibleEl = document.getElementById('visible-count');
  var totalEl = document.getElementById('total-count');
  var rows = Array.prototype.slice.call(document.querySelectorAll('tr[data-pipeline]'));
  if (totalEl) totalEl.textContent = '{total}';

  // Fixed ordering for the CI dropdown so the four buckets always appear in
  // the same order regardless of which jobs are present.
  var CI_ORDER = ['ready', 'merge', 'nightly', 'weekly'];
  // Preferred ordering for the State dropdown. Jobs whose state is not in
  // this list are appended alphabetically.
  var STATE_ORDER = ['passed', 'failed', 'running', 'canceled', 'scheduled',
                     'blocked', 'skipped', 'not_run', 'broken', 'unknown'];

  // Helper: append a single <option> to a <select>.
  function appendOpt(sel, value, label) {{
    var opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    sel.appendChild(opt);
  }}

  // Helper: build a counts map (key = attr value, value = count) from the
  // rows that match the given predicate. The predicate receives the row.
  function countsFromRows(predicate, attr) {{
    var counts = Object.create(null);
    rows.forEach(function(r) {{
      if (predicate(r)) {{
        var v = r.getAttribute(attr);
        counts[v] = (counts[v] || 0) + 1;
      }}
    }});
    return counts;
  }}

  // hasOwn() works on null-prototype objects (unlike `.hasOwnProperty`).
  var hasOwn = Object.prototype.hasOwnProperty;

  // Helper: refill a <select> from a counts map using the given preferred
  // order. Keys not in the order list are appended alphabetically.
  function refillSelectWithCounts(sel, counts, order, allLabel) {{
    var prev = sel.value;
    var keys = Object.keys(counts);
    var inOrder = order.filter(function(k) {{ return hasOwn.call(counts, k); }});
    var rest = keys.filter(function(k) {{ return order.indexOf(k) < 0; }})
                    .sort(function(a, b) {{ return a.localeCompare(b); }});
    sel.innerHTML = '';
    appendOpt(sel, '__ALL__', allLabel);
    inOrder.concat(rest).forEach(function(k) {{
      appendOpt(sel, k, k + '  (' + counts[k] + ')');
    }});
    if (hasOwn.call(counts, prev)) {{
      sel.value = prev;
    }} else {{
      sel.value = '__ALL__';
    }}
  }}

  // Predicate helpers — match rows that pass all upstream filters.
  function pipelineOk(r) {{
    return pipelineSel.value === '__ALL__' || r.getAttribute('data-pipeline') === pipelineSel.value;
  }}
  function branchOk(r) {{
    return branchSel.value === '__ALL__' || r.getAttribute('data-branch') === branchSel.value;
  }}
  function ciOk(r) {{
    return ciSel.value === '__ALL__' || r.getAttribute('data-ci-bucket') === ciSel.value;
  }}
  function stateOk(r) {{
    return stateSel.value === '__ALL__' || r.getAttribute('data-state') === stateSel.value;
  }}

  // Rebuild the Branch dropdown so it lists only branches that exist under
  // the currently selected Pipeline.
  function rebuildBranchOptions() {{
    var counts = countsFromRows(pipelineOk, 'data-branch');
    var keys = Object.keys(counts).sort(function(a, b) {{ return a.localeCompare(b); }});
    var prev = branchSel.value;
    branchSel.innerHTML = '';
    appendOpt(branchSel, '__ALL__', 'All branches');
    keys.forEach(function(b) {{ appendOpt(branchSel, b, b); }});
    branchSel.value = hasOwn.call(counts, prev) ? prev : '__ALL__';
  }}

  // Rebuild the CI dropdown so it lists only CI buckets that have at least
  // one job surviving Pipeline + Branch. Order is fixed (CI_ORDER).
  function rebuildCiOptions() {{
    var counts = countsFromRows(function(r) {{ return pipelineOk(r) && branchOk(r); }}, 'data-ci-bucket');
    refillSelectWithCounts(ciSel, counts, CI_ORDER, 'All CI buckets');
  }}

  // Rebuild the State dropdown so it lists only states that have at least
  // one job surviving Pipeline + Branch + CI.
  function rebuildStateOptions() {{
    var counts = countsFromRows(function(r) {{ return pipelineOk(r) && branchOk(r) && ciOk(r); }}, 'data-state');
    refillSelectWithCounts(stateSel, counts, STATE_ORDER, 'All states');
  }}

  // Rebuild the Job Name dropdown so it lists only job names that exist
  // under the currently selected Pipeline + Branch + CI + State combination.
  function rebuildJobNameOptions() {{
    var counts = countsFromRows(
      function(r) {{ return pipelineOk(r) && branchOk(r) && ciOk(r) && stateOk(r); }},
      'data-job-name'
    );
    refillSelectWithCounts(jobSel, counts, [], 'All job names');
  }}

  function applyFilter() {{
    var visible = 0;
    rows.forEach(function(r) {{
      var show = pipelineOk(r) && branchOk(r) && ciOk(r) && stateOk(r) &&
                 (jobSel.value === '__ALL__' || r.getAttribute('data-job-name') === jobSel.value);
      r.style.display = show ? '' : 'none';
      if (show) visible++;
    }});
    if (visibleEl) visibleEl.textContent = visible;
    // Toggle a "no matches" placeholder row so the user gets visual feedback
    // that the filter is working even when 0 rows match.
    var placeholder = document.getElementById('filter-empty-row');
    if (placeholder) {{
      placeholder.style.display = (visible === 0) ? '' : 'none';
    }}
  }}

  // `change` is the canonical event for <select>, but some browsers / mobile
  // keyboards prefer `input`. Wire both so the filter always feels live.
  function wireLive(sel, rebuildFns) {{
    var handler = function() {{
      (rebuildFns || []).forEach(function(fn) {{ fn(); }});
      applyFilter();
    }};
    sel.addEventListener('change', handler);
    sel.addEventListener('input', handler);
  }}

  wireLive(pipelineSel, [rebuildBranchOptions, rebuildCiOptions, rebuildStateOptions, rebuildJobNameOptions]);
  wireLive(branchSel,   [rebuildCiOptions, rebuildStateOptions, rebuildJobNameOptions]);
  wireLive(ciSel,       [rebuildStateOptions, rebuildJobNameOptions]);
  wireLive(stateSel,    [rebuildJobNameOptions]);
  wireLive(jobSel,      []);
  if (resetBtn) {{
    resetBtn.addEventListener('click', function() {{
      // Clear every filter, not just Pipeline — otherwise a previously-picked
      // CI / State / Job Name value would still hide rows.
      pipelineSel.value = '__ALL__';
      branchSel.value = '__ALL__';
      ciSel.value = '__ALL__';
      stateSel.value = '__ALL__';
      jobSel.value = '__ALL__';
      rebuildBranchOptions();
      rebuildCiOptions();
      rebuildStateOptions();
      rebuildJobNameOptions();
      applyFilter();
    }});
  }}
  // Initialize
  rebuildBranchOptions();
  rebuildCiOptions();
  rebuildStateOptions();
  rebuildJobNameOptions();
  applyFilter();
}})();
</script>
"""


def render_html(
    records: list[JobRecord],
    date_str: str,
    pipelines: list[str],
    *,
    skipped_summary: dict[str, int] | None = None,
    build_records: list[BuildRecord] | None = None,
) -> str:
    """
    Render the full self-contained HTML page.

    records:         list of JobRecord sorted however the caller likes.
    date_str:        YYYY-MM-DD (CST calendar date shown in the title and filename).
    pipelines:       pipeline slugs the data was drawn from.
    skipped_summary: optional {job_name: count} map of infrastructure jobs
                     that were filtered out before aggregation.
    build_records:   optional list of BuildRecord for build-level stats in
                     the CI Aggregate panel. Defaults to an empty list.
    """
    cards = _summary_cards(records)
    pipelines_listed = sorted(pipelines)
    branches_listed = sorted({r.branch for r in records})
    # Counts for the filter dropdowns — server-side so the dropdowns are
    # populated even before JS runs. JS still rebuilds on filter changes.
    job_name_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for r in records:
        job_name_counts[r.job_name] = job_name_counts.get(r.job_name, 0) + 1
        state_counts[r.state] = state_counts.get(r.state, 0) + 1

    # ── Aggregate cards: per pipeline, per CI bucket
    by_pipeline: dict[str, AggregateStats] = defaultdict(AggregateStats)
    by_ci: dict[str, AggregateStats] = defaultdict(AggregateStats)
    by_ci_pipeline: dict[tuple[str, str], AggregateStats] = defaultdict(AggregateStats)
    for r in records:
        by_pipeline[r.pipeline].add(r)
        by_ci[r.ci_bucket].add(r)
        by_ci_pipeline[(r.ci_bucket, r.pipeline)].add(r)

    # ── Build-level aggregate per pipeline and per CI bucket
    # (independent of job infra filter; one row per Buildkite build)
    by_pipeline_build: dict[str, AggregateStats] = defaultdict(AggregateStats)
    by_ci_build: dict[str, AggregateStats] = defaultdict(AggregateStats)
    by_ci_pipeline_build: dict[tuple[str, str], AggregateStats] = defaultdict(AggregateStats)
    for b in build_records or []:
        by_pipeline_build[b.pipeline].add(b)
        by_ci_build[b.ci_bucket].add(b)
        by_ci_pipeline_build[(b.ci_bucket, b.pipeline)].add(b)

    # CI Aggregate cards: split by (CI bucket × pipeline). One card per
    # combination, in fixed bucket order then pipeline order. Each card
    # carries a Job level / Build level toggle. The card's data-ci-bucket
    # attribute is set to "<bucket>-<pipeline>" so the toggle id is unique.
    ci_cards_parts: list[str] = []
    for b in CI_BUCKET_ORDER:
        for p in pipelines_listed:
            ci_cards_parts.append(
                _aggregate_card_html(
                    f"{CI_BUCKET_DISPLAY.get(b, b)} · {PIPELINE_DISPLAY.get(p, p)}",
                    by_ci_pipeline.get((b, p), AggregateStats()),
                    data_ci_bucket=f"{b}-{p}",
                    build_stats=by_ci_pipeline_build.get((b, p), AggregateStats()),
                )
            )
    ci_cards_html = "\n".join(ci_cards_parts)

    pipeline_cards_html = "\n".join(
        _aggregate_card_html(
            PIPELINE_DISPLAY.get(p, p),
            by_pipeline.get(p, AggregateStats()),
            build_stats=by_pipeline_build.get(p, AggregateStats()),
            toggle_id_suffix=f"pipeline-{p}",
        )
        for p in pipelines_listed
    )

    # ── Job table rows
    rows = [_row_html(rec, i) for i, rec in enumerate(records)]
    # Always include a "no matches" placeholder row at the end so the filter
    # has something to show when the user narrows the table to 0 rows. The
    # placeholder is hidden by default (style="display:none") and the JS
    # toggles it on when visible === 0.
    empty_row_html = (
        '<tr id="filter-empty-row" style="display:none">'
        '<td colspan="10" class="no-data">'
        "No jobs match the current filters. "
        'Try resetting one of the dropdowns above or click "↺ Reset".'
        "</td></tr>"
    )
    if not rows:
        rows_html = '<tr><td colspan="10" class="no-data">No script jobs found in the specified date range.</td></tr>'
    else:
        rows_html = "\n".join(rows) + "\n" + empty_row_html

    table_html = (
        '<div class="table-scroll">\n'
        '<table class="job-analysis">\n'
        "<thead>\n<tr>\n"
        "  <th>Pipeline</th>\n"
        "  <th>Branch</th>\n"
        "  <th>CI</th>\n"
        "  <th>Build #</th>\n"
        "  <th>Job</th>\n"
        "  <th>State</th>\n"
        "  <th>Duration</th>\n"
        "  <th>Started (CST)</th>\n"
        "  <th>Finished (CST)</th>\n"
        "  <th>Link</th>\n"
        "</tr>\n</thead>\n"
        "<tbody>\n" + rows_html + "\n"
        "</tbody>\n</table>\n</div>"
    )

    filter_html = _filter_options_html(
        pipelines_listed,
        branches_listed,
        job_name_counts,
        state_counts,
    )
    filter_script = _filter_script(total=len(records))

    legend_html = (
        '<div class="legend">\n'
        "<dl>\n"
        "  <dt>Pipeline</dt>\n"
        "  <dd>Buildkite pipeline slug (<code>vllm-omni</code>, <code>vllm-omni-npu-ci</code>).</dd>\n"
        "  <dt>Branch</dt>\n"
        "  <dd>Source branch from the parent build, e.g. <code>main</code>, "
        "<code>alice/add-thing</code>. All branches from the date range are listed; "
        "the dropdown filters the table live.</dd>\n"
        "  <dt>Job Name</dt>\n"
        "  <dd>Buildkite job label (e.g. <code>unit-tests</code>, <code>lint</code>, "
        "<code>diffusion-model-test</code>). The dropdown only shows names that "
        "survive the current Pipeline + Branch filters, with a per-name count.</dd>\n"
        "  <dt>State</dt>\n"
        "  <dd>Job state from the Buildkite API. <code>passed</code> / "
        "<code>failed</code> drive the success rate; "
        "<code>canceled</code>, <code>running</code>, and <code>other</code> are excluded "
        "from the rate so they don't drag it down.</dd>\n"
        "  <dt>Duration</dt>\n"
        "  <dd><code>finished_at − started_at</code> for jobs that have both timestamps. "
        "Jobs still running or missing timestamps show <code>N/A</code>.</dd>\n"
        "  <dt>CI Bucket</dt>\n"
        "  <dd>Per-job classification derived from the parent build:\n"
        "    <code>ready</code> = non-<code>main</code> branch;\n"
        "    <code>merge</code> = <code>main</code>, ordinary run, not scheduled nightly/weekly;\n"
        "    <code>nightly</code> = <code>main</code>, scheduled nightly "
        "(message contains <code>nightly</code>, or <code>source=schedule</code>, "
        "or <code>scheduled</code>+<code>build</code>);\n"
        "    <code>weekly</code> = <code>main</code>, scheduled weekly "
        "(message contains <code>scheduled weekly</code>).</dd>\n"
        "  <dt>Infra filter</dt>\n"
        "  <dd>Buildkite orchestration jobs (default patterns: "
        "<code>:pipeline: init</code>, anything matching <code>Resolve skip-ci</code>, "
        "<code>Upload … Pipeline</code>, <code>Collect results?</code>) are "
        "filtered out of the report so they don't pollute success rate and "
        "duration. Override with <code>--include-infra</code> to keep them, "
        'or <code>--exclude-jobs "regex1,regex2"</code> to extend the list. '
        "Skipped counts are shown in the source footer.</dd>\n"
        "  <dt>Build #</dt>\n"
        "  <dd>Buildkite build number. Click the <strong>open ↗</strong> link to view "
        "the job in Buildkite.</dd>\n"
        "</dl>\n</div>"
    )

    pipeline_label_html = ", ".join(html.escape(PIPELINE_DISPLAY.get(p, p)) for p in pipelines_listed)

    page = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Buildkite Daily Job Analysis — {html.escape(date_str)} CST (UTC+8)</title>\n"
        "<style>\n" + ANALYSIS_CSS + "\n</style>\n"
        "</head>\n"
        "<body>\n"
        '<header class="top-bar">\n'
        '<div class="top-bar-inner">\n'
        '<div class="brand">\n'
        f'  <div class="brand-mark">{ICON_JOBS}</div>\n'
        '  <div class="brand-copy">\n'
        f"    <h1>Buildkite Daily Job Analysis</h1>\n"
        f'    <p class="tagline">{html.escape(date_str)} CST (UTC+8) · Pipelines: {pipeline_label_html}</p>\n'
        "  </div>\n"
        "</div>\n"
        "</div>\n"
        "</header>\n"
        '<div class="shell">\n' + _summary_cards_html(cards) + "\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_TREND}</span>'
        f" Per-Pipeline Aggregate</span></h2>\n"
        f'  <div class="aggregate-grid">\n{pipeline_cards_html}\n  </div>\n'
        "</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_FILTER}</span>'
        f' CI Aggregate <span class="meta" style="border:none;padding:0;margin-left:.4rem;">'
        f"(ready · merge · nightly · weekly · split by pipeline)</span></span></h2>\n"
        f'  <div class="aggregate-grid">\n{ci_cards_html}\n  </div>\n'
        "</div>\n"
        '<div class="panel panel-bk">\n'
        f'  <h2><span class="heading-row"><span class="heading-ico">{ICON_FILTER}</span>'
        f" Job-Level Detail (filterable)</span></h2>\n" + filter_html + "\n" + table_html + "\n" + legend_html + "\n"
        "</div>\n"
        f'<p class="meta">Source: <code>scripts/ci_daily_analysis.py</code>; '
        f"pipelines: {pipeline_label_html}; "
        f"window: <code>{html.escape(date_str)}</code> (00:00 — 23:59 CST, UTC+8)"
        + (
            "; filtered infra jobs: "
            + ", ".join(f"<code>{html.escape(name)}</code>×{count}" for name, count in skipped_summary.items())
            if skipped_summary
            else ""
        )
        + ".</p>\n"
        "</div>\n" + filter_script + "\n" + _toggle_script() + "\n"
        "</body>\n</html>"
    )
    return page


def _toggle_script() -> str:
    """Wire up the Job level / Build level pill-button toggle on each
    CI Aggregate card. When the user clicks a pill, the matching section
    becomes visible and the other is hidden. Default state (Job level)
    is set by the checked attribute in the markup."""
    return """
<script>
(function() {
  var toggles = document.querySelectorAll('.aggregate-card .agg-toggle');
  toggles.forEach(function(toggle) {
    toggle.addEventListener('change', function(ev) {
      var sel = ev.target;
      if (!sel || sel.type !== 'radio') return;
      var card = toggle.closest('.aggregate-card');
      if (!card) return;
      var want = sel.value;
      card.querySelectorAll('.agg-section').forEach(function(sec) {
        sec.style.display = (sec.getAttribute('data-section') === want) ? '' : 'none';
      });
    });
  });
})();
</script>
"""


# ── Markdown / JSON helpers (used only when explicitly requested) ───────


def render_markdown(records: list[JobRecord], date_str: str) -> str:
    if not records:
        return f"# Buildkite Daily Job Analysis ({date_str} CST, UTC+8)\n\nNo jobs found.\n"
    by_pipeline: dict[str, AggregateStats] = defaultdict(AggregateStats)
    by_ci: dict[str, AggregateStats] = defaultdict(AggregateStats)
    for r in records:
        by_pipeline[r.pipeline].add(r)
        by_ci[r.ci_bucket].add(r)

    lines: list[str] = []
    lines.append(f"# Buildkite Daily Job Analysis ({date_str} CST, UTC+8)")
    lines.append("")
    cards = _summary_cards(records)
    for c in cards:
        lines.append(f"- **{c['title']}**: {c['value']} ({c['detail']})")
    lines.append("")

    lines.append("## Per-Pipeline")
    lines.append("")
    lines.append("| Pipeline | Jobs | Passed | Failed | Success | Avg | P50 | P90 | Max |")
    lines.append("|----------|-----:|-------:|-------:|--------:|----:|----:|----:|----:|")
    for p in sorted(by_pipeline):
        agg = by_pipeline[p]
        sr = agg.sr()
        lines.append(
            f"| {PIPELINE_DISPLAY.get(p, p)} | {agg.job_count} | {agg.passed} | {agg.failed} | "
            f"{f'{sr:.1f}%' if sr is not None else 'N/A'} | "
            f"{format_duration(agg.avg_duration())} | "
            f"{format_duration(agg.p50_duration())} | "
            f"{format_duration(agg.p90_duration())} | "
            f"{format_duration(agg.max_duration())} |"
        )
    lines.append("")
    lines.append("## CI Aggregate (ready · merge · nightly · weekly)")
    lines.append("")
    lines.append("| CI Bucket | Jobs | Passed | Failed | Success | Avg | P50 | P90 | Max |")
    lines.append("|-----------|-----:|-------:|-------:|--------:|----:|----:|----:|----:|")
    for b in CI_BUCKET_ORDER:
        agg = by_ci.get(b, AggregateStats())
        sr = agg.sr()
        lines.append(
            f"| `{b}` ({CI_BUCKET_DISPLAY.get(b, b)}) | {agg.job_count} | {agg.passed} | "
            f"{agg.failed} | "
            f"{f'{sr:.1f}%' if sr is not None else 'N/A'} | "
            f"{format_duration(agg.avg_duration())} | "
            f"{format_duration(agg.p50_duration())} | "
            f"{format_duration(agg.p90_duration())} | "
            f"{format_duration(agg.max_duration())} |"
        )
    return "\n".join(lines) + "\n"


def render_json(records: list[JobRecord], date_str: str) -> str:
    by_pipeline: dict[str, AggregateStats] = defaultdict(AggregateStats)
    by_ci: dict[str, AggregateStats] = defaultdict(AggregateStats)
    for r in records:
        by_pipeline[r.pipeline].add(r)
        by_ci[r.ci_bucket].add(r)

    def agg_dict(a: AggregateStats) -> dict:
        sr = a.sr()
        return {
            "job_count": a.job_count,
            "passed": a.passed,
            "failed": a.failed,
            "canceled": a.canceled,
            "running": a.running,
            "other": a.other,
            "success_rate_pct": round(sr, 2) if sr is not None else None,
            "avg_duration_seconds": round(a.avg_duration(), 2) if a.avg_duration() is not None else None,
            "p50_duration_seconds": round(a.p50_duration(), 2) if a.p50_duration() is not None else None,
            "p90_duration_seconds": round(a.p90_duration(), 2) if a.p90_duration() is not None else None,
            "max_duration_seconds": round(a.max_duration(), 2) if a.max_duration() is not None else None,
        }

    payload = {
        "date_cst": date_str,
        "pipelines": {p: agg_dict(by_pipeline.get(p, AggregateStats())) for p in sorted(by_pipeline)},
        "ci_buckets": {b: agg_dict(by_ci.get(b, AggregateStats())) for b in CI_BUCKET_ORDER},
        "jobs": [
            {
                "pipeline": r.pipeline,
                "branch": r.branch,
                "ci_bucket": r.ci_bucket,
                "build_number": r.build_number,
                "job_name": r.job_name,
                "state": r.state,
                "bucket": r.bucket,
                "duration_seconds": r.duration_seconds,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "finished_at": r.finished_at.isoformat() if r.finished_at else None,
                "job_url": r.job_url,
            }
            for r in records
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze yesterday's Buildkite builds for vllm-omni and "
        "vllm-omni-npu-ci (the default date window is the previous "
        "full Beijing Time / CST calendar day, 00:00 — 23:59 CST "
        "= UTC (date-1) 16:00 → date 15:59:59), and emit an HTML "
        "report with Pipeline / Branch / CI / State / Job Name "
        "filter dropdowns plus a CI Aggregate panel broken down "
        "by ready / merge / nightly / weekly buckets. Default "
        "output is HTML."
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="CST calendar date to analyze. Default: yesterday CST (00:00 — 23:59 CST).",
    )
    parser.add_argument(
        "--today", dest="use_today", action="store_true", help="Analyze today CST instead of yesterday CST."
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
        help="Output format. Default: html (writes a self-contained file).",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        metavar="PATH",
        help="Output file path for HTML. Default: ci-daily-YYYY-MM-DD.html.",
    )
    parser.add_argument(
        "--include-infra",
        dest="include_infra",
        action="store_true",
        help="Keep Buildkite orchestration/setup jobs (init, Resolve skip-ci, "
        "Upload pipeline, Collect results) instead of filtering them out.",
    )
    parser.add_argument(
        "--exclude-jobs",
        dest="exclude_jobs",
        default=None,
        metavar="REGEX1,REGEX2",
        help="Comma-separated regex patterns (case-insensitive) added to the "
        "infra-skip list. Job labels matching any pattern are dropped.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print each build's job count and state during fetch."
    )
    args = parser.parse_args()

    if args.date:
        date_str = args.date
    elif args.use_today:
        date_str = today_range_cst()[0]
    else:
        # Default to yesterday CST (UTC+8) — the previous full Beijing-time
        # calendar day, i.e. 00:00 — 23:59 CST = (yesterday-1) 16:00 UTC →
        # yesterday 15:59:59 UTC. Saying "today's report" naturally maps to
        # "the most recent fully completed CI day".
        date_str = yesterday_range_cst()[0]

    pipeline_slugs: list[str]
    if args.pipelines:
        pipeline_slugs = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    else:
        pipeline_slugs = list(DEFAULT_PIPELINES)

    extra_patterns = compile_extra_patterns(args.exclude_jobs)

    token = get_api_token()
    if not token:
        print("BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is not set; cannot call the Buildkite API.", file=sys.stderr)
        print("Set one in the environment and retry.", file=sys.stderr)
        return 1

    all_records: list[JobRecord] = []
    all_build_records: list[BuildRecord] = []
    all_skipped: list[tuple[str, str]] = []
    for slug in pipeline_slugs:
        try:
            recs, builds, skipped = collect_jobs(
                token,
                slug,
                date_str,
                verbose=args.verbose,
                include_infra=args.include_infra,
                extra_skip_patterns=extra_patterns,
            )
            all_records.extend(recs)
            all_build_records.extend(builds)
            all_skipped.extend(skipped)
        except requests.RequestException as e:
            print(f"API request failed for {slug}: {e}", file=sys.stderr)
            if hasattr(e, "response") and e.response is not None:
                print(f"HTTP status: {e.response.status_code}", file=sys.stderr)
                print(e.response.text[:500], file=sys.stderr)

    if all_skipped and not args.include_infra:
        # Summarize skipped jobs by name so the user can see what was filtered.
        from collections import Counter

        skipped_counts = Counter(name for name, _ in all_skipped)
        sample = ", ".join(f"{html.escape(name)}({count})" for name, count in skipped_counts.most_common())
        print(
            f"Filtered {len(all_skipped)} infrastructure job(s): {sample}",
            file=sys.stderr,
        )

    # Sort: passed first, then failed, then others; within a bucket, longest duration first
    bucket_order = {
        STATE_FAILED: 0,
        STATE_RUNNING: 1,
        STATE_CANCELED: 2,
        STATE_OTHER: 3,
        STATE_PASSED: 4,
    }
    all_records.sort(
        key=lambda r: (
            bucket_order.get(r.bucket, 9),
            -(r.duration_seconds or 0.0),
            r.pipeline,
            r.branch,
            -r.build_number,
            r.job_name,
        )
    )

    if args.output_format == "html":
        skipped_summary: dict[str, int] | None = None
        if all_skipped and not args.include_infra:
            from collections import Counter

            skipped_summary = dict(Counter(name for name, _ in all_skipped).most_common())
        html_content = render_html(
            all_records,
            date_str,
            pipeline_slugs,
            skipped_summary=skipped_summary,
            build_records=all_build_records,
        )
        out_path = Path(args.output_path) if args.output_path else Path(f"ci-daily-{date_str}.html")
        out_path.write_text(html_content, encoding="utf-8")
        print(f"HTML report written to {out_path}")
    elif args.output_format == "markdown":
        print(render_markdown(all_records, date_str))
    elif args.output_format == "json":
        print(render_json(all_records, date_str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
