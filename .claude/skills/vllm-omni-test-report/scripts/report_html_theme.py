#!/usr/bin/env python3
"""Shared editorial HTML theme (nightly + release reports)."""

from __future__ import annotations

# Full theme: CSS variables, top-bar, panels, tables, nightly-only components.
EDITORIAL_THEME_CSS = """
:root {
  /* Dashboard palette (aligned with omni docs / slate home — cool gray, blue accent) */
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
  --omni-baseline-line: #64748b;
  --omni-baseline-label-bg: rgba(241, 245, 249, 0.96);
  --omni-baseline-label-fg: #0f172a;
  --omni-baseline-label-border: rgba(100, 116, 139, 0.45);
  --dashboard-tooltip-bg: rgba(15, 23, 42, 0.92);
  --dashboard-tooltip-border: rgba(148, 163, 184, 0.28);
  --dashboard-tooltip-text: #f8fafc;
  --dashboard-healthy: #1f9d63;
  --dashboard-alert: #d14343;
  --dashboard-healthy-bg: rgba(31, 157, 99, 0.1);
  --dashboard-alert-bg: rgba(209, 67, 67, 0.11);
  --dashboard-warning: #d97706;
  --dashboard-warning-bg: rgba(217, 119, 6, 0.12);
  --dashboard-violet: #4f46e5;
  --dashboard-violet-bg: rgba(79, 70, 229, 0.1);
  /* Internal aliases (report markup) */
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
  --ci: #2563eb;
  --ci-soft: rgba(59, 130, 246, 0.16);
  --danger: var(--dashboard-alert);
  --danger-strong: #b91c1c;
  --danger-bg: rgba(209, 67, 67, 0.08);
  --danger-edge: var(--dashboard-alert);
  --ok: var(--dashboard-healthy);
  --ok-bg: var(--dashboard-healthy-bg);
  --ok-edge: var(--dashboard-healthy);
  --fail-bg: var(--dashboard-alert-bg);
  --fail-edge: var(--dashboard-alert);
  --unknown-bg: rgba(148, 163, 184, 0.08);
  --unknown-edge: #64748b;
  --code-bg-top: #334155;
  --code-bg-bottom: #1e293b;
  --code-fg: #e2e8f0;
  --radius: 12px;
  --radius-sm: 8px;
  --shadow-bar: none;
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
    --omni-baseline-line: #94a3b8;
    --omni-baseline-label-bg: rgba(30, 41, 55, 0.94);
    --omni-baseline-label-fg: #f1f5f9;
    --omni-baseline-label-border: rgba(148, 163, 184, 0.4);
    --dashboard-tooltip-bg: rgba(15, 23, 42, 0.96);
    --dashboard-tooltip-border: rgba(148, 163, 184, 0.24);
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
    --ci: #60a5fa;
    --ci-soft: rgba(96, 165, 250, 0.2);
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
  border-bottom: 3px solid var(--accent);
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
.brand-mark .ico { stroke: var(--accent); }
.brand-copy h1 {
  margin: 0;
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.15;
  color: var(--dashboard-text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.tagline {
  margin: 0.35rem 0 0;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.top-bar .tagline {
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
  border-color: rgba(59, 130, 246, 0.28);
}
.panel h2 {
  margin: 0 0 1rem;
  font-size: 1.12rem;
  font-weight: 800;
  color: var(--dashboard-text);
  letter-spacing: -0.02em;
  border-bottom: 2px solid var(--dashboard-border);
  padding-bottom: 0.55rem;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
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
  color: var(--accent);
}
.panel-bk .heading-ico {
  color: var(--ci);
}
.heading-text {
  display: flex;
  flex-direction: column;
  gap: 0.12rem;
  min-width: 0;
}
.heading-sub {
  font-size: 0.78em;
  font-weight: 500;
  color: var(--muted);
}
h3.panel-sub {
  margin: 1.2rem 0 0.65rem;
  font-size: 1.04rem;
  color: var(--dashboard-soft-text);
  font-weight: 700;
}
h3.panel-sub:first-child {
  margin-top: 0;
}
h3.panel-sub .heading-ico {
  color: var(--dashboard-muted);
}
h3.section-failures {
  margin: 0 0 0.85rem;
  font-size: 1.02rem;
  font-weight: 700;
  color: var(--danger-strong);
}
h3.section-failures .heading-ico {
  color: var(--danger);
}
section.job-fail-bk h3.section-failures .heading-ico {
  color: var(--ci);
}
.panel-fail-analysis {
  border-top: 3px solid var(--dashboard-warning);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-warning-bg) 65%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 42%
  );
}
.panel-fail-analysis > h2 {
  border-bottom-color: color-mix(in srgb, var(--dashboard-warning) 35%, transparent);
}
.panel-fail-analysis .heading-ico {
  color: var(--dashboard-warning);
}
.perf-delta {
  font-size: 0.78em;
  font-weight: 750;
  margin-left: 0.35rem;
  white-space: nowrap;
}
.perf-delta--up {
  color: var(--dashboard-warning);
}
.perf-delta--down {
  color: var(--dashboard-healthy);
}
details.job-fail-details {
  margin-bottom: 1.25rem;
  border-radius: var(--radius);
  overflow: hidden;
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-left: 4px solid var(--danger-edge);
  background: var(--surface);
}
details.job-fail-details:last-child {
  margin-bottom: 0;
}
summary.job-fail-details-summary {
  list-style: none;
  cursor: pointer;
  background: var(--danger-bg);
  padding: 1.05rem 1.1rem 1.05rem 2.65rem;
  border-bottom: 1px solid color-mix(in srgb, var(--dashboard-warning) 38%, transparent);
  position: relative;
}
summary.job-fail-details-summary::-webkit-details-marker {
  display: none;
}
summary.job-fail-details-summary::before {
  content: "▸";
  position: absolute;
  left: 0.95rem;
  top: 1.2rem;
  font-size: 0.95rem;
  font-weight: 800;
  color: var(--danger-strong);
  line-height: 1;
  pointer-events: none;
}
details.job-fail-details[open] > summary.job-fail-details-summary::before {
  content: "▾";
}
summary.job-fail-details-summary h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 750;
  color: var(--danger-strong);
}
summary.job-fail-details-summary .heading-ico {
  color: var(--danger-strong);
}
summary.job-fail-details-summary .meta {
  margin: 0.5rem 0 0;
}
.job-fail-details-body {
  padding: 1.1rem 1.35rem 1.35rem;
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-alert-bg) 40%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 48%
  );
}
.full-log-wrap {
  margin: 0.55rem 0 0;
  padding-top: 0.4rem;
  border-top: 1px dashed color-mix(in srgb, var(--dashboard-warning) 42%, transparent);
}
.btn-view-full-log {
  display: inline-flex;
  align-items: center;
  padding: 0.42rem 0.88rem;
  border-radius: 9px;
  border: 1px solid color-mix(in srgb, var(--dashboard-warning) 45%, var(--dashboard-border));
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--dashboard-warning-bg) 55%, var(--dashboard-panel-bg)) 100%
  );
  color: var(--danger-strong);
  font-size: 0.86rem;
  font-weight: 650;
  cursor: pointer;
  box-shadow: 0 1px 2px color-mix(in srgb, var(--dashboard-warning) 12%, transparent);
}
.btn-view-full-log:hover {
  border-color: var(--dashboard-warning);
  background: var(--dashboard-panel-bg);
}
.full-log-panel {
  margin-top: 0.65rem;
}
pre.log-full {
  margin: 0;
  padding: 0.85rem 1rem;
  background: linear-gradient(180deg, var(--code-bg-top) 0%, var(--code-bg-bottom) 100%);
  color: var(--code-fg);
  border-radius: var(--radius-sm);
  border: 1px solid color-mix(in srgb, var(--dashboard-muted) 38%, transparent);
  font-size: 0.78rem;
  line-height: 1.48;
  max-height: min(75vh, 42rem);
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
ul.full-log-paths {
  margin: 0.35rem 0 0;
  padding-left: 1.25rem;
  font-size: 0.88rem;
  color: var(--text);
}
.full-log-oversize {
  margin: 0;
}
summary.job-fail-details-summary-bk {
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--ci-soft) 65%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-strong)
  );
  border-bottom-color: color-mix(in srgb, var(--ci) 42%, transparent);
}
summary.job-fail-details-summary-bk::before {
  color: var(--ci);
}
details.job-fail-details-bk {
  border-left-color: var(--ci);
}
summary.job-fail-details-summary-bk h2 {
  color: var(--dashboard-chart-text);
}
summary.job-fail-details-summary-bk .heading-ico {
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
.meta a {
  color: var(--accent);
  font-weight: 600;
  text-decoration: none;
  border-bottom: 1px solid var(--accent-soft);
}
.meta a:hover {
  color: var(--accent-hover);
  border-bottom-color: var(--accent);
}
.hint {
  color: var(--muted);
  font-size: 0.88rem;
  margin: 0.55rem 0 0;
  display: flex;
  align-items: flex-start;
  gap: 0.45rem;
}
.hint::before {
  content: "→";
  color: var(--accent);
  font-weight: 800;
  flex-shrink: 0;
  margin-top: 0.08rem;
}
p.summary-legend {
  margin: 0.35rem 0 0;
  font-size: 0.84rem;
}
.summary-legend strong.summary-legend--ok { color: var(--dashboard-healthy); }
.summary-legend strong.summary-legend--fail { color: var(--dashboard-alert); }
.summary-legend strong.summary-legend--unk { color: var(--omni-baseline-line); }
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
  min-width: 620px;
}
table.summary {
  border-collapse: collapse;
  font-size: 0.92rem;
}
table.summary th, table.summary td {
  border: 1px solid var(--border);
  padding: 0.65rem 0.8rem;
  text-align: left;
  vertical-align: top;
}
table.summary th {
  background: var(--dashboard-panel-strong);
  font-weight: 650;
  color: var(--dashboard-chart-text);
  white-space: nowrap;
}
table.summary tbody tr.summary-row--ok td {
  background: color-mix(in srgb, var(--dashboard-healthy-bg) 92%, var(--dashboard-panel-bg));
}
table.summary tbody tr.summary-row--ok td:first-child {
  border-left: 4px solid var(--ok-edge);
  padding-left: calc(0.8rem - 3px);
  font-weight: 650;
  color: var(--dashboard-healthy);
}
table.summary tbody tr.summary-row--fail td {
  background: color-mix(in srgb, var(--dashboard-alert-bg) 88%, var(--dashboard-panel-bg));
}
table.summary tbody tr.summary-row--fail td:first-child {
  border-left: 4px solid var(--fail-edge);
  padding-left: calc(0.8rem - 3px);
  font-weight: 650;
  color: var(--danger-strong);
}
table.summary tbody tr.summary-row--unknown td {
  background: var(--unknown-bg);
}
table.summary tbody tr.summary-row--unknown td:first-child {
  border-left: 4px solid var(--unknown-edge);
  padding-left: calc(0.8rem - 3px);
  color: var(--dashboard-chart-text);
}
table.summary tbody tr.summary-row--ok:hover td {
  background: color-mix(in srgb, var(--dashboard-healthy-bg) 98%, var(--dashboard-panel-bg));
}
table.summary tbody tr.summary-row--fail:hover td {
  background: color-mix(in srgb, var(--dashboard-alert-bg) 95%, var(--dashboard-panel-bg));
}
table.summary tbody tr.summary-row--unknown:hover td {
  background: var(--dashboard-badge-bg);
}
/* Consecutive-failure streak rows (Nightly focus "Days failing" column). */
/* Row tint applies to every cell; Days-failing chip keeps saturated bg. */
table.summary tr.regression-streak--1d td {
  background-color: #fef3c7 !important;
}
table.summary tr.regression-streak--1d td.regression-streak--1d {
  background-color: #fde68a !important;
  color: #854d0e;
  font-weight: 650;
  border-left: 3px solid #facc15;
}
table.summary tr.regression-streak--2d td {
  background-color: #ffedd5 !important;
}
table.summary tr.regression-streak--2d td.regression-streak--2d {
  background-color: #fdba74 !important;
  color: #7c2d12;
  font-weight: 650;
  border-left: 3px solid #fb923c;
}
table.summary tr.regression-streak--3d td {
  background-color: #fee2e2 !important;
}
table.summary tr.regression-streak--3d td.regression-streak--3d {
  background-color: #fca5a5 !important;
  color: #7f1d1d;
  font-weight: 700;
  border-left: 3px solid #ef4444;
}
/* Days-failing cell capped at "3 days+" (>= 4 actual streak). Purple marks
   "chronic regression" — escalated beyond the normal red --3d bucket because
   the streak has lasted longer than the visible 3-day cap. */
table.summary tr.regression-streak--3d-plus td {
  background-color: #ede9fe !important;
}
table.summary tr.regression-streak--3d-plus td.regression-streak--3d-plus {
  background-color: #c4b5fd !important;
  color: #4c1d95;
  font-weight: 750;
  border-left: 3px solid #7c3aed;
}
@media (prefers-color-scheme: dark) {
  table.summary tr.regression-streak--1d td {
    background-color: rgba(250, 204, 21, 0.18) !important;
  }
  table.summary tr.regression-streak--1d td.regression-streak--1d {
    background-color: rgba(250, 204, 21, 0.5) !important;
    color: #fde68a;
    border-left-color: #facc15;
  }
  table.summary tr.regression-streak--2d td {
    background-color: rgba(251, 146, 60, 0.18) !important;
  }
  table.summary tr.regression-streak--2d td.regression-streak--2d {
    background-color: rgba(251, 146, 60, 0.5) !important;
    color: #fdba74;
    border-left-color: #fb923c;
  }
  table.summary tr.regression-streak--3d td {
    background-color: rgba(239, 68, 68, 0.18) !important;
  }
  table.summary tr.regression-streak--3d td.regression-streak--3d {
    background-color: rgba(239, 68, 68, 0.5) !important;
    color: #fecaca;
    border-left-color: #ef4444;
  }
  table.summary tr.regression-streak--3d-plus td {
    background-color: rgba(124, 58, 237, 0.18) !important;
  }
  table.summary tr.regression-streak--3d-plus td.regression-streak--3d-plus {
    background-color: rgba(124, 58, 237, 0.5) !important;
    color: #ddd6fe;
    border-left-color: #a78bfa;
  }
}
section.job-fail {
  background: var(--surface);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  margin-bottom: 1.5rem;
  overflow: hidden;
  border-left: 4px solid var(--danger-edge);
}
section.job-fail header {
  background: var(--danger-bg);
  padding: 1.05rem 1.35rem;
  border-bottom: 1px solid color-mix(in srgb, var(--dashboard-warning) 38%, transparent);
}
section.job-fail header .heading-ico {
  color: var(--danger-strong);
}
section.job-fail h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 750;
  color: var(--danger-strong);
}
section.job-fail .body {
  padding: 1.1rem 1.35rem 1.35rem;
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-alert-bg) 35%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 48%
  );
}
table.fail-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
table.fail-table th, table.fail-table td {
  border: 1px solid var(--border);
  padding: 0.72rem 0.8rem;
  vertical-align: top;
}
table.fail-table th {
  background: var(--dashboard-panel-strong);
  font-weight: 650;
  text-align: left;
  color: var(--dashboard-text);
}
.th-lbl {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
}
.th-ico {
  flex-shrink: 0;
  opacity: 0.88;
  color: var(--dashboard-chart-text);
}
table.fail-table tr.row-error {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--dashboard-warning-bg) 70%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 100%
  );
}
td.mono {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 0.84rem;
  word-break: break-all;
}
td.reason {
  color: var(--dashboard-text);
}
td.analysis {
  color: var(--dashboard-healthy);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-healthy-bg) 55%, var(--dashboard-panel-bg)) 0%,
    color-mix(in srgb, var(--dashboard-healthy-bg) 88%, var(--dashboard-panel-bg)) 100%
  );
  font-size: 0.86rem;
  border-left: 3px solid var(--dashboard-healthy) !important;
}
table.fail-table th.excerpt-col {
  text-align: center;
}
table.fail-table th.excerpt-col .th-lbl {
  justify-content: center;
}
td.excerpt-cell {
  padding: 0.55rem 0.65rem !important;
  background: var(--surface-muted);
  width: 9.5rem;
  vertical-align: middle !important;
}
.excerpt-cell-inner {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
}
.btn-view-log-excerpt {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.42rem 0.75rem;
  border-radius: 9px;
  border: 1px solid color-mix(in srgb, var(--dashboard-warning) 45%, var(--dashboard-border));
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--dashboard-warning-bg) 55%, var(--dashboard-panel-bg)) 100%
  );
  color: var(--danger-strong);
  font-size: 0.82rem;
  font-weight: 650;
  cursor: pointer;
  white-space: nowrap;
  box-shadow: 0 1px 2px color-mix(in srgb, var(--dashboard-warning) 12%, transparent);
}
.btn-view-log-excerpt:hover {
  border-color: var(--dashboard-warning);
  background: var(--dashboard-panel-bg);
}
pre.log-excerpt--stored {
  display: none !important;
}
body.log-modal-open {
  overflow: hidden;
}
.log-excerpt-modal[hidden] {
  display: none !important;
}
.log-excerpt-modal {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
}
.log-excerpt-modal-backdrop {
  position: absolute;
  inset: 0;
  background: color-mix(in srgb, var(--dashboard-text) 42%, transparent);
  backdrop-filter: blur(2px);
}
.log-excerpt-modal-panel {
  position: relative;
  z-index: 1;
  width: min(92vw, 56rem);
  max-height: min(92vh, 48rem);
  display: flex;
  flex-direction: column;
  background: var(--dashboard-panel-bg);
  border-radius: var(--radius-md);
  border: 1px solid var(--dashboard-border);
  box-shadow: 0 20px 50px color-mix(in srgb, var(--dashboard-text) 22%, transparent);
}
.log-excerpt-modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem 1.1rem;
  border-bottom: 1px solid var(--dashboard-border);
  background: linear-gradient(180deg, var(--dashboard-panel-strong) 0%, var(--dashboard-panel-bg) 100%);
}
.log-excerpt-modal-header h2 {
  margin: 0;
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--dashboard-text);
  word-break: break-word;
  line-height: 1.35;
}
.log-excerpt-modal-close {
  flex-shrink: 0;
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--dashboard-border);
  border-radius: 8px;
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  font-size: 1.35rem;
  line-height: 1;
  cursor: pointer;
}
.log-excerpt-modal-close:hover {
  border-color: var(--dashboard-warning);
  color: var(--danger-strong);
}
.log-excerpt-modal-body {
  overflow: hidden;
  flex: 1;
  min-height: 0;
}
.log-excerpt-modal-body pre {
  margin: 0;
  padding: 1rem 1.1rem;
  background: linear-gradient(180deg, var(--code-bg-top) 0%, var(--code-bg-bottom) 100%);
  color: var(--code-fg);
  font-size: 0.78rem;
  line-height: 1.48;
  max-height: min(78vh, 40rem);
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
pre.log-excerpt {
  margin: 0;
  padding: 0.85rem 1rem;
  background: linear-gradient(180deg, var(--code-bg-top) 0%, var(--code-bg-bottom) 100%);
  color: var(--code-fg);
  border-radius: var(--radius-sm);
  border: 1px solid color-mix(in srgb, var(--dashboard-muted) 38%, transparent);
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--dashboard-panel-bg) 18%, transparent);
  font-size: 0.78rem;
  line-height: 1.48;
  max-height: 28rem;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
.note {
  color: var(--muted);
  font-size: 0.9rem;
  margin-top: 0.35rem;
}
.panel-bk {
  border-top: 4px solid var(--ci);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--ci-soft) 35%, var(--dashboard-panel-bg)) 100%
  );
}
section.job-fail-bk {
  border-left-color: var(--ci);
}
section.job-fail-bk header {
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--ci-soft) 55%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-strong)
  );
  border-bottom: 1px solid color-mix(in srgb, var(--ci) 38%, transparent);
}
section.job-fail-bk header .heading-ico {
  color: var(--ci);
}
section.job-fail-bk h2 {
  color: var(--dashboard-chart-text);
  font-weight: 750;
}
.issue-action-cell {
  width: 8.5rem;
  text-align: center;
  vertical-align: middle !important;
  background: var(--surface-muted);
}
.btn-github-issue {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.35rem;
  padding: 0.4rem 0.65rem;
  border-radius: 9px;
  border: 1px solid var(--border);
  background: var(--dashboard-panel-strong);
  color: var(--dashboard-text);
  font-size: 0.82rem;
  font-weight: 650;
  cursor: pointer;
  box-shadow: 0 1px 2px color-mix(in srgb, var(--dashboard-shadow) 25%, transparent);
  transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
}
.btn-github-issue:hover {
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--accent-tint) 90%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-strong)
  );
  border-color: color-mix(in srgb, var(--accent) 35%, var(--dashboard-border));
  color: var(--accent-hover);
}
.btn-github-issue:active { transform: scale(0.98); }
.btn-issue-ico { stroke: var(--accent); }
.btn-issue-text { white-space: nowrap; }

/* Status column — two interactive buttons (filed / not-issue) backed by localStorage. */
th.status-col, td.fail-status-cell {
  width: 11.5rem;
  text-align: left;
  vertical-align: middle !important;
  background: var(--surface-muted);
}
.fail-status-cell {
  padding: 0.55rem 0.65rem !important;
}
.fail-status-label {
  display: block;
  font-size: 0.7rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.25rem;
  font-weight: 650;
}
.fail-status-buttons {
  display: inline-flex;
  gap: 0.3rem;
  flex-wrap: wrap;
}
.fail-status-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 0.32rem 0.65rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--dashboard-panel-strong);
  color: var(--dashboard-text);
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
}
.fail-status-btn--filed:hover {
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--accent-tint) 88%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-strong)
  );
  border-color: color-mix(in srgb, var(--accent) 40%, var(--dashboard-border));
  color: var(--accent-hover);
}
.fail-status-btn--not-issue:hover {
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-warning-bg) 88%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-strong)
  );
  border-color: color-mix(in srgb, var(--dashboard-warning) 45%, var(--dashboard-border));
  color: var(--dashboard-warning);
}
.fail-status-btn:active { transform: scale(0.97); }
.fail-status-display {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.82rem;
  font-weight: 650;
}
.fail-status-display--filed {
  color: var(--accent-hover);
}
.fail-status-display--not-issue {
  color: var(--dashboard-warning);
}
.fail-status-issue-link {
  color: inherit;
  text-decoration: none;
  border-bottom: 1px dashed currentColor;
  padding-bottom: 1px;
}
.fail-status-issue-link:hover {
  color: var(--accent);
  border-bottom-style: solid;
}
.fail-status-reset {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 0.45rem;
  padding: 0.2rem 0.5rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--dashboard-panel-bg);
  color: var(--muted);
  font-size: 0.72rem;
  cursor: pointer;
}
.fail-status-reset:hover {
  background: var(--dashboard-panel-strong);
  color: var(--dashboard-text);
}

/* Row-level tinting driven by the Status column. Default (un-confirmed)
   failure-analysis rows have neutral background by default; rows whose Status
   has been confirmed (filed / not-issue) are tinted green. The CSS uses :has()
   so it doesn't require extra markup in the report body. */
tr:has(td.fail-status-cell[data-status="unset"]) {
  background: transparent;
}
tr:has(td.fail-status-cell[data-status="unset"]):hover {
  background: rgba(0, 0, 0, 0.04);
}
tr:has(td.fail-status-cell[data-status="filed"]),
tr:has(td.fail-status-cell[data-status="not-issue"]) {
  background: color-mix(in srgb, #2ea043 12%, transparent);
}
tr:has(td.fail-status-cell[data-status="filed"]):hover,
tr:has(td.fail-status-cell[data-status="not-issue"]):hover {
  background: color-mix(in srgb, #2ea043 18%, transparent);
}
td.fail-status-cell[data-status="unset"] {
  box-shadow: inset 4px 0 0 0 #6e7681;
}
td.fail-status-cell[data-status="filed"],
td.fail-status-cell[data-status="not-issue"] {
  box-shadow: inset 4px 0 0 0 #2ea043;
}
.gh-modal[hidden] { display: none !important; }
.gh-modal:not([hidden]) {
  position: fixed;
  inset: 0;
  z-index: 99990;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}
.gh-modal-backdrop {
  position: absolute;
  inset: 0;
  background: color-mix(in srgb, var(--dashboard-tooltip-bg) 55%, transparent);
  backdrop-filter: blur(3px);
}
.gh-modal-panel {
  position: relative;
  z-index: 1;
  max-width: 720px;
  width: 100%;
  max-height: min(92vh, 920px);
  display: flex;
  flex-direction: column;
  background: var(--surface);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  box-shadow: var(--dashboard-shadow);
  overflow: hidden;
}
.gh-modal-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem 1.15rem;
  background: var(--dashboard-panel-strong);
  border-bottom: 1px solid var(--border);
}
.gh-modal-head h2 {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 750;
  color: var(--dashboard-text);
}
.gh-modal-x {
  border: none;
  background: transparent;
  font-size: 1.5rem;
  line-height: 1;
  cursor: pointer;
  color: var(--muted);
  padding: 0 0.25rem;
}
.gh-modal-x:hover { color: var(--danger-strong); }
.gh-modal-hint {
  margin: 0;
  padding: 0.85rem 1.15rem 0;
  font-size: 0.88rem;
  color: var(--muted);
  line-height: 1.55;
}
.gh-modal-hint a { color: var(--accent); font-weight: 600; }
.gh-issue-textarea {
  margin: 0.75rem 1.15rem;
  flex: 1;
  min-height: 200px;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 0.8rem;
  line-height: 1.45;
  padding: 0.75rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  resize: vertical;
  color: var(--dashboard-text);
  background: color-mix(in srgb, var(--dashboard-badge-bg) 40%, var(--dashboard-panel-bg));
}
.gh-modal-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
  padding: 0.85rem 1.15rem 1.1rem;
  border-top: 1px solid var(--border);
  background: var(--surface-muted);
}
.btn-gh-copy {
  padding: 0.5rem 1rem;
  border-radius: 9px;
  border: 1px solid var(--border);
  background: var(--dashboard-panel-bg);
  font-weight: 650;
  cursor: pointer;
  color: var(--dashboard-soft-text);
}
.btn-gh-copy:hover { border-color: var(--dashboard-border-strong); }
.btn-gh-open {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1.1rem;
  border-radius: var(--radius-sm);
  background: var(--accent);
  color: var(--dashboard-tooltip-text) !important;
  font-weight: 700;
  text-decoration: none !important;
  border: 1px solid var(--accent-hover);
  box-shadow: none;
}
.btn-gh-open:hover { background: var(--accent-hover); filter: none; }
/* Nightly report: Buildkite Test / Local Test outer cards + collapsible subcards */
.nightly-focus {
  border-top: 4px solid var(--accent);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--accent-tint) 65%, var(--dashboard-panel-bg)) 100%
  );
}
.nightly-focus--fail {
  border-top-color: var(--dashboard-alert);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--dashboard-alert-bg) 62%, var(--dashboard-panel-bg)) 100%
  );
}
.nightly-focus--normal {
  border-top-color: var(--dashboard-warning);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--dashboard-warning-bg) 58%, var(--dashboard-panel-bg)) 100%
  );
}
.nightly-focus--ok {
  border-top-color: var(--dashboard-healthy);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--dashboard-healthy-bg) 62%, var(--dashboard-panel-bg)) 100%
  );
}
.nightly-focus--unknown {
  border-top-color: var(--unknown-edge);
}
.nightly-focus .heading-ico {
  color: var(--accent);
}
.nightly-focus--fail .heading-ico {
  color: var(--dashboard-alert);
}
.nightly-focus--normal .heading-ico {
  color: var(--dashboard-warning);
}
.nightly-focus--ok .heading-ico {
  color: var(--dashboard-healthy);
}
.focus-conclusion {
  margin: 0 0 0.9rem;
  font-size: 1rem;
  font-weight: 720;
  color: var(--dashboard-text);
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
.focus-card--fail {
  border-left: 4px solid var(--dashboard-alert);
  background: color-mix(in srgb, var(--dashboard-alert-bg) 76%, var(--dashboard-panel-bg));
}
.focus-card--ok {
  border-left: 4px solid var(--dashboard-healthy);
  background: color-mix(in srgb, var(--dashboard-healthy-bg) 72%, var(--dashboard-panel-bg));
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
.focus-table-title {
  margin: 1rem 0 0.45rem;
  color: var(--dashboard-soft-text);
  font-size: 1rem;
}
table.focus-table {
  table-layout: fixed;
  font-size: 0.86rem;
  min-width: 1080px;
}
table.focus-table th,
table.focus-table td {
  padding: 0.48rem 0.55rem;
  overflow-wrap: anywhere;
  white-space: normal;
  line-height: 1.35;
}
table.focus-table th:nth-child(1),
table.focus-table td:nth-child(1) {
  width: 5.8rem;
}
table.focus-table th:nth-child(2),
table.focus-table td:nth-child(2) {
  width: 13rem;
}
table.focus-table th:nth-child(3),
table.focus-table td:nth-child(3) {
  width: 6.8rem;
}
table.focus-table th:nth-child(4),
table.focus-table td:nth-child(4) {
  width: 12rem;
}
table.focus-table th:nth-child(5),
table.focus-table td:nth-child(5) {
  width: 12rem;
}
table.focus-table th:nth-child(6),
table.focus-table td:nth-child(6) {
  width: 8rem;
}
table.focus-table th:nth-child(7),
table.focus-table td:nth-child(7),
table.focus-table th:nth-child(8),
table.focus-table td:nth-child(8),
table.focus-table th:nth-child(9),
table.focus-table td:nth-child(9) {
  width: 6.4rem;
  text-align: right;
  white-space: nowrap;
}
table.focus-table th:nth-child(10),
table.focus-table td:nth-child(10) {
  width: 5rem;
  white-space: nowrap;
}
table.focus-table th:nth-child(11),
table.focus-table td:nth-child(11) {
  width: 6.5rem;
  white-space: nowrap;
  text-align: center;
}
.focus-filter-scope .table-scroll {
  max-height: 13rem;
  overflow: auto;
}
.focus-filter-scope--expanded .table-scroll {
  max-height: none;
}
.focus-filter-scope table.focus-table th {
  position: sticky;
  top: 0;
  z-index: 1;
}
.focus-model-filter {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem 0.7rem;
  max-width: min(58rem, 100%);
  margin: 0;
  padding: 0.55rem 0.7rem;
  border: 1px solid color-mix(in srgb, var(--dashboard-border) 82%, var(--ci) 18%);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--dashboard-panel-bg) 86%, var(--ci-soft) 14%);
}
.focus-model-filter legend {
  padding: 0 0.25rem;
  color: var(--dashboard-muted);
  font-size: 0.78rem;
  font-weight: 760;
}
.focus-model-filter legend span {
  font-weight: 520;
}
.focus-model-check {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  max-width: 18rem;
  color: var(--dashboard-text);
  font-size: 0.82rem;
  font-weight: 640;
}
.focus-model-check input {
  margin: 0;
  flex-shrink: 0;
}
.focus-model-check span {
  overflow-wrap: anywhere;
}
.nightly-root.panel {
  margin-bottom: 1.35rem;
}
.nightly-root--buildkite {
  border-top: 4px solid var(--ci);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--ci-soft) 28%, var(--dashboard-panel-bg)) 100%
  );
}
.nightly-root--local {
  border-top: 4px solid var(--accent);
  background: linear-gradient(
    180deg,
    var(--dashboard-panel-bg) 0%,
    color-mix(in srgb, var(--accent-tint) 70%, var(--dashboard-panel-bg)) 100%
  );
}
details.report-subcard {
  margin-bottom: 0.9rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-bg);
  box-shadow: var(--dashboard-shadow);
  overflow: hidden;
}
details.report-subcard:last-child {
  margin-bottom: 0;
}
summary.report-subcard-summary {
  list-style: none;
  cursor: pointer;
  padding: 0.78rem 1rem 0.78rem 2.45rem;
  position: relative;
  font-weight: 780;
  font-size: 1.02rem;
  color: var(--dashboard-text);
  background: linear-gradient(180deg, var(--dashboard-panel-strong) 0%, var(--dashboard-panel-bg) 100%);
  border-bottom: 1px solid var(--dashboard-border);
}
.report-subcard-summary-inner {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  min-width: 0;
}
.report-subcard-summary-inner .report-subcard-ico {
  flex-shrink: 0;
  opacity: 0.92;
}
.report-subcard-title {
  font-weight: inherit;
}
.report-subcard-summary .report-subcard-ico {
  color: var(--accent);
}
.report-subcard--bk .report-subcard-summary .report-subcard-ico {
  color: var(--ci);
}
.report-subcard--bk-fail .report-subcard-summary .report-subcard-ico {
  color: var(--dashboard-warning);
}
.report-subcard--bk-perf .report-subcard-summary .report-subcard-ico,
.report-subcard--bk-perf-model .report-subcard-summary .report-subcard-ico {
  color: var(--ci);
}
.report-subcard--local-fail .report-subcard-summary .report-subcard-ico {
  color: var(--danger-strong);
}
summary.report-subcard-summary::-webkit-details-marker {
  display: none;
}
summary.report-subcard-summary::before {
  content: "▸";
  position: absolute;
  left: 0.88rem;
  top: 0.9rem;
  font-size: 0.95rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
  pointer-events: none;
}
details.report-subcard[open] > summary.report-subcard-summary::before {
  content: "▾";
}
.report-subcard--bk summary.report-subcard-summary::before {
  color: var(--ci);
}
.report-subcard--bk-fail summary.report-subcard-summary::before {
  color: var(--dashboard-warning);
}
.report-subcard--bk-perf summary.report-subcard-summary::before,
.report-subcard--bk-perf-model summary.report-subcard-summary::before {
  color: var(--ci);
}
.report-subcard--local-fail summary.report-subcard-summary::before {
  color: var(--danger-strong);
}
.report-subcard-body {
  padding: 1rem 1.2rem 1.2rem;
  background: var(--dashboard-panel-bg);
}
.report-subcard-body > .table-scroll:first-child {
  margin-top: 0;
}
.local-summary-pillar-body {
  padding: 0.35rem 0.15rem 0.65rem;
}
details.local-summary-pillar {
  margin-bottom: 0.75rem;
  border-radius: var(--radius-sm);
  border: 1px solid var(--dashboard-border);
  background: color-mix(in srgb, var(--dashboard-panel-bg) 90%, var(--accent) 10%);
}
details.local-summary-pillar:last-child {
  margin-bottom: 0;
}
summary.local-summary-pillar-summary {
  list-style: none;
  cursor: pointer;
  padding: 0.55rem 0.85rem 0.55rem 2rem;
  position: relative;
  font-weight: 750;
  font-size: 0.98rem;
  color: var(--dashboard-text);
  background: var(--dashboard-panel-strong);
  border-bottom: 1px solid var(--dashboard-border);
}
summary.local-summary-pillar-summary::-webkit-details-marker {
  display: none;
}
summary.local-summary-pillar-summary::before {
  content: "▸";
  position: absolute;
  left: 0.52rem;
  top: 0.62rem;
  font-size: 0.92rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
  pointer-events: none;
}
details.local-summary-pillar[open] > summary.local-summary-pillar-summary::before {
  content: "▾";
}
details.local-summary-dim {
  margin: 0.5rem 0.35rem 0.45rem 0.65rem;
  border-radius: var(--radius-sm);
  border: 1px dashed color-mix(in srgb, var(--dashboard-border) 82%, var(--accent) 18%);
  background: var(--dashboard-panel-bg);
}
summary.local-summary-dim-summary {
  list-style: none;
  cursor: pointer;
  padding: 0.42rem 0.75rem 0.42rem 1.85rem;
  position: relative;
  font-weight: 680;
  font-size: 0.9rem;
  color: var(--dashboard-muted);
  background: color-mix(in srgb, var(--dashboard-panel-bg) 96%, var(--accent) 4%);
  border-bottom: 1px solid var(--dashboard-border);
}
summary.local-summary-dim-summary::-webkit-details-marker {
  display: none;
}
summary.local-summary-dim-summary::before {
  content: "▸";
  position: absolute;
  left: 0.45rem;
  top: 0.48rem;
  font-size: 0.85rem;
  font-weight: 800;
  color: var(--accent);
  line-height: 1;
  pointer-events: none;
}
details.local-summary-dim[open] > summary.local-summary-dim-summary::before {
  content: "▾";
}
.local-summary-dim-body {
  padding: 0.55rem 0.45rem 0.65rem;
}
.local-summary-dim-body > .table-scroll:first-child {
  margin-top: 0;
}
.report-subcard--bk-perf-model {
  background: color-mix(in srgb, var(--dashboard-panel-bg) 94%, var(--ci-soft) 6%);
}
.perf-filter-scope {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.perf-filter-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 0.65rem;
  padding: 0.75rem;
  border: 1px solid color-mix(in srgb, var(--dashboard-border) 82%, var(--ci) 18%);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--dashboard-panel-strong) 88%, var(--ci-soft) 12%);
}
.perf-filter-label {
  display: inline-flex;
  flex-direction: column;
  gap: 0.25rem;
  min-width: 10rem;
  font-size: 0.78rem;
  font-weight: 720;
  color: var(--dashboard-muted);
}
.perf-filter-select,
.perf-filter-input {
  min-height: 2rem;
  max-width: min(22rem, 70vw);
  padding: 0.35rem 2rem 0.35rem 0.55rem;
  border: 1px solid var(--dashboard-border);
  border-radius: 0.55rem;
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  font: inherit;
}
.perf-filter-select:focus,
.perf-filter-input:focus {
  outline: 2px solid color-mix(in srgb, var(--ci) 45%, transparent);
  outline-offset: 2px;
}
.focus-expand-btn {
  min-height: 2rem;
  padding: 0.35rem 0.85rem;
  border: 1px solid color-mix(in srgb, var(--accent) 35%, var(--dashboard-border));
  border-radius: 0.55rem;
  background: color-mix(in srgb, var(--accent-tint) 72%, var(--dashboard-panel-bg));
  color: var(--accent-hover);
  cursor: pointer;
  font: inherit;
  font-weight: 720;
}
.focus-expand-btn:hover {
  border-color: var(--accent);
  background: color-mix(in srgb, var(--accent-tint) 45%, var(--dashboard-panel-bg));
}
.perf-filter-table td,
.perf-filter-table th {
  white-space: nowrap;
}
.perf-filter-empty {
  margin: 0;
}
.ico {
  display: block;
}
"""

# Scoped rules for release Markdown → HTML (compose_full_report body).
RELEASE_MARKDOWN_DOC_CSS = """
.top-bar-actions {
  flex-shrink: 0;
}
.btn-release-archive {
  margin-top: 0.2rem;
  padding: 0.52rem 1.05rem;
  border-radius: 999px;
  border: 1px solid rgba(59, 130, 246, 0.35);
  background: rgba(59, 130, 246, 0.12);
  color: var(--accent-hover);
  font-weight: 700;
  font-size: 0.88rem;
  cursor: pointer;
  font-family: inherit;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.btn-release-archive:hover {
  background: rgba(59, 130, 246, 0.2);
  border-color: var(--accent);
  color: var(--accent);
}
.release-doc {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.release-doc .release-section-card {
  margin-bottom: 0;
}
.release-doc .release-section-h2 {
  margin: 0 0 1rem;
  padding-bottom: 0.65rem;
  border-bottom: 2px solid
    color-mix(
      in srgb,
      var(--section-accent, var(--dashboard-border)) 42%,
      var(--dashboard-border)
    );
  font-size: 1.12rem;
  font-weight: 800;
  color: var(--dashboard-text);
  letter-spacing: -0.02em;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
}
.release-doc .release-section-h2-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 0;
}
.release-doc .release-section-h2-ico {
  flex-shrink: 0;
  width: 2.65rem;
  height: 2.65rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius-sm);
  background: color-mix(
    in srgb,
    var(--section-ico-bg, var(--dashboard-badge-bg)) 100%,
    transparent
  );
  color: var(--section-accent, var(--accent));
  border: 1px solid
    color-mix(
      in srgb,
      var(--section-accent, var(--border)) 25%,
      var(--dashboard-border)
    );
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}
.release-doc .release-section-h2-label {
  min-width: 0;
  line-height: 1.28;
}
.release-doc .release-section-card--intro {
  border-top: 3px dashed
    color-mix(in srgb, var(--dashboard-muted) 52%, var(--dashboard-border));
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-badge-bg) 72%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 48%
  );
}
.release-doc .release-section-card--conclusion {
  --section-accent: var(--ok);
  --section-ico-bg: var(--ok-bg);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--section-ico-bg) 48%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--metrics {
  --section-accent: var(--dashboard-healthy);
  --section-ico-bg: var(--dashboard-healthy-bg);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-healthy-bg) 52%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--failure {
  --section-accent: var(--dashboard-warning);
  --section-ico-bg: var(--dashboard-warning-bg);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-warning-bg) 52%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--tests {
  --section-accent: var(--dashboard-violet);
  --section-ico-bg: var(--dashboard-violet-bg);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-violet-bg) 42%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--tracking {
  --section-accent: var(--dashboard-warning);
  --section-ico-bg: var(--dashboard-warning-bg);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--dashboard-warning-bg) 52%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--open-issues {
  --section-accent: color-mix(in srgb, var(--danger) 82%, #991b1b);
  --section-ico-bg: var(--danger-bg);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--danger-bg) 52%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--data {
  --section-accent: #64748b;
  --section-ico-bg: rgba(148, 163, 184, 0.18);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--section-ico-bg) 75%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
.release-doc .release-section-card--default {
  --section-accent: var(--accent);
  --section-ico-bg: var(--accent-tint);
  border-top: 4px solid var(--section-accent);
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--accent-tint) 88%, var(--dashboard-panel-bg)) 0%,
    var(--dashboard-panel-bg) 46%
  );
}
/* H2 major sections (Test conclusion / Metrics / Test Result / …): entire block collapsible */
.release-doc details.panel.release-section-card.release-section-details {
  padding: 0;
  overflow: hidden;
}
.release-doc summary.release-section-fold-summary {
  list-style: none;
  cursor: pointer;
  margin: 0;
  padding: 0.95rem 1.2rem 0.95rem 3rem;
  position: relative;
}
.release-doc summary.release-section-fold-summary::-webkit-details-marker {
  display: none;
}
.release-doc summary.release-section-fold-summary::before {
  content: "▸";
  position: absolute;
  left: 0.82rem;
  top: 1.05rem;
  font-size: 0.95rem;
  font-weight: 800;
  color: var(--section-accent, var(--accent));
  line-height: 1;
  pointer-events: none;
}
.release-doc details.release-section-details[open] > summary.release-section-fold-summary::before {
  content: "▾";
}
.release-doc details.release-section-details[open] > summary.release-section-fold-summary {
  border-bottom: 1px solid var(--dashboard-border);
}
.release-doc details.release-section-details > summary.release-section-fold-summary .release-section-h2 {
  margin: 0 0 0.35rem;
  padding-bottom: 0.5rem;
}
.release-doc .release-section-fold-body {
  padding: 1rem 1.3rem 1.25rem;
}
.release-doc .release-section-fold-body > *:first-child,
.release-doc .release-section-fold-body > details:first-child {
  margin-top: 0;
}
.release-doc .release-gpu-summary-row {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  min-width: 0;
}
.release-doc .release-gpu-summary-ico {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  color: var(--release-gpu-ico, var(--accent));
}
.release-doc .release-h4-fold {
  border-left: 3px solid
    color-mix(in srgb, var(--accent) 50%, var(--dashboard-border));
}
.release-doc .release-h4-fold > summary.report-subcard-summary {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent-soft) 22%, var(--dashboard-panel-strong)) 0%,
    var(--dashboard-panel-bg) 100%
  );
}
.release-doc .release-h5-fold {
  border-left: 3px solid
    color-mix(in srgb, var(--dashboard-violet) 55%, var(--dashboard-border));
  background: color-mix(
    in srgb,
    var(--dashboard-violet-bg) 14%,
    var(--dashboard-panel-bg)
  );
}
.release-doc .release-h5-fold > summary.report-subcard-summary {
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--dashboard-violet-bg) 55%, var(--dashboard-panel-strong)) 0%,
    var(--dashboard-panel-bg) 100%
  );
}
.release-doc .release-h5-fold > summary.report-subcard-summary::before {
  color: var(--dashboard-violet);
}
/* GPU rows: ``details.panel`` inherited global ``.panel`` padding — inset summary looked flat. Reset + accent edge. */
.release-doc details.panel.test-result-gpu-card.release-gpu-details {
  padding: 0;
  overflow: hidden;
  margin: 0 0 0.95rem;
  background: var(--dashboard-panel-bg);
  border-radius: var(--radius);
  border: 1px solid
    color-mix(in srgb, var(--release-gpu-ico, var(--accent)) 28%, var(--dashboard-border));
  border-left: 4px solid var(--release-gpu-ico, var(--accent));
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.09);
  --release-gpu-ico: var(--accent);
}
.release-doc details.panel.test-result-gpu-card.release-gpu-details.release-gpu-details--h200 {
  --release-gpu-ico: #7c3aed;
}
.release-doc details.panel.test-result-gpu-card.release-gpu-details.release-gpu-details--h800 {
  --release-gpu-ico: #2563eb;
}
.release-doc details.panel.test-result-gpu-card.release-gpu-details.release-gpu-details--a100 {
  --release-gpu-ico: #059669;
}
.release-doc details.panel.test-result-gpu-card.release-gpu-details.release-gpu-details--h100 {
  --release-gpu-ico: var(--ci);
}
.release-doc details.panel.test-result-gpu-card.release-gpu-details:last-child {
  margin-bottom: 0;
}
.release-doc details.panel.test-result-gpu-card.release-gpu-details:hover {
  border-color: color-mix(in srgb, var(--release-gpu-ico, var(--accent)) 45%, var(--dashboard-border));
  border-left-color: var(--release-gpu-ico, var(--accent));
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
}
.release-doc details.release-gpu-details[open] > summary.release-gpu-details-summary {
  border-bottom: 1px solid var(--dashboard-border);
}
.release-doc summary.release-gpu-details-summary {
  list-style: none;
  cursor: pointer;
  padding: 0.92rem 1.05rem 0.92rem 3.15rem;
  position: relative;
  font-weight: 800;
  font-size: 1.08rem;
  color: var(--dashboard-text);
  letter-spacing: -0.02em;
  background: linear-gradient(
    105deg,
    color-mix(in srgb, var(--release-gpu-ico, var(--accent)) 14%, var(--dashboard-panel-strong)) 0%,
    var(--dashboard-panel-bg) 62%
  );
  margin: 0;
}
.release-doc summary.release-gpu-details-summary::-webkit-details-marker {
  display: none;
}
.release-doc summary.release-gpu-details-summary::before {
  content: "▸";
  position: absolute;
  left: 0.88rem;
  top: 0.98rem;
  font-size: 0.95rem;
  font-weight: 800;
  color: var(--release-gpu-ico, var(--accent));
  line-height: 1;
  pointer-events: none;
}
.release-doc details.release-gpu-details[open] > summary.release-gpu-details-summary::before {
  content: "▾";
}
.release-doc .release-gpu-details-body {
  padding: 1rem 1.2rem 1.2rem;
  background: color-mix(in srgb, var(--dashboard-badge-bg) 32%, var(--dashboard-panel-bg));
}
.release-doc .release-gpu-details-body > .release-h4-fold:first-child,
.release-doc .release-gpu-details-body > p:first-child {
  margin-top: 0;
}
.release-doc .release-h5-fold .report-subcard-body {
  font-size: 0.98rem;
}
/* Clicks must reach <summary>; decorative SVGs must not steal the toggle. */
.release-doc details > summary svg,
.release-doc details > summary .ico {
  pointer-events: none;
}
.release-doc > .meta.generated-meta {
  margin: 0;
  padding-bottom: 0.85rem;
  border-bottom: 1px dashed color-mix(in srgb, var(--dashboard-border-strong) 55%, transparent);
  flex-shrink: 0;
}
.release-doc h3 {
  margin: 1.15rem 0 0.55rem;
  font-size: 1.04rem;
  color: var(--dashboard-soft-text);
  font-weight: 700;
}
.release-doc h4, .release-doc h5, .release-doc h6 {
  margin: 1rem 0 0.45rem;
  font-size: 1rem;
  color: var(--dashboard-soft-text);
  font-weight: 700;
}
.release-doc p {
  margin: 0.65rem 0;
}
.release-doc p:first-of-type {
  margin-top: 0;
}
.release-doc ul {
  margin: 0.5rem 0 0.85rem 1.25rem;
  padding: 0;
}
.release-doc li {
  margin: 0.3rem 0;
}
.release-doc .table-scroll {
  margin: 0.85rem 0;
}
.release-doc .table-scroll > table {
  min-width: 480px;
}
.release-doc table {
  border-collapse: collapse;
  width: 100%;
  margin: 0;
  font-size: 0.92rem;
}
.release-doc th, .release-doc td {
  border: 1px solid var(--border);
  padding: 0.65rem 0.8rem;
  text-align: left;
  vertical-align: top;
}
.release-doc th {
  background: var(--dashboard-panel-strong);
  font-weight: 650;
  color: var(--dashboard-chart-text);
}
.release-doc tbody tr:nth-child(even) td {
  background: color-mix(in srgb, var(--dashboard-badge-bg) 45%, var(--dashboard-panel-bg));
}
.release-doc code {
  background: color-mix(in srgb, var(--dashboard-badge-bg) 55%, var(--dashboard-panel-bg));
  padding: 0.12em 0.38em;
  border-radius: 4px;
  font-size: 0.92em;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}
.release-doc a {
  color: var(--accent);
  font-weight: 600;
  text-decoration: none;
  border-bottom: 1px solid var(--accent-soft);
}
.release-doc a:hover {
  color: var(--accent-hover);
  border-bottom-color: var(--accent);
}
.release-conclusion-wrap {
  margin: 0.5rem 0 1rem;
}
.release-conclusion-table {
  min-width: 320px;
}
.release-conclusion-table td:last-child {
  white-space: normal;
}
.conc-btns {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  align-items: center;
}
.conc-btns.conc-auto {
  pointer-events: none;
  opacity: 0.9;
}
.conc-auto-hint {
  margin-top: 0.35rem;
  font-size: 0.82rem;
  line-height: 1.35;
  color: color-mix(in srgb, var(--dashboard-text) 65%, transparent);
}
.conc-btn {
  font: inherit;
  font-size: 0.86rem;
  font-weight: 650;
  padding: 0.38rem 0.95rem;
  border-radius: 8px;
  cursor: pointer;
  border: 1px solid var(--border);
  background: color-mix(in srgb, var(--dashboard-panel-bg) 88%, var(--dashboard-badge-bg));
  color: var(--dashboard-text);
  transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease;
}
.conc-btn:hover {
  border-color: var(--accent-soft);
}
.conc-btn.conc-pass.is-on {
  background: color-mix(in srgb, #16a34a 22%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #16a34a 45%, var(--border));
  color: #15803d;
}
.conc-btn.conc-fail.is-on {
  background: color-mix(in srgb, #dc2626 20%, var(--dashboard-panel-bg));
  border-color: color-mix(in srgb, #dc2626 42%, var(--border));
  color: #b91c1c;
}
.release-verdict-line {
  margin: 1rem 0 0.25rem;
  font-size: 1.02rem;
  font-weight: 650;
}
.release-verdict {
  font-weight: 800;
  letter-spacing: 0.02em;
}

/* Development-report Metrics overview: highlight at-risk snapshot rows in red.
   Used by compose_full_report.py --kind development when the helper wraps an
   alert-bearing cell in <span class="dev-snapshot-alert">…</span>. */
.release-doc .dev-snapshot-alert {
  display: inline-block;
  padding: 0.05em 0.55em;
  border-radius: 6px;
  background: color-mix(in srgb, var(--dashboard-alert-bg, rgba(209,67,67,0.11)) 88%,
    var(--dashboard-panel-bg, #ffffff));
  color: var(--dashboard-alert, #d14343);
  font-weight: 700;
}

/* UT coverage editable cell — sits inline inside the Result column of the
   Development Metrics overview. The cell renders **only the value** (no
   inline editor / reset / saved-hint buttons) so the row reads the same
   as every other Result cell. Clicking the value opens the in-page modal
   below (works in iframes where ``window.prompt`` is blocked). */
.ut-coverage-cell {
  display: inline;
}
.ut-coverage-btn {
  display: inline;
  padding: 0;
  margin: 0;
  background: transparent;
  border: 0;
  border-bottom: 1px dashed color-mix(in srgb, var(--dashboard-muted) 65%, transparent);
  color: inherit;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}
.ut-coverage-btn:hover {
  border-bottom-style: solid;
  color: var(--accent);
}

/* UT coverage modal — used by the Development Metrics overview snapshot
   (works in iframe contexts where ``window.prompt`` is blocked). The modal
   owns Save / Cancel / Reset so the cell itself stays clean. */
.ut-coverage-modal[hidden] {
  display: none !important;
}
.ut-coverage-modal {
  position: fixed;
  inset: 0;
  z-index: 10002;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
}
.ut-coverage-modal-backdrop {
  position: absolute;
  inset: 0;
  background: color-mix(in srgb, var(--dashboard-text) 42%, transparent);
  backdrop-filter: blur(2px);
}
.ut-coverage-modal-panel {
  position: relative;
  z-index: 1;
  width: min(92vw, 32rem);
  display: flex;
  flex-direction: column;
  background: var(--dashboard-panel-bg);
  border-radius: var(--radius-md);
  border: 1px solid var(--dashboard-border);
  box-shadow: 0 20px 50px color-mix(in srgb, var(--dashboard-text) 22%, transparent);
  overflow: hidden;
}
.ut-coverage-modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem 1.1rem;
  border-bottom: 1px solid var(--dashboard-border);
  background: linear-gradient(180deg, var(--dashboard-panel-strong) 0%, var(--dashboard-panel-bg) 100%);
}
.ut-coverage-modal-header h2 {
  margin: 0;
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--dashboard-text);
}
.ut-coverage-modal-close {
  flex-shrink: 0;
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--dashboard-border);
  border-radius: 8px;
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  font-size: 1.35rem;
  line-height: 1;
  cursor: pointer;
}
.ut-coverage-modal-body {
  padding: 1rem 1.1rem;
}
.ut-coverage-modal-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.85rem;
  color: var(--dashboard-text);
}
.ut-coverage-modal-label {
  font-weight: 600;
  color: var(--dashboard-text);
}
.ut-coverage-modal-hint {
  font-size: 0.75rem;
  color: var(--dashboard-muted);
}
.ut-coverage-modal-input {
  font: inherit;
  font-size: 0.9rem;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--dashboard-border);
  border-radius: 8px;
  background: var(--dashboard-panel-bg);
  color: var(--text);
  width: 100%;
  box-sizing: border-box;
}
.ut-coverage-modal-input:focus {
  outline: 2px solid color-mix(in srgb, var(--accent) 35%, transparent);
  outline-offset: 1px;
  border-color: var(--accent);
}
.ut-coverage-modal-footer {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.85rem 1.1rem;
  border-top: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-strong);
}
.ut-coverage-modal-footer-spacer {
  flex: 1 1 auto;
}
.ut-coverage-modal-reset,
.ut-coverage-modal-cancel,
.ut-coverage-modal-save {
  font: inherit;
  font-size: 0.85rem;
  padding: 0.45rem 0.95rem;
  border-radius: 8px;
  border: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  cursor: pointer;
}
.ut-coverage-modal-save {
  background: var(--dashboard-healthy);
  color: #fff;
  border-color: var(--dashboard-healthy);
  font-weight: 600;
}
.ut-coverage-modal-save:hover {
  filter: brightness(0.95);
}
.ut-coverage-modal-cancel:hover {
  border-color: var(--dashboard-warning);
  color: var(--danger-strong);
}
.ut-coverage-modal-reset:hover {
  border-color: var(--dashboard-warning);
  color: var(--danger-strong);
}
.ut-coverage-modal-open {
  overflow: hidden;
}

/* Fail-status modal — used by the failure-analysis tables when the report is
   embedded via iframe (kanban Reports page). The legacy ``window.prompt()``
   flow is blocked by iframe sandbox lacking ``allow-modals``; this in-page
   modal works in both standalone and iframe contexts. */
.fail-status-modal[hidden] {
  display: none !important;
}
.fail-status-modal {
  position: fixed;
  inset: 0;
  z-index: 10001;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.25rem;
}
.fail-status-modal-backdrop {
  position: absolute;
  inset: 0;
  background: color-mix(in srgb, var(--dashboard-text) 42%, transparent);
  backdrop-filter: blur(2px);
}
.fail-status-modal-panel {
  position: relative;
  z-index: 1;
  width: min(92vw, 38rem);
  display: flex;
  flex-direction: column;
  background: var(--dashboard-panel-bg);
  border-radius: var(--radius-md);
  border: 1px solid var(--dashboard-border);
  box-shadow: 0 20px 50px color-mix(in srgb, var(--dashboard-text) 22%, transparent);
  overflow: hidden;
}
.fail-status-modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.85rem 1.1rem;
  border-bottom: 1px solid var(--dashboard-border);
  background: linear-gradient(180deg, var(--dashboard-panel-strong) 0%, var(--dashboard-panel-bg) 100%);
}
.fail-status-modal-header h2 {
  margin: 0;
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--dashboard-text);
}
.fail-status-modal-close {
  flex-shrink: 0;
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--dashboard-border);
  border-radius: 8px;
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  font-size: 1.35rem;
  line-height: 1;
  cursor: pointer;
}
.fail-status-modal-body {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem 1.1rem;
}
.fail-status-modal-field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.85rem;
  color: var(--dashboard-text);
}
.fail-status-modal-label {
  font-weight: 600;
  color: var(--dashboard-text);
}
.fail-status-modal-hint {
  font-size: 0.75rem;
  color: var(--dashboard-muted);
}
.fail-status-modal-input,
.fail-status-modal-textarea {
  font: inherit;
  font-size: 0.9rem;
  padding: 0.5rem 0.6rem;
  border: 1px solid var(--dashboard-border);
  border-radius: 8px;
  background: var(--dashboard-panel-bg);
  color: var(--text);
  width: 100%;
  box-sizing: border-box;
}
.fail-status-modal-textarea {
  font-family: ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 0.8rem;
  min-height: 5.5rem;
  resize: vertical;
}
.fail-status-modal-input:focus,
.fail-status-modal-textarea:focus {
  outline: 2px solid color-mix(in srgb, var(--accent) 35%, transparent);
  outline-offset: 1px;
  border-color: var(--accent);
}
.fail-status-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 0.55rem;
  padding: 0.85rem 1.1rem;
  border-top: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-strong);
}
.fail-status-modal-cancel,
.fail-status-modal-save {
  font: inherit;
  font-size: 0.85rem;
  padding: 0.45rem 0.95rem;
  border-radius: 8px;
  border: 1px solid var(--dashboard-border);
  background: var(--dashboard-panel-bg);
  color: var(--dashboard-text);
  cursor: pointer;
}
.fail-status-modal-save {
  background: var(--dashboard-healthy);
  color: #fff;
  border-color: var(--dashboard-healthy);
  font-weight: 600;
}
.fail-status-modal-save:hover {
  filter: brightness(0.95);
}
.fail-status-modal-cancel:hover {
  border-color: var(--dashboard-warning);
  color: var(--danger-strong);
}
.fail-status-modal-open {
  overflow: hidden;
}
.fail-status-note {
  color: var(--dashboard-muted);
  font-style: italic;
}
"""
