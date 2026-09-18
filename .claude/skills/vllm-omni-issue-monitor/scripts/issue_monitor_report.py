#!/usr/bin/env python3
"""
Generate a standalone HTML report for **Outstanding DI Top 20 (Issue Monitor)**.

This skill is the focused twin of the Daily-focus sub-card that lives inside
`vllm-omni-test-report`'s nightly HTML. It emits a single self-contained
HTML file (no Buildkite / local log / kanban dependency) with:

  * An **Outstanding DI** metric card (red when total DI > 30)
  * A collapsible **Top 20 DI contributors** table with editable Assignee /
    Maintainer cells, expanded to Top 40 when more rows exist.

The DI formula and table layout are byte-for-byte aligned with the parent
skill's nightly section so triaging work can move between reports without
context switching.

Usage:
    python issue_monitor_report.py                     # writes issue-monitor-report-YYYY-MM-DD.html
    python issue_monitor_report.py --preview          # empty dataset, offline smoke test
    python issue_monitor_report.py --mock-data        # synthetic 23-issue dataset (no GitHub calls)
    python issue_monitor_report.py --date 2026-08-26  # explicit UTC date
    python issue_monitor_report.py --out ./my.html    # override output path
    python issue_monitor_report.py --gh-token TOKEN   # explicit token (or env GITHUB_TOKEN/GH_TOKEN)
    python issue_monitor_report.py --no-token         # unauthenticated REST (degraded)
    python issue_monitor_report.py --archive          # copy (overwrite) + commit + push to vllm-omni-kanban
    python issue_monitor_report.py --archive-dry-run  # preview archive plan without touching git

GitHub token auth is identical to the parent skill: ``GITHUB_TOKEN`` or
``GH_TOKEN`` from the environment, never via chat.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

# Pull the shared theme CSS + naming helpers from the parent skill's scripts/
# so the standalone HTML looks identical to the parent nightly's Daily focus.
_PARENT_SKILL_SCRIPTS = (
    Path(__file__).resolve().parent.parent.parent
    / "vllm-omni-test-report"
    / "scripts"
)
if str(_PARENT_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PARENT_SKILL_SCRIPTS))

from report_html_theme import EDITORIAL_THEME_CSS  # noqa: E402
from report_naming import (  # noqa: E402
    default_issue_monitor_html_path,
    issue_monitor_report_basename,
    issue_monitor_report_preview_basename,
    issue_monitor_report_title,
    resolve_report_date_iso,
)

# ---------------------------------------------------------------------------
# Constants — copied verbatim from nightly_local_log_report.py to keep math
# identical between this standalone report and the parent nightly's focus
# card (see SKILL.md "Daily focus DI card (SLO-escalating model)").
# ---------------------------------------------------------------------------

BUG_DI_TABLE: dict[str, dict[str, Any]] = {
    "critical": {"base": 10.0, "slo_days": 1, "order": 0},
    "high priority": {"base": 3.0, "slo_days": 5, "order": 1},
    "medium priority": {"base": 1.0, "slo_days": 10, "order": 2},
    "low priority": {"base": 0.1, "slo_days": 14, "order": 3},
    "invalid": {"base": 0.0, "slo_days": None, "order": 4},
}
BUG_DI_LABEL_ORDER = ("critical", "high priority", "medium priority", "low priority", "invalid")
BUG_DI_RED_THRESHOLD = 30.0  # total DI > 30 ⇒ focus-card--fail
STALE_BUG_DAYS: float = 60.0  # "open > 2 months" = the stale-bug threshold
# Label names that mark an issue as "do not rank" — the issue is still counted
# toward the Outstanding DI total but never appears in the Top table.
_DROPPED_DI_LABELS: frozenset[str] = frozenset(
    {"wontfix", "won't fix", "won’t fix", "invalid"}
)

_REPO = "vllm-project/vllm-omni"
_REPO_URL = "https://github.com/vllm-project/vllm-omni"

# ---------------------------------------------------------------------------
# Kanban archive constants (used only when --archive / 归档报告 is invoked).
# ---------------------------------------------------------------------------

KANBAN_REPO_URL = "https://github.com/hsliuustc0106/vllm-omni-kanban"
KANBAN_DEFAULT_REPO_ROOT = Path("/home/wy/vllm-omni-kanban")
KANBAN_ARCHIVE_DIR = Path("data") / "issue_monitor"
KANBAN_ARCHIVE_FILENAME = "issue-monitor-report.html"  # no date — always overwrites
GH_GIT_CREDENTIAL_HELPER = "!gh auth git-credential"
GH_CLI_INSTALL_HINT = (
    "GitHub CLI (gh) is not installed. Archive push requires gh:\n"
    "  Linux:   https://github.com/cli/cli/blob/trunk/docs/install_linux.md\n"
    "After install, log in: gh auth login"
)
GH_AUTH_HINT = (
    "gh is not logged in or the token is invalid. Run: gh auth login\n"
    "Or set GH_TOKEN / GITHUB_TOKEN in the environment (requires repo scope)."
)


def _resolve_kanban_repo_root(explicit: str | None = None) -> Path:
    """Locate the local kanban checkout.

    Precedence: explicit `--kanban-repo-root` > ``$KANBAN_REPO_ROOT`` >
    ``/home/wy/vllm-omni-kanban`` (HOME-mismatch fallback scans
    ``/home/*/vllm-omni-kanban`` when HOME does not contain one).
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_raw = (os.environ.get("KANBAN_REPO_ROOT") or "").strip()
    if env_raw:
        return Path(env_raw).expanduser().resolve()
    default = KANBAN_DEFAULT_REPO_ROOT.expanduser().resolve()
    if default.is_dir():
        return default
    home_root = Path("/home")
    if home_root.is_dir():
        for candidate in sorted(home_root.iterdir()):
            ck = candidate / "vllm-omni-kanban"
            if ck.is_dir():
                return ck.resolve()
    return default  # downstream callers treat missing as no-op

# ---------------------------------------------------------------------------
# GitHub HTTP helpers (mirrors nightly_local_log_report.py::_http_get_json_nightly
# and ::_github_fetch_open_bug_issues_nightly).
# ---------------------------------------------------------------------------


def _http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> object:
    h = dict(headers or {})
    try:
        import requests  # type: ignore

        try:
            resp = requests.get(url, headers=h, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            pass  # fall through to urllib
    except ImportError:
        pass
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _resolve_github_token(explicit: str | None = None, *, allow_none: bool = False) -> str | None:
    if explicit:
        return explicit.strip() or None
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return None if allow_none else None


def _fetch_open_bug_issues(gh_token: str | None) -> list[dict[str, Any]]:
    """Paginate ``GET /repos/{owner}/{repo}/issues?state=open&labels=bug``.

    PRs (``pull_request`` field present) are excluded — the bug label is
    shared between them but the issue-vs-PR distinction matters for the
    Top contributors table.
    """
    base = f"https://api.github.com/repos/{_REPO}/issues"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-issue-monitor-report",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{base}?state=open&labels=bug&per_page=100&page={page}"
        try:
            batch = _http_get_json(url, headers=headers, timeout=60)
        except Exception:
            return out  # partial OK
        if not batch:
            break
        for issue in batch:
            if issue.get("pull_request"):
                continue
            out.append(issue)
        if len(batch) < 100:
            break
        page += 1
    return out


def _fetch_bugfix_pr_links_via_graphql(
    gh_token: str | None,
    issue_numbers: list[int],
) -> dict[int, list[int]]:
    """Single GraphQL pass for ``Issue.closedByPullRequestsReferences``.

    Mirrors ``_fetch_bugfix_pr_links`` in the parent nightly script — the
    canonical mirror of GitHub's Development panel. Network/API failure
    degrades gracefully to ``{}`` (every Bugfix cell then shows ``—``).

    Note: the alias query must be wrapped in
    ``query($owner: String!, $name: String!) { repository(...) { ... } }``
    — the legacy root-level ``issue(number:)`` field was removed from
    GitHub's GraphQL schema, and querying it returns
    ``"Field 'issue' doesn't exist on type 'Query'"`` which previously
    caused every Bugfix cell to silently degrade to ``—``.
    """
    if not issue_numbers:
        return {}
    if not gh_token:
        return {}
    parts: list[str] = []
    for n in issue_numbers:
        parts.append(
            f"i{n}: issue(number: {n}) {{ number closedByPullRequestsReferences(first: 10) {{ nodes {{ number }} }} }}"
        )
    query = (
        "query($owner: String!, $name: String!) { "
        "repository(owner: $owner, name: $name) { "
        + " ".join(parts)
        + " } }"
    )
    url = "https://api.github.com/graphql"
    body = json.dumps(
        {
            "query": query,
            "variables": {"owner": _REPO.split("/")[0], "name": _REPO.split("/")[1]},
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {gh_token}",
        "Content-Type": "application/json",
        "User-Agent": "vllm-omni-issue-monitor-report",
    }
    try:
        # GraphQL via urllib directly (the aliases build is too dynamic for
        # the helper above).
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return {}
    out: dict[int, list[int]] = {}
    if not isinstance(data, dict):
        return out
    # Surface API-level errors (e.g. "Field 'issue' doesn't exist" if the
    # schema regresses again) so the empty-result mystery is debuggable.
    if data.get("errors"):
        return out
    repo = (data.get("data") or {}).get("repository") or {}
    if not isinstance(repo, dict):
        return out
    for n in issue_numbers:
        node = repo.get(f"i{n}") or {}
        if not isinstance(node, dict):
            continue
        ref_nodes = ((node.get("closedByPullRequestsReferences") or {}).get("nodes")) or []
        nums: list[int] = []
        for r in ref_nodes:
            try:
                rn = int((r or {}).get("number") or 0)
            except (TypeError, ValueError):
                continue
            if rn:
                nums.append(rn)
        if nums:
            out[n] = list(dict.fromkeys(nums))
    return out


# ---------------------------------------------------------------------------
# DI math (mirror of nightly_local_log_report.py::_compute_outstanding_di).
# ---------------------------------------------------------------------------


def _bug_di_priority_class(issue: dict[str, Any]) -> str:
    label_names = [str(label.get("name") or "") for label in issue.get("labels", []) or []]
    if "invalid" in label_names:
        return "invalid"
    for label in BUG_DI_LABEL_ORDER:
        if label == "invalid":
            continue
        if label in label_names:
            return label
    return "unclassified"


def _compute_issue_di(
    issue: dict[str, Any],
    now: datetime | None = None,
) -> tuple[float, str, float, int, str, str]:
    if now is None:
        now = datetime.now(timezone.utc)
    priority = _bug_di_priority_class(issue)
    created = str(issue.get("created_at") or "").strip()
    days_open = 0.0
    if created:
        try:
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            days_open = max(0.0, (now - created_dt).total_seconds() / 86400.0)
        except Exception:
            days_open = 0.0
    issue_number = int(issue.get("number") or 0)
    title = str(issue.get("title") or "").strip()
    assignees = issue.get("assignees") or []
    assignee = (
        ", ".join("@" + str(a.get("login", "")) for a in assignees if isinstance(a, dict) and a.get("login"))
        if assignees
        else ""
    )
    if priority not in BUG_DI_TABLE:
        return 0.0, priority, days_open, issue_number, title, assignee
    info = BUG_DI_TABLE[priority]
    if info["slo_days"] is None or info["base"] == 0:
        return 0.0, priority, days_open, issue_number, title, assignee
    n_slos = max(0, ceil(days_open / info["slo_days"]))
    return info["base"] * n_slos, priority, days_open, issue_number, title, assignee


def _filter_dropped_di_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``wontfix`` / ``invalid`` issues from the Top ranking."""
    out: list[dict[str, Any]] = []
    for issue in issues:
        labels = [str(label.get("name") or "").lower() for label in issue.get("labels", []) or []]
        if any(label in _DROPPED_DI_LABELS for label in labels):
            continue
        out.append(issue)
    return out


def _compute_stale_bug_rows(
    issues: list[dict[str, Any]],
    now: datetime,
) -> list[tuple[float, str, float, int, str, str]]:
    """Per-issue rows for bugs open more than ``STALE_BUG_DAYS`` days.

    Excludes ``wontfix`` / ``invalid`` (mirrors Top 20 filtering) and
    returns rows sorted by ``days_open`` descending — oldest first, so
    the most-neglected issues surface at the top of the sub-card.

    Tuple shape matches ``_compute_issue_di`` so callers can re-use the
    same downstream columns: ``(di, priority, days_open, number, title,
    assignee)``.
    """
    rows: list[tuple[float, str, float, int, str, str]] = []
    for issue in _filter_dropped_di_issues(issues):
        di, prio, days, num, title, assignee = _compute_issue_di(issue, now=now)
        if days > STALE_BUG_DAYS:
            rows.append((di, prio, days, num, title, assignee))
    rows.sort(key=lambda r: -r[2])
    return rows


def _format_di_value(value: float) -> str:
    if value == 0:
        return "0"
    s = f"{value:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _shorten_title(title: str, max_len: int = 60) -> str:
    if len(title) <= max_len:
        return title
    return title[: max_len - 1].rstrip() + "…"


def _build_mock_issues(now: datetime) -> list[dict[str, Any]]:
    """Synthesize ~25 representative issues so the preview exercises every
    rendering path (all 5 priorities, red threshold crossed, Expand-to-Top-40
    visible, mix of with/without bugfix links, plus a ``wontfix`` count).

    Days-open is anchored to ``now`` so the SLO-escalating math produces the
    same numbers the parent nightly would.
    """

    def _issue(
        number: int,
        title: str,
        labels: list[str],
        assignees: list[str] | None = None,
        created_offset_days: int = 10,
    ) -> dict[str, Any]:
        from datetime import timedelta

        created = (now - timedelta(days=created_offset_days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        return {
            "number": number,
            "title": title,
            "created_at": created,
            "labels": [{"name": n} for n in (["bug"] + labels)],
            "assignees": [{"login": a} for a in (assignees or [])],
            "pull_request": None,
        }

    return [
        # 4 critical (each contributes 10 × ceil(days/1))
        _issue(101, "Pipeline stalls when KV cache is reloaded mid-generation (regression after #94)",
               ["critical"], ["alice"], created_offset_days=3),
        _issue(102, "Crash: tensor parallel worker OOM on >4 GPU configs", ["critical"], ["bob"], created_offset_days=2),
        _issue(103, "Audio tokenizer produces NaNs for empty inputs", ["critical"], [], created_offset_days=1),
        _issue(104, "Inference server hangs on graceful shutdown", ["critical"], ["carol"], created_offset_days=4),
        # 6 high priority (base 3.0, slo 5d)
        _issue(110, "LoRA adapter weights leak between concurrent requests in Stream API",
               ["high priority"], ["alice", "dave"], created_offset_days=12),
        _issue(111, "Tokenizer trims trailing whitespace inconsistently across models",
               ["high priority"], ["bob"], created_offset_days=7),
        _issue(112, "Diffusion scheduler miscalculates CFG at batch_size > 8",
               ["high priority"], [], created_offset_days=15),
        _issue(113, "Chat template fails on multi-turn tool messages with image content",
               ["high priority"], ["carol"], created_offset_days=11),
        _issue(114, "Engine version mismatch between worker and frontend (404 on /v1/models)",
               ["high priority"], ["dave"], created_offset_days=9),
        _issue(115, "Vision encoder miscomputes RoPE positions for non-square inputs",
               ["high priority"], ["alice"], created_offset_days=18),
        # 8 medium priority (base 1.0, slo 10d)
        _issue(120, "Slow first-token latency when KV cache exceeds 80% capacity",
               ["medium priority"], ["bob"], created_offset_days=22),
        _issue(121, "Paged attention swaps to CPU under bursty request patterns",
               ["medium priority"], ["alice"], created_offset_days=25),
        _issue(122, "Speculative decoding draft model not warmed on hot path",
               ["medium priority"], ["dave"], created_offset_days=14),
        _issue(123, "Tokenizer cache key collides for similar but distinct unicode inputs",
               ["medium priority"], [], created_offset_days=30),
        _issue(124, "Logging middleware doubles latency on /v1/chat/completions",
               ["medium priority"], ["bob"], created_offset_days=12),
        _issue(125, "Worker health probe misreports during graceful shutdown window",
               ["medium priority"], ["carol"], created_offset_days=19),
        _issue(126, "Diffusion sampler diverges when guidance_rescale > 0.7",
               ["medium priority"], [], created_offset_days=28),
        _issue(127, "ASR endpoint returns 200 with empty body for partial failures",
               ["medium priority"], ["dave"], created_offset_days=17),
        # 4 low priority (base 0.1, slo 14d)
        _issue(130, "CLI help text truncates after --max-model-len flag",
               ["low priority"], [], created_offset_days=16),
        _issue(131, "Deprecation warnings emitted by httpx pollute worker logs",
               ["low priority"], ["alice"], created_offset_days=22),
        _issue(132, "Image preview thumbnails miss for some .webp variants",
               ["low priority"], ["bob"], created_offset_days=10),
        _issue(133, "Documentation link 404s for legacy v0.3 examples",
               ["low priority"], [], created_offset_days=31),
        # 1 invalid (counts toward total but never ranks)
        _issue(140, "Outdated: legacy checkpoint format no longer supported",
               ["invalid"], ["carol"], created_offset_days=5),
        # 3 stale bugs (> STALE_BUG_DAYS = 60) — exercise the Stale Bugs
        # sub-card under --mock-data. Sorted by age (oldest first).
        _issue(150, "[Bug]: Async request batching drops cancellation tokens on retry",
               ["high priority"], ["alice"], created_offset_days=75),
        _issue(151, "[Bug]: Diffusion cfg_rescale > 1.0 emits NaN gradients after warmup",
               ["medium priority"], ["bob", "dave"], created_offset_days=95),
        _issue(152, "[Bug]: Legacy `--enable-prefix-caching` flag silently ignored on omni models",
               ["low priority"], [], created_offset_days=120),
    ]


def _compute_outstanding_di(
    gh_token: str | None,
    now: datetime | None = None,
    *,
    mock: bool = False,
) -> dict[str, Any]:
    if mock:
        if now is None:
            now = datetime.now(timezone.utc)
        issues = _build_mock_issues(now)
        # Synthetic Bugfix PR links for a few issues so the Bugfix column has
        # both populated and empty rows (matches parent nightly shape).
        bugfix_links: dict[int, list[int]] = {
            101: [205, 207],
            110: [198],
            111: [203],
            113: [],
            120: [212],
            124: [],
        }
    else:
        issues = _fetch_open_bug_issues(gh_token)
    rankable = _filter_dropped_di_issues(issues)
    if now is None:
        now = datetime.now(timezone.utc)
    counts = {label: 0 for label in BUG_DI_TABLE}
    counts["unclassified"] = 0
    total = 0.0
    for issue in issues:
        di, priority, _, _, _, _ = _compute_issue_di(issue, now=now)
        total += di
        if priority in counts:
            counts[priority] += 1
        else:
            counts["unclassified"] += 1
    per_issue_raw: list[tuple[float, str, float, int, str, str]] = []
    for issue in rankable:
        di, priority, days_open, issue_number, title, assignee = _compute_issue_di(issue, now=now)
        per_issue_raw.append((di, priority, days_open, issue_number, title, assignee))
    per_issue_sorted = sorted(per_issue_raw, key=lambda x: -x[0])
    top_issue_numbers = [t[3] for t in per_issue_sorted[:40] if t[3]]
    if mock:
        # Don't actually call GraphQL in mock mode — the synthetic map above
        # is sufficient and keeps the smoke test fully offline.
        pass
    else:
        bugfix_links = _fetch_bugfix_pr_links_via_graphql(gh_token, top_issue_numbers)
    per_issue_with_bugfix: list[tuple[float, str, float, int, str, str, list[int]]] = []
    for di, priority, days_open, issue_number, title, assignee in per_issue_sorted:
        per_issue_with_bugfix.append(
            (di, priority, days_open, issue_number, title, assignee, bugfix_links.get(issue_number, []))
        )
    # Stale bugs (> STALE_BUG_DAYS days) — merge into the same bugfix-link
    # GraphQL pass so we make at most one HTTP call for the section.
    stale_rows = _compute_stale_bug_rows(issues, now=now)
    stale_numbers = [r[3] for r in stale_rows if r[3]]
    already = set(top_issue_numbers)
    extra_numbers = [n for n in stale_numbers if n not in already]
    if extra_numbers and not mock:
        # In mock mode the synthetic map above already covers the stale rows;
        # skip the GraphQL call so the smoke test stays offline.
        bugfix_links.update(_fetch_bugfix_pr_links_via_graphql(gh_token, extra_numbers))
    stale_with_bugfix: list[tuple[float, str, float, int, str, str, list[int]]] = [
        (di, prio, days, num, title, assignee, bugfix_links.get(num, []))
        for (di, prio, days, num, title, assignee) in stale_rows
    ]
    severity = "fail" if total > BUG_DI_RED_THRESHOLD else "ok"
    detail_parts = [f"{p}={counts[p]}" for p in BUG_DI_TABLE if counts.get(p, 0)]
    if counts.get("unclassified", 0):
        detail_parts.append(f"unclassified={counts['unclassified']}")
    detail = ", ".join(detail_parts) if detail_parts else "no open bug in snapshot"
    return {
        "total_di": total,
        "n_issues": len(issues),
        "counts": counts,
        "per_issue": per_issue_with_bugfix,
        "stale_bugs": stale_with_bugfix,
        "stale_count": len(stale_rows),
        "value": _format_di_value(total),
        "detail": detail,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# HTML rendering — focus card + Top 20 table + sub-card wrapper.
# ---------------------------------------------------------------------------

_SVG_LIST = '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>'
_SVG_SPARK = '<path d="m12 3-1.9 5.8L4 10l5.8 1.9L12 18l1.9-5.8L20 10l-6.2-1.9L12 3z"/>'
_SVG_CLIPBOARD = (
    '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
    '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
)


def _svg_icon(inner: str, *, size: int = 20, extra_class: str = "") -> str:
    c = f"ico {extra_class}".strip()
    return (
        f'<svg class="{c}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
        f"{inner}</svg>"
    )


def _heading_html(tag: str, icon_paths: str, label_inner: str, *, sub: str | None = None) -> str:
    sub_html = f'<span class="heading-sub">{html.escape(sub)}</span>' if sub else ""
    return (
        f"<{tag}>"
        '<span class="heading-row">'
        f'<span class="heading-ico">{_svg_icon(icon_paths, size=22)}</span>'
        f'<span class="heading-text"><span class="heading-label">{label_inner}</span>{sub_html}</span>'
        "</span>"
        f"</{tag}>"
    )


def _details_subcard(
    title: str,
    body_html: str,
    *,
    open_default: bool = False,
    details_class: str = "",
) -> str:
    """Render a collapsible sub-card with a dedicated arrow slot.

    The arrow is rendered as an explicit ``.report-subcard-arrow`` span
    (instead of relying solely on ``summary::before``) so the layout has
    three flex children of fixed widths:

        [▸ arrow]  [📋 icon]  [title]

    Combined with the CSS that hides both ``::-webkit-details-marker``
    and ``details > summary::marker``, this prevents the default
    disclosure triangle from leaking through (Firefox / Chromium) and
    overlapping with the custom arrow + SVG icon.
    """
    op = " open" if open_default else ""
    extra = f" {details_class.strip()}" if details_class.strip() else ""
    te = html.escape(title)
    return (
        f'<details class="report-subcard{extra}"{op}>'
        f'<summary class="report-subcard-summary">'
        f'<span class="report-subcard-arrow" aria-hidden="true"></span>'
        f'<span class="report-subcard-summary-inner">'
        f'{_svg_icon(_SVG_LIST, size=18, extra_class="report-subcard-ico")}'
        f'<span class="report-subcard-title">{te}</span>'
        f"</span></summary>"
        f'<div class="report-subcard-body">{body_html}</div>'
        "</details>"
    )


def _render_focus_metric_card(title: str, value: str, detail: str, severity: str) -> str:
    return (
        f'<div class="focus-card focus-card--{html.escape(severity)}">'
        f'<div class="focus-card-title">{html.escape(title)}</div>'
        f'<div class="focus-card-value">{html.escape(value)}</div>'
        f'<div class="focus-card-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def _di_assignee_cell_html(issue_number: int, assignee: str) -> str:
    """Same persistence + DOM-attribute semantics as the parent nightly."""
    key = f"#{issue_number}"
    key_attr = html.escape(key, quote=True)
    if assignee:
        text = html.escape(assignee, quote=True)
        return (
            f'<td class="di-assignee-cell is-filled">'
            f'<span class="di-assignee-text" data-da-key="{key_attr}">'
            f"{text}</span></td>"
        )
    placeholder = html.escape("+ Add assignee", quote=True)
    return (
        f'<td class="di-assignee-cell is-empty">'
        f'<input type="text" class="di-assignee-input" '
        f'data-da-key="{key_attr}" data-da-value="" '
        f'data-da-persisted="0" '
        f'placeholder="{placeholder}" value="" />'
        f"</td>"
    )


def _di_maintainer_cell_html(issue_number: int) -> str:
    key = f"#{issue_number}"
    key_attr = html.escape(key, quote=True)
    placeholder = html.escape("+ Add maintainer", quote=True)
    return (
        f'<td class="di-maintainer-cell is-empty">'
        f'<input type="text" class="di-maintainer-input" '
        f'data-dm-key="{key_attr}" data-dm-value="" '
        f'data-dm-persisted="0" '
        f'placeholder="{placeholder}" value="" />'
        f"</td>"
    )


def _di_bugfix_cell_html(linked: list[int]) -> str:
    if linked:
        parts = [
            f'<a href="{_REPO_URL}/pull/{n}" target="_blank" rel="noopener">#{n}</a>'
            for n in linked
        ]
        return f'<td class="di-bugfix-cell">{", ".join(parts)}</td>'
    return '<td class="di-bugfix-cell di-bugfix-cell--none">—</td>'


def _render_top_di_table_html(
    per_issue: list[tuple[float, str, float, int, str, str, list[int]]],
) -> str:
    """Top 20 default; expands to Top 40 via toggle button."""
    if not per_issue:
        return ""

    def _row_html(
        di: float,
        priority: str,
        days_open: float,
        issue_number: int,
        title: str,
        assignee: str,
        linked: list[int],
    ) -> str:
        title_disp = html.escape(_shorten_title(title))
        issue_url = f"{_REPO_URL}/issues/{issue_number}"
        days_disp = f"{int(days_open)}"
        di_disp = html.escape(_format_di_value(di))
        priority_disp = html.escape(priority)
        return (
            "<tr>"
            f'<td><a href="{issue_url}" target="_blank" rel="noopener">#{issue_number}</a></td>'
            f"<td>{title_disp}</td>"
            f"<td>{priority_disp}</td>"
            f'<td class="di-days">{days_disp}</td>'
            f'<td class="di-value">{di_disp}</td>'
            f"{_di_assignee_cell_html(issue_number, assignee)}"
            f"{_di_maintainer_cell_html(issue_number)}"
            f"{_di_bugfix_cell_html(linked)}"
            "</tr>"
        )

    visible_rows = per_issue[:20]
    extra_rows = per_issue[20:40]
    main_tbody = "".join(_row_html(*row) for row in visible_rows)

    parts: list[str] = [
        '<div class="focus-top-table" data-top-di-table>',
        '<table class="summary top-di-table">',
        "<thead><tr>",
        "<th>#</th><th>Title</th><th>Priority</th><th>Days</th>",
        "<th>DI</th><th>Assignee</th><th>Maintainer</th><th>Bugfix</th>",
        "</tr></thead>",
        f"<tbody>{main_tbody}</tbody>",
    ]
    if extra_rows:
        extra_tbody = "".join(_row_html(*row) for row in extra_rows)
        table_id = "di-top-table-extra"
        parts.extend(
            [
                f'<tbody class="di-top-table-collapsed" id="{table_id}" hidden>',
                extra_tbody,
                "</tbody>",
                "</table>",
                "</div>",
                '<div class="di-top-table-toggle-row">',
                '<button type="button" class="di-top-table-toggle" '
                f'aria-expanded="false" aria-controls="{table_id}" '
                f'onclick="var t=document.getElementById(\'{table_id}\');'
                "if(t.hasAttribute('hidden')){t.removeAttribute('hidden');"
                "this.setAttribute('aria-expanded','true');"
                "this.textContent='Collapse to Top 20';"
                "this.classList.add('is-open');"
                "}else{t.setAttribute('hidden','');"
                "this.setAttribute('aria-expanded','false');"
                "this.textContent='Expand to show Top 40';"
                "this.classList.remove('is-open');"
                '}">Expand to show Top 40</button>',
                "</div>",
            ]
        )
    else:
        parts.extend(["</table>", "</div>"])
    return "".join(parts)


def _render_stale_bugs_subcard_html(
    rows: list[tuple[float, str, float, int, str, str, list[int]]],
    total_open_bugs: int,
) -> str:
    """Render the **Stale Bugs (open > 2 months)** sub-card.

    Reuses the Top 20 table styling (via the shared ``.top-di-table`` class)
    with two visual tweaks that signal "stale / needs attention":

    * The table carries an extra ``stale-bugs-table`` class so the header
      gets a warning-tinted background (see ``_DI_SECTION_CSS``).
    * Every row's Days cell carries ``di-days--stale`` so the value
      renders in the alert-red palette — mirrors the
      ``legacy-pr-card-days--stale`` convention from
      ``vllm-omni-pr-monitor``.

    Empty-state contract:

    * ``total_open_bugs == 0`` → return ``""`` (the parent panel already
      shows the "no issues fetched" hint; an empty sub-card would be noise).
    * ``total_open_bugs > 0`` and ``rows`` empty → render the sub-card
      with a positive "all bugs < 2 months" indicator (the metric was
      computed, the result happens to be zero — confirm explicitly).
    * ``rows`` non-empty → render the full table.
    """
    if total_open_bugs == 0:
        return ""

    title = "Stale Bugs (open > 2 months)"
    sub_text = (
        f"Open bug-labeled issues older than {int(STALE_BUG_DAYS)} days, "
        "sorted by age (oldest first)."
    )

    if not rows:
        body = (
            f'<p class="focus-table-sub">{html.escape(sub_text)}</p>'
            '<p class="hint hint-positive">'
            "✓ No stale bugs — every open <code>label:bug</code> issue is "
            f"under {int(STALE_BUG_DAYS)} days old.</p>"
        )
        return _details_subcard(
            title,
            body,
            open_default=False,
            details_class="report-subcard--stale-bugs",
        )

    def _row_html(
        di: float,
        priority: str,
        days_open: float,
        issue_number: int,
        title: str,
        assignee: str,
        linked: list[int],
    ) -> str:
        title_disp = html.escape(_shorten_title(title))
        issue_url = f"{_REPO_URL}/issues/{issue_number}"
        days_disp = f"{int(days_open)}"
        di_disp = html.escape(_format_di_value(di))
        priority_disp = html.escape(priority)
        return (
            "<tr>"
            f'<td><a href="{issue_url}" target="_blank" rel="noopener">#{issue_number}</a></td>'
            f"<td>{title_disp}</td>"
            f"<td>{priority_disp}</td>"
            f'<td class="di-days di-days--stale">{days_disp}</td>'
            f'<td class="di-value">{di_disp}</td>'
            f"{_di_assignee_cell_html(issue_number, assignee)}"
            f"{_di_maintainer_cell_html(issue_number)}"
            f"{_di_bugfix_cell_html(linked)}"
            "</tr>"
        )

    tbody = "".join(_row_html(*row) for row in rows)
    table_html = (
        '<div class="focus-top-table" data-stale-bugs-table>'
        '<table class="summary top-di-table stale-bugs-table">'
        "<thead><tr>"
        "<th>#</th><th>Title</th><th>Priority</th><th>Days</th>"
        "<th>DI</th><th>Assignee</th><th>Maintainer</th><th>Bugfix</th>"
        "</tr></thead>"
        f"<tbody>{tbody}</tbody>"
        "</table>"
        "</div>"
    )
    body = (
        f'<p class="focus-table-sub">{html.escape(sub_text)}</p>'
        + table_html
    )
    return _details_subcard(
        title,
        body,
        open_default=True,
        details_class="report-subcard--stale-bugs",
    )


def _inline_edit_script() -> str:
    """Editable Assignee / Maintainer cells with localStorage persistence."""
    return """
<script>
(function () {
  "use strict";
  var mem = {};
  function lsGet(k) {
    try { return localStorage.getItem(k); }
    catch (e) { return Object.prototype.hasOwnProperty.call(mem, k) ? mem[k] : null; }
  }
  function lsSet(k, v) {
    try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); }
    catch (e) { if (v) mem[k] = v; else delete mem[k]; }
  }
  function aKey(k) { return "di-assignee:" + k; }
  function mKey(k) { return "di-maintainer:" + k; }
  function hydrate(input, kind, key) {
    var attr = input.getAttribute("data-" + kind + "-value");
    if (attr === null || attr === undefined) {
      var ls = lsGet(key);
      if (ls !== null && ls !== undefined && ls !== "") {
        attr = ls;
        input.value = ls;
      }
    } else if (attr) {
      input.value = attr;
    }
    if (input.value) input.setAttribute("data-" + kind + "-persisted", "1");
    function persist() {
      input.setAttribute("data-" + kind + "-value", input.value || "");
      lsSet(key, input.value || "");
    }
    input.addEventListener("input", persist);
    input.addEventListener("blur", persist);
  }
  function initAll() {
    var aInputs = document.querySelectorAll("input.di-assignee-input");
    for (var i = 0; i < aInputs.length; i++) {
      var k = aInputs[i].getAttribute("data-da-key") || "";
      hydrate(aInputs[i], "da", aKey(k));
    }
    var mInputs = document.querySelectorAll("input.di-maintainer-input");
    for (var j = 0; j < mInputs.length; j++) {
      var k2 = mInputs[j].getAttribute("data-dm-key") || "";
      hydrate(mInputs[j], "dm", mKey(k2));
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
</script>
"""


# ---------------------------------------------------------------------------
# Section-specific CSS — copied from nightly_local_log_report.py so the
# standalone HTML matches the parent nightly's Daily focus styling.
# ---------------------------------------------------------------------------

_DI_SECTION_CSS = """
/* Top 20 DI contributors table (under the Outstanding DI card) */
.focus-top-table { overflow-x: auto; margin: 0.5rem 0 1rem; }
.focus-top-table table.top-di-table {
  width: 100%;
  font-size: 0.9rem;
  border-collapse: collapse;
}
.focus-top-table table.top-di-table th,
.focus-top-table table.top-di-table td {
  padding: 0.35rem 0.6rem;
  border-bottom: 1px solid var(--omni-divider, #e5e7eb);
  text-align: left;
  vertical-align: top;
}
.focus-top-table table.top-di-table th {
  background: var(--dashboard-card-tint, #f6f8fa);
  font-weight: 600;
  color: var(--dashboard-soft-text, #475569);
}
.focus-top-table table.top-di-table td a {
  color: var(--dashboard-link, #1d4ed8);
  text-decoration: none;
}
.focus-top-table table.top-di-table td.di-value {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--omni-critical-fg, #b91c1c);
  font-weight: 600;
}
.focus-top-table table.top-di-table td.di-days {
  text-align: right;
  font-variant-numeric: tabular-nums;
  color: var(--dashboard-soft-text, #64748b);
}
.focus-top-table table.top-di-table td.di-bugfix-cell {
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}
.focus-top-table table.top-di-table td.di-bugfix-cell a {
  color: var(--dashboard-link, #1d4ed8);
  text-decoration: none;
}
.focus-top-table table.top-di-table td.di-bugfix-cell a:hover {
  text-decoration: underline;
}
.focus-top-table table.top-di-table td.di-bugfix-cell--none {
  color: var(--dashboard-soft-text, #94a3b8);
  font-style: italic;
}
/* Expand / collapse toggle for rows 21-40 */
.focus-top-table + .di-top-table-toggle-row {
  margin: -0.5rem 0 1rem;
  text-align: right;
}
.di-top-table-toggle {
  appearance: none;
  background: var(--dashboard-card-tint, #f6f8fa);
  color: var(--dashboard-link, #1d4ed8);
  border: 1px dashed var(--dashboard-link, #1d4ed8);
  border-radius: 6px;
  padding: 0.35rem 0.85rem;
  font-size: 0.85rem;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 120ms ease, color 120ms ease;
}
.di-top-table-toggle:hover {
  background: var(--dashboard-link, #1d4ed8);
  color: #fff;
}
.di-top-table-toggle.is-open {
  background: var(--dashboard-link, #1d4ed8);
  color: #fff;
}
/* Focus card (Outstanding DI) */
.focus-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem;
  margin: 0.75rem 0 1.25rem;
}
.focus-card {
  border: 1px solid var(--dashboard-border, #d9e2ec);
  border-radius: var(--radius-sm, 8px);
  background: var(--dashboard-panel-bg, #ffffff);
  padding: 0.85rem 1rem;
  box-shadow: var(--dashboard-shadow, 0 2px 6px rgba(15, 23, 42, 0.06));
}
.focus-card--fail {
  border-color: var(--dashboard-alert, #d14343);
  background: var(--dashboard-alert-bg, rgba(209, 67, 67, 0.06));
}
.focus-card-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--dashboard-soft-text, #475569);
  margin-bottom: 0.35rem;
  font-weight: 700;
}
.focus-card-value {
  font-size: 1.7rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--dashboard-text, #26323f);
  line-height: 1.2;
}
.focus-card--fail .focus-card-value {
  color: var(--dashboard-alert, #d14343);
}
.focus-card-detail {
  font-size: 0.85rem;
  color: var(--dashboard-soft-text, #475569);
  margin-top: 0.4rem;
  line-height: 1.4;
}
/* Sub-card (collapsible Top DI table) */
details.report-subcard {
  border: 1px solid var(--dashboard-border, #d9e2ec);
  border-radius: var(--radius-sm, 8px);
  background: var(--dashboard-panel-bg, #ffffff);
  margin: 0.75rem 0;
}
details.report-subcard > summary.report-subcard-summary {
  list-style: none;
  cursor: pointer;
  padding: 0.7rem 1rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.55rem;
}
/* Hide the native disclosure marker in Chromium/Safari AND Firefox /
  // modern standard, so only the explicit .report-subcard-arrow span is
  visible (otherwise the default triangle overlaps the SVG icon). */
details.report-subcard > summary.report-subcard-summary::-webkit-details-marker {
  display: none;
}
details.report-subcard > summary.report-subcard-summary::marker {
  content: "";
}
details.report-subcard > summary.report-subcard-summary > .report-subcard-arrow {
  flex: 0 0 1rem;
  width: 1rem;
  height: 1em;
  position: relative;
  color: var(--dashboard-soft-text, #475569);
  transition: transform 120ms ease;
}
details.report-subcard > summary.report-subcard-summary > .report-subcard-arrow::before {
  content: "▸";
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85em;
  line-height: 1;
}
details.report-subcard[open] > summary.report-subcard-summary > .report-subcard-arrow {
  transform: rotate(90deg);
}
details.report-subcard--di-top > .report-subcard-body,
details.report-subcard--legacy-pr > .report-subcard-body {
  padding: 0.85rem 1rem 1.05rem;
}
.focus-table-sub {
  font-size: 0.85rem;
  color: var(--dashboard-soft-text, #475569);
  margin: 0 0 0.6rem;
  line-height: 1.45;
}
/* DI table polish (zebra, hover, tighter numerics) */
.focus-top-table table.top-di-table th {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.72rem;
  padding: 0.55rem 0.6rem;
  border-bottom: 2px solid var(--omni-divider, #e5e7eb);
}
.focus-top-table table.top-di-table tbody tr:nth-child(odd) td {
  background: color-mix(in srgb, var(--dashboard-card-tint, #f6f8fa) 60%, var(--dashboard-panel-bg));
}
.focus-top-table table.top-di-table tbody tr:hover td {
  background: color-mix(
    in srgb,
    var(--accent-soft, rgba(59, 130, 246, 0.18)) 50%,
    var(--dashboard-panel-bg)
  );
}
.focus-top-table table.top-di-table td {
  padding: 0.5rem 0.6rem;
  border-bottom: 1px solid color-mix(in srgb, var(--omni-divider, #e5e7eb) 70%, transparent);
}
.focus-top-table table.top-di-table td:first-child {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: var(--dashboard-text);
}
/* Stale Bugs sub-card — warning-tint header + red Days accent so the
   section reads as "old / needs attention" next to the clean Top 20
   table. Mirrors the legacy-pr-card-days--stale convention in
   vllm-omni-pr-monitor. */
.focus-top-table table.stale-bugs-table thead th {
  background: color-mix(in srgb, var(--dashboard-warning-bg, #fef3c7) 60%, var(--dashboard-card-tint, #f6f8fa));
}
.focus-top-table table.top-di-table td.di-days--stale {
  color: var(--dashboard-alert, #d14343);
  font-weight: 700;
}
.hint-positive {
  color: var(--dashboard-healthy, #15803d);
  font-weight: 600;
}
/* Inline editable cells */
.di-assignee-cell.is-empty,
.di-maintainer-cell.is-empty {
  padding: 0 !important;
}
.di-assignee-input,
.di-maintainer-input {
  width: 100%;
  border: 1px dashed var(--dashboard-border, #d9e2ec);
  background: transparent;
  padding: 0.35rem 0.45rem;
  border-radius: 4px;
  font-family: inherit;
  font-size: 0.85rem;
  color: var(--dashboard-text);
}
.di-assignee-input:focus,
.di-maintainer-input:focus {
  outline: none;
  border-style: solid;
  border-color: var(--dashboard-link, #1d4ed8);
  background: var(--dashboard-panel-bg, #ffffff);
}
.di-assignee-text {
  font-size: 0.85rem;
}
.panel {
  background: var(--dashboard-panel-bg, #ffffff);
  border: 1px solid var(--dashboard-border, #d9e2ec);
  border-radius: var(--radius, 12px);
  padding: 1.25rem 1.4rem 1.5rem;
  margin: 1rem 0;
  box-shadow: var(--dashboard-shadow);
}
.data-source {
  margin-top: 1.5rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--dashboard-border, #d9e2ec);
  font-size: 0.8rem;
  color: var(--dashboard-soft-text, #475569);
}
"""


# ---------------------------------------------------------------------------
# Document assembly.
# ---------------------------------------------------------------------------


def _html_document(title: str, css: str, body: str, *, tail: str = "") -> str:
    t = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
{css}
</style>
</head>
<body>
{body}
{tail}
</body>
</html>
"""


def render_html(
    *,
    title: str,
    di: dict[str, Any],
    gh_token_state: str,  # "token", "no-token", "preview"
    now: datetime,
) -> str:
    css = EDITORIAL_THEME_CSS + _DI_SECTION_CSS
    body_parts: list[str] = [
        '<div class="top-bar"><div class="shell top-bar-inner">'
        '<div class="brand">'
        f'<div class="brand-mark">{_svg_icon(_SVG_CLIPBOARD, size=30, extra_class="brand-ico")}</div>'
        '<div class="brand-copy">'
        f"<h1>{html.escape(title)}</h1>"
        '<p class="tagline">Standalone Outstanding DI report</p>'
        "</div></div></div></div>",
        '<div class="shell">',
        '<section class="panel nightly-focus">',
        _heading_html(
            "h2",
            _SVG_SPARK,
            html.escape("Outstanding DI"),
            sub=html.escape("Top open bug issues ranked by SLO-escalating Defect Index"),
        ),
    ]

    # Metric cards — Outstanding DI + N open bugs
    cards = [
        _render_focus_metric_card("Outstanding DI", di["value"], di["detail"], di["severity"]),
        _render_focus_metric_card(
            "Open bugs",
            str(di["n_issues"]),
            f"total labeled bug(s) on {_REPO}",
            "ok" if di["n_issues"] > 0 else "ok",
        ),
    ]
    body_parts.append('<div class="focus-card-grid">')
    body_parts.append("\n".join(cards))
    body_parts.append("</div>")

    # Sub-card with the Top DI contributors table
    di_table_html = _render_top_di_table_html(di.get("per_issue") or [])
    if di_table_html:
        di_body = (
            '<p class="focus-table-sub">Top open <code>bug</code> issues '
            "ranked by SLO-escalating Defect Index. Use this to triage "
            "which issues are accumulating the most backlog weight.</p>"
            + di_table_html
        )
        body_parts.append(
            _details_subcard(
                "Top 20 DI contributors (Outstanding DI)",
                di_body,
                open_default=True,
                details_class="report-subcard--di-top",
            )
        )
    else:
        body_parts.append(
            '<p class="hint">No open <code>label:bug</code> issues fetched. '
            "Confirm <code>GITHUB_TOKEN</code> / <code>GH_TOKEN</code> is set "
            "in the environment and that the token has <code>public_repo</code> "
            "scope for <code>vllm-project/vllm-omni</code>.</p>"
        )

    # Sub-card with the Stale Bugs table (open > STALE_BUG_DAYS).
    # The renderer returns "" when there are no open bugs at all — the
    # empty-state hint above already covers that case.
    stale_html = _render_stale_bugs_subcard_html(
        di.get("stale_bugs") or [],
        total_open_bugs=di.get("n_issues") or 0,
    )
    if stale_html:
        body_parts.append(stale_html)

    body_parts.append("</section>")
    body_parts.append("</div>")

    # Data source footer
    token_line = {
        "token": "authenticated via <code>$GITHUB_TOKEN</code> / <code>$GH_TOKEN</code>",
        "no-token": "unauthenticated REST (degraded — rate-limited; set <code>$GITHUB_TOKEN</code> for stable results)",
        "preview": "preview mode — empty dataset, no GitHub calls",
        "mock": "mock-data mode — synthetic 23-issue dataset (no GitHub calls)",
    }[gh_token_state]
    body_parts.append(
        f'<div class="shell"><div class="data-source">'
        f"Data source: <code>{_REPO_URL}/issues</code> "
        f"({token_line}). Generated <code>{html.escape(now.isoformat(timespec='seconds'))}</code> UTC."
        f"</div></div>"
    )

    return _html_document(title, css, "\n".join(body_parts), tail=_inline_edit_script())


# ---------------------------------------------------------------------------
# Kanban archive workflow (--archive / 归档报告).
# ---------------------------------------------------------------------------


def _gh_cli_path() -> str | None:
    return shutil.which("gh")


def _ensure_gh_cli() -> None:
    if _gh_cli_path() is None:
        raise RuntimeError(GH_CLI_INSTALL_HINT)


def _ensure_gh_authenticated() -> None:
    _ensure_gh_cli()
    proc = subprocess.run(
        ["gh", "auth", "status"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        msg = GH_AUTH_HINT
        if detail:
            msg = f"{GH_AUTH_HINT}\n{detail}"
        raise RuntimeError(msg)


def _run_git(
    repo: Path,
    *args: str,
    check: bool = True,
    gh_credential: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd: list[str] = ["git"]
    if gh_credential:
        cmd.extend(["-c", f"credential.helper={GH_GIT_CREDENTIAL_HELPER}"])
    cmd.extend(args)
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed: {detail}")
    return proc


def _git_current_branch(repo: Path) -> str:
    proc = _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    branch = (proc.stdout or "").strip()
    if not branch or branch == "HEAD":
        raise RuntimeError(f"Unable to determine current branch in {repo}")
    return branch


def _git_commit_identity_args(repo: Path) -> list[str]:
    """``git -c`` overrides only for identity fields not configured in the repo."""
    name = _run_git(repo, "config", "--get", "user.name", check=False).stdout.strip()
    email = _run_git(repo, "config", "--get", "user.email", check=False).stdout.strip()
    args: list[str] = []
    if not name:
        args.extend(["-c", "user.name=vllm-omni-issue-monitor"])
    if not email:
        args.extend(["-c", "user.email=vllm-omni-issue-monitor@users.noreply.github.com"])
    return args


def _kanban_archive_dest(kanban_repo: Path) -> Path:
    return kanban_repo / KANBAN_ARCHIVE_DIR / KANBAN_ARCHIVE_FILENAME


def _archive_to_kanban(
    report_path: Path,
    kanban_repo_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One-shot copy → ``git pull --rebase`` → ``git add`` → commit → push.

    ``report_path`` is the HTML file produced by this skill; it is copied
    to ``<kanban_repo>/data/issue_monitor/issue-monitor-report.html``
    (overwrite) and pushed. Returns a dict with the archived path, commit
    hash, branch, and push note for the caller to print.

    Raises ``RuntimeError`` (or ``FileNotFoundError`` /
    ``NotADirectoryError``) on any failure; the caller surfaces the exact
    message and does **not** claim archival completed.
    """
    report_path = report_path.resolve()
    kanban_repo = kanban_repo_root.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")
    if not kanban_repo.is_dir():
        raise NotADirectoryError(f"Kanban repo root not found: {kanban_repo}")
    if not (kanban_repo / ".git").exists():
        raise RuntimeError(f"Not a git repository: {kanban_repo}")

    dest = _kanban_archive_dest(kanban_repo)
    rel_dest = dest.relative_to(kanban_repo).as_posix()
    branch = _git_current_branch(kanban_repo)
    now = datetime.now(timezone.utc)
    commit_message = f"data: archive issue monitor report (UTC {now.isoformat(timespec='seconds')})"

    plan_lines = [
        "",
        "=" * 60,
        "Issue-monitor kanban archive",
        "=" * 60,
        f"Source report : {report_path}",
        f"Kanban repo   : {kanban_repo}",
        f"Remote        : origin -> {KANBAN_REPO_URL}",
        f"Branch        : {branch}",
        f"Destination   : {rel_dest}  (no date; overwrites same-name)",
        f"Commit msg    : {commit_message}",
        "=" * 60,
    ]
    plan_text = "\n".join(plan_lines)

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "dest": str(dest),
            "rel_dest": rel_dest,
            "branch": branch,
            "commit_message": commit_message,
            "plan_text": plan_text,
            "commit_hash": "",
            "push_note": "[dry-run] no git changes were made",
        }

    # 1. Verify gh CLI is authenticated BEFORE touching git so the user sees
    #    a clean failure if credentials are missing.
    _ensure_gh_authenticated()

    # 2. Defensive: clear any leftover modification to the destination file
    #    from a previous failed attempt (a half-completed archive leaves the
    #    file in an unstaged state, which would block `git pull --rebase`).
    #    Only restore when the destination is the ONLY dirty file — any
    #    other modified/untracked file is treated as unrelated work the user
    #    must resolve themselves.
    status_proc = _run_git(kanban_repo, "status", "--porcelain", check=False)
    dirty = [
        line[3:].split(" -> ", 1)[-1].strip()
        for line in (status_proc.stdout or "").splitlines()
        if line.strip()
    ]
    if dirty and dirty != [rel_dest]:
        raise RuntimeError(
            f"Kanban working tree has unrelated changes: {dirty}. "
            "Commit, stash, or discard them before re-running the archive."
        )
    if dirty == [rel_dest]:
        _run_git(kanban_repo, "checkout", "--", rel_dest)

    # 3. git pull --rebase (use gh credential helper) — on a clean tree now.
    pull = _run_git(
        kanban_repo,
        "pull",
        "--rebase",
        "origin",
        branch,
        check=False,
        gh_credential=True,
    )
    if pull.returncode != 0:
        detail = (pull.stderr or pull.stdout or "").strip()
        raise RuntimeError(f"git pull --rebase origin {branch} failed: {detail}")

    # 4. Copy the report (overwrite).
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, dest)

    # 5. Stage only the intended file (defensive — anything else is unexpected).
    _run_git(kanban_repo, "add", "--", rel_dest)
    staged_proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--name-only",
        check=False,
    )
    staged = sorted({
        raw.strip().replace("\\", "/")
        for raw in (staged_proc.stdout or "").splitlines()
        if raw.strip()
    })
    expected = {rel_dest}
    extra = [s for s in staged if s not in expected]
    missing = [s for s in expected if s not in staged]
    if extra or missing:
        # Roll back the accidental stage so the user sees a clean tree.
        _run_git(kanban_repo, "reset", "HEAD", "--", *staged, check=False)
        raise RuntimeError(
            f"Refusing to commit — unexpected staged changes. extra={extra} missing={missing}. "
            "Run `git status` in the kanban repo and retry after cleaning unrelated files."
        )

    # 5. Commit.
    identity_args = _git_commit_identity_args(kanban_repo)
    commit_args = identity_args + ["commit", "-m", commit_message, "--", rel_dest]
    _run_git(kanban_repo, *commit_args)
    commit_hash = _run_git(
        kanban_repo,
        "rev-parse",
        "--short",
        "HEAD",
        check=False,
    ).stdout.strip()

    # 6. Push (gh credential helper).
    push = _run_git(
        kanban_repo,
        "push",
        "origin",
        branch,
        check=False,
        gh_credential=True,
    )
    if push.returncode != 0:
        detail = (push.stderr or push.stdout or "").strip()
        raise RuntimeError(
            f"git push origin {branch} failed (commit {commit_hash} is local but not pushed): {detail}"
        )

    push_note = (
        f"Pushed issue-monitor report to origin/{branch} ({KANBAN_REPO_URL}) "
        f"as commit {commit_hash}."
    )
    return {
        "ok": True,
        "dry_run": False,
        "dest": str(dest),
        "rel_dest": rel_dest,
        "branch": branch,
        "commit_hash": commit_hash,
        "commit_message": commit_message,
        "plan_text": plan_text,
        "push_note": push_note,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path (default: dated file in skill dir)")
    parser.add_argument("--date", default=None, help="UTC date YYYY-MM-DD for filename/title (default: today UTC)")
    parser.add_argument("--gh-token", default=None, help="Explicit GitHub token (else read $GITHUB_TOKEN / $GH_TOKEN)")
    parser.add_argument("--no-token", action="store_true", help="Skip auth entirely (degraded REST)")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write preview HTML with empty dataset (no GitHub calls). Filename ends in -preview-YYYY-MM-DD.html.",
    )
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Render with a synthetic 23-issue dataset (no GitHub calls). Exercises every rendering path; filename ends in -mock-YYYY-MM-DD.html.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help=(
            "After writing the report, copy it to <kanban>/data/issue_monitor/issue-monitor-report.html "
            "(no date, overwrite) and run git pull --rebase / commit / push in the kanban repo. "
            "Activated by the explicit '归档报告' intent."
        ),
    )
    parser.add_argument(
        "--archive-dry-run",
        action="store_true",
        help=(
            "Print the planned archive + commit + push without touching git or copying the file. "
            "Use to preview the destination path, branch, and commit message."
        ),
    )
    parser.add_argument(
        "--kanban-repo-root",
        type=Path,
        default=None,
        help=(
            f"Local checkout of {KANBAN_REPO_URL}. Default: $KANBAN_REPO_ROOT or "
            f"{KANBAN_DEFAULT_REPO_ROOT}. Used only with --archive / --archive-dry-run."
        ),
    )
    args = parser.parse_args(argv)

    skill_dir = Path(__file__).resolve().parent.parent
    date_iso = resolve_report_date_iso(args.date)

    if args.mock_data:
        di = _compute_outstanding_di(gh_token=None, mock=True)
        token_state = "mock"
        out_path = args.out or skill_dir / f"issue-monitor-report-mock-{date_iso}.html"
        title = "vLLM-Omni Issue Monitor (mock data)"
    elif args.preview:
        di = {
            "total_di": 0.0,
            "n_issues": 0,
            "counts": {label: 0 for label in BUG_DI_TABLE} | {"unclassified": 0},
            "per_issue": [],
            "value": "0",
            "detail": "preview mode — empty dataset",
            "severity": "ok",
        }
        token_state = "preview"
        out_path = args.out or skill_dir / issue_monitor_report_preview_basename(date_iso)
        title = "vLLM-Omni Issue Monitor (preview)"
    else:
        gh_token: str | None = None
        if not args.no_token:
            gh_token = _resolve_github_token(args.gh_token)
        di = _compute_outstanding_di(gh_token)
        token_state = "token" if gh_token else "no-token"
        out_path = args.out or default_issue_monitor_html_path(skill_dir, date_iso)
        title = issue_monitor_report_title(date_iso)

    now = datetime.now(timezone.utc)
    document = render_html(title=title, di=di, gh_token_state=token_state, now=now)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")
    print(f"Wrote {out_path}")

    # Archive to vllm-omni-kanban (opt-in: --archive / 归档报告).
    if args.archive_dry_run and args.archive:
        print(
            "--archive and --archive-dry-run are mutually exclusive; "
            "use one or the other.",
            file=sys.stderr,
        )
        return 2
    if args.archive or args.archive_dry_run:
        if args.mock_data or args.preview:
            print(
                f"Refusing to archive a {'mock' if args.mock_data else 'preview'} "
                "report — regenerate without --mock-data/--preview for a real snapshot.",
                file=sys.stderr,
            )
            return 2
        kanban_repo = _resolve_kanban_repo_root(
            str(args.kanban_repo_root) if args.kanban_repo_root else None
        )
        result = _archive_to_kanban(
            out_path,
            kanban_repo,
            dry_run=args.archive_dry_run,
        )
        print(result["plan_text"])
        print(result["push_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())