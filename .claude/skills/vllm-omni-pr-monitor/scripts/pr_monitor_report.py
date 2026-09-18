#!/usr/bin/env python3
"""
Generate a standalone HTML report for **Top 20 Stale PRs by Bot-Mentioned
Reviewers (PR Monitor)**.

This skill is the focused twin of the second sub-card inside
`vllm-omni-test-report`'s nightly Daily-focus section. It emits a single
self-contained HTML file (no Buildkite / local log / kanban dependency)
with the same per-reviewer bucketing and "Show all N" toggle as the parent
nightly.

The reviewer bucket key is the GitHub login that the
``vllm-omni-review-bot`` mentions in its triage comment (e.g.
``Module owners: @alice @bob``); high-priority PRs surface first within
each bucket.

    python scripts/pr_monitor_report.py                 # writes pr-monitor-report-YYYY-MM-DD.html
    python scripts/pr_monitor_report.py --preview    # empty dataset, offline smoke test
    python scripts/pr_monitor_report.py --date 2026-08-26  # explicit UTC date
    python scripts/pr_monitor_report.py --out ./my.html
    python scripts/pr_monitor_report.py --gh-token TOKEN
    python scripts/pr_monitor_report.py --no-token   # degraded
    python scripts/pr_monitor_report.py --archive          # copy (overwrite) + commit + push to vllm-omni-kanban
    python scripts/pr_monitor_report.py --archive-dry-run  # preview archive plan without touching git
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
from pathlib import Path
from typing import Any

_PARENT_SKILL_SCRIPTS = (
    Path(__file__).resolve().parent.parent.parent
    / "vllm-omni-test-report"
    / "scripts"
)
if str(_PARENT_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PARENT_SKILL_SCRIPTS))

from report_html_theme import EDITORIAL_THEME_CSS  # noqa: E402
from report_naming import (  # noqa: E402
    default_pr_monitor_html_path,
    pr_monitor_report_basename,
    pr_monitor_report_preview_basename,
    pr_monitor_report_title,
    resolve_report_date_iso,
)

# ---------------------------------------------------------------------------
# Constants — mirror of nightly_local_log_report.py lines ~1211–1240.
# ---------------------------------------------------------------------------

_LEGACY_PR_PER_REVIEWER_LIMIT = 20
_LEGACY_PR_DEFAULT_VISIBLE = 2
_LEGACY_PR_BOT_LOGIN = "vllm-omni-review-bot"

_LEGACY_PR_HIGH_PRIORITY_LABELS = frozenset(
    {"high priority", "critical", "p0", "p1"}
)
_LEGACY_PR_PRIORITY_DISPLAY_ORDER = (
    "critical",
    "p0",
    "p1",
    "high priority",
)

_REPO = "vllm-project/vllm-omni"
_REPO_URL = "https://github.com/vllm-project/vllm-omni"

# ---------------------------------------------------------------------------
# Kanban archive constants (used only when --archive / 归档报告 is invoked).
# ---------------------------------------------------------------------------

KANBAN_REPO_URL = "https://github.com/hsliuustc0106/vllm-omni-kanban"
KANBAN_DEFAULT_REPO_ROOT = Path("/home/wy/vllm-omni-kanban")
KANBAN_ARCHIVE_DIR = Path("data") / "pr_monitor"
KANBAN_ARCHIVE_FILENAME = "pr-monitor-report.html"  # no date — always overwrites
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

# Bot triage comment line: ``Module owners: @alice @bob`` (and Model /
# Issue / Component / File variants). The regex captures only the first
# owners line — the self-review reminder in the next paragraph is
# intentionally excluded so the PR author doesn't appear as their own
# reviewer.
_OWNERS_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:Module|Model|Issue|Component|File)\s+owners?\s*:\s*"
    r"(?P<owners>[^\n]+)",
    re.IGNORECASE | re.MULTILINE,
)

# GitHub login rules: 1–39 chars, alphanumeric or single hyphens. The
# leading/trailing negative lookarounds prevent matching email addresses
# (``foo@bar``) or org references (``@vllm-project/...``).
_GITHUB_MENTION_RE = re.compile(
    r"(?<![\w-])@([A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38})(?![\w/-])"
)
_MENTION_ALIAS_BLACKLIST = frozenset({"here", "everyone", "channel"})
_KNOWN_BOT_LOGINS = frozenset(
    {
        "dependabot",
        "dependabot-preview",
        "renovate",
        "renovate-bot",
        "github-actions",
        "codecov",
        "codecov-bot",
        "codecov-commenter",
        "sonarcloud",
        "snyk-bot",
        "mergify",
        "imgbot",
        "greenkeeper",
        "pull-bot",
    }
)


# ---------------------------------------------------------------------------
# GitHub HTTP helpers (mirrors nightly_local_log_report.py::_http_get_json_nightly).
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
            pass
    except ImportError:
        pass
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _resolve_github_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit.strip() or None
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return None


def _fetch_open_pulls(gh_token: str | None) -> list[dict[str, Any]]:
    """Paginate ``GET /repos/{owner}/{repo}/pulls?state=open``.

    Hits ``/pulls`` (not ``/issues``) so each entry carries
    ``requested_reviewers[]`` / ``requested_teams[]`` — not present on issues
    but useful for downstream grouping.
    """
    base = f"https://api.github.com/repos/{_REPO}/pulls"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-pr-monitor-report",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{base}?state=open&per_page=100&page={page}"
        try:
            batch = _http_get_json(url, headers=headers, timeout=60)
        except Exception:
            return out  # partial OK
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return out


def _search_open_prs_with_bot_comments(gh_token: str | None, bot_login: str) -> set[int]:
    """``GET /search/issues?q=commenter:<bot>+is:open+is:pr+...`` — bounds
    the per-PR comment fetch to PRs the bot actually commented on.
    """
    out: set[int] = set()
    if not gh_token:
        return out
    base = "https://api.github.com/search/issues"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-pr-monitor-report",
        "Authorization": f"Bearer {gh_token}",
    }
    q = f"commenter:{bot_login} is:open is:pr repo:{_REPO}"
    for page in range(1, 4):  # hard cap at 300 results
        url = f"{base}?q={urllib.parse.quote(q, safe=': -')}&per_page=100&page={page}"
        try:
            j = _http_get_json(url, headers=headers, timeout=60)
        except Exception:
            return out
        if not isinstance(j, dict):
            return out
        items = j.get("items") or []
        if not items:
            break
        for it in items:
            try:
                n = int(it.get("number"))
            except (TypeError, ValueError):
                continue
            out.add(n)
        if len(items) < 100:
            break
    return out


def _fetch_pr_issue_comments(gh_token: str | None, pr_number: int) -> list[dict[str, Any]]:
    """Per-PR ``GET /issues/{n}/comments`` — paginated, capped at 100/page.

    Only top-level (issue) comments are fetched; inline review comments
    (``/pulls/{n}/comments``) rarely contain reviewer ``@`` tags.
    """
    if not gh_token or not pr_number:
        return []
    base = f"https://api.github.com/repos/{_REPO}/issues/{int(pr_number)}/comments"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-pr-monitor-report",
        "Authorization": f"Bearer {gh_token}",
    }
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        url = f"{base}?per_page=100&page={page}"
        try:
            batch = _http_get_json(url, headers=headers, timeout=60)
        except Exception:
            return out  # partial OK
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return out


# ---------------------------------------------------------------------------
# Mention + bot triage parsing.
# ---------------------------------------------------------------------------


def _github_extract_mentions(body: str) -> list[str]:
    if not body:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _GITHUB_MENTION_RE.finditer(body):
        login = m.group(1)
        if not login or login.endswith("[bot]"):
            continue
        if login.lower() in _MENTION_ALIAS_BLACKLIST:
            continue
        if login.lower() in _KNOWN_BOT_LOGINS:
            continue
        if login in seen:
            continue
        seen.add(login)
        out.append(login)
    return out


def _extract_bot_owners(body: str, *, exclude_logins: tuple[str, ...] = ()) -> list[str]:
    if not body:
        return []
    m = _OWNERS_LINE_RE.search(body)
    if not m:
        return []
    mentions = _github_extract_mentions(m.group("owners"))
    if exclude_logins:
        excl = {login.lower() for login in exclude_logins}
        mentions = [login for login in mentions if login.lower() not in excl]
    return mentions


# ---------------------------------------------------------------------------
# PR grouping (mirror of nightly_local_log_report.py::_compute_top_legacy_prs).
# ---------------------------------------------------------------------------


def _format_utc_date(iso: str) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "-"
    return dt.strftime("%Y-%m-%d")


def _shorten_title(title: str, max_len: int = 60) -> str:
    if len(title) <= max_len:
        return title
    return title[: max_len - 1].rstrip() + "…"


def _legacy_pr_priority_label(pr: dict[str, Any]) -> str | None:
    labels = pr.get("labels") or []
    if not isinstance(labels, list):
        return None
    label_names_lower = [str(label.get("name") or "").lower() for label in labels if isinstance(label, dict)]
    for preferred in _LEGACY_PR_PRIORITY_DISPLAY_ORDER:
        if preferred in label_names_lower:
            for label in labels:
                if isinstance(label, dict) and str(label.get("name") or "").lower() == preferred:
                    return str(label.get("name") or preferred)
            return preferred
    return None


def _compute_top_legacy_prs(
    gh_token: str | None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Group open PRs by user mentioned in ``vllm-omni-review-bot`` comments."""
    if not gh_token:
        return []
    prs = _fetch_open_pulls(gh_token)
    if not prs:
        return []
    if now is None:
        now = datetime.now(timezone.utc)

    def _days_open(iso: str) -> float:
        if not iso:
            return 0.0
        try:
            created_dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, (now - created_dt).total_seconds() / 86400.0)

    bot_pr_numbers = _search_open_prs_with_bot_comments(gh_token, _LEGACY_PR_BOT_LOGIN)
    if not bot_pr_numbers:
        return []

    pr_comments: dict[int, list[dict[str, Any]]] = {}
    targets = sorted(bot_pr_numbers)
    max_workers = min(8, max(1, len(targets)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_pr_issue_comments, gh_token, n): n for n in targets}
        for fut in futures:
            n = futures[fut]
            try:
                pr_comments[n] = fut.result() or []
            except Exception:
                pr_comments[n] = []

    bot_login = _LEGACY_PR_BOT_LOGIN
    pr_to_mentions: dict[int, list[str]] = {}
    for n, comments in pr_comments.items():
        seen: set[str] = set()
        out: list[str] = []
        for c in comments:
            user = c.get("user") or {}
            if (user.get("login") or "").strip() != bot_login:
                continue
            body = c.get("body") or ""
            for login in _extract_bot_owners(body, exclude_logins=(bot_login,)):
                if login in seen:
                    continue
                seen.add(login)
                out.append(login)
        if out:
            pr_to_mentions[n] = out

    groups: dict[str, list[dict[str, Any]]] = {}
    for pr in prs:
        if pr.get("draft"):
            continue
        try:
            n = int(pr.get("number") or 0)
        except (TypeError, ValueError):
            continue
        mentioned = pr_to_mentions.get(n)
        if not mentioned:
            continue
        created = (pr.get("created_at") or "").strip()
        author_obj = pr.get("user") or {}
        item = {
            "number": pr.get("number"),
            "html_url": pr.get("html_url") or "",
            "title": (pr.get("title") or "").strip(),
            "author": (author_obj.get("login") or "").strip(),
            "author_avatar": (author_obj.get("avatar_url") or "").strip(),
            "created_at": created,
            "days_open": _days_open(created),
            "draft": bool(pr.get("draft")),
            "priority": _legacy_pr_priority_label(pr),
        }
        for reviewer in mentioned:
            groups.setdefault(reviewer, []).append(item)

    out: list[dict[str, Any]] = []
    for reviewer, items in groups.items():
        items_sorted = sorted(
            items,
            key=lambda p: (
                0 if p.get("priority") else 1,
                p["created_at"] or "",
            ),
        )
        max_days = max((p["days_open"] for p in items_sorted), default=0.0)
        out.append(
            {
                "reviewer": reviewer,
                "max_days_open": max_days,
                "pr_count": len(items_sorted),
                "shown_pr_count": min(len(items_sorted), _LEGACY_PR_PER_REVIEWER_LIMIT),
                "prs": items_sorted[:_LEGACY_PR_PER_REVIEWER_LIMIT],
            }
        )
    out.sort(key=lambda g: (-g["max_days_open"], g["reviewer"].lower()))
    return out


# ---------------------------------------------------------------------------
# HTML rendering (mirror of nightly_local_log_report.py::_render_top_legacy_pr_section_html).
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
    summary_extras: str = "",
) -> str:
    """Render a collapsible sub-card with a dedicated arrow slot.

    The arrow is rendered as an explicit ``.report-subcard-arrow`` span
    (instead of relying solely on ``summary::before``) so the layout has
    three flex children of fixed widths:

        [▸ arrow]  [📋 icon]  [@reviewer]

    Combined with the CSS that hides both ``::-webkit-details-marker``
    and ``details > summary::marker``, this prevents the default
    disclosure triangle from leaking through (Firefox / Chromium) and
    overlapping with the custom arrow + SVG icon.

    ``summary_extras`` is an optional HTML chunk appended after the title
    inside the summary (e.g. an avatar + PR-count badge for reviewer rows).
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
        f"{summary_extras}"
        f"</span></summary>"
        f'<div class="report-subcard-body">{body_html}</div>'
        "</details>"
    )


def _github_avatar_url(login: str, *, size: int = 56) -> str:
    """Public GitHub avatar CDN URL for ``login``.

    ``https://github.com/<login>.png`` redirects to the user's actual
    avatar on every public profile, so no API call is required.
    """
    safe = (login or "").strip()
    return f"https://github.com/{urllib.parse.quote(safe, safe='')}.png?s={int(size)}"


def _render_focus_metric_card(title: str, value: str, detail: str, severity: str) -> str:
    return (
        f'<div class="focus-card focus-card--{html.escape(severity)}">'
        f'<div class="focus-card-title">{html.escape(title)}</div>'
        f'<div class="focus-card-value">{html.escape(value)}</div>'
        f'<div class="focus-card-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def _render_legacy_pr_card_html(pr: dict[str, Any], *, collapsed: bool) -> str:
    """Render one slim PR row (no author/avatar) for the reviewer bucket list."""
    classes = ["legacy-pr-card"]
    if pr.get("priority"):
        classes.append("legacy-pr-card--high-priority")
    if pr.get("draft"):
        classes.append("legacy-pr-card--draft")
    if collapsed:
        classes.append("legacy-pr-card--collapsed")
    short_title = _shorten_title(pr["title"], max_len=64)
    pr_number = int(pr.get("number") or 0)
    priority_label = pr.get("priority") or ""
    priority_html = (
        f'<span class="legacy-pr-card-priority">{html.escape(priority_label)}</span>'
        if priority_label
        else ""
    )
    draft_html = (
        f'<span class="legacy-pr-card-draft" title="Draft PR">DRAFT</span>'
        if pr.get("draft")
        else ""
    )
    days_open = pr["days_open"]
    days_class = "legacy-pr-card-days"
    if days_open >= 30:
        days_class += " legacy-pr-card-days--stale"
    head_html = (
        f'<div class="legacy-pr-card-head">'
        f'<span class="legacy-pr-card-number">#{pr_number}</span>'
        f'<span class="{days_class}">{days_open:.1f}d</span>'
        f"</div>"
    )
    title_html = (
        f'<div class="legacy-pr-card-title">{html.escape(short_title)}</div>'
    )
    if priority_html or draft_html:
        body_html = (
            f'<div class="legacy-pr-card-foot">'
            f"{priority_html}{draft_html}"
            f"</div>"
        )
    else:
        body_html = ""
    return (
        f'<a class="{" ".join(classes)}" '
        f'href="{html.escape(pr["html_url"])}" '
        f'target="_blank" rel="noopener">'
        f"{head_html}{title_html}{body_html}"
        f"</a>"
    )


def _render_legacy_pr_group_html(group: dict[str, Any]) -> str:
    shown_prs = group["prs"]
    total_rows = len(shown_prs)
    cards: list[str] = []
    for row_idx, pr in enumerate(shown_prs):
        cards.append(
            _render_legacy_pr_card_html(
                pr,
                collapsed=(row_idx >= _LEGACY_PR_DEFAULT_VISIBLE),
            )
        )
    collapsed_attr = ' data-collapsed="true"' if total_rows > _LEGACY_PR_DEFAULT_VISIBLE else ""
    grid_html = (
        f'<div class="legacy-pr-grid"{collapsed_attr}>'
        f"{''.join(cards)}"
        f"</div>"
    )
    expand_btn_html = ""
    if total_rows > _LEGACY_PR_DEFAULT_VISIBLE:
        expand_btn_html = (
            f'<button type="button" class="legacy-pr-expand-btn" '
            f'data-total-rows="{total_rows}">Show all {total_rows}</button>'
        )
    reviewer = html.escape(group["reviewer"])
    reviewer_avatar = _github_avatar_url(group["reviewer"], size=40)
    reviewer_summary_extras = (
        f'<img class="legacy-pr-reviewer-avatar" loading="lazy" '
        f'src="{html.escape(reviewer_avatar)}" alt="" '
        f'onerror="this.onerror=null;this.removeAttribute(&quot;src&quot;);">'
        f'<span class="legacy-pr-reviewer-count" '
        f'title="{total_rows} PR(s) bucketed">×{total_rows}</span>'
    )
    return _details_subcard(
        f"@{reviewer}",
        grid_html + expand_btn_html,
        open_default=True,
        details_class="legacy-pr-group",
        summary_extras=reviewer_summary_extras,
    )


def _render_top_legacy_pr_section_html(top_prs: list[dict[str, Any]]) -> str:
    if not top_prs:
        return ""
    parts = ['<div class="legacy-pr-bucket-grid">']
    parts.extend(_render_legacy_pr_group_html(g) for g in top_prs)
    parts.append("</div>")
    parts.append(_legacy_pr_expand_script())
    return "\n".join(parts)


def _legacy_pr_expand_script() -> str:
    return """
<script>
(function () {
  function setCollapsed(grid, btn, collapsed) {
    if (!grid || !btn) return;
    if (collapsed) {
      grid.setAttribute('data-collapsed', 'true');
      btn.textContent = 'Show all ' + btn.getAttribute('data-total-rows');
    } else {
      grid.setAttribute('data-collapsed', 'false');
      btn.textContent = 'Show fewer';
    }
  }
  document.querySelectorAll('.legacy-pr-expand-btn').forEach(function (btn) {
    var body = btn.closest('.report-subcard-body');
    if (!body) return;
    var grid = body.querySelector('.legacy-pr-grid');
    setCollapsed(grid, btn, true);
    btn.addEventListener('click', function () {
      var collapsed = grid.getAttribute('data-collapsed') === 'true';
      setCollapsed(grid, btn, !collapsed);
    });
  });
})();
</script>
"""


# ---------------------------------------------------------------------------
# Section-specific CSS — copied from nightly_local_log_report.py.
# ---------------------------------------------------------------------------

_LEGACY_PR_SECTION_CSS = """
/* Reviewer bucket avatar + PR-count badge in summary */
.legacy-pr-reviewer-avatar {
  width: 22px;
  height: 22px;
  border-radius: 50%;
  vertical-align: middle;
  margin-left: 0.4rem;
  border: 1px solid color-mix(in srgb, var(--dashboard-border, #d9e2ec) 70%, transparent);
  background: var(--dashboard-panel-strong, #f1f5f9);
  flex: 0 0 auto;
}
.legacy-pr-reviewer-count {
  display: inline-block;
  margin-left: 0.35rem;
  padding: 0.05rem 0.45rem;
  background: color-mix(in srgb, var(--dashboard-warning) 14%, transparent);
  color: color-mix(in srgb, var(--dashboard-warning) 80%, #1f2937);
  border-radius: 999px;
  font-size: 0.78em;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
/* Reviewer bucket — 3-column PR card grid (the @-reviewer row) */
.legacy-pr-group { margin: 0.5rem 0 1rem; }
/* Inside the bucket grid: stack PRs vertically (1 column) so each card spans the narrow tile */
.legacy-pr-bucket-grid .legacy-pr-grid {
  grid-template-columns: 1fr;
  gap: 0.55rem;
}
/* Reviewer bucket tiles — compact 2-3 per row grid; stays in grid even when opened */
.legacy-pr-bucket-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  row-gap: 0.9rem;
  column-gap: 0.9rem;
  margin: 0.8rem 0 0.6rem;
  padding: 0.15rem;
}
.legacy-pr-bucket-grid > details.legacy-pr-group {
  margin: 0;
  align-self: start;
  border: 1px solid color-mix(in srgb, var(--dashboard-soft-text, #475569) 35%, var(--dashboard-border, #d9e2ec));
  border-left: 3px solid var(--dashboard-soft-text, #475569);
}
@media (max-width: 480px) {
  .legacy-pr-bucket-grid { grid-template-columns: 1fr; }
}
/* Compact summary when the reviewer tile is collapsed inside the bucket grid */
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary {
  padding: 0.55rem 0.7rem;
  font-size: 0.88rem;
  gap: 0.45rem;
  border-radius: var(--radius-sm, 8px);
  background: linear-gradient(180deg,
    color-mix(in srgb, var(--dashboard-soft-text, #475569) 14%, var(--dashboard-panel-bg, #ffffff)) 0%,
    color-mix(in srgb, var(--dashboard-soft-text, #475569) 6%, var(--dashboard-panel-bg, #ffffff)) 100%);
}
/* Inside the bucket grid: each PR card is a horizontal pill — content laid out inline (#number + days + title + badge) */
.legacy-pr-bucket-grid .legacy-pr-card {
  flex-direction: row;
  align-items: center;
  gap: 0.55rem;
  min-height: 32px;
  min-width: 0;
  padding: 0.3rem 0.85rem;
  border-radius: 999px;
  background: color-mix(in srgb, var(--dashboard-panel-strong, #f1f5f9) 60%, var(--dashboard-panel-bg, #ffffff));
  border-color: color-mix(in srgb, var(--dashboard-soft-text, #475569) 25%, var(--dashboard-border, #d9e2ec));
}
.legacy-pr-bucket-grid .legacy-pr-card-head {
  margin-bottom: 0;
  font-size: 0.72rem;
  flex: 0 0 auto;
  gap: 0.35rem;
}
.legacy-pr-bucket-grid .legacy-pr-card-title {
  display: block;
  -webkit-line-clamp: initial;
  -webkit-box-orient: initial;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 0.82rem;
  line-height: 1.25;
  flex: 1 1 auto;
  min-width: 0;
}
.legacy-pr-bucket-grid .legacy-pr-card-foot {
  margin-top: 0;
  margin-left: 0;
  flex: 0 0 auto;
}
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary > .report-subcard-arrow {
  flex: 0 0 0.8rem;
  width: 0.8rem;
  font-size: 0.8em;
  color: var(--dashboard-soft-text, #475569);
}
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary > .report-subcard-summary-inner {
  gap: 0.35rem;
  flex-wrap: wrap;
  min-width: 0;
}
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary > .report-subcard-summary-inner > .report-subcard-ico {
  width: 14px;
  height: 14px;
}
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary .report-subcard-title {
  font-size: 0.92rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
  font-weight: 700;
  color: color-mix(in srgb, var(--dashboard-soft-text, #475569) 80%, var(--dashboard-text, #26323f));
}
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary .legacy-pr-reviewer-avatar {
  width: 18px;
  height: 18px;
  margin-left: 0.25rem;
}
.legacy-pr-bucket-grid details.legacy-pr-group > summary.report-subcard-summary .legacy-pr-reviewer-count {
  font-size: 0.72em;
  padding: 0.04rem 0.4rem;
  margin-left: 0.25rem;
  background: color-mix(in srgb, var(--dashboard-soft-text, #475569) 18%, transparent);
  color: color-mix(in srgb, var(--dashboard-soft-text, #475569) 90%, #1f2937);
}
/* When the bucket is opened, give the inner body breathing room */
.legacy-pr-bucket-grid details.legacy-pr-group[open] > .report-subcard-body {
  padding: 0.7rem 0.35rem 0.4rem;
}
.legacy-pr-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.6rem;
  margin: 0.1rem 0 0.1rem;
}
@media (max-width: 820px) {
  .legacy-pr-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 480px) {
  .legacy-pr-grid { grid-template-columns: 1fr; }
}
/* Default: first 3 cards visible; cards 4+ hidden until "Show all N" toggled */
.legacy-pr-grid[data-collapsed="true"] .legacy-pr-card--collapsed { display: none; }
.legacy-pr-card {
  display: flex;
  flex-direction: column;
  min-height: 132px;
  padding: 0.55rem 0.7rem 0.5rem;
  border: 1px solid color-mix(in srgb, var(--dashboard-border, #d9e2ec) 80%, transparent);
  border-radius: 8px;
  background: var(--dashboard-panel-bg, #ffffff);
  text-decoration: none;
  color: inherit;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
  transition: border-color 120ms ease, box-shadow 120ms ease, background-color 120ms ease;
}
.legacy-pr-card:hover {
  border-color: color-mix(in srgb, var(--dashboard-warning) 55%, var(--dashboard-border, #d9e2ec));
  background: color-mix(in srgb, var(--dashboard-warning-soft, rgba(217, 119, 6, 0.18)) 22%, var(--dashboard-panel-bg));
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
.legacy-pr-card:focus-visible {
  outline: 2px solid var(--dashboard-link, #1d4ed8);
  outline-offset: 2px;
}
.legacy-pr-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.78rem;
  margin-bottom: 0.3rem;
}
.legacy-pr-card-number {
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: var(--dashboard-text, #26323f);
}
.legacy-pr-card-days {
  font-variant-numeric: tabular-nums;
  color: var(--dashboard-soft-text, #475569);
  font-weight: 600;
}
.legacy-pr-card-days--stale { color: var(--dashboard-alert, #d14343); }
.legacy-pr-card-title {
  font-size: 0.92rem;
  line-height: 1.3;
  font-weight: 500;
  color: var(--dashboard-text, #26323f);
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  word-break: break-word;
  flex: 1 1 auto;
}
.legacy-pr-card-foot {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin-top: 0.45rem;
  font-size: 0.78rem;
  color: var(--dashboard-soft-text, #475569);
}
.legacy-pr-card-avatar {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--dashboard-panel-strong, #f1f5f9);
  border: 1px solid color-mix(in srgb, var(--dashboard-border, #d9e2ec) 70%, transparent);
  flex: 0 0 auto;
}
.legacy-pr-card-author {
  font-weight: 600;
  color: var(--dashboard-text, #26323f);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.legacy-pr-card-priority {
  margin-left: auto;
  padding: 0.05rem 0.45rem;
  background: #fee2e2;
  color: #b91c1c;
  border-radius: 4px;
  font-size: 0.78em;
  font-weight: 700;
  white-space: nowrap;
}
.legacy-pr-card-draft {
  margin-left: auto;
  padding: 0.05rem 0.45rem;
  background: color-mix(in srgb, var(--dashboard-soft-text, #475569) 18%, transparent);
  color: var(--dashboard-soft-text, #475569);
  border-radius: 4px;
  font-size: 0.72em;
  font-weight: 700;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.legacy-pr-card--draft { opacity: 0.85; }
.legacy-pr-card--draft .legacy-pr-card-title {
  text-decoration: line-through;
  text-decoration-color: color-mix(in srgb, var(--dashboard-soft-text, #475569) 55%, transparent);
}
.legacy-pr-card--high-priority {
  border-left: none;
}
.legacy-pr-count {
  display: inline-block;
  margin: 0.5rem 0 0;
  color: var(--dashboard-soft-text, #475569);
  font-size: 0.85em;
}
/* Show all N toggle */
.legacy-pr-expand-btn {
  display: inline-block;
  margin: 0.6rem 0 0;
  padding: 0.3rem 0.9rem;
  background: var(--dashboard-soft-text, #475569);
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85em;
  font-family: inherit;
}
.legacy-pr-expand-btn:hover { opacity: 0.85; }
.legacy-pr-expand-btn:focus-visible {
  outline: 2px solid var(--dashboard-link, #1d4ed8);
  outline-offset: 2px;
}
details.legacy-pr-group { margin-top: 0.6rem; }
details.legacy-pr-group:first-of-type { margin-top: 0.25rem; }
details.legacy-pr-group > summary.report-subcard-summary {
  background: linear-gradient(180deg,
    color-mix(in srgb, var(--dashboard-warning) 6%, var(--dashboard-panel-strong)) 0%,
    var(--dashboard-panel-bg) 100%);
}
/* Focus card grid */
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
.focus-card-detail {
  font-size: 0.85rem;
  color: var(--dashboard-soft-text, #475569);
  margin-top: 0.4rem;
  line-height: 1.4;
}
/* Sub-card (collapsible) */
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
.focus-table-sub {
  font-size: 0.85rem;
  color: var(--dashboard-soft-text, #475569);
  margin: 0 0 0.6rem;
  line-height: 1.45;
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
    top_prs: list[dict[str, Any]],
    gh_token_state: str,  # "token", "no-token", "preview"
    now: datetime,
    bot_login: str,
) -> str:
    css = EDITORIAL_THEME_CSS + _LEGACY_PR_SECTION_CSS
    reviewers = len({g["reviewer"] for g in top_prs})
    total_prs = sum(g["pr_count"] for g in top_prs)
    shown_prs = sum(g["shown_pr_count"] for g in top_prs)
    max_days = max((g["max_days_open"] for g in top_prs), default=0.0)
    high_priority_total = sum(
        1 for g in top_prs for pr in g["prs"] if pr.get("priority")
    )
    severity = "fail" if max_days >= 30 else ("fail" if reviewers == 0 and total_prs == 0 else "ok")
    cards = [
        _render_focus_metric_card(
            "Pending reviewers",
            str(reviewers),
            f"users @{0 if reviewers == 1 else ''} mentioned by {bot_login}",
            "ok" if reviewers > 0 else "fail",
        ),
        _render_focus_metric_card(
            "Pending PRs",
            str(total_prs),
            f"{shown_prs} shown (top {min(20, shown_prs)} per reviewer)",
            "ok",
        ),
        _render_focus_metric_card(
            "Oldest pending",
            f"{max_days:.1f}d",
            f"{high_priority_total} high-priority PR(s) across all buckets",
            "fail" if max_days >= 30 else "ok",
        ),
    ]
    body_parts: list[str] = [
        '<div class="top-bar"><div class="shell top-bar-inner">'
        '<div class="brand">'
        f'<div class="brand-mark">{_svg_icon(_SVG_CLIPBOARD, size=30, extra_class="brand-ico")}</div>'
        '<div class="brand-copy">'
        f"<h1>{html.escape(title)}</h1>"
        '<p class="tagline">Standalone Stale PRs report</p>'
        "</div></div></div></div>",
        '<div class="shell">',
        '<section class="panel nightly-focus">',
        _heading_html(
            "h2",
            _SVG_SPARK,
            html.escape("Stale PRs"),
            sub=html.escape("Open PRs grouped by bot-mentioned reviewers"),
        ),
        '<div class="focus-card-grid">',
        "\n".join(cards),
        "</div>",
    ]

    pr_section_html = _render_top_legacy_pr_section_html(top_prs)
    if pr_section_html:
        body_parts.append(
            '<p class="focus-table-sub">Open PRs whose '
            f"<code>{bot_login}</code> comment mentions the "
            "reviewer (the bot posts a cc-style triage comment per PR). "
            "High-priority PRs surface first; remaining slots are filled "
            "by the longest-pending PRs.</p>"
            + pr_section_html
        )
    else:
        body_parts.append(
            '<p class="hint">No open PRs with bot-mentioned reviewers fetched. '
            "Confirm <code>GITHUB_TOKEN</code> / <code>GH_TOKEN</code> is set "
            "in the environment and that the token has <code>public_repo</code> "
            f"scope for <code>{_REPO}</code> "
            f"(the bot <code>{bot_login}</code> must have commented on the PRs).</p>"
        )

    body_parts.append("</section>")
    body_parts.append("</div>")

    token_line = {
        "token": "authenticated via <code>$GITHUB_TOKEN</code> / <code>$GH_TOKEN</code>",
        "no-token": "unauthenticated REST (degraded — set <code>$GITHUB_TOKEN</code> for stable results)",
        "preview": "preview mode — empty dataset, no GitHub calls",
    }[gh_token_state]
    body_parts.append(
        f'<div class="shell"><div class="data-source">'
        f"Data source: <code>{_REPO_URL}/pulls</code> + <code>{bot_login}</code> "
        f"comment scan ({token_line}). Generated "
        f"<code>{html.escape(now.isoformat(timespec='seconds'))}</code> UTC."
        f"</div></div>"
    )

    return _html_document(title, css, "\n".join(body_parts))


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
        args.extend(["-c", "user.name=vllm-omni-pr-monitor"])
    if not email:
        args.extend(["-c", "user.email=vllm-omni-pr-monitor@users.noreply.github.com"])
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
    to ``<kanban_repo>/data/pr_monitor/pr-monitor-report.html`` (overwrite)
    and pushed. Returns a dict with the archived path, commit hash,
    branch, and push note for the caller to print.

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
    commit_message = f"data: archive pr monitor report (UTC {now.isoformat(timespec='seconds')})"

    plan_lines = [
        "",
        "=" * 60,
        "PR-monitor kanban archive",
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
        f"Pushed pr-monitor report to origin/{branch} ({KANBAN_REPO_URL}) "
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
        "--bot-login",
        default=_LEGACY_PR_BOT_LOGIN,
        help=f"GitHub login of the triage bot (default: {_LEGACY_PR_BOT_LOGIN}).",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help=(
            "After writing the report, copy it to <kanban>/data/pr_monitor/pr-monitor-report.html "
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

    if args.preview:
        top_prs: list[dict[str, Any]] = []
        token_state = "preview"
        out_path = args.out or skill_dir / pr_monitor_report_preview_basename(date_iso)
        title = "vLLM-Omni PR Monitor (preview)"
    else:
        gh_token: str | None = None
        if not args.no_token:
            gh_token = _resolve_github_token(args.gh_token)
        top_prs = _compute_top_legacy_prs(gh_token, now=datetime.now(timezone.utc))
        token_state = "token" if gh_token else "no-token"
        out_path = args.out or default_pr_monitor_html_path(skill_dir, date_iso)
        title = pr_monitor_report_title(date_iso)

    now = datetime.now(timezone.utc)
    document = render_html(
        title=title,
        top_prs=top_prs,
        gh_token_state=token_state,
        now=now,
        bot_login=args.bot_login,
    )
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
        if args.preview:
            print(
                "Refusing to archive a preview report — regenerate without --preview "
                "for a real snapshot.",
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