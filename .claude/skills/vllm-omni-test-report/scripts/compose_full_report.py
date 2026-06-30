#!/usr/bin/env python3
"""
Compose a full test report (default **HTML**).

Agents and users should **emit HTML by default**; pass ``--format markdown`` only when a Markdown
file is explicitly required (e.g. hand-editing, ``patch_report_*.py``).

  - Test conclusion: interactive checklist (HTML) / static MD; auto rows: "L2&L3…" = latest finished
    **ready** + **merge** (same buckets as metrics) have no failed/broken jobs;
    "critical issues…" = no open ``critical``;
    "Remaining DI…" = open ``bug`` in stats window weighted by priority labels < 30;
    "bug assignees…" = open ``bug`` all have assignee
  - Metrics overview: buildkite_build_stats.py --markdown
  - Test Result: Common stack from references/local-test-matrix.md; H200/H800/A100 from
    optional --log-dir-h* (nightly-style Summary); H100 = Buildkite scheduled nightly (**Build** table
    with build link, branch, commit only; **Summary** + failed-jobs table — no per-job pytest or
    bug+ci-failure analysis)
  - Issue tracking: GitHub Search label:ci-failure + local test in:title (stats window)
  - Open issues: GitHub open bugs (label:bug); filter `created_at` to --stats-from..--stats-to (UTC)

Requires BUILDKITE_TOKEN or BUILDKITE_API_TOKEN in the environment.
Run from skill dir: python scripts/compose_full_report.py
(Use ``--format markdown`` only when a ``.md`` artifact is explicitly needed, e.g. for ``patch_report_*.py``.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from md_table import render_markdown_table  # noqa: E402
from nightly_local_log_report import markdown_local_summary_from_log_dir  # noqa: E402
from release_md_to_html import (  # noqa: E402
    RELEASE_CONCLUSION_PLACEHOLDER,
    convert_release_report_markdown,
    materialize_release_conclusion_in_markdown,
)
from report_naming import (  # noqa: E402
    release_report_basename,
    release_report_preview_basename,
    resolve_report_date_iso,
)

CI_FAILURE_LABEL = "ci-failure"  # matches GitHub label on vllm-project/vllm-omni
BUG_DI_THRESHOLD_TENTHS = 300  # "Remaining DI < 30"; store DI in tenths to avoid float drift.
BUG_DI_WEIGHTS_TENTHS: dict[str, int] = {
    "critical": 100,
    "high priority": 30,
    "medium priority": 10,
    "low priority": 1,
    "invalid": 0,
}
BUG_DI_LABEL_ORDER: tuple[str, ...] = (
    "invalid",
    "critical",
    "high priority",
    "medium priority",
    "low priority",
)

ORG = "vllm"
PIPELINE = "vllm-omni"
BRANCH = "main"
UPLOAD_PIPELINE_RE = re.compile(r"^Upload .+ Pipeline$", re.IGNORECASE)


def _github_tls_verify() -> bool:
    """Same as ``GITHUB_INSECURE_SSL`` in ``buildkite_build_stats.py`` (GitHub API only)."""
    v = (os.environ.get("GITHUB_INSECURE_SSL") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return False
    return True


if not _github_tls_verify():
    try:
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


def http_get_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
) -> object:
    """
    GET JSON from ``url``. Prefer ``requests`` when installed (better TLS/proxy behavior on some
    Windows setups); fall back to :mod:`urllib`.

    For ``api.github.com``, TLS verification follows ``GITHUB_INSECURE_SSL`` (see
    ``buildkite_build_stats.py``). Other hosts always verify.
    """
    h = dict(headers or {})
    verify = True
    if "api.github.com" in url:
        verify = _github_tls_verify()
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        last_err: Exception | None = None
        for attempt in range(12):
            try:
                r = requests.get(url, headers=h, timeout=timeout, verify=verify)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After", "60")
                    try:
                        wait_s = int(float(ra)) + 1
                    except ValueError:
                        wait_s = 61
                    time.sleep(min(180, max(1, wait_s)))
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_err = e
                if attempt < 11:
                    time.sleep(min(8, 2 ** min(attempt, 3)))
        assert last_err is not None
        raise last_err
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def http_json(url: str, token: str | None = None) -> object:
    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return http_get_json(url, headers=headers, timeout=120)


def latest_scheduled_nightly_number(token: str) -> int:
    url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds?branch={BRANCH}&per_page=50"
    builds = http_json(url, token)
    assert isinstance(builds, list)
    for b in builds:
        if re.search(r"scheduled\s+nightly", (b.get("message") or ""), re.I):
            return int(b["number"])
    sys.exit("No scheduled nightly build found on main (per_page=50).")


def _issue_created_date_utc(issue: dict) -> str | None:
    """``YYYY-MM-DD`` from GitHub ``created_at`` or ``None``."""
    ca = issue.get("created_at")
    if not ca or not isinstance(ca, str):
        return None
    s = ca.strip().replace("Z", "+00:00")
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def _github_fetch_open_issues_with_label(gh_token: str | None, label: str) -> list[dict]:
    """Paginate **open** issues with a single GitHub label (PR entries excluded)."""
    base = "https://api.github.com/repos/vllm-project/vllm-omni/issues"
    lab = urllib.parse.quote(label)
    all_items: list = []
    page = 1
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-compose-report",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    while True:
        url = f"{base}?state=open&labels={lab}&per_page=100&page={page}"
        batch = http_get_json(url, headers=headers, timeout=60)
        if not batch:
            break
        for i in batch:
            if i.get("pull_request"):
                continue
            all_items.append(i)
        if len(batch) < 100:
            break
        page += 1
    return all_items


def _github_fetch_open_bug_issues(gh_token: str | None) -> list[dict]:
    """Paginate **open** issues with label ``bug`` (PR entries excluded)."""
    return _github_fetch_open_issues_with_label(gh_token, "bug")


def _format_di_tenths(value: int) -> str:
    """Display a tenths-based DI integer without floating-point formatting."""
    sign = "-" if value < 0 else ""
    v = abs(value)
    whole, frac = divmod(v, 10)
    if frac == 0:
        return f"{sign}{whole}"
    return f"{sign}{whole}.{frac}"


def _issue_label_names(issue: dict) -> set[str]:
    """Normalized label names on a GitHub issue."""
    names: set[str] = set()
    labels = issue.get("labels") or []
    for label in labels:
        raw_name = label.get("name") if isinstance(label, dict) else str(label)
        if raw_name:
            names.add(str(raw_name).strip().lower())
    return names


def _bug_di_label_and_value(issue: dict) -> tuple[str, int]:
    """Return the DI priority label and tenths value for one open bug issue."""
    labels = _issue_label_names(issue)
    if "invalid" in labels:
        return "invalid", BUG_DI_WEIGHTS_TENTHS["invalid"]
    for label in BUG_DI_LABEL_ORDER:
        if label == "invalid":
            continue
        if label in labels:
            return label, BUG_DI_WEIGHTS_TENTHS[label]
    return "unclassified", 0


def _bug_di_summary(issues: list[dict]) -> tuple[int, dict[str, int]]:
    """Sum DI for stats-window open bugs and count labels used by the rule."""
    counts = {label: 0 for label in BUG_DI_LABEL_ORDER}
    counts["unclassified"] = 0
    total = 0
    for issue in issues:
        label, value = _bug_di_label_and_value(issue)
        counts[label] += 1
        total += value
    return total, counts


def _bug_di_detail(total_tenths: int, counts: dict[str, int]) -> str:
    """Human-readable DI calculation detail for the release conclusion row."""
    parts = [f"{label}={counts[label]}" for label in BUG_DI_LABEL_ORDER if counts.get(label, 0)]
    if counts.get("unclassified", 0):
        parts.append(f"unclassified={counts['unclassified']}")
    detail = ", ".join(parts) if parts else "no open bug in stats window"
    return f"Auto DI={_format_di_tenths(total_tenths)} ({detail})"


def _bug_di_conclusion(issues: list[dict]) -> tuple[bool, str]:
    total_tenths, counts = _bug_di_summary(issues)
    return total_tenths < BUG_DI_THRESHOLD_TENTHS, _bug_di_detail(total_tenths, counts)


def no_open_critical_labeled_issues(
    gh_token: str | None,
) -> tuple[bool, str]:
    """
    For **Test conclusion** row "No remaining critical issues": pass iff there is **no** open (non-PR)
    issue with label ``critical``.
    """
    try:
        issues = _github_fetch_open_issues_with_label(gh_token, "critical")
    except Exception as exc:
        return False, f"Unable to check label critical ({exc})"
    if not issues:
        return True, ""
    nums: list[int] = []
    for i in issues:
        try:
            nums.append(int(i["number"]))
        except (KeyError, TypeError, ValueError):
            continue
    nums.sort()
    show = nums[:15]
    tail = f" ({len(nums)} total)" if len(nums) > len(show) else ""
    lst = ", ".join(f"#{n}" for n in show)
    return False, f"Open issues with label **critical** still exist: {lst}{tail}"


def open_bug_assignees_all_assigned(
    gh_token: str | None,
) -> tuple[bool, str]:
    """
    For **Test conclusion** auto row: pass iff every open ``bug`` issue has at least one assignee.

    Returns ``(ok, detail)`` — ``detail`` is empty on success; on failure, a short English
    note listing unassigned issue numbers (or an error reason if the API call fails).
    """
    try:
        issues = _github_fetch_open_bug_issues(gh_token)
    except Exception as exc:
        return False, f"Unable to check assignee ({exc})"
    unassigned: list[int] = []
    for i in issues:
        assignees = i.get("assignees")
        if assignees is None:
            assignees = []
        if not assignees:
            try:
                unassigned.append(int(i["number"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not unassigned:
        return True, ""
    unassigned.sort()
    show = unassigned[:15]
    tail = f" ({len(unassigned)} total)" if len(unassigned) > len(show) else ""
    nums = ", ".join(f"#{n}" for n in show)
    return False, f"The following open bugs have no assignee: {nums}{tail}"


def github_open_bug_rows_in_range(
    gh_token: str | None,
    date_from: str,
    date_to: str,
) -> tuple[int, int, str, list[dict]]:
    """
    Paginate **open** issues with label ``bug`` (PR entries excluded).

    Return ``(total_open_bug_fetched, count_in_created_range, markdown_table, issues_in_range)``.
    ``count_in_created_range`` = issues whose **UTC calendar date** of ``created_at``
    lies in ``[date_from, date_to]`` inclusive (``YYYY-MM-DD`` strings).
    """
    all_items = _github_fetch_open_bug_issues(gh_token)

    in_range = [i for i in all_items if (d := _issue_created_date_utc(i)) is not None and date_from <= d <= date_to]
    in_range.sort(key=lambda x: x["created_at"], reverse=True)
    row_cells: list[list[str]] = []
    for i in in_range:
        t = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        u = (i.get("user") or {}).get("login", "")
        di_label, di_tenths = _bug_di_label_and_value(i)
        row_cells.append(
            [
                f"[#{i['number']}](https://github.com/vllm-project/vllm-omni/issues/{i['number']})",
                t,
                str(i["created_at"])[:10],
                di_label,
                _format_di_tenths(di_tenths),
                "open",
                f"@{u}",
            ]
        )
    body = render_markdown_table(
        ["Issue", "Title", "Opened at", "Priority", "DI", "Status", "Owner"],
        row_cells,
    )
    return len(all_items), len(in_range), body, in_range


def render_open_issues_section_with_di(
    stats_from: str,
    stats_to: str,
    gh_token: str | None,
) -> tuple[str, bool | None, str]:
    """Markdown for ``## Open issues`` plus DI conclusion data when GitHub fetch succeeds."""
    github_open_error = ""
    di_row_ok: bool | None = None
    di_row_detail = ""
    try:
        open_total, open_range_n, issue_rows, issues_in_range = github_open_bug_rows_in_range(
            gh_token, stats_from, stats_to
        )
        di_row_ok, di_row_detail = _bug_di_conclusion(issues_in_range)
    except Exception as exc:
        open_total = 0
        open_range_n = 0
        issue_rows = render_markdown_table(
            ["Issue", "Title", "Opened at", "Priority", "DI", "Status", "Owner"],
            [
                [
                    "*—*",
                    "*Failed to fetch; set `GITHUB_TOKEN` or fill in manually*",
                    "*—*",
                    "*—*",
                    "*—*",
                    "*—*",
                    "*—*",
                ]
            ],
        )
        github_open_error = str(exc)

    github_open_note = f" **Note:** open-bugs fetch failed (`{github_open_error}`)." if github_open_error else ""
    return (
        (
            f"## Open issues (stats window)\n\n"
            f"Open issues labeled **bug**, state **open**, excluding PRs, with `created_at` "
            f"(UTC date) in **{stats_from}** … **{stats_to}** (same as Buildkite `--stats-from` / "
            f"`--stats-to`): **{open_range_n}** (total open `bug` issues when fetched: "
            f"**{open_total}**). DI uses priority labels: `critical` = 10, `high priority` = 3, "
            f"`medium priority` = 1, `low priority` = 0.1, `invalid` = 0.{github_open_note}\n\n"
            f"{issue_rows}\n"
        ),
        di_row_ok,
        di_row_detail,
    )


def render_open_issues_section(
    stats_from: str,
    stats_to: str,
    gh_token: str | None,
) -> str:
    """Markdown for ``## Open issues`` block (GitHub REST, open ``label:bug`` only)."""
    section, _, _ = render_open_issues_section_with_di(stats_from, stats_to, gh_token)
    return section


def github_ci_failure_analysis_rows(
    created_from: str,
    created_to: str,
    gh_token: str | None,
) -> tuple[int, str]:
    """
    Issues with labels ``bug`` and ``ci-failure``, ``created_at`` (UTC) in
    ``created_from`` .. ``created_to`` (inclusive, YYYY-MM-DD).

    Same date window as ``compose_full_report.py`` ``--stats-from`` / ``--stats-to``
    (Buildkite metrics window).
    """
    q = f"repo:vllm-project/vllm-omni is:issue label:bug label:{CI_FAILURE_LABEL} created:{created_from}..{created_to}"
    base = "https://api.github.com/search/issues?q=" + urllib.parse.quote(q)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-compose-report",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    collected: list[dict] = []
    page = 1
    while True:
        url = f"{base}&per_page=100&page={page}"
        data = http_get_json(url, headers=headers, timeout=120)
        items = data.get("items") or []
        if not items:
            break
        for i in items:
            if i.get("pull_request"):
                continue
            collected.append(i)
        if len(items) < 100:
            break
        page += 1

    collected.sort(key=lambda x: int(x.get("number", 0)), reverse=True)
    row_cells: list[list[str]] = []
    for i in collected:
        num = i["number"]
        title = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        st = (i.get("state") or "").lower()
        status_label = "Closed" if st == "closed" else "Open"
        link = f"https://github.com/vllm-project/vllm-omni/issues/{num}"
        row_cells.append([f"[#{num}]({link})", title, status_label])
    if not row_cells:
        return 0, ""
    return len(collected), render_markdown_table(["Issue #", "Title", "Status"], row_cells)


def render_ci_failure_section(
    stats_from: str,
    stats_to: str,
    gh_token: str | None,
) -> str:
    """
    Markdown for ``### Analysis (CI Failure)`` … (GitHub Search only; no Buildkite).

    Used by ``compose_full_report.py`` and ``patch_report_ci_failure.py``.
    """
    try:
        ci_fail_n, ci_fail_rows = github_ci_failure_analysis_rows(stats_from, stats_to, gh_token)
        ci_fail_error = ""
    except Exception as exc:
        ci_fail_n = -1
        ci_fail_rows = ""
        ci_fail_error = str(exc)

    ci_filter_note = (
        f"**Filter:** `label:bug` and `label:{CI_FAILURE_LABEL}`, "
        f"`created` (UTC) **{stats_from}** … **{stats_to}** (same window as Buildkite metrics / "
        f"`--stats-from` / `--stats-to`). "
        f"**Cross-check:** "
        f"[issues · bug + ci-failure](https://github.com/vllm-project/vllm-omni/issues?q=is%3Aissue+label%3Abug+label%3Aci-failure)."
    )
    if ci_fail_error:
        return (
            f"### Analysis (CI Failure)\n\n"
            f"*GitHub Search API unavailable: {ci_fail_error}.* Fill in manually per "
            f"[references/ci-github-ci-failure-issues.md](references/ci-github-ci-failure-issues.md) "
            f"from [open bugs](https://github.com/vllm-project/vllm-omni/issues/"
            f"?q=is%3Aissue%20state%3Aopen%20label%3Abug) "
            f"and [closed bugs](https://github.com/vllm-project/vllm-omni/issues/"
            f"?q=is%3Aissue%20state%3Aclosed%20label%3Abug).\n"
        )
    if ci_fail_n == 0:
        return f"### Analysis (CI Failure)\n\n{ci_filter_note}\n\n*No matching issues in this date range.*\n"
    return f"### Analysis (CI Failure)\n\n{ci_filter_note} **Rows in table:** {ci_fail_n}.\n\n{ci_fail_rows}\n"


def github_issue_tracking_local_test_rows(
    created_from: str,
    created_to: str,
    gh_token: str | None,
) -> tuple[int, str]:
    """
    GitHub Search: ``label:ci-failure``, ``created`` in date range, **title** contains
    ``local test``. Excludes PR entries; post-filters title case-insensitively.
    """
    q = (
        f"repo:vllm-project/vllm-omni is:issue label:{CI_FAILURE_LABEL} "
        f"created:{created_from}..{created_to} "
        f'in:title "local test"'
    )
    base = "https://api.github.com/search/issues?q=" + urllib.parse.quote(q)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-compose-report",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    collected: list[dict] = []
    page = 1
    while True:
        url = f"{base}&per_page=100&page={page}"
        data = http_get_json(url, headers=headers, timeout=120)
        items = data.get("items") or []
        if not items:
            break
        for i in items:
            if i.get("pull_request"):
                continue
            title = (i.get("title") or "").lower()
            if "local test" not in title:
                continue
            collected.append(i)
        if len(items) < 100:
            break
        page += 1

    collected.sort(key=lambda x: int(x.get("number", 0)), reverse=True)
    row_cells: list[list[str]] = []
    for i in collected:
        num = i["number"]
        title = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        st = (i.get("state") or "").lower()
        status_label = "closed" if st == "closed" else "open"
        ca = str(i.get("created_at") or "")[:10]
        link = f"https://github.com/vllm-project/vllm-omni/issues/{num}"
        row_cells.append([f"[#{num}]({link})", title, status_label, ca])
    body = (
        render_markdown_table(
            ["Issue", "Title", "State", "Created (UTC date)"],
            row_cells,
        )
        if row_cells
        else ""
    )
    return len(collected), body


def render_issue_tracking_section(
    stats_from: str,
    stats_to: str,
    gh_token: str | None,
) -> str:
    """Markdown for ``## Issue tracking`` (ci-failure + *local test* in title)."""
    try:
        n, table_rows = github_issue_tracking_local_test_rows(stats_from, stats_to, gh_token)
        err_note = ""
    except Exception as exc:
        n = -1
        table_rows = ""
        err_note = str(exc)

    filt = (
        f"**Filter:** GitHub Search — `label:{CI_FAILURE_LABEL}`, `created` (UTC) "
        f"**{stats_from}** … **{stats_to}**, title contains `local test` (case-insensitive). "
        f"**Cross-check:** "
        f"[search · ci-failure + local in title](https://github.com/search?q=repo%3Avllm-project%2Fvllm-omni+is%3Aissue+label%3Aci-failure+local+test+in%3Atitle&type=issues).\n\n"
    )
    if err_note:
        return (
            "## Issue tracking\n\n"
            f"{filt}"
            f"*GitHub Search API unavailable: {err_note}.* Configure `GITHUB_TOKEN` / `GH_TOKEN` and retry, "
            f"or search manually using the link above.\n"
        )
    if n == 0:
        return f"## Issue tracking\n\n{filt}*No matching issues in this window.*\n"
    return f"## Issue tracking\n\n{filt}*Matching issues: **{n}**.*\n\n{table_rows}\n"


def extract_common_stack_from_matrix(skill_dir: Path) -> str:
    """Body text under ``## Common stack (all rows)`` in ``local-test-matrix.md``."""
    ref = skill_dir / "references" / "local-test-matrix.md"
    if not ref.is_file():
        return "*(`references/local-test-matrix.md` not found.)*\n"
    raw = ref.read_text(encoding="utf-8")
    m = re.search(
        r"(?ms)^## Common stack \(all rows\)\s*\n(.*?)(?=^\#\# |\Z)",
        raw,
    )
    if not m:
        return "*Could not find `## Common stack (all rows)` section; check reference.*\n"
    body = (m.group(1) or "").strip()
    return (body + "\n") if body else "*Common stack section is empty.*\n"


def _gpu_log_placeholder(gpu_flag: str) -> str:
    return (
        f"*`{gpu_flag}` not provided: no summary table matching nightly local sections.* "
        f"Pass the cluster/machine `nightly_jobs` log root to compose (see `--help`)."
    )


def build_h100_ci_markdown_body(
    *,
    build_table_md: str,
    passed: int,
    failed: int,
    skipped: int,
    failed_section: str,
) -> str:
    return (
        f"#### Build\n\n{build_table_md}\n\n"
        f"#### Summary (reportable jobs only)\n\n"
        f"- **Passed**: {passed} jobs\n"
        f"- **Failed / broken**: {failed} jobs\n"
        f"- **Skipped / blocked / not_run**: {skipped} (if any)\n\n"
        f"#### Failed test jobs (if any)\n\n{failed_section}\n"
    )


def render_test_result_section(
    skill_dir: Path,
    *,
    log_h200: Path | None,
    log_h800: Path | None,
    log_a100: Path | None,
    h100_ci_markdown: str,
) -> str:
    chunks: list[str] = [
        "## Test Result",
        "",
        "Layout: `### Common stack` from reference; "
        "`### H200` / `### H800` / `### A100` match nightly local **Summary** grouping (pass `--log-dir-*`); "
        "`### H100` is **Buildkite scheduled nightly** (Build / Summary / failed job table; "
        "no per-job pytest or bug+ci-failure analysis).",
        "",
        "### Common stack (all rows)",
        "",
        extract_common_stack_from_matrix(skill_dir).rstrip(),
        "",
        "### H200",
        "",
    ]
    chunks.append(markdown_local_summary_from_log_dir(log_h200) if log_h200 else _gpu_log_placeholder("--log-dir-h200"))
    chunks.extend(["", "### H800", ""])
    chunks.append(markdown_local_summary_from_log_dir(log_h800) if log_h800 else _gpu_log_placeholder("--log-dir-h800"))
    chunks.extend(["", "### A100", ""])
    chunks.append(markdown_local_summary_from_log_dir(log_a100) if log_a100 else _gpu_log_placeholder("--log-dir-a100"))
    chunks.extend(
        [
            "",
            "### H100（CI — Buildkite scheduled nightly）",
            "",
            h100_ci_markdown.rstrip(),
            "",
        ]
    )
    return "\n".join(chunks)


def render_test_conclusion_section() -> str:
    """``## Test conclusion`` + placeholder for interactive widget (HTML) or static MD."""
    return f"## Test conclusion\n\n{RELEASE_CONCLUSION_PLACEHOLDER}\n\n"


def run_script(py: Path, args: list[str], cwd: Path, env: dict[str, str]) -> str:
    cmd = [sys.executable, str(py)] + args
    child_env = dict(env)
    child_env.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform == "win32":
        child_env.setdefault("PYTHONUTF8", "1")
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=child_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
    )
    if p.returncode != 0:
        sys.stderr.write(p.stderr or "")
        sys.exit(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout or ""


def extract_ci_markdown(stats_stdout: str) -> str:
    heading = "## Metrics overview"
    if heading not in stats_stdout:
        return stats_stdout.strip()
    part = stats_stdout.split(heading, 1)[1]
    if "Done." in part:
        part = part.split("Done.", 1)[0]
    return (heading + part).strip()


def _job_scope_ref_lookup_key(cell: str) -> str:
    """First column of a scope table row -> lookup key (matches Buildkite `job.name`)."""
    t = (cell or "").replace("**", "").strip()
    if " (" in t:
        t = t.split(" (", 1)[0].strip()
    return t


def load_job_scope_lookup(ref_path: Path) -> dict[str, str]:
    """
    Parse pipe tables in ``ci-job-test-scope.md`` -> job name -> scope / intent (second column).

    Skips separator rows and header cells ``Typical job name`` / ``Source``.
    """
    if not ref_path.is_file():
        return {}
    lookup: dict[str, str] = {}
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        parts = [p.strip() for p in s.split("|")[1:-1]]
        if len(parts) < 2:
            continue
        k_raw, scope = parts[0], parts[1]
        if not k_raw or re.match(r"^:?-+:?$", k_raw):
            continue
        key = _job_scope_ref_lookup_key(k_raw)
        low = key.lower()
        if low in ("typical job name", "source"):
            continue
        if not key:
            continue
        lookup[key] = scope.replace("|", "/")
    return lookup


def render_job_scope_section(build: dict, build_no: int, skill_dir: Path) -> str:
    """
    ``## Test content (job scope)``: one row per **reportable** job in this nightly
    (same rule as Summary: omit ``Upload * Pipeline``), scope text from reference lookup.
    """
    ref = skill_dir / "references" / "ci-job-test-scope.md"
    lookup = load_job_scope_lookup(ref)
    jobs = build.get("jobs") or []
    reportable = [j for j in jobs if not UPLOAD_PIPELINE_RE.match((j.get("name") or "").strip())]
    reportable.sort(key=lambda x: (x.get("name") or ""))
    missing = (
        "*—* *(not in reference; add to [references/ci-job-test-scope.md](references/ci-job-test-scope.md) or see log)*"
    )
    rows: list[list[str]] = []
    for j in reportable:
        name = (j.get("name") or "").replace("|", "/")
        st = (j.get("state") or "").replace("|", "/")
        jid = j.get("id") or ""
        link = f"[open](https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#{jid})"
        scope = lookup.get(name.strip(), missing)
        rows.append([name, st, link, scope])
    table = render_markdown_table(
        ["Job (this nightly)", "State", "Step link", "Scope / intent"],
        rows,
    )
    return (
        "## Test content (job scope)\n\n"
        f"Jobs match **scheduled nightly** "
        f"[#{build_no}](https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}) "
        "(**reportable** only: `Upload * Pipeline` omitted). "
        "**Scope / intent** is looked up from "
        "[references/ci-job-test-scope.md](references/ci-job-test-scope.md) "
        "by exact job name (see categorized reference for maintenance).\n\n"
        f"{table}\n"
    )


PREVIEW_BUILD_NO = 12880


def preview_report_markdown(
    skill_dir: Path,
    *,
    stats_from: str,
    stats_to: str,
    build_no: int = PREVIEW_BUILD_NO,
) -> str:
    """
    Same section layout as the live **release** report (minus any hand-only sections), but **no network**
    and no subprocess calls.

    Embeds real ``references/local-test-matrix.md`` Common stack when present.
    """
    conclusion = render_test_conclusion_section()
    ci_md = (
        "## Metrics overview\n\n"
        "*This section uses **preview placeholder data**: `buildkite_build_stats.py` was not run; "
        "values below are layout demos only.*\n\n"
        + render_markdown_table(
            ["Metric (example)", "Value"],
            [
                ["**Stats window**", f"`{stats_from}` … `{stats_to}`"],
                ["**Pipeline**", f"`{ORG}/{PIPELINE}` · branch `{BRANCH}`"],
                ["**Job success rate (window)**", "97.4%"],
                ["**UT coverage (example)**", "84.6%"],
                ["**Bug avg first response (h)**", "6.2"],
                ["**New bugs in window (example)**", "5"],
                ["**L4 / nightly reach**", "✓ Example: last 7 scheduled builds completed"],
                [
                    "**Note**",
                    "Remove `--preview` and configure tokens to replace with real `buildkite_build_stats.py` output.",
                ],
            ],
        )
    )

    demo_link_a = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#step-demo-jid-a"
    demo_link_b = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#step-demo-jid-b"

    build_table_md = render_markdown_table(
        ["Field", "Value"],
        [
            [
                "**Build**",
                f"[{build_no}](https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no})",
            ],
            ["**Branch**", BRANCH],
            [
                "**Commit**",
                "`c0ffee1` ([full](https://github.com/vllm-project/vllm-omni/commit/c0ffee1deadbeefcafe000000000000000000001))",
            ],
        ],
    )

    failed_section = render_markdown_table(
        ["Step / Job", "State", "Notes", "Step link"],
        [
            [
                "L2_Diffusion_Accuracy_Test",
                "failed",
                "AssertionError: max diff 0.08 > 0.05 *(example)*",
                f"[open]({demo_link_a})",
            ],
            [
                "L3_Merge_Example_Suite",
                "failed",
                "Timeout after 45m *(example)*",
                f"[open]({demo_link_b})",
            ],
        ],
    )

    h100_body = build_h100_ci_markdown_body(
        build_table_md=build_table_md,
        passed=11,
        failed=2,
        skipped=1,
        failed_section=failed_section,
    )

    test_result = render_test_result_section(
        skill_dir,
        log_h200=None,
        log_h800=None,
        log_a100=None,
        h100_ci_markdown=h100_body,
    )

    issue_tracking = (
        "## Issue tracking\n\n"
        "**Filter:** GitHub Search — `label:ci-failure`, `created` (UTC) "
        f"**{stats_from}** … **{stats_to}**, title contains `local test`.\n\n"
        "*Preview placeholder data below (columns match the live report).*\n\n"
        "*Matching issues: **2**.*\n\n"
        + render_markdown_table(
            ["Issue", "Title", "State", "Created (UTC date)"],
            [
                [
                    "[#10042](https://github.com/vllm-project/vllm-omni/issues/10042)",
                    "local test · H100 diffusion batch flaky",
                    "open",
                    stats_to,
                ],
                [
                    "[#10018](https://github.com/vllm-project/vllm-omni/issues/10018)",
                    "Regression in local test matrix for A100 path",
                    "closed",
                    stats_from,
                ],
            ],
        )
        + "\n"
    )

    open_issues_block = (
        "## Open issues (stats window)\n\n"
        f"Open issues labeled **bug**, state **open**, excluding PRs, with `created_at` "
        f"(UTC date) in **{stats_from}** … **{stats_to}**. "
        "*Preview placeholder data; live report uses paginated GitHub results.*\n\n"
        + render_markdown_table(
            ["Issue", "Title", "Opened at", "Priority", "DI", "Status", "Owner"],
            [
                [
                    "[#10055](https://github.com/vllm-project/vllm-omni/issues/10055)",
                    "OOM when loading Qwen-Omni with FP8 on 40GB",
                    "2026-05-14",
                    "high priority",
                    "3",
                    "open",
                    "@alice-preview",
                ],
                [
                    "[#10042](https://github.com/vllm-project/vllm-omni/issues/10042)",
                    "Intermittent timeout on L2 diffusion accuracy",
                    stats_to,
                    "medium priority",
                    "1",
                    "open",
                    "@bob-preview",
                ],
                [
                    "[#10030](https://github.com/vllm-project/vllm-omni/issues/10030)",
                    "Docs: wrong env var for TEE cache",
                    "2026-05-10",
                    "low priority",
                    "0.1",
                    "open",
                    "@carol-preview",
                ],
            ],
        )
        + "\n"
    )

    return f"""# vLLM-Omni Test Report - Scheduled Nightly

* **Preview mode:** No Buildkite / GitHub / pytest log fetch; content below is layout preview only.*

{conclusion}{ci_md}

{test_result}
{issue_tracking}{open_issues_block}
## Data source

- **Mode:** `compose_full_report.py --preview` (sample tables only)
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-*`; H100 is Buildkite block
- **Issue tracking:** `label:ci-failure` + title **local test**; Open issues still paginated `label:bug`
- Live report: `buildkite_build_stats.py`, GitHub REST/Search
"""


def local_testing_markdown(skill_dir: Path) -> str:
    """Backward-compatible stub for patch scripts: Test Result without log dirs, dummy H100."""
    return render_test_result_section(
        skill_dir,
        log_h200=None,
        log_h800=None,
        log_a100=None,
        h100_ci_markdown=(
            "*Insert full H100 / Buildkite block here; run `compose_full_report.py` to regenerate the report.*\n"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose full vllm-omni test report (HTML default; optional Markdown).",
    )
    parser.add_argument(
        "--format",
        choices=("html", "markdown"),
        default="html",
        help="Output format (default: html). Use markdown for patch_report_*.py workflows.",
    )
    parser.add_argument(
        "--report-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC date for default --out filename (default: today UTC). "
        "Never derived from --log-dir-h* or nightly_jobs_* suffixes.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. Default: <skill-dir>/vllm-omni-test-report-YYYY-MM-DD.html (or .md when --format markdown)."
        ),
    )
    parser.add_argument(
        "--stats-from",
        default=None,
        help=(
            "buildkite_build_stats.py --from (UTC YYYY-MM-DD). "
            "Default: first day of current UTC month (month-to-date, matches SKILL)."
        ),
    )
    parser.add_argument(
        "--stats-to",
        default=None,
        help="buildkite_build_stats.py --to (default: today UTC)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Emit sample data only (no Buildkite, GitHub, or pytest log fetch). "
            "Default output: vllm-omni-test-report-preview-YYYY-MM-DD.html"
        ),
    )
    parser.add_argument(
        "--log-dir-h200",
        type=Path,
        default=None,
        help=(
            "Optional. Root directory of nightly job logs for **Test Result → H200** "
            "(same layout as nightly `nightly_jobs`; see references/nightly-local-log-layout.md)."
        ),
    )
    parser.add_argument(
        "--log-dir-h800",
        type=Path,
        default=None,
        help="Optional. Log root for **Test Result → H800** (same layout as --log-dir-h200).",
    )
    parser.add_argument(
        "--log-dir-a100",
        type=Path,
        default=None,
        help="Optional. Log root for **Test Result → A100** (same layout as --log-dir-h200).",
    )
    args = parser.parse_args()

    skill_dir = Path(__file__).resolve().parent.parent
    scripts_dir = skill_dir / "scripts"
    report_date = resolve_report_date_iso(args.report_date)

    token = (os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN") or "").strip()
    if not args.preview and not token:
        print(
            "BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is not set.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.preview:
        today_utc = report_date
        stats_to = args.stats_to or today_utc
        stats_from = args.stats_from or datetime.strptime(today_utc, "%Y-%m-%d").date().replace(day=1).isoformat()
        md = preview_report_markdown(skill_dir, stats_from=stats_from, stats_to=stats_to)
        out_path = args.out
        if out_path is None:
            ext = ".html" if args.format == "html" else ".md"
            base = release_report_preview_basename(report_date)
            out_path = skill_dir / (base.replace(".html", ext) if ext == ".md" else base)
        else:
            out_path = Path(args.out)
        if args.format == "html":
            archive_name = out_path.with_suffix(".md").name
            out_path.write_text(
                convert_release_report_markdown(
                    md,
                    archive_download_name=archive_name,
                    l2_l3_row_ok=True,
                    l2_l3_row_detail="",
                    di_row_ok=True,
                    di_row_detail="Auto DI=4.1 (high priority=1, medium priority=1, low priority=1)",
                    critical_row_ok=True,
                    critical_row_detail="",
                    assignee_row_ok=True,
                    assignee_row_detail="(Preview: GitHub / Buildkite gates not run; auto rows placeholder Pass)",
                ),
                encoding="utf-8",
            )
        else:
            out_path.write_text(
                materialize_release_conclusion_in_markdown(
                    md,
                    l2_l3_row_ok=True,
                    l2_l3_row_detail="",
                    di_row_ok=True,
                    di_row_detail="Auto DI=4.1 (high priority=1, medium priority=1, low priority=1)",
                    critical_row_ok=True,
                    critical_row_detail="",
                    assignee_row_ok=True,
                    assignee_row_detail="(Preview: GitHub / Buildkite gates not run; auto rows placeholder Pass)",
                ),
                encoding="utf-8",
            )
        print(f"Wrote {out_path}")
        return

    today_utc = report_date
    stats_to = args.stats_to or today_utc
    stats_from = args.stats_from or datetime.strptime(today_utc, "%Y-%m-%d").date().replace(day=1).isoformat()

    build_no = latest_scheduled_nightly_number(token)
    build_url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds/{build_no}"
    build = http_json(build_url, token)
    assert isinstance(build, dict)

    jobs = build.get("jobs") or []
    reportable = [j for j in jobs if not UPLOAD_PIPELINE_RE.match((j.get("name") or "").strip())]
    states = [(j.get("state") or "").lower() for j in reportable]
    passed = sum(1 for s in states if s == "passed")
    failed = sum(1 for s in states if s in ("failed", "broken"))
    skipped = sum(1 for s in states if s in ("skipped", "not_run", "blocked"))

    commit = build.get("commit") or ""
    short = commit[:7] if len(commit) >= 7 else commit
    env = os.environ.copy()

    stats_raw = run_script(
        scripts_dir / "buildkite_build_stats.py",
        ["--from", stats_from, "--to", stats_to, "--markdown"],
        skill_dir,
        env,
    )
    ci_md = extract_ci_markdown(stats_raw)

    gh_token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip() or None

    try:
        from buildkite_build_stats import l2_l3_ready_merge_gate

        l2_l3_row_ok, l2_l3_row_detail = l2_l3_ready_merge_gate(token)
    except ImportError:
        l2_l3_row_ok, l2_l3_row_detail = (
            False,
            "Unable to import buildkite_build_stats (pip install requests)",
        )
    except Exception as exc:
        l2_l3_row_ok, l2_l3_row_detail = False, f"L2&L3 check failed ({exc})"

    critical_row_ok, critical_row_detail = no_open_critical_labeled_issues(gh_token)
    assignee_row_ok, assignee_row_detail = open_bug_assignees_all_assigned(gh_token)
    out_path = args.out
    if out_path is None:
        ext = ".html" if args.format == "html" else ".md"
        base = release_report_basename(report_date)
        out_path = skill_dir / (base.replace(".html", ext) if ext == ".md" else base)
    else:
        out_path = Path(args.out)

    failed_jobs_rows: list[list[str]] = []
    for j in reportable:
        st = (j.get("state") or "").lower()
        if st in ("failed", "broken"):
            name = (j.get("name") or "").replace("|", "/")
            jid = j.get("id") or ""
            link = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#{jid}"
            failed_jobs_rows.append([name, st, "See step log", f"[open]({link})"])

    failed_section = (
        render_markdown_table(
            ["Step / Job", "State", "Notes", "Step link"],
            failed_jobs_rows,
        )
        if failed_jobs_rows
        else "*None.*"
    )

    build_table_md = render_markdown_table(
        ["Field", "Value"],
        [
            [
                "**Build**",
                f"[{build_no}](https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no})",
            ],
            ["**Branch**", build.get("branch") or "main"],
            [
                "**Commit**",
                f"`{short}` ([full](https://github.com/vllm-project/vllm-omni/commit/{commit}))",
            ],
        ],
    )

    conclusion = render_test_conclusion_section()
    h100_body = build_h100_ci_markdown_body(
        build_table_md=build_table_md,
        passed=passed,
        failed=failed,
        skipped=skipped,
        failed_section=failed_section,
    )
    test_result = render_test_result_section(
        skill_dir,
        log_h200=args.log_dir_h200,
        log_h800=args.log_dir_h800,
        log_a100=args.log_dir_a100,
        h100_ci_markdown=h100_body,
    )
    issue_tracking_block = render_issue_tracking_section(stats_from, stats_to, gh_token)
    open_issues_block, di_row_ok, di_row_detail = render_open_issues_section_with_di(stats_from, stats_to, gh_token)

    md = f"""# vLLM-Omni Test Report - Scheduled Nightly

{conclusion}{ci_md}

{test_result}
{issue_tracking_block}{open_issues_block}
## Data source

- **Test conclusion (auto):** (1) Buildkite **ready** (non-main) and **merge** (main non-nightly/weekly)
  each latest **finished** build has no `failed`/`broken` job; (2) open `label:bug` in stats window
  weighted by priority labels **DI < 30**; (3) no open `critical`; (4) open `label:bug` all have assignee
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-h200` / `--log-dir-h800` / `--log-dir-a100`; H100 = Buildkite scheduled nightly
  (this build #{build_no}: **Build** table link/branch/commit only + Summary + failed jobs)
- **Issue tracking:** GitHub Search — `label:ci-failure`, `local test` in:title,
  `created` in `{stats_from}`..`{stats_to}` (UTC)
- **Open issues:** REST `label:bug`, `created_at` UTC date in `{stats_from}`..`{stats_to}`
- Buildkite API: `{ORG}/{PIPELINE}` branch `main`
- `scripts/buildkite_build_stats.py --from {stats_from} --to {stats_to} --markdown` (**bugs (first response, …)** =
  GitHub `label:bug` issues with `created_at` UTC date in the same `--from`..`--to` window)
"""
    if args.format == "html":
        archive_name = out_path.with_suffix(".md").name
        out_path.write_text(
            convert_release_report_markdown(
                md,
                archive_download_name=archive_name,
                l2_l3_row_ok=l2_l3_row_ok,
                l2_l3_row_detail=l2_l3_row_detail,
                di_row_ok=di_row_ok,
                di_row_detail=di_row_detail,
                critical_row_ok=critical_row_ok,
                critical_row_detail=critical_row_detail,
                assignee_row_ok=assignee_row_ok,
                assignee_row_detail=assignee_row_detail,
            ),
            encoding="utf-8",
        )
    else:
        out_path.write_text(
            materialize_release_conclusion_in_markdown(
                md,
                l2_l3_row_ok=l2_l3_row_ok,
                l2_l3_row_detail=l2_l3_row_detail,
                di_row_ok=di_row_ok,
                di_row_detail=di_row_detail,
                critical_row_ok=critical_row_ok,
                critical_row_detail=critical_row_detail,
                assignee_row_ok=assignee_row_ok,
                assignee_row_detail=assignee_row_detail,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
