#!/usr/bin/env python3
"""
Compose a full test report (default **HTML**).

Agents and users should **emit HTML by default**; pass ``--format markdown`` only when a Markdown
file is explicitly required (e.g. hand-editing, ``patch_report_*.py``).

Two report kinds (``--kind``):

  - ``release`` (default): full release layout.
      - Test conclusion: interactive checklist (HTML) / static MD; auto rows: "L2&L3…" = latest
        finished **ready** + **merge** (same buckets as metrics) have no failed/broken jobs;
        "critical issues…" = no open ``critical``;
        "Remaining DI…" = open ``bug`` in stats window weighted by priority labels < 30;
        "bug assignees…" = open ``bug`` all have assignee
      - Metrics overview: ``buildkite_build_stats.py --markdown``; **UT coverage** rows
        (``ut`` and ``ut (exclude models)``) are **manual-edit** cells (click to enter value,
        persisted via localStorage), matching the Development variant's editable cell pattern
      - Test Result: Common stack from ``references/local-test-matrix.md``; H200/H800/A100 from
        optional ``--log-dir-h*`` (nightly-style Summary); H100 = Buildkite scheduled nightly
      - Failure Analysis: top-level section with per-GPU (H200/H800/A100 from local logs;
        H100 from Buildkite) collapsible subsections; interactive **Status** column (Filed /
        Not an issue) backed by localStorage, mirroring the Development variant's layout
      - Issue tracking: GitHub Search ``label:ci-failure`` + ``local test`` in:title (stats window)
      - Open issues: GitHub open bugs (``label:bug``); filter ``created_at`` to
        ``--stats-from..--stats-to`` (UTC)

  - ``development``: same as ``release`` but **drops Test conclusion** and **Issue tracking**,
    and replaces **Metrics overview** with a Development-flavored block focused on:
      - Outstanding DI (cumulative DI = sum of priority weights for **all** open ``label:bug``)
      - Open Critical Issue (count of issues with label ``critical`` that are still open)
      - Latest merge CI result (all pass / fail) (Buildkite latest finished merge build, all-pass/fail)
      - All unassigned outstanding issues (table of all open ``label:bug`` with no assignee)

Requires BUILDKITE_TOKEN or BUILDKITE_API_TOKEN in the environment.
Run from skill dir: ``python scripts/compose_full_report.py`` (release) or
``python scripts/compose_full_report.py --kind development`` (Development).
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
    development_report_basename,
    development_report_preview_basename,
    release_report_basename,
    release_report_preview_basename,
    resolve_report_date_iso,
)
from skip_issue_monitor import (  # noqa: E402
    render_skip_issue_monitor_preview_section,
    render_skip_issue_monitor_section,
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

# Buildkite jobs that are pure orchestration / report-collection steps and
# must NEVER count as a real failure or appear in the failure table. They
# run after the test matrix is already complete, so a red status here just
# means "we couldn't package the report" — not "the tests failed". The
# canonical example is ``Nightly Collection&Email`` which collects nightly
# results and sends an email; if SMTP or the kanban is down it shows up as
# ``failed`` / ``broken`` and pollutes the H100 summary.
_NON_REPORTABLE_BK_JOB_NAMES: frozenset[str] = frozenset(
    {
        # Match both ``&`` and ``and`` forms to be robust.
        "nightly collection&email",
        "nightly collection and email",
    }
)

# Default build number used by ``preview_report_markdown`` / ``render_development_report_markdown_preview``
# when no Buildkite call is made. Defined early so development-preview can use it as a kwarg default.
PREVIEW_BUILD_NO = 12880


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


def _github_fetch_open_issues_with_labels(gh_token: str | None, *labels: str) -> list[dict]:
    """Paginate **open** issues that carry **all** of the given GitHub labels (AND filter; PR entries excluded).

    GitHub REST ``GET /repos/.../issues?state=open&labels=a,b`` returns issues with both labels
    attached (commas = logical AND across label names). Pass one or more label names.
    """
    base = "https://api.github.com/repos/vllm-project/vllm-omni/issues"
    lab = urllib.parse.quote(",".join(labels)) if labels else ""
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
    return _github_fetch_open_issues_with_labels(gh_token, "bug")


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
    issue with labels ``bug`` **and** ``critical`` (both required — RFC / Feature tickets carrying
    ``critical`` alone are intentionally excluded).
    """
    try:
        issues = _github_fetch_open_issues_with_labels(gh_token, "bug", "critical")
    except Exception as exc:
        return False, f"Unable to check label bug,critical ({exc})"
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
    return False, f"Open issues with labels **bug** + **critical** still exist: {lst}{tail}"


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


# ---------------------------------------------------------------------------
# Development report helpers (``compose_full_report.py --kind development``)
# ---------------------------------------------------------------------------


def _bug_di_detail_str(total_tenths: int, counts: dict[str, int], n_issues: int) -> str:
    """Human-readable DI detail string shared by DI calculation helpers."""
    parts = [f"{label}={counts[label]}" for label in BUG_DI_LABEL_ORDER if counts.get(label, 0)]
    if counts.get("unclassified", 0):
        parts.append(f"unclassified={counts['unclassified']}")
    detail = ", ".join(parts) if parts else "no open bug"
    return f"Auto Outstanding DI={_format_di_tenths(total_tenths)} ({n_issues} open bugs; {detail})"


def legacy_open_bug_di_total(gh_token: str | None) -> tuple[int | None, str]:
    """
    Sum of priority-label-weighted DI for **all open** ``label:bug`` issues
    (``critical``=10, ``high priority``=3, ``medium priority``=1, ``low priority``=0.1,
    ``invalid``=0). No stats-window filter — this is the **cumulative / legacy** DI
    snapshot for the Development report's Metrics overview row.

    Returns ``(total_tenths, detail_str)``. ``total_tenths`` is ``None`` when the
    GitHub fetch fails; ``detail_str`` is a short, human-readable note shown in the
    Metrics overview cell.
    """
    try:
        issues = _github_fetch_open_bug_issues(gh_token)
    except Exception as exc:
        return None, f"Unable to fetch open bugs ({exc})"
    total_tenths, counts = _bug_di_summary(issues)
    return total_tenths, _bug_di_detail_str(total_tenths, counts, len(issues))


def release_open_bug_di_total(
    gh_token: str | None,
    stats_to: str,
) -> tuple[int | None, str]:
    """
    Sum of priority-label-weighted DI for **open** ``label:bug`` issues whose
    ``created_at`` UTC date is **on or before** ``stats_to`` (``YYYY-MM-DD``).

    Issues created *after* the stats window end are excluded so that the DI
    reflects only the bug backlog that existed during the release period.
    The start date is intentionally unbounded — bugs from any earlier date
    are included as long as they are still open.

    Returns ``(total_tenths, detail_str)``. ``total_tenths`` is ``None`` when the
    GitHub fetch fails.
    """
    try:
        issues = _github_fetch_open_bug_issues(gh_token)
    except Exception as exc:
        return None, f"Unable to fetch open bugs ({exc})"
    filtered = [i for i in issues if (d := _issue_created_date_utc(i)) is not None and d <= stats_to]
    total_tenths, counts = _bug_di_summary(filtered)
    return total_tenths, _bug_di_detail_str(total_tenths, counts, len(filtered))


def open_critical_labeled_issue_count(gh_token: str | None) -> tuple[int | None, str]:
    """
    Count of **open** issues that carry both ``bug`` and ``critical`` labels (excludes PRs).
    RFC / Feature issues tagged only with ``critical`` are intentionally excluded so that the
    Development snapshot's red-alert row reflects genuine critical bugs only.
    Returns ``(count, detail_str)`` — ``count`` is ``None`` when fetch fails.
    """
    try:
        issues = _github_fetch_open_issues_with_labels(gh_token, "bug", "critical")
    except Exception as exc:
        return None, f"Unable to fetch label=bug,critical ({exc})"
    nums: list[int] = []
    for i in issues:
        try:
            nums.append(int(i["number"]))
        except (KeyError, TypeError, ValueError):
            continue
    nums.sort()
    if not nums:
        return 0, "No open issues with labels `bug` + `critical`"
    show = nums[:10]
    tail = f" ({len(nums)} total)" if len(nums) > len(show) else ""
    listing = ", ".join(f"#{n}" for n in show)
    return len(nums), f"Open issues with labels `bug` + `critical`: {listing}{tail}"


def unassigned_open_bug_issue_rows(gh_token: str | None) -> tuple[str, list[dict]]:
    """
    Markdown table of all **open** ``label:bug`` issues that have **no assignee**,
    plus the underlying issue list (so callers can also count them). If GitHub is
    unavailable, the table is an empty placeholder row.
    """
    try:
        all_items = _github_fetch_open_bug_issues(gh_token)
    except Exception as exc:
        return (
            render_markdown_table(
                ["Issue", "Title", "Opened at", "Priority", "DI", "Status"],
                [
                    [
                        "*—*",
                        f"*Failed to fetch open bugs; set `GITHUB_TOKEN`. ({exc})*",
                        "*—*",
                        "*—*",
                        "*—*",
                        "*—*",
                    ]
                ],
            ),
            [],
        )
    unassigned: list[dict] = []
    for i in all_items:
        assignees = i.get("assignees") or []
        if not assignees:
            unassigned.append(i)
    unassigned.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    if not unassigned:
        return (
            render_markdown_table(
                ["Issue", "Title", "Opened at", "Priority", "DI", "Status"],
                [["*—*", "*No unassigned open bugs.*", "*—*", "*—*", "*—*", "*—*"]],
            ),
            [],
        )
    rows: list[list[str]] = []
    for i in unassigned:
        title = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        di_label, di_tenths = _bug_di_label_and_value(i)
        rows.append(
            [
                f"[#{i['number']}](https://github.com/vllm-project/vllm-omni/issues/{i['number']})",
                title,
                str(i.get("created_at") or "")[:10],
                di_label,
                _format_di_tenths(di_tenths),
                "open",
            ]
        )
    return (
        render_markdown_table(
            ["Issue", "Title", "Opened at", "Priority", "DI", "Status"],
            rows,
        ),
        unassigned,
    )


def _dev_alert_cell(value: str, alert: bool) -> str:
    """Wrap a snapshot cell value in ``<span class=\"dev-snapshot-alert\">`` when alert=True.

    In the rendered HTML this is colored red via the CSS rule in
    ``RELEASE_MARKDOWN_DOC_CSS`` (``.release-doc .dev-snapshot-alert``).
    When alert=False the raw value is returned unchanged; the Markdown stays
    plain ``**value**`` (no inline HTML).
    """
    if not alert:
        return value
    return f'<span class="dev-snapshot-alert">{value}</span>'


def render_development_metrics_overview(
    token: str,
    gh_token: str | None,
) -> tuple[str, int, int, list[dict], dict[str, bool]]:
    """
    Markdown body for the **Development** report's ``## Metrics overview`` section.

    Layout (per spec for ``--kind development``):

      A. **Key snapshot table** — 5 rows, each row turns red via
         ``<span class="dev-snapshot-alert">…</span>`` when its threshold is breached:

         | Row label | Alert condition (red) |
         |-----------|-----------------------|
         | Outstanding DI | DI > 30 (i.e. ``total_tenths > BUG_DI_THRESHOLD_TENTHS``) |
         | Open Critical Issue | open critical issues > 0 |
         | merge CI result | latest finished merge build is NOT all-pass |
         | nightly CI result | latest scheduled nightly has any failed/broken reportable job |
         | Unassigned Open Issue | count > 0 |

      B. **Unassigned open bug issues** — full table (all open ``label:bug`` with no assignee).

    Returns ``(markdown, unassigned_count, critical_count, unassigned_issue_list, alerts)``
    where ``alerts`` maps row key → bool (True ⇔ red).
    """
    # 1) Legacy / cumulative DI  (alert when DI > BUG_DI_THRESHOLD_TENTHS, i.e. > 30 displayed)
    di_tenths, di_detail = legacy_open_bug_di_total(gh_token)
    di_alert = bool(di_tenths is not None and di_tenths > BUG_DI_THRESHOLD_TENTHS)
    di_value = f"**{_format_di_tenths(di_tenths)}**" if di_tenths is not None else "*N/A*"
    di_cell = _dev_alert_cell(di_value + (f" — {di_detail}" if di_detail else ""), di_alert)
    if di_tenths is None:
        di_cell = _dev_alert_cell(f"*N/A* ({di_detail})", False)

    # 2) Open critical issues (alert when count > 0)
    crit_n, crit_detail = open_critical_labeled_issue_count(gh_token)
    crit_alert = bool(crit_n and crit_n > 0)
    if crit_n is None:
        crit_value = f"*N/A* ({crit_detail})"
    elif crit_n == 0:
        crit_value = "**0** (no open critical issues)"
    else:
        crit_value = f"**{crit_n}** (open): {crit_detail.split('Open issues with labels `bug` + `critical`: ', 1)[-1]}"
    crit_cell = _dev_alert_cell(crit_value, crit_alert)

    # 3) merge CI result (alert when latest finished merge build is NOT all-pass)
    merge_alert = False
    merge_value = "*N/A*"
    try:
        from buildkite_build_stats import (
            fetch_latest_finished_merge_build,
            summarize_build_all_pass,
        )

        mb = fetch_latest_finished_merge_build(token)
        if mb is None:
            merge_value = "*N/A (no finished merge build found)*"
        else:
            mb_full = mb
            if not mb_full.get("jobs"):
                from buildkite_build_stats import ensure_build_with_jobs

                mb_full = ensure_build_with_jobs(token, mb)
            ok, label, passed, failed = summarize_build_all_pass(mb_full)
            mn = mb_full.get("number")
            web = mb_full.get("web_url") or f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{mn}"
            verdict = "✅ **All pass**" if ok else f"❌ **{label}**"
            merge_value = f"{verdict} — [{mn}]({web}) (passed={passed}, failed={failed})"
            merge_alert = not bool(ok)
    except Exception as exc:
        merge_value = f"*N/A* ({exc})"
    merge_cell = _dev_alert_cell(merge_value, merge_alert)

    # 4) Unassigned Open Issue (alert when count > 0)
    table_md, unassigned_issues = unassigned_open_bug_issue_rows(gh_token)
    unassigned_n = len(unassigned_issues)
    unassigned_alert = unassigned_n > 0
    if unassigned_n:
        unassigned_value = f"**{unassigned_n}** unassigned outstanding issues (see table below)"
    else:
        unassigned_value = "**0** unassigned outstanding issues"
    unassigned_cell = _dev_alert_cell(unassigned_value, unassigned_alert)

    # 5) nightly CI result (alert when any reportable job failed/broken)
    nightly_alert = False
    nightly_value = "*N/A*"
    try:
        nb_no = latest_scheduled_nightly_number(token)
        nb_url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds/{nb_no}"
        nb = http_json(nb_url, token)
        assert isinstance(nb, dict)
        nb_jobs = nb.get("jobs") or []
        nb_reportable = [j for j in nb_jobs if not UPLOAD_PIPELINE_RE.match((j.get("name") or "").strip())]
        nb_states = [(j.get("state") or "").lower() for j in nb_reportable]
        nb_passed = sum(1 for s in nb_states if s == "passed")
        # `broken` is excluded from the nightly-CI failure count, matching
        # the H100 failure-analysis table (which only surfaces `failed`).
        nb_failed = sum(1 for s in nb_states if s == "failed")
        nb_skipped = sum(1 for s in nb_states if s in ("skipped", "not_run", "blocked"))
        nb_total = nb_passed + nb_failed + nb_skipped
        nb_web = nb.get("web_url") or f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{nb_no}"
        nb_ok = nb_failed == 0
        nb_verdict = "✅ **All pass**" if nb_ok else f"❌ **{nb_failed} failed**"
        nightly_value = (
            f"{nb_verdict} — [{nb_no}]({nb_web}) "
            f"(total={nb_total}, passed={nb_passed}, failed={nb_failed}, skipped={nb_skipped})"
        )
        nightly_alert = not nb_ok
    except SystemExit as exc:
        nightly_value = f"*N/A* ({exc})"
    except Exception as exc:
        nightly_value = f"*N/A* ({exc})"
    nightly_cell = _dev_alert_cell(nightly_value, nightly_alert)

    snapshot_header = ["Metric (Development)", "Result"]
    # Build the snapshot as a regular Markdown table so it survives the
    # markdown-to-HTML pass cleanly. The UT coverage row uses the
    # ``<!--UT-CELL-INSERTION-POINT-->`` placeholder, which
    # ``release_md_to_html`` substitutes with the editable raw HTML cell
    # (registered via ``_ut_coverage_submit_script``).
    snapshot_rows = [
        ["**Outstanding DI** (all open `label:bug`, weighted by priority)", di_cell],
        ["**Open Critical Issue** (labels `bug` + `critical`, open)", crit_cell],
        ["**merge CI result** (Buildkite latest `main` non-nightly)", merge_cell],
        ["**nightly CI result** (Buildkite latest scheduled nightly)", nightly_cell],
        ["**Unassigned Open Issue** (open `label:bug` with no assignee)", unassigned_cell],
        [
            "**UT coverage** (Unit Test coverage; click the cell to edit & persist locally)",
            "@@UT_CELL_INSERTION_POINT@@",
        ],
    ]
    snapshot_table = render_markdown_table(snapshot_header, snapshot_rows)

    detail_lines: list[str] = []
    if di_tenths is not None and di_detail:
        detail_lines.append(f"- DI detail: {di_detail}")
    if crit_n:
        detail_lines.append(
            "- Critical list source: [open critical](https://github.com/vllm-project/vllm-omni/issues?q=is%3Aissue+state%3Aopen+label%3Acritical)."
        )
    detail_lines.append(
        "- Merge CI link: [vllm-omni main builds](https://buildkite.com/vllm/vllm-omni/builds?branch=main)."
    )
    detail_lines.append(
        "- Nightly CI link: [vllm-omni scheduled nightly builds]"
        "(https://buildkite.com/vllm/vllm-omni/builds?branch=main)"
        " (filter by `Scheduled nightly build` message)."
    )
    detail_lines.append(
        "- The complete unassigned issue list comes from GitHub REST pagination"
        " `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=bug&per_page=100`."
    )
    detail_lines.append(
        "- Red highlight rules: DI > 30 => red; open critical issue > 0 => red; merge CI not all passing => red;"
        " nightly CI has any failed/broken reportable job => red; Unassigned Open Issue > 0 => red"
        " (see ``references/development-metrics-alerts.md``)."
    )
    snapshot_md = (
        "**Quick Overview (Development report only — red highlight indicates alert)**\n\n"
        f"{snapshot_table}\n\n" + "\n".join(detail_lines)
    )

    section_md = (
        f"## Metrics overview\n\n"
        f"Source: `scripts/compose_full_report.py --kind development`; "
        f"Based on the release report layout, the **Test conclusion** and "
        f"**Issue tracking** sections are removed, and this section is expanded to "
        f"reflect outstanding development-side issues and the latest merge CI status "
        f"(4-row snapshot + full unassigned owner issue table).\n\n"
        f"{snapshot_md}\n\n"
        f"### Unassigned Open Issue (open `label:bug` with no assignee)\n\n"
        f"{table_md}\n"
    )
    alerts = {
        "di": di_alert,
        "critical": crit_alert,
        "merge": merge_alert,
        "nightly": nightly_alert,
        "unassigned": unassigned_alert,
    }
    return (
        section_md,
        unassigned_n,
        (crit_n if crit_n is not None else 0),
        unassigned_issues,
        alerts,
    )


def render_development_report_markdown_preview(
    skill_dir: Path,
    *,
    stats_from: str,
    stats_to: str,
    build_no: int = PREVIEW_BUILD_NO,
) -> str:
    """
    Same layout as the live **development** report, but no network / subprocess calls.
    Used by ``--preview --kind development``.
    """
    demo_link = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#step-demo"

    # Preview snapshot rows are contrived to demonstrate **all five** red-alert conditions,
    # so the rendered preview visibly shows the ``dev-snapshot-alert`` styling for each row.
    snapshot_rows = [
        [
            "**Outstanding DI** (all open `label:bug`, weighted by priority)",
            _dev_alert_cell("**42.3** *(Preview: intentionally > 30, demonstrating red highlight)*", True),
        ],
        [
            "**Open Critical Issue** (labels `bug` + `critical`, open)",
            _dev_alert_cell(
                "**1** open: [#10077](https://github.com/vllm-project/vllm-omni/issues/10077)",
                True,
            ),
        ],
        [
            "**merge CI result** (Buildkite latest `main` non-nightly)",
            _dev_alert_cell(
                f"❌ **Failed (passed=10, failed=2)** — [{build_no}]({demo_link})",
                True,
            ),
        ],
        [
            "**nightly CI result** (Buildkite latest scheduled nightly)",
            _dev_alert_cell(
                f"❌ **1 failed** — [{build_no}]({demo_link}) (total=14, passed=11, failed=1, skipped=2)",
                True,
            ),
        ],
        [
            "**Unassigned Open Issue** (open `label:bug` with no assignee)",
            _dev_alert_cell(
                "**2** unassigned outstanding issues (see table below)",
                True,
            ),
        ],
    ]
    snapshot_table = render_markdown_table(["Metric (Development)", "Result"], snapshot_rows)

    unassigned_table = render_markdown_table(
        ["Issue", "Title", "Opened at", "Priority", "DI", "Status"],
        [
            [
                "[#10055](https://github.com/vllm-project/vllm-omni/issues/10055)",
                "OOM when loading Qwen-Omni with FP8 on 40GB *(example, unassigned)*",
                "2026-05-14",
                "high priority",
                "3",
                "open",
            ],
            [
                "[#10030](https://github.com/vllm-project/vllm-omni/issues/10030)",
                "Docs: wrong env var for TEE cache *(example, unassigned)*",
                "2026-05-10",
                "low priority",
                "0.1",
                "open",
            ],
        ],
    )

    metrics_block = (
        "## Metrics overview\n\n"
        "*This section uses **preview placeholder data**: `compose_full_report.py --kind development` "
        "was not run; values below are layout demos only.*\n\n"
        "**Quick Overview (Development report only — red highlight indicates alert)**\n\n"
        f"{snapshot_table}\n\n"
        "### Unassigned Open Issue (open `label:bug` with no assignee)\n\n"
        f"{unassigned_table}\n"
    )

    # H100 (CI/Buildkite) is intentionally excluded from the development
    # preview. The local-GPU nightly summary layout is shown via placeholder
    # text in the per-GPU panels below.

    # Preview placeholder for per-GPU Performance Data Comparison. Live path
    # passes the real block (computed from `--kanban-repo-root` / `--perf-assets-dir`).
    def _dev_perf_preview_note(gpu: str) -> str:
        return (
            f"#### {gpu}\n\n"
            f"*Preview placeholder for `{gpu}`. Live path reuses the nightly Local "
            "performance baseline comparison (`nightly_local_log_report._buildkite_perf_rows` + "
            "`_filter_perf_summary_for_local`) against kanban `docs/assets/charts/*_history.json`; "
            "**no kanban writes** (no `prepare_kanban_before_report.py`, no `mkdocs build`, no push). "
            "Configure `--kanban-repo-root <vllm-omni-kanban>` (or `--perf-assets-dir`) to populate "
            "this subsection from real data.*\n"
        )

    # 1) Test Result: Overall test execution summary table + per-GPU nightly
    #    summaries. H100 is intentionally excluded from the development variant
    #    (it lives in the Buildkite CI side, not the local nightly log roll-up).
    preview_overall_table = render_overall_test_execution_summary_table(
        log_h200=None,
        log_h800=None,
        log_a100=None,
    )
    test_result = render_test_result_section(
        skill_dir,
        log_h200=None,
        log_h800=None,
        log_a100=None,
        h100_ci_markdown="",
        overall_summary_table_md=preview_overall_table,
    )

    # 2) Failure Analysis: top-level section, one collapsible sub-section per
    #    local GPU. H100 is dropped for the development variant.
    failure_analysis = render_failure_analysis_section(
        log_h200=None,
        log_h800=None,
        log_a100=None,
        include_h100=False,
    )

    # 3) Performance Data Comparison: top-level section, per-GPU sub-folds.
    pdc_section = render_performance_data_comparison_section(
        dev_perf_h200=_dev_perf_preview_note("H200"),
        dev_perf_h800=_dev_perf_preview_note("H800"),
        dev_perf_a100=_dev_perf_preview_note("A100"),
    )

    # 4) Skip Test Case Monitoring: hardcoded preview rows (no git pull, no
    #    AST scan, no GitHub API call). Two rows share one issue number so the
    #    HTML per-issue collapsible grouping is visible in preview mode.
    skip_monitor_preview = render_skip_issue_monitor_preview_section()

    # Open issues (stats window) preview block: same column layout as `release`.
    open_issues_preview = (
        f"## Open issues (stats window)\n\n"
        f"Open issues labeled **bug**, state **open**, excluding PRs, with `created_at` "
        f"(UTC date) in **{stats_from}** … **{stats_to}** (same as Buildkite `--stats-from` / "
        f"`--stats-to`): placeholder preview rows.\n\n"
        + render_markdown_table(
            OPEN_ISSUES_HEADERS,
            [
                [
                    "[#10042](https://github.com/vllm-project/vllm-omni/issues/10042)",
                    "Intermittent timeout on L2 diffusion accuracy *(example)*",
                    stats_to,
                    "medium priority",
                    "1",
                    "open",
                    "@preview-dev-1",
                    *OPEN_ISSUE_ACTION_CELLS,
                ],
                [
                    "[#10018](https://github.com/vllm-project/vllm-omni/issues/10018)",
                    "Regression in test matrix for A100 path *(example)*",
                    stats_from,
                    "high priority",
                    "3",
                    "open",
                    "@preview-dev-2",
                    *OPEN_ISSUE_ACTION_CELLS,
                ],
            ],
        )
        + "\n"
    )

    # Bugfix Monitor preview block: mocked layout so the operator can see
    # the section's two collapsible sub-folds + verdict column.
    from datetime import datetime, timedelta, timezone

    _today = datetime.now(timezone.utc).date()
    _date_from = (_today - timedelta(days=6)).isoformat()
    _date_to = _today.isoformat()
    _date_minus1 = (_today - timedelta(days=1)).isoformat()
    bugfix_monitor_preview = (
        f"## Bugfix Monitor  ({_date_from} → {_date_to}, last 7d)\n\n"
        "*This section uses **preview placeholder data**: `compose_full_report.py "
        "--kind development` was not run; the GitHub fetch was skipped. "
        "Numbers below are layout demos only.*\n\n"
        "Bugfix PRs on `vllm-project/vllm-omni` (matched by title prefix "
        "`[Bugfix]` / `[BugFix]` / `[bugfix]` or label `bug` / `bugfix`). "
        "Each row's **Analysis** column explains whether supplementary test "
        "cases are needed and what kind.\n\n"
        "- Open bugfix PRs: **6** (of which **3** lack tests/)\n"
        "- Closed bugfix PRs: **6** (of which **1** lacks tests/)\n\n"
        f"### Open bugfix PRs (6)\n\n"
        + render_markdown_table(
            ["#", "Title", "Created", "Author", "Analysis"],
            [
                [
                    "[#4950](https://github.com/vllm-project/vllm-omni/pull/4950)",
                    "[Bugfix] Helios Cholesky positive-definite crashes *(example)*",
                    _date_to,
                    "@alice",
                    "Test case needed (Other). Add a minimal regression under `tests/` that "
                    "reproduces the bug from the PR description (or a small script), "
                    "asserting the corrected behavior after the fix.",
                ],
                [
                    "[#4941](https://github.com/vllm-project/vllm-omni/pull/4941)",
                    "[Bugfix] Accept kv_prefetch_jobs in ARDiffusionModelRunner *(example)*",
                    _date_to,
                    "@bob",
                    "Already covered — all or most of the 3 changed files are in `tests/`. No new test case needed.",
                ],
                [
                    "[#4928](https://github.com/vllm-project/vllm-omni/pull/4928)",
                    "[Bugfix][Qwen3-Omni]Repair async Code2Wav streaming chunk boundary sample loss *(example)*",
                    _date_to,
                    "@carol",
                    "Test case needed (Async/Streaming). Add a streaming unit test under "
                    "`tests/core/sched/` that simulates chunk boundary / preemption, "
                    "asserting sample continuity and that the prefix cache is not corrupted after the fix.",
                ],
            ],
        )
        + "\n\n"
        "### Closed bugfix PRs (6)\n\n"
        + render_markdown_table(
            ["#", "Title", "Created", "Author", "Analysis"],
            [
                [
                    "[#4910](https://github.com/vllm-project/vllm-omni/pull/4910)",
                    "[Bugfix] Fix full-payload mm splitting for dual hidden/scheduled batch axes *(example)*",
                    _date_to,
                    "@dave",
                    "Partially covered — 1/2 files are in `tests/`, the rest are source changes. "
                    "Suggest reviewing edge cases on the Other path (e.g. error inputs / "
                    "concurrency / numerical extremes) to make sure nothing slipped through.",
                ],
                [
                    "[#4881](https://github.com/vllm-project/vllm-omni/pull/4881)",
                    "[Bugfix] Sync JoyVL interaction layer with upstream reference fixes *(example)*",
                    _date_minus1,
                    "@eve",
                    "Partially covered — 2/8 files are in `tests/`, the rest are source changes. "
                    "Suggest reviewing edge cases on the Other path (e.g. error inputs / "
                    "concurrency / numerical extremes) to make sure nothing slipped through.",
                ],
            ],
        )
        + "\n"
    )

    return f"""# vLLM-Omni Test Report - Development (Preview)

{metrics_block}

{test_result}

{failure_analysis}

{pdc_section}
{skip_monitor_preview}
{open_issues_preview}

{bugfix_monitor_preview}
## Data source

- **Mode:** `compose_full_report.py --preview --kind development` (sample tables only)
- **Test Result:** Overall test execution summary (Total / Passed / Failed across H200 /
   H800 / A100; Failed cell links to matching Failure Analysis subsection) + per-GPU
   nightly summaries. **H100 is excluded** in the development variant.
- **Failure Analysis:** Top-level section; one collapsible subsection per GPU. Mirrors
   the original failure-analysis pattern (per-job Failures & errors table; H100 lists
   failed Buildkite steps).
- **Performance Data Comparison:** Top-level section; read-only against kanban
   `docs/assets/charts/*_history.json` (no kanban writes).
- **Skip Test Case Monitoring:** top-level section; one hardcoded 5-row preview
   (no AST scan, no `git pull`, no GitHub API call). Two preview rows share one
   issue number so the per-issue collapsible grouping is visible.
- **Metrics overview:** Buildkite latest finished merge build
   (`buildkite_build_stats.fetch_latest_finished_merge_build`) + GitHub REST
   (`label:bug`, `label:bug+critical` AND filter, assignee scan). 4-row snapshot; each
   row turns red via `<span class="dev-snapshot-alert">` when threshold breached.
- **Open issues (stats window):** Same as `release` —
   `compose_full_report.render_open_issues_section(stats_from, stats_to, gh_token)`; the
   **Follow-up action** dropdown + **Remarks** note columns are interactive in HTML
   (localStorage, keyed by issue number).
- Live report: `buildkite_build_stats`, GitHub REST
"""


#: Column layout of the ``## Open issues`` table (release *and* development).
#: The last two columns are **manual-entry** cells: in HTML they are upgraded by
#: ``release_md_to_html._upgrade_open_issue_action_cells`` into a ``<select>``
#: (Follow-up action) and a click-to-edit note box (Remarks), both persisted in
#: ``localStorage`` keyed by the row's issue number. In Markdown they stay ``—``.
OPEN_ISSUES_HEADERS: list[str] = [
    "Issue",
    "Title",
    "Opened at",
    "Priority",
    "DI",
    "Status",
    "Owner",
    "Follow-up action",
    "Remarks",
]

#: Placeholder cells for the two manual-entry columns above.
OPEN_ISSUE_ACTION_CELLS: list[str] = ["—", "—"]


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
                *OPEN_ISSUE_ACTION_CELLS,
            ]
        )
    body = render_markdown_table(
        OPEN_ISSUES_HEADERS,
        row_cells,
    )
    return len(all_items), len(in_range), body, in_range


def github_open_bug_issues_all(gh_token: str | None) -> list[dict]:
    """Paginate every **open** ``label:bug`` issue in ``vllm-project/vllm-omni`` (PRs excluded).

    Used by ``render_open_issues_section_with_di(..., all_open=True)`` for the
    Development variant so the report owner sees the full backlog, not just the
    month-to-date slice.
    """
    return _github_fetch_open_bug_issues(gh_token)


def github_open_bug_issue_rows(issues: list[dict]) -> str:
    """Render the per-issue Markdown table shared by all-open and stats-window variants."""
    sorted_items = sorted(issues, key=lambda x: x.get("created_at") or "", reverse=True)
    row_cells: list[list[str]] = []
    for i in sorted_items:
        t = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        u = (i.get("user") or {}).get("login", "")
        di_label, di_tenths = _bug_di_label_and_value(i)
        row_cells.append(
            [
                f"[#{i['number']}](https://github.com/vllm-project/vllm-omni/issues/{i['number']})",
                t,
                str(i.get("created_at") or "")[:10],
                di_label,
                _format_di_tenths(di_tenths),
                "open",
                f"@{u}",
                *OPEN_ISSUE_ACTION_CELLS,
            ]
        )
    return render_markdown_table(
        OPEN_ISSUES_HEADERS,
        row_cells,
    )


def render_open_issues_section_with_di(
    stats_from: str,
    stats_to: str,
    gh_token: str | None,
    *,
    all_open: bool = False,
) -> tuple[str, bool | None, str]:
    """Markdown for ``## Open issues`` plus DI conclusion data when GitHub fetch succeeds.

    When ``all_open=True``, the rendered section lists **every** open ``label:bug``
    issue in the ``vllm-project/vllm-omni`` repository (no ``created_at`` window
    filter) — used by the Development variant of ``compose_full_report.py`` so
    the report owner sees the whole backlog, not just the month-to-date slice.
    When ``all_open=False`` (default; release variant), the table is restricted to
    issues whose ``created_at`` UTC date falls in ``stats_from``..``stats_to``.
    """
    github_open_error = ""
    di_row_ok: bool | None = None
    di_row_detail = ""
    try:
        if all_open:
            # Paginate every open `label:bug` issue, no `created_at` filter.
            issues_all = github_open_bug_issues_all(gh_token)
            open_total = len(issues_all)
            open_range_n = open_total
            issues_in_range = issues_all
            issue_rows = github_open_bug_issue_rows(issues_all)
        else:
            open_total, open_range_n, issue_rows, issues_in_range = github_open_bug_rows_in_range(
                gh_token, stats_from, stats_to
            )
        di_row_ok, di_row_detail = _bug_di_conclusion(issues_in_range)
    except Exception as exc:
        open_total = 0
        open_range_n = 0
        issue_rows = render_markdown_table(
            OPEN_ISSUES_HEADERS,
            [
                [
                    "*—*",
                    "*Failed to fetch; set `GITHUB_TOKEN` or fill in manually*",
                    "*—*",
                    "*—*",
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
    if all_open:
        heading = "## Open issues (all open in repo)"
        body_intro = (
            f"Open issues labeled **bug**, state **open**, excluding PRs — every open "
            f"bug in [`vllm-project/vllm-omni`]("
            f"https://github.com/vllm-project/vllm-omni/issues"
            f"?q=is%3Aissue+state%3Aopen+label%3Abug) "
            f"at report time (no `created_at` window filter): **{open_range_n}**. DI uses priority "
            f"labels: `critical` = 10, `high priority` = 3, `medium priority` = 1, `low priority` = 0.1, "
            f"`invalid` = 0.{github_open_note}"
        )
    else:
        heading = "## Open issues (stats window)"
        body_intro = (
            f"Open issues labeled **bug**, state **open**, excluding PRs, with `created_at` "
            f"(UTC date) in **{stats_from}** … **{stats_to}** (same as Buildkite `--stats-from` / "
            f"`--stats-to`): **{open_range_n}** (total open `bug` issues when fetched: "
            f"**{open_total}**). DI uses priority labels: `critical` = 10, `high priority` = 3, "
            f"`medium priority` = 1, `low priority` = 0.1, `invalid` = 0.{github_open_note}"
        )
    return (
        (f"{heading}\n\n{body_intro}\n\n{issue_rows}\n"),
        di_row_ok,
        di_row_detail,
    )


def render_open_issues_section(
    stats_from: str,
    stats_to: str,
    gh_token: str | None,
    *,
    all_open: bool = False,
) -> str:
    """Markdown for ``## Open issues`` block (GitHub REST, open ``label:bug`` only)."""
    section, _, _ = render_open_issues_section_with_di(stats_from, stats_to, gh_token, all_open=all_open)
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


def render_bugfix_monitor_section(
    gh_token: str | None,
    *,
    days_back: int = 7,
    max_prs: int = 200,
) -> str:
    """Render the **Bugfix Monitor** section for the development report.

    Lists every Bugfix PR (by ``[Bugfix]`` / ``[BugFix]`` / ``[bugfix]`` title
    prefix **or** the ``bug`` / ``bugfix`` label) on
    https://github.com/vllm-project/vllm-omni created in the last ``days_back``
    days. Each PR row carries an **Analysis** cell explaining either:

      - **why no new test case is needed** (e.g. the fix only updates a
        constant, a comment, a build-time config, or a deprecated path; or
        the test is upstream in the framework and an offline manual smoke
        is acceptable), **or**
      - **what kind of test should be added** (regression / accuracy /
        unit / e2e), inferred from the PR area (TTS / diffusion / frontend /
        deployment / numerical), with a one-line suggestion.

    Sub-sections (Open / Closed) are emitted as ``### h3`` headings so
    :func:`_wrap_bugfix_monitor_h3_in_details` (in
    ``release_md_to_html.py``) can convert them to ``<details>`` cards. On any
    GitHub error the section falls back to a single line noting the failure
    so the report still renders.
    """
    from datetime import datetime, timedelta, timezone

    import requests as _req

    # Re-use the same TLS-verify rule as the rest of the skill.
    try:
        from buildkite_build_stats import _github_tls_verify as _tls_verify

        _verify = _tls_verify()
    except Exception:
        _verify = True

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-test-report-bugfix-monitor",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    today = datetime.now(timezone.utc).date()
    date_from = (today - timedelta(days=days_back - 1)).isoformat()
    date_to = today.isoformat()

    def _search(q: str) -> list[dict]:
        out: list[dict] = []
        for page in range(1, 6):
            try:
                r = _req.get(
                    "https://api.github.com/search/issues",
                    params={"q": q, "per_page": 100, "page": page, "advanced_search": "true"},
                    headers=headers,
                    timeout=60,
                    verify=_verify,
                )
            except Exception:
                return out
            if r.status_code == 403:
                return out
            if r.status_code != 200:
                return out
            batch = r.json().get("items") or []
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
        return out

    q_open = f"repo:vllm-project/vllm-omni is:pr is:open created:{date_from}..{date_to}"
    q_closed = f"repo:vllm-project/vllm-omni is:pr is:closed created:{date_from}..{date_to}"
    open_items = _search(q_open)
    closed_items = _search(q_closed)

    def _looks_bugfix(it: dict) -> bool:
        labels = {label["name"].lower() for label in (it.get("labels") or [])}
        if "bug" in labels or "bugfix" in labels:
            return True
        title = (it.get("title") or "").lower()
        return "[bugfix]" in title or "[bug fix]" in title or title.startswith("bugfix")

    def _shape_pr(it: dict) -> dict:
        title = (it.get("title") or "").strip()
        n = it["number"]
        state = (it.get("state") or "").lower()
        created = (it.get("created_at") or "")[:10]
        user = (it.get("user") or {}).get("login", "?")
        labels = [label["name"] for label in (it.get("labels") or [])]
        url = it.get("html_url") or f"https://github.com/vllm-project/vllm-omni/pull/{n}"
        return {"n": n, "title": title, "state": state, "created": created, "user": user, "labels": labels, "url": url}

    def _area_from_title(title: str) -> str:
        """Infer which test area this PR belongs to from the title."""
        t = title.lower()
        if any(
            k in t
            for k in [
                "tts",
                "cosyvoice",
                "voxcpm",
                "voxtral",
                "higgs",
                "moss-tts",
                "ming-tts",
                "aura",
                "speech",
                "audio",
            ]
        ):
            return "TTS"
        if any(
            k in t
            for k in [
                "hunyuan",
                "qwen-image",
                "wan",
                "cosmos",
                "bagel",
                "joyvl",
                "krea",
                "mammothmoda",
                "ominivoice",
                "helios",
                "ltx",
                "diffusion",
                "imag",
                "video",
                "magi",
            ]
        ):
            return "Diffusion/Image/Video"
        if any(
            k in t
            for k in [
                "v1/chat",
                "v1/audio",
                "v1/image",
                "endpoint",
                "request",
                "validator",
                "logprobs",
                "modalit",
                "prompt",
                "input",
                "openai",
            ]
        ):
            return "API/Frontend"
        if any(k in t for k in ["deploy", "stage", "multi-replica", "engine_extras", "connector", "final_stage"]):
            return "Deploy/Stage"
        if any(
            k in t
            for k in [
                "async",
                "chunk",
                "preempt",
                "resume",
                "replay",
                "mtp",
                "sampling",
                "talker",
                "token",
                "kv cache",
                "prefix cache",
            ]
        ):
            return "Async/Streaming"
        if any(k in t for k in ["tensor parallel", "hspd", "usp", "cfg parallel", "shard", "lapis"]):
            return "Distributed/Parallel"
        if any(k in t for k in ["np ", "npu", "ascend", "310p"]):
            return "NPU/Ascend"
        if any(k in t for k in ["fp8", "nvfp4", "quack", "quantization"]):
            return "Quantization"
        return "Other"

    def _analysis(pr: dict) -> str:
        """Build the per-PR Analysis text.

        Logic:
          - 0 tests/ files added
            - If all changes are docs/ comments/ ci-config/ build-only → no
              test case needed; explain why.
            - Otherwise → suggest a test type keyed off the PR area.
          - 1+ tests/ files added
            - If only test files changed (and PR is small) → already covered.
            - If test files mix with non-test changes (config / refactor /
              build) → partial coverage; suggest the missing test type.
        """
        title = pr["title"]
        n_test = pr["n_test_files"]
        n_total = pr["n_total_files"]
        non_test_paths = pr.get("non_test_paths", [])
        area = _area_from_title(title)

        # Categorize the non-test changes for explanation text
        def _cat(p: str) -> str:
            if p.startswith("docs/"):
                return "docs"
            if p.startswith(".github/"):
                return "ci-config"
            if "/test_" in p and p.endswith(".py"):
                return "test-source"  # the file path itself is test-related code
            if p.startswith("examples/") or p.startswith("tools/"):
                return "example/tool"
            if p.startswith("docker/") or "Dockerfile" in p:
                return "docker"
            if p.startswith("scripts/"):
                return "script"
            if p.endswith(".md") or p.endswith(".rst"):
                return "docfile"
            return "source"

        non_test_cats = {_cat(p) for p in non_test_paths}
        only_meta = non_test_cats <= {"docs", "docfile", "ci-config", "docker", "example/tool", "script"}

        if n_test == 0:
            if only_meta:
                return (
                    f"No new test case needed — changes are limited to "
                    f"{', '.join(sorted(non_test_cats))} only; no functional/logic change, "
                    f"no regression risk."
                )
            # Suggest a test by area
            suggestions = {
                "TTS": (
                    "Add a unit test under `tests/model_executor/models/<model>/` that "
                    "reproduces the originally failing input (e.g. varying batch size / "
                    "speaker-embedding boundary), asserting the talker no longer crashes after the fix."
                ),
                "Diffusion/Image/Video": (
                    "Add an e2e under `tests/diffusion/` that runs the failing config "
                    "(seed / prompt / resolution / steps), asserting the generated image / "
                    "video stays within the accuracy threshold of the baseline after the fix."
                ),
                "API/Frontend": (
                    "Add an e2e under `tests/entrypoints/openai_api/` that sends the "
                    "rejected payload from the PR description (empty prompt / invalid "
                    "modality / out-of-range value), asserting the endpoint returns 422 "
                    "instead of 500 after the fix."
                ),
                "Deploy/Stage": (
                    "Add a multi-replica deploy smoke under `tests/deploy/` or "
                    "`tests/e2e/online_serving/` that exercises the stage-identity path, "
                    "asserting the stage_id is preserved across stages after the fix."
                ),
                "Async/Streaming": (
                    "Add a streaming unit test under `tests/core/sched/` that simulates "
                    "chunk boundary / preemption, asserting sample continuity and that the "
                    "prefix cache is not corrupted after the fix."
                ),
                "Distributed/Parallel": (
                    "Add a TP/HSDP/CFG-Parallel unit test under `tests/distributed/` that "
                    "enables the relevant parallel config and runs the originally failing "
                    "tensor shape, asserting no shape errors after the fix."
                ),
                "NPU/Ascend": (
                    "Extend an existing NPU e2e (or add a 310P/Ascend-targeted test, skip "
                    "if hardware unavailable) under `tests/`, asserting the affected op no "
                    "longer crashes on the NPU backend after the fix."
                ),
                "Quantization": (
                    "Add an accuracy / performance regression under `tests/quantization/` "
                    "for the fp8/nvfp4 path under batched serving, asserting numerical "
                    "consistency after the fix."
                ),
                "Other": (
                    "Add a minimal regression under `tests/` that reproduces the bug from "
                    "the PR description (or a small script), asserting the corrected behavior "
                    "after the fix."
                ),
            }
            return f"Test case needed ({area}). " + suggestions.get(area, suggestions["Other"])

        # has at least 1 test/ file
        if n_test >= max(1, n_total // 2) and not non_test_paths:
            return (
                f"Already covered — all or most of the {n_total} changed files are in `tests/`. "
                f"No new test case needed."
            )
        if non_test_cats <= {"docs", "docfile", "ci-config", "docker", "example/tool", "script"}:
            return (
                f"Already covered — {n_test}/{n_total} files are in `tests/`, the rest are docs / CI config. "
                f"No new test case needed."
            )
        # Mixed: tests + non-trivial source changes
        return (
            f"Partially covered — {n_test}/{n_total} files are in `tests/`, "
            f"the rest are source changes. Suggest reviewing edge cases on the "
            f"{area} path (e.g. error inputs / concurrency / numerical extremes) "
            f"to make sure nothing slipped through."
        )

    def _enrich(pr: dict) -> dict:
        # Fetch files list to compute the test-coverage verdict.
        try:
            r = _req.get(
                f"https://api.github.com/repos/vllm-project/vllm-omni/pulls/{pr['n']}/files",
                params={"per_page": 100},
                headers=headers,
                timeout=60,
                verify=_verify,
            )
        except Exception:
            r = None
        test_files: list[str] = []
        non_test_paths: list[str] = []
        if r is not None and r.status_code == 200:
            for f in r.json() or []:
                p = f.get("filename") or ""
                if p.startswith("tests/") or "/tests/" in p:
                    test_files.append(p)
                else:
                    non_test_paths.append(p)
        pr["test_files"] = test_files
        pr["non_test_paths"] = non_test_paths
        pr["n_test_files"] = len(test_files)
        pr["n_total_files"] = len(test_files) + len(non_test_paths)
        pr["analysis"] = _analysis(pr)
        return pr

    def _to_row(pr: dict) -> list[str]:
        title_safe = pr["title"].replace("|", "\\|")
        # analysis may itself contain pipes — escape them
        analysis_safe = pr["analysis"].replace("|", "\\|")
        return [
            f"[#{pr['n']}]({pr['url']})",
            title_safe,
            pr["created"],
            pr["user"],
            analysis_safe,
        ]

    open_prs = [_shape_pr(it) for it in open_items if _looks_bugfix(it)]
    closed_prs = [_shape_pr(it) for it in closed_items if _looks_bugfix(it)]
    # Newest first
    open_prs.sort(key=lambda p: p["n"], reverse=True)
    closed_prs.sort(key=lambda p: p["n"], reverse=True)
    if max_prs:
        open_prs = open_prs[:max_prs]
        closed_prs = closed_prs[:max_prs]

    if not open_prs and not closed_prs and not open_items and not closed_items:
        return (
            "## Bugfix Monitor\n\n"
            f"_No data — GitHub search returned no results for window "
            f"{date_from}..{date_to}. Check the BUILDKITE_API_TOKEN / GITHUB_TOKEN env vars._\n"
        )

    # Enrich a limited subset to keep GitHub API usage modest.
    enriched_open = [_enrich(p) for p in open_prs]
    enriched_closed = [_enrich(p) for p in closed_prs]

    def _render_table(prs):
        if not prs:
            return "_None._"
        rows = [_to_row(p) for p in prs]
        return render_markdown_table(
            ["#", "Title", "Created", "Author", "Analysis"],
            rows,
        )

    open_table = _render_table(enriched_open)
    closed_table = _render_table(enriched_closed)

    n_open_total = len(enriched_open)
    n_closed_total = len(enriched_closed)
    n_open_no_tests = sum(1 for p in enriched_open if p["n_test_files"] == 0)
    n_closed_no_tests = sum(1 for p in enriched_closed if p["n_test_files"] == 0)

    header = (
        f"## Bugfix Monitor  ({date_from} → {date_to}, last {days_back}d)\n\n"
        f"Bugfix PRs on `vllm-project/vllm-omni` (matched by title prefix "
        f"`[Bugfix]` / `[BugFix]` / `[bugfix]` or label `bug` / `bugfix`). "
        f"Each row's **Analysis** column explains whether supplementary test "
        f"cases are needed and what kind.\n\n"
        f"- Open bugfix PRs: **{n_open_total}** (of which **{n_open_no_tests}** lack tests/)\n"
        f"- Closed bugfix PRs: **{n_closed_total}** (of which **{n_closed_no_tests}** lack tests/)\n\n"
    )

    # The two sub-sections are emitted as ``### h3`` headings so the
    # markdown→HTML converter can wrap them as ``<details>`` cards (see
    # ``_wrap_bugfix_monitor_h3_in_details`` in release_md_to_html.py).
    return (
        header
        + f"### Open bugfix PRs ({n_open_total})\n\n"
        + open_table
        + "\n\n"
        + f"### Closed bugfix PRs ({n_closed_total})\n\n"
        + closed_table
        + "\n"
    )


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


def _render_local_gpu_failure_section(
    gpu: str,
    log_dir: Path,
) -> str:
    """Render a per-GPU failure-analysis block in the style of nightly_local_log_report.

    Reuses ``discover_job_logs`` + ``_local_job_rows_with_info`` + ``_job_is_clean`` from
    ``nightly_local_log_report`` to obtain the parsed ``info`` dict (failed_nodes,
    failed_reasons, failure_analyses, failure_excerpts) and emits a Markdown
    ``#### {gpu} failures`` heading + ``Failures & errors`` table. Mirrors the
    ``nightly_local_log_report._append_local_summary_failure_markdown`` pattern.
    """
    try:
        from nightly_local_log_report import (  # local import: keep top-level deps lean
            _excerpt_md_cell,
            _job_is_clean,
            _local_job_rows_with_info,
            _md_cell,
            discover_job_logs,
            render_markdown_table,
        )
    except Exception as exc:
        return (
            f"#### {gpu} failures\n\n"
            f"*Failure-analysis helpers unavailable (`{exc}`). "
            "Run from the skill directory so `nightly_local_log_report.py` is importable.*\n"
        )

    groups = discover_job_logs(log_dir)
    if not groups:
        return (
            f'<a id="failure-analysis-{gpu.lower()}"></a>\n'
            f"#### {gpu} failures\n\n"
            f"*No job logs found under `{log_dir}`.*\n"
        )

    job_rows = _local_job_rows_with_info(groups)
    failed_rows = [(name, paths, info) for name, paths, info in job_rows if not _job_is_clean(info)]
    if not failed_rows:
        return (
            f'<a id="failure-analysis-{gpu.lower()}"></a>\n'
            f"#### {gpu} failures\n\n"
            f"*No failed or errored jobs in `{log_dir}`.*\n"
        )

    chunks: list[str] = [
        f'<a id="failure-analysis-{gpu.lower()}"></a>',
        f"#### {gpu} failures",
        "",
        f"Log root: `{log_dir}` — {len(failed_rows)} failed/errored job(s) out of {len(job_rows)} total.",
        "",
    ]
    for job_name, paths, info in failed_rows:
        chunks.append(f"##### Local job: `{_md_cell(job_name)}`")
        chunks.append("")
        rel = ", ".join(f"`{p.name}`" for p in paths)
        chunks.append(f"- Log files: {rel}")
        chunks.append("")
        fail_rows: list[list[str]] = []
        for node in info["failed_nodes"]:
            fail_rows.append(
                [
                    _md_cell(node),
                    _md_cell(info["failed_reasons"].get(node, "")),
                    _md_cell(info["failure_analyses"].get(node, "")),
                    _excerpt_md_cell(info["failure_excerpts"].get(node, "")),
                    "Submit issue",
                    "Filed / Not an issue",
                ]
            )
        for node in info["error_nodes"]:
            fail_rows.append(
                [
                    _md_cell(node) + " (ERROR)",
                    _md_cell(info["error_reasons"].get(node, "")),
                    _md_cell(info["error_analyses"].get(node, "")),
                    _excerpt_md_cell(info["error_excerpts"].get(node, "")),
                    "Submit issue",
                    "Filed / Not an issue",
                ]
            )
        chunks.append("###### Failures & errors")
        chunks.append("")
        chunks.append(
            render_markdown_table(
                ["Test node", "Log reason", "Analysis", "Excerpt (truncated)", "Submit Issue", "Status"],
                fail_rows,
            )
        )
        chunks.append("")
    return "\n".join(chunks)


def _render_local_gpu_job_counts(log_dir) -> tuple:
    """Return ``(total_jobs, failed_jobs)`` for one local GPU's log dir.

    Mirrors :func:`_render_local_gpu_failure_section` gating so the **Execution
    Results** summary table matches the failure analysis section. Returns
    ``(0, 0)`` if the directory is missing, unreadable, or job-row helpers are
    unavailable.
    """
    from pathlib import Path

    if not log_dir or not log_dir.exists() or not any(Path(log_dir).iterdir()):
        return 0, 0
    try:
        from nightly_local_log_report import (
            _job_is_clean,
            _local_job_rows_with_info,
            discover_job_logs,
        )
    except Exception:
        return 0, 0
    groups = discover_job_logs(Path(log_dir))
    if not groups:
        return 0, 0
    job_rows = _local_job_rows_with_info(groups)
    failed = sum(1 for _name, _paths, info in job_rows if not _job_is_clean(info))
    return len(job_rows), failed


def render_per_gpu_summary_table(
    *,
    gpu: str,
    anchor_id: str,
    total: int | str,
    passed: int | str,
    failed: int | str,
) -> str:
    """Return Markdown for a single GPU's Total/Passed/Failed summary table.

    The *Failed* column links to ``#{anchor_id}`` (one of the IDs added by the
    failure-analysis helpers: ``failure-analysis-h200`` etc.). When *total* /
    *passed* / *failed* are integers the row reads ``N / N / [N](#anchor)``;
    pass ``"—"`` (str) when the data is unavailable and the link still renders
    as a non-navigating anchor.
    """
    from nightly_local_log_report import render_markdown_table

    def _fmt(value: int | str) -> str:
        if isinstance(value, int):
            return str(value)
        return value or "—"

    failed_cell = f"[{_fmt(failed)}](#{anchor_id})"
    return render_markdown_table(
        ["Total cases", "Passed", "Failed"],
        [[_fmt(total), _fmt(passed), failed_cell]],
    )


def render_execution_results_section(
    *,
    log_h200,
    log_h800,
    log_a100,
    h100_passed: int,
    h100_failed: int,
    h100_skipped: int,
    h100_ci_markdown: str,
) -> str:
    """Emit the **## Execution Results** section with one panel per GPU.

    Each ``### HXXX`` heading is followed by a per-GPU **Total / Passed / Failed**
    summary table whose *Failed* cell links to the corresponding Failure Analysis
    anchor. For local GPUs (H200/H800/A100) the counts come from ``discover_job_logs``
    via :func:`_render_local_gpu_job_counts`; for H100 the counts are the
    Buildkite reportable-jobs bucketing already collected in the live path.

    Layout mirrors the previous Test Result list (Common stack + per-GPU nightly
    Summary + H100) but lives in its own top-level section so the Test Result
    section can stay focused on **Common stack + Failure Analysis** for the
    development variant.
    """
    # _gpu_log_placeholder is defined at module scope in compose_full_report.py
    pass

    chunks: list[str] = [
        "## Execution Results",
        "",
        "Per-GPU nightly job logs and their aggregate **Total / Passed / Failed** counters. "
        "Click the *Failed* number to jump straight to the corresponding subsection in "
        "[Failure Analysis](#failure-analysis).",
        "",
    ]

    panels = [
        (
            "H200",
            "failure-analysis-h200",
            log_h200,
        ),
        (
            "H800",
            "failure-analysis-h800",
            log_h800,
        ),
        (
            "A100",
            "failure-analysis-a100",
            log_a100,
        ),
    ]
    for gpu, anchor_id, log_dir in panels:
        chunks.extend(["", f"### {gpu}", ""])
        if log_dir:
            total, failed_count = _render_local_gpu_job_counts(log_dir)
            passed = max(total - failed_count, 0)
            chunks.append(
                render_per_gpu_summary_table(
                    gpu=gpu,
                    anchor_id=anchor_id,
                    total=total,
                    passed=passed,
                    failed=failed_count,
                )
            )
            chunks.append("")
            chunks.append(markdown_local_summary_from_log_dir(log_dir))
        else:
            chunks.append(
                render_per_gpu_summary_table(
                    gpu=gpu,
                    anchor_id=anchor_id,
                    total="—",
                    passed="—",
                    failed="—",
                )
            )
            chunks.append("")
            chunks.append(_gpu_log_placeholder(f"--log-dir-{gpu.lower()}"))

    # H100 Buildkite summary table (passed/failed Buildkite reportable jobs)
    chunks.extend(
        [
            "",
            "### H100 (CI — Buildkite scheduled nightly)",
            "",
        ]
    )
    chunks.append(
        render_per_gpu_summary_table(
            gpu="H100",
            anchor_id="failure-analysis-h100",
            total=h100_passed + h100_failed + h100_skipped,
            passed=h100_passed,
            failed=h100_failed,
        )
    )
    chunks.append("")
    chunks.append(h100_ci_markdown.rstrip())
    chunks.append("")
    return "\n".join(chunks)


def render_performance_data_comparison_section(
    *,
    dev_perf_h200: str | None,
    dev_perf_h800: str | None,
    dev_perf_a100: str | None,
) -> str:
    """Top-level **## Performance Data Comparison** section.

    Aggregates the per-GPU ``#### Performance Data Comparison`` blocks
    produced by :func:`render_dev_perf_baseline_local_md` into a single section
    so the layout sits at the same level as Failure Analysis and Execution
    Results. Returns ``""`` if every input is ``None``.
    """
    chunks: list[str] = []
    for gpu, block in (
        ("H200", dev_perf_h200),
        ("H800", dev_perf_h800),
        ("A100", dev_perf_a100),
    ):
        if block:
            chunks.append(block.rstrip())
            chunks.append("")
    if not chunks:
        return ""
    intro = (
        "## Performance Data Comparison\n\n"
        "Per-GPU performance baseline comparison against kanban "
        "`docs/assets/charts/*_history.json`. Purely **read-only** — no "
        "`prepare_kanban_before_report.py`, no `mkdocs build`, no push.\n"
    )
    return intro + "\n".join(chunks)


def _render_buildkite_step_failure_section(
    *,
    build_no: int | None,
    build_url: str | None,
    failed_steps: list[tuple[str, str, str]],
) -> str:
    """Render a compact H100 (Buildkite) failed-step block for the Failure Analysis section.

    ``failed_steps`` is a list of ``(name, state, step_link)`` tuples — only Buildkite
    steps in ``failed`` state (``broken`` is intentionally excluded; it's a transient
    pipeline-execution state, not a real test failure). Step log parsing
    (failed_nodes / reasons / analysis) is left to the dedicated nightly Buildkite
    pipeline and intentionally omitted here to keep the Summary read-only against
    cached build metadata.
    """
    if not failed_steps:
        return (
            '<a id="failure-analysis-h100"></a>\n'
            "#### H100 (CI — Buildkite scheduled nightly) failures\n\n"
            "*No failed Buildkite steps in the latest finished scheduled nightly.*\n"
        )
    chunks: list[str] = [
        '<a id="failure-analysis-h100"></a>',
        "#### H100 (CI — Buildkite scheduled nightly) failures",
        "",
    ]
    if build_no is not None and build_url:
        chunks.append(f"- Latest finished scheduled nightly: build [{build_no}]({build_url}).")
        chunks.append("")
    chunks.append(f"{len(failed_steps)} failed step(s):")
    chunks.append("")
    rows: list[list[str]] = []
    for name, state, link in failed_steps:
        rows.append([name, state, f"[open]({link})" if link else "—", "—", "Filed / Not an issue"])
    chunks.append(
        render_markdown_table(
            ["Step / Job", "State", "Step link", "Submit Issue", "Status"],
            rows,
        )
    )
    chunks.append("")
    return "\n".join(chunks)


def render_failure_summary_md(
    *,
    log_h200: Path | None,
    log_h800: Path | None,
    log_a100: Path | None,
    h100_build_no: int | None,
    h100_build_url: str | None,
    h100_failed_steps: list[tuple[str, str, str]],
) -> str:
    """Aggregate failed jobs from each GPU into a single ``### Failure Analysis`` Markdown block.

    Mirrors the failure-analysis pattern of ``nightly_local_log_report.py`` (per-job
    ``Failures & errors`` table with ``Test node | Log reason | Analysis | Excerpt``).
    Local GPUs (H200/H800/A100) reuse the parsed ``info`` dict via
    ``nightly_local_log_report._local_job_rows_with_info``; H100 (Buildkite) lists
    failed/broken steps with their Buildkite step link (raw-log parsing is left to the
    nightly Buildkite pipeline to keep this summary cheap).
    """
    return "### Failure Analysis\n\n" + _render_failure_summary_blocks(
        log_h200=log_h200,
        log_h800=log_h800,
        log_a100=log_a100,
        h100_build_no=h100_build_no,
        h100_build_url=h100_build_url,
        h100_failed_steps=h100_failed_steps,
    )


def _render_local_gpu_failure_placeholder(gpu: str) -> str:
    """Render an empty placeholder block for a local GPU without --log-dir-*.

    Always emits the ``#failure-analysis-{gpu}`` anchor so the *Failed* column
    links in :func:`render_overall_test_execution_summary_table` always land
    on a real target, even when no data was supplied.
    """
    return (
        f'<a id="failure-analysis-{gpu.lower()}"></a>\n'
        f"#### {gpu} failures\n\n"
        f"*No `--log-dir-{gpu.lower()}` supplied - failure analysis skipped.*\n"
    )


def _render_failure_summary_blocks(
    *,
    log_h200,
    log_h800,
    log_a100,
    h100_build_no=None,
    h100_build_url=None,
    h100_failed_steps=None,
    include_h100: bool = True,
) -> str:
    """Per-GPU failure detail blocks (no top-level heading).

    Always emits an anchor for every local GPU (H200/H800/A100) so the
    *Failed* column links in the Overall test execution summary table land
    on a real target. ``include_h100=False`` skips the Buildkite H100 block
    (used by the development variant, which has no H100 data).
    """
    local_pairs = [
        ("H200", log_h200),
        ("H800", log_h800),
        ("A100", log_a100),
    ]
    gpu_blocks: list[str] = []
    for gpu, log_dir in local_pairs:
        if log_dir:
            gpu_blocks.append(_render_local_gpu_failure_section(gpu, log_dir))
        else:
            gpu_blocks.append(_render_local_gpu_failure_placeholder(gpu))
    if include_h100:
        gpu_blocks.append(
            _render_buildkite_step_failure_section(
                build_no=h100_build_no,
                build_url=h100_build_url,
                failed_steps=h100_failed_steps or [],
            )
        )
    return "\n".join(gpu_blocks)


def build_h100_ci_markdown_body(
    *,
    build_table_md: str,
    passed: int,
    failed: int,
    skipped: int,
    failed_section: str,
    compact: bool = False,
) -> str:
    """Render the H100 (CI — Buildkite scheduled nightly) body.

    The H100 panel intentionally keeps only the ``#### Build`` subsection. The
    per-build ``Summary (reportable jobs only)`` and ``Failed test jobs`` blocks
    were dropped because:

    * The same Total / Passed / Failed numbers belong in the Overall test
      execution summary table at the top of Test Result (which now includes an
      H100 row).
    * ``Failed test jobs`` previously surfaced the ``:email: Nightly
      Collection & Email`` orchestration step (state ``broken``) as if it were
      a real test failure — that's an SMTP / kanban-sync side effect, not a
      CI regression.

    ``failed_section`` is kept as a parameter so older callers continue to
    compile, but it is no longer rendered in either layout.
    """
    return f"#### Build\n\n{build_table_md}\n"


def render_overall_test_execution_summary_table(
    *,
    log_h200,
    log_h800,
    log_a100,
    h100_passed: int | None = None,
    h100_failed: int | None = None,
    h100_skipped: int | None = None,
) -> str:
    """Emit the combined Total / Passed / Failed table at the top of Test Result.

    Includes one row per local GPU (H200, H800, A100) plus an H100 row when
    ``h100_passed`` / ``h100_failed`` are supplied (Buildkite scheduled
    nightly counts; ``broken`` steps are excluded so the totals stay aligned
    with the per-GPU failure detail). H100 totals stay blank in the
    development variant — see ``buildkite_build_stats.py`` for the same rule.

    The *Failed* column links to ``#failure-analysis-hXXX`` (the matching
    subsection inside the top-level Failure Analysis section).
    """
    from nightly_local_log_report import render_markdown_table

    def _row(gpu, log_dir):
        anchor_id = f"failure-analysis-{gpu.lower()}"
        if log_dir:
            total, failed_count = _render_local_gpu_job_counts(log_dir)
            passed = max(total - failed_count, 0)
            total_s, passed_s, failed_s = str(total), str(passed), str(failed_count)
        else:
            total_s = passed_s = failed_s = "—"
        return [gpu, total_s, passed_s, f"[{failed_s}](#{anchor_id})"]

    rows = [
        _row("H200", log_h200),
        _row("H800", log_h800),
        _row("A100", log_a100),
    ]
    if h100_passed is not None or h100_failed is not None:
        total = (h100_passed or 0) + (h100_failed or 0) + (h100_skipped or 0)
        rows.append(
            [
                "H100",
                str(total),
                str(h100_passed or 0),
                f"[{h100_failed or 0}](#failure-analysis-h100)",
            ]
        )
    return render_markdown_table(
        ["GPU", "Total cases", "Passed", "Failed"],
        rows,
    )


def render_test_result_section(
    skill_dir: Path,
    *,
    log_h200,
    log_h800,
    log_a100,
    h100_ci_markdown: str,
    h100_passed=None,
    h100_failed=None,
    h100_skipped=None,
    dev_perf_h200=None,
    dev_perf_h800=None,
    dev_perf_a100=None,
    include_failure_summary: bool = False,
    h100_build_no=None,
    h100_build_url=None,
    h100_failed_steps=None,
    overall_summary_table_md: str | None = None,
) -> str:
    """Render the Test Result section.

    Layout (per spec): Common stack + Overall test execution summary table
    (combined Total / Passed / Failed across all GPUs) + per-GPU nightly
    summaries (### H200 / ### H800 / ### A100 / ### H100). The Failed column
    in the combined summary links to the matching Failure Analysis subsection.

    The H100 panel (### H100 (CI — Buildkite scheduled nightly)) is rendered
    from the ``h100_ci_markdown`` argument and always emits the heading —
    callers without a live Buildkite result pass an empty string and get a
    friendly placeholder instead.

    The Failure Analysis aggregate no longer lives inside this function —
    Development variants render it as its own top-level ``## Failure Analysis``
    section via :func:`render_failure_analysis_section`. ``include_failure_summary``
    is preserved as a legacy no-op (release path callers that still pass it
    get the legacy appended block).
    """
    summary_md = overall_summary_table_md
    if summary_md is None:
        summary_md = render_overall_test_execution_summary_table(
            log_h200=log_h200,
            log_h800=log_h800,
            log_a100=log_a100,
            h100_passed=h100_passed,
            h100_failed=h100_failed,
            h100_skipped=h100_skipped,
        )

    chunks: list[str] = [
        "## Test Result",
        "",
        "### Overall test execution summary",
        "",
        "Combined Total / Passed / Failed across the local machine types "
        "(H200 / H800 / A100) **and the H100 Buildkite scheduled nightly build**. "
        "H100 counts come from the latest scheduled nightly build fetched via "
        "the Buildkite API; `Upload * Pipeline` and orchestration-only steps "
        "like `Nightly Collection&Email` are excluded from both Total and "
        "Failed. The Failed cell links to the matching subsection under the "
        "next Failure Analysis section.",
        "",
        summary_md,
        "",
    ]
    if include_failure_summary:
        chunks.append("")
        chunks.append(
            render_failure_summary_md(
                log_h200=log_h200,
                log_h800=log_h800,
                log_a100=log_a100,
                h100_build_no=h100_build_no,
                h100_build_url=h100_build_url,
                h100_failed_steps=h100_failed_steps or [],
            ).rstrip()
        )
        chunks.append("")
    chunks.extend(["", "### H200", ""])
    chunks.append(markdown_local_summary_from_log_dir(log_h200) if log_h200 else _gpu_log_placeholder("--log-dir-h200"))
    if dev_perf_h200:
        chunks.extend(["", dev_perf_h200.rstrip(), ""])
    chunks.extend(["", "### H800", ""])
    chunks.append(markdown_local_summary_from_log_dir(log_h800) if log_h800 else _gpu_log_placeholder("--log-dir-h800"))
    if dev_perf_h800:
        chunks.extend(["", dev_perf_h800.rstrip(), ""])
    chunks.extend(["", "### A100", ""])
    chunks.append(markdown_local_summary_from_log_dir(log_a100) if log_a100 else _gpu_log_placeholder("--log-dir-a100"))
    if dev_perf_a100:
        chunks.extend(["", dev_perf_a100.rstrip(), ""])
    # H100 (CI — Buildkite scheduled nightly). Always render a `### H100` heading
    # so the section is visible even when the live Buildkite call is unavailable
    # (the caller passes an empty `h100_ci_markdown` placeholder in that case).
    chunks.extend(["", "### H100 (CI — Buildkite scheduled nightly)", ""])
    if h100_ci_markdown:
        chunks.append(h100_ci_markdown.rstrip())
    else:
        chunks.append(
            "*No H100 (Buildkite scheduled nightly) result for this build. "
            "Re-run `compose_full_report.py` after the latest scheduled nightly finishes.*"
        )
    chunks.append("")
    return "\n".join(chunks)


def render_failure_analysis_section(
    *,
    log_h200,
    log_h800,
    log_a100,
    h100_build_no=None,
    h100_build_url=None,
    h100_failed_steps=None,
    include_h100: bool = True,
) -> str:
    """Emit a top-level ## Failure Analysis section.

    Each local GPU (H200/H800/A100) gets its own collapsible sub-section.
    ``include_h100=True`` (default; release path) also renders the H100
    Buildkite failed-steps block; ``include_h100=False`` (development
    variant) skips H100 entirely.

    Anchors named ``failure-analysis-hXXX`` (always emitted, even for
    placeholders) so the *Failed* cells in the
    **Test Result → Overall test execution summary** table jump here.
    """
    summary_section = (
        "### Summary\n\n"
        '<details class="report-subcard release-h-fold release-h4-fold">'
        '<summary class="report-subcard-summary">'
        '<span class="report-subcard-title">Summary</span></summary>'
        '<div class="report-subcard-body">'
        '<div class="fa-summary-editable" data-oi-key="failure-analysis-summary" '
        'data-oi-value="" data-oi-state="empty">'
        '<button type="button" class="oi-note-btn oi-note-empty" '
        'data-oi-note-action="edit" title="Click to add a summary">'
        "Click to add a summary</button></div></div></details>\n\n"
    )
    intro = (
        "## Failure Analysis\n\n"
        "Per-machine failure detail. Click the *Failed* cell in the "
        "Test Result summary table to jump to the matching subsection below."
        "\n\n" + summary_section
    )
    return intro + _render_failure_summary_blocks(
        log_h200=log_h200,
        log_h800=log_h800,
        log_a100=log_a100,
        h100_build_no=h100_build_no,
        h100_build_url=h100_build_url,
        h100_failed_steps=h100_failed_steps,
        include_h100=include_h100,
    )


def render_dev_perf_baseline_local_md(
    gpu_log_dir: Path,
    *,
    assets_dir: Path | None,
    gpu_name: str,
) -> str:
    """Render a ``#### {gpu_name}`` block for one GPU under Performance Data Comparison.
    Reuses the **nightly Local performance baseline comparison** logic
    (``nightly_local_log_report._buildkite_perf_rows`` +
    ``_filter_perf_summary_for_local``), but is **read-only**: when the caller
    has no kanban checkout handy the helper passes ``repo_root=None`` and reads
    ``docs/assets/charts/*_history.json`` directly. No
    ``prepare_kanban_before_report.py`` / ``mkdocs build`` / push is invoked.

    Returns a Markdown ``#### {gpu_name}`` heading + grouped perf table.
    Empty / missing data renders as a friendly note (never raises).
    """
    heading = f"#### {gpu_name}"
    try:
        from nightly_local_log_report import (  # local import: keep top-level deps lean
            KanbanAssetsConfig,
            _append_buildkite_perf_markdown,
            _buildkite_perf_rows,
        )
    except Exception as exc:
        return (
            f"{heading}\n\n"
            f"*Performance baseline helpers unavailable (`{exc}`). "
            "Run from the skill directory so `nightly_local_log_report.py` is importable.*\n"
        )

    kanban_cfg = KanbanAssetsConfig(assets_dir=assets_dir, repo_root=None)
    if assets_dir is None:
        return (
            f"{heading}\n\n"
            "*`--kanban-repo-root` / `--perf-assets-dir` not provided — skipping perf "
            "baseline comparison (read-only; nothing is written to kanban).*\n"
        )

    try:
        summary, grouped_rows = _buildkite_perf_rows(kanban_cfg, log_dir=gpu_log_dir)
    except Exception as exc:
        return f"{heading}\n\n*Failed to compute perf baseline rows for `{gpu_log_dir}`: `{exc}`.*\n"

    lines: list[str] = [heading, ""]
    # Use `#####` for model headings so they nest **inside** the `#### {gpu}`
    # collapsible details (H200 / H800 / A100) instead of becoming siblings.
    _append_buildkite_perf_markdown(lines, summary, grouped_rows, model_heading_level=5)
    return "\n".join(lines)


def _resolve_perf_assets_dir(
    kanban_repo_root: Path | None,
    perf_assets_dir: Path | None,
) -> Path | None:
    """Pick the assets dir from explicit flag → KANBAN_REPO_ROOT env → default ~/vllm-omni-kanban.

    Falls back to ``<kanban_repo_root>/docs/assets/charts`` when only the
    kanban root is set. Returns ``None`` if nothing usable is configured.
    """
    if perf_assets_dir is not None:
        return perf_assets_dir
    if kanban_repo_root is None:
        env_root = os.environ.get("KANBAN_REPO_ROOT") or os.environ.get("VLLM_OMNI_KANBAN_ROOT")
        if env_root:
            try:
                kanban_repo_root = Path(env_root).expanduser()
            except Exception:
                kanban_repo_root = None
    # Falls back to ~/vllm-omni-kanban if neither flag nor env var is set
    if kanban_repo_root is None:
        default_kanban = Path.home() / "vllm-omni-kanban"
        if default_kanban.exists():
            kanban_repo_root = default_kanban
    if kanban_repo_root is not None and kanban_repo_root.exists():
        candidate = kanban_repo_root / "docs/assets/charts"
        if candidate.is_dir():
            return candidate
    return None


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


def replace_ut_coverage_with_manual_edit(ci_md: str) -> str:
    """Replace the auto-computed UT coverage cell in the release Metrics overview
    with the ``@@UT_CELL_INSERTION_POINT@@`` placeholder so it renders as
    interactive manual-edit cells in HTML (matching the development variant).

    The ``buildkite_build_stats.py --markdown`` table has two UT coverage rows:
    - ``| ut | <pct> | <dur> | <count> | - |``
    - ``| ut (exclude models) | <pct> | - | <count> | - |``

    The ``ut (exclude models)`` row is **dropped** from the release report
    (only the canonical ``ut`` row is kept). The remaining row's second column
    (``Success rate/UT coverage``) is replaced with the placeholder. The HTML
    post-processor (``release_md_to_html``) substitutes the placeholder with
    the editable cell widget.
    """
    lines = ci_md.splitlines()
    out_lines: list[str] = []
    in_table = False
    ut_col_idx = -1  # index of "Success rate/UT coverage" column

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
                # Parse header to find the UT coverage column
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                for ci, c in enumerate(cells):
                    if "UT coverage" in c or "Success rate/UT" in c:
                        ut_col_idx = ci
                        break
                out_lines.append(line)
                continue
            # Separator row: just pass through
            if all(bool(re.match(r"^:?-{3,}:?$", (c or "").strip())) for c in stripped.split("|")[1:-1]):
                out_lines.append(line)
                continue
            # Data row: check if first column starts with "ut"
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            first_cell = cells[0] if cells else ""
            first_lower = first_cell.lower()
            if ut_col_idx >= 0 and first_lower.startswith("ut"):
                # Drop the "ut (exclude models)" row entirely — only the
                # canonical "ut" row is kept in the release report.
                if first_lower.startswith("ut (exclude models)"):
                    continue
                cells[ut_col_idx] = "@@UT_CELL_INSERTION_POINT@@"
                rebuilt = "|" + "|".join(cells) + "|"
                out_lines.append(rebuilt)
                continue
            out_lines.append(line)
        else:
            if in_table:
                in_table = False
            out_lines.append(line)

    return "\n".join(out_lines)


def _job_scope_ref_lookup_key(cell: str) -> str:
    """First column of a scope table row -> lookup key (matches Buildkite `job.name`)."""
    t = (cell or "").replace("**", "").strip()
    if " (" in t:
        t = t.split(" (", 1)[0].strip()
    return t


def ci_issue_detection_rate(
    gh_token: str | None,
    date_from: str,
    date_to: str,
) -> tuple[int, int, str]:
    """Compute the **CI issue detection rate** for the release Metrics overview.

    Returns ``(ci_failure_count, total_bug_count, detail)`` where the rate is
    ``ci_failure_count / total_bug_count`` as a percentage. Both the numerator
    and denominator are GitHub issues whose ``created_at`` UTC date falls in
    ``[date_from, date_to]`` and which carry the ``bug`` label. The numerator
    additionally requires the ``ci-failure`` label.

    The total is taken across all states (open + closed) so the rate reflects
    the share of bugs that the CI pipeline correctly tagged during the window,
    not just the ones that are still open. On any GitHub error the function
    returns ``(0, 0, "...")`` and the caller should render the row as
    unavailable.
    """
    try:
        import requests
    except ImportError:
        return 0, 0, "requests not installed"

    # Re-use the same TLS-verify rule the rest of the skill applies for
    # GitHub calls. Setting ``GITHUB_INSECURE_SSL=1`` lets the call succeed
    # in environments whose OS trust store can't validate ``api.github.com``
    # (corporate proxies, custom CAs not installed as root, etc.).
    try:
        from buildkite_build_stats import _github_tls_verify as _tls_verify

        verify = _tls_verify()
    except Exception:
        verify = True

    per_page = 100
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-test-report",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"

    def _page(params: dict) -> list[dict]:
        r = requests.get(
            "https://api.github.com/repos/vllm-project/vllm-omni/issues",
            params={**params, "per_page": per_page},
            headers=headers,
            timeout=60,
            verify=verify,
        )
        if r.status_code == 403:
            raise RuntimeError("GitHub API 403 (rate limit)")
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []

    def _filter(batch: list[dict]) -> list[dict]:
        out: list[dict] = []
        for i in batch:
            if i.get("pull_request"):
                continue
            ca = str(i.get("created_at") or "")
            if len(ca) < 10:
                continue
            d = ca[0:10]
            if date_from <= d <= date_to:
                out.append(i)
        return out

    total = 0
    ci_fail = 0
    page = 1
    try:
        while True:
            batch = _page({"state": "all", "labels": "bug", "page": page, "sort": "created", "direction": "desc"})
            if not batch:
                break
            kept = _filter(batch)
            total += len(kept)
            for i in kept:
                names = _issue_label_names(i)
                if "ci-failure" in names:
                    ci_fail += 1
            # Stop once this page was entirely outside the window.
            if not kept and all(
                str(i.get("created_at") or "")[0:10] < date_from for i in batch if not i.get("pull_request")
            ):
                break
            if len(batch) < per_page:
                break
            page += 1
    except Exception as exc:
        return 0, 0, f"GitHub fetch failed ({exc})"

    detail = f"{ci_fail}/{total} bugs in {date_from}..{date_to} carry `ci-failure`"
    return ci_fail, total, detail


def append_ci_issue_detection_rate_row(
    ci_md: str,
    gh_token: str | None,
    date_from: str,
    date_to: str,
) -> str:
    """Append a **CI issue detection rate** row to the release Metrics overview.

    The release Metrics overview is a 5-column Markdown table emitted by
    ``buildkite_build_stats.py --markdown``. We insert a new row whose first
    column names the metric and whose **Success rate/UT coverage** cell
    carries the formatted percentage + numerator/denominator.
    """
    ci_fail, total, detail = ci_issue_detection_rate(gh_token, date_from, date_to)
    if total <= 0:
        rate_cell = f"_(unavailable — {detail})_"
    else:
        pct = (ci_fail / total) * 100
        rate_cell = f"{pct:.1f}% ({ci_fail}/{total})"

    label = f"**CI issue detection rate** ({date_from}..{date_to})"
    new_row = f"| {label} | {rate_cell} | - | - | - |"

    lines = ci_md.splitlines()
    # Find the table region: starts at the header (a "|" line that contains
    # "CI category" — the canonical header emitted by buildkite_build_stats.py)
    # and ends at the last data row (the "bugs (first response, ...)" row,
    # which is always the last data row in the metrics table).
    out: list[str] = []
    inserted = False
    in_table = False
    for i, line in enumerate(lines):
        out.append(line)
        if not in_table:
            if line.lstrip().startswith("|") and "CI category" in line:
                in_table = True
            continue
        # We're inside the table. Detect the last data row: a "|" line that
        # mentions "first response" (the canonical final row).
        if "first response" in line.lower():
            out.append(new_row)
            inserted = True
            in_table = False  # we've appended, stop tracking the table
    if not inserted:
        # Fallback: append at the end of the file.
        out.append(new_row)
    return "\n".join(out)


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
                ["**UT coverage**", "@@UT_CELL_INSERTION_POINT@@"],
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

    # Failure Analysis (preview): per-GPU placeholder blocks + H100 preview.
    demo_step_a = "L2_Diffusion_Accuracy_Test"
    demo_step_b = "L3_Merge_Example_Suite"
    h100_failed_steps_preview: list[tuple[str, str, str]] = [
        (demo_step_a, "failed", demo_link_a),
        (demo_step_b, "failed", demo_link_b),
    ]
    failure_analysis = render_failure_analysis_section(
        log_h200=None,
        log_h800=None,
        log_a100=None,
        h100_build_no=build_no,
        h100_build_url=f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}",
        h100_failed_steps=h100_failed_steps_preview,
        include_h100=True,
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
            OPEN_ISSUES_HEADERS,
            [
                [
                    "[#10055](https://github.com/vllm-project/vllm-omni/issues/10055)",
                    "OOM when loading Qwen-Omni with FP8 on 40GB",
                    "2026-05-14",
                    "high priority",
                    "3",
                    "open",
                    "@alice-preview",
                    *OPEN_ISSUE_ACTION_CELLS,
                ],
                [
                    "[#10042](https://github.com/vllm-project/vllm-omni/issues/10042)",
                    "Intermittent timeout on L2 diffusion accuracy",
                    stats_to,
                    "medium priority",
                    "1",
                    "open",
                    "@bob-preview",
                    *OPEN_ISSUE_ACTION_CELLS,
                ],
                [
                    "[#10030](https://github.com/vllm-project/vllm-omni/issues/10030)",
                    "Docs: wrong env var for TEE cache",
                    "2026-05-10",
                    "low priority",
                    "0.1",
                    "open",
                    "@carol-preview",
                    *OPEN_ISSUE_ACTION_CELLS,
                ],
            ],
        )
        + "\n"
    )

    return f"""# vLLM-Omni Test Report - Scheduled Nightly

{conclusion}{ci_md}

{test_result}

{failure_analysis}
{issue_tracking}{open_issues_block}
## Data source

- **Mode:** `compose_full_report.py --preview` (sample tables only)
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-*`; H100 is Buildkite block
- **Failure Analysis:** Per-GPU failure detail. Interactive **Status** column
  (Filed / Not an issue) backed by `localStorage`.
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
        "--kind",
        choices=("release", "development"),
        default="release",
        help=(
            "Report kind. ``release`` (default) — full release layout with Test conclusion + "
            "Metrics overview (UT coverage rows are manual-edit cells) + "
            "Failure Analysis (per-GPU with interactive Status column) + "
            "Issue tracking. ``development`` — same Test Result + "
            "Open issues (stats window) layout as release, but **Test conclusion** and "
            "**Issue tracking** sections are omitted and **Metrics overview** is replaced with "
            "a Development-flavored 4-row snapshot (legacy DI · Open Critical Issue · merge CI result · "
            'Unassigned Open Issue). Each row turns red via ``<span class="dev-snapshot-alert">`` '
            "when its threshold is breached: DI>30, open critical issue>0, merge CI not all passing, "
            "Unassigned Open Issue>0."
        ),
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
            "Output path. Default: <skill-dir>/vllm-omni-test-report-YYYY-MM-DD.html "
            "(or vllm-omni-test-report-development-YYYY-MM-DD.html for --kind development); "
            ".md when --format markdown."
        ),
    )
    parser.add_argument(
        "--stats-from",
        default=None,
        help=(
            "buildkite_build_stats.py --from (UTC YYYY-MM-DD). "
            "Default: first day of current UTC month (month-to-date, matches SKILL). "
            "Used only by --kind release (for Metrics overview + DI threshold). --kind "
            "development uses GitHub snapshot + Buildkite 'latest finished' pulls (no date "
            "window)."
        ),
    )
    parser.add_argument(
        "--stats-to",
        default=None,
        help="buildkite_build_stats.py --to (default: today UTC). See --stats-from for kind notes.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Emit sample data only (no Buildkite, GitHub, or pytest log fetch). "
            "Default output: vllm-omni-test-report-preview-YYYY-MM-DD.html "
            "(or vllm-omni-test-report-development-preview-YYYY-MM-DD.html for --kind development)."
        ),
    )
    parser.add_argument(
        "--log-dir-h200",
        type=Path,
        default=None,
        help=(
            "Optional. Root directory of nightly job logs for **Test Result → H200** "
            "(same layout as nightly `nightly_jobs`; see references/nightly-local-log-layout.md). "
            "Applies to both --kind release and --kind development."
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
    parser.add_argument(
        "--kanban-repo-root",
        type=Path,
        default=None,
        help=(
            "Optional. Path to the vllm-omni-kanban checkout (only used by --kind "
            "development to read kanban `docs/assets/charts/*_history.json` for the "
            "per-GPU `Performance Data Comparison` subsection under H200/H800/A100). "
            "Resolution order: (1) this flag, (2) $KANBAN_REPO_ROOT env var, "
            "(3) $VLLM_OMNI_KANBAN_ROOT env var, (4) ~/vllm-omni-kanban default. "
            "No `prepare_kanban_before_report.py` / mkdocs build / push is invoked — "
            "the report is read-only against the kanban tree."
        ),
    )
    parser.add_argument(
        "--perf-assets-dir",
        type=Path,
        default=None,
        help=(
            "Optional. Explicit override for the kanban assets dir "
            "(`docs/assets/charts`) used by --kind development's perf subsection. "
            "Takes precedence over `--kanban-repo-root` when both are supplied."
        ),
    )
    parser.add_argument(
        "--omni-repo-root",
        type=Path,
        default=None,
        help=(
            "Optional. Path to the vllm-omni checkout used by --kind "
            "development to scan `tests/**` for issue-linked pytest skips in "
            "the `Skip Test Case Monitoring` section. Resolution order: (1) "
            "this flag, (2) $OMNI_REPO_ROOT env, (3) $REPO_ROOT env, (4) "
            "the skill's containing checkout, (5) ~/vllm-omni. The repo's "
            "`tests/` directory must exist or the section renders a note."
        ),
    )
    parser.add_argument(
        "--no-repo-pull",
        action="store_true",
        help=(
            "Optional. Skip the fast-forward `git pull` step in the `Skip "
            "Test Case Monitoring` section. The section will scan the "
            "on-disk tree as-is. No-op when --omni-repo-root is not "
            "resolvable to a git checkout."
        ),
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

    today_utc = report_date
    stats_to = args.stats_to or today_utc
    stats_from = args.stats_from or datetime.strptime(today_utc, "%Y-%m-%d").date().replace(day=1).isoformat()

    # Resolve default output filename based on kind and format.
    def _default_output_path() -> Path:
        ext = ".html" if args.format == "html" else ".md"
        if args.kind == "development":
            base = (
                development_report_preview_basename(report_date)
                if args.preview
                else development_report_basename(report_date)
            )
        else:
            base = (
                release_report_preview_basename(report_date) if args.preview else release_report_basename(report_date)
            )
        out = skill_dir / (base.replace(".html", ext) if ext == ".md" else base)
        return out

    out_path = Path(args.out) if args.out else _default_output_path()

    if args.preview:
        if args.kind == "development":
            md = render_development_report_markdown_preview(
                skill_dir,
                stats_from=stats_from,
                stats_to=stats_to,
            )
        else:
            md = preview_report_markdown(skill_dir, stats_from=stats_from, stats_to=stats_to)
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

    # ---- Live (non-preview) path ----

    if args.kind == "development":
        # Development report: shares Test Result layout with release, but
        # - skips Test conclusion and Issue tracking
        # - replaces Metrics overview with Development-flavored block
        gh_token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip() or None
        # Reuse release's H100 build fetch (Buildkite scheduled nightly) so the Test
        # Result block stays structurally identical to ``--kind release``.
        build_no = latest_scheduled_nightly_number(token)
        build_url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds/{build_no}"
        build = http_json(build_url, token)
        assert isinstance(build, dict)
        jobs = build.get("jobs") or []
        reportable = [j for j in jobs if not UPLOAD_PIPELINE_RE.match((j.get("name") or "").strip())]
        states = [(j.get("state") or "").lower() for j in reportable]
        passed = sum(1 for s in states if s == "passed")
        # `broken` is intentionally excluded from the H100 failure counters
        # so it stays consistent with the H100 failure-analysis table below
        # (which only surfaces `failed` steps).
        failed = sum(1 for s in states if s == "failed")
        skipped = sum(1 for s in states if s in ("skipped", "not_run", "blocked"))
        commit = build.get("commit") or ""
        short = commit[:7] if len(commit) >= 7 else commit

        failed_jobs_rows: list[list[str]] = []
        h100_failed_steps: list[tuple[str, str, str]] = []
        for j in reportable:
            st = (j.get("state") or "").lower()
            # Only `failed` is counted as a real failure; `broken` is a
            # transient Buildkite pipeline-execution state and should not be
            # surfaced in the H100 failure analysis. The summary counts
            # above also stop including `broken` so the Total / Passed /
            # Failed numbers stay aligned with the failure-detail table.
            if st == "failed":
                name = (j.get("name") or "").replace("|", "/")
                jid = j.get("id") or ""
                link = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#{jid}"
                failed_jobs_rows.append([name, st, "See step log", f"[open]({link})"])
                h100_failed_steps.append((name, st, link))
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
        h100_body = build_h100_ci_markdown_body(
            build_table_md=build_table_md,
            passed=passed,
            failed=failed,
            skipped=skipped,
            failed_section=failed_section,
            compact=True,
        )

        # Per-GPU perf baseline blocks for the Development variant. Each block is
        # only computed when the corresponding --log-dir-h* is supplied AND the
        # caller configured a kanban assets dir (--kanban-repo-root / --perf-assets-dir /
        # $KANBAN_REPO_ROOT). No kanban writes — purely read against the existing
        # `docs/assets/charts/*_history.json` tree.
        assets_dir = _resolve_perf_assets_dir(args.kanban_repo_root, args.perf_assets_dir)
        dev_perf_h200 = (
            render_dev_perf_baseline_local_md(args.log_dir_h200, assets_dir=assets_dir, gpu_name="H200")
            if args.log_dir_h200
            else None
        )
        dev_perf_h800 = (
            render_dev_perf_baseline_local_md(args.log_dir_h800, assets_dir=assets_dir, gpu_name="H800")
            if args.log_dir_h800
            else None
        )
        dev_perf_a100 = (
            render_dev_perf_baseline_local_md(args.log_dir_a100, assets_dir=assets_dir, gpu_name="A100")
            if args.log_dir_a100
            else None
        )

        # Test Result: Overall test execution summary table + per-GPU nightly
        # summaries. H100 is intentionally excluded from the development variant
        # (it lives in the Buildkite CI side, not the local nightly log roll-up).
        test_result = render_test_result_section(
            skill_dir,
            log_h200=args.log_dir_h200,
            log_h800=args.log_dir_h800,
            log_a100=args.log_dir_a100,
            h100_ci_markdown="",
        )

        # Failure Analysis: top-level section, one collapsible subsection per
        # local GPU. H100 is dropped for the development variant.
        failure_analysis = render_failure_analysis_section(
            log_h200=args.log_dir_h200,
            log_h800=args.log_dir_h800,
            log_a100=args.log_dir_a100,
            include_h100=False,
        )

        pdc_section = render_performance_data_comparison_section(
            dev_perf_h200=dev_perf_h200,
            dev_perf_h800=dev_perf_h800,
            dev_perf_a100=dev_perf_a100,
        )

        dev_metrics_md, unassigned_n, critical_n, _unassigned_issues, _alerts = render_development_metrics_overview(
            token, gh_token
        )
        open_issues_block = render_open_issues_section(stats_from, stats_to, gh_token, all_open=True)

        # Bugfix Monitor: the last 7 days of [Bugfix] PRs on vllm-project/vllm-omni,
        # grouped into Open / Closed, with a per-PR "needs more tests?" verdict.
        bugfix_monitor_block = render_bugfix_monitor_section(gh_token, days_back=7)

        # Skip Test Case Monitoring: static scan of `tests/**` for pytest skips
        # whose reason references a GitHub issue, cross-referenced via the
        # GitHub REST API. The pull is fast-forward-only and non-fatal so a
        # dirty working tree never blocks report generation.
        skip_monitor_block = render_skip_issue_monitor_section(
            repo_root=args.omni_repo_root,
            gh_token=gh_token,
            pull=not args.no_repo_pull,
        )

        md = f"""# vLLM-Omni Test Report - Development

* **Report date (UTC):** {today_utc}

{dev_metrics_md}

{test_result}

{failure_analysis}

{pdc_section}
{skip_monitor_block}
{open_issues_block}

{bugfix_monitor_block}
## Data source

- **Kind:** `compose_full_report.py --kind development`
- **Metrics overview (Development, key 4 items + red threshold):**
  - Outstanding DI = sum of priority-label weights (`critical=10` / `high priority=3` /
    `medium priority=1` / `low priority=0.1` / `invalid=0`) for **all** open `label:bug`
    (no stats-window filter; snapshot at report time). **Red threshold:** > 30 (i.e. > `BUG_DI_THRESHOLD_TENTHS`).
  - Open Critical Issue = GitHub REST `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=bug,critical`
    (AND filter — only issues with both `bug` AND `critical` labels; RFC / Feature tickets tagged
    only `critical` are excluded). **Red threshold:** count > 0.
  - merge CI result = `buildkite_build_stats.fetch_latest_finished_merge_build`; all reportable
    jobs (excluding `Upload * Pipeline`) must be `passed` for "✅ All pass".
    **Red threshold:** not all passing.
  - Unassigned Open Issue = open `label:bug` with empty `assignees[]` (REST paginated).
    **Red threshold:** count > 0.
- **Red highlight implementation:** Row cells are wrapped in
  `<span class="dev-snapshot-alert">…</span>`; see CSS in
  `release_html_theme.RELEASE_MARKDOWN_DOC_CSS` for `.release-doc .dev-snapshot-alert`.
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-h200` / `--log-dir-h800` / `--log-dir-a100`; H100 = Buildkite scheduled nightly
  (this build #{build_no}; reportable jobs only — upload steps excluded).
- **Skip Test Case Monitoring:** static AST scan of `vllm-omni/tests/**` for
  `pytest.mark.{{skip,skipif,xfail}}(reason=...)` / `pytest.skip("…")` whose
  reason text references a GitHub issue. Idioms recognised: full
  `https://github.com/<owner>/<repo>/issues/N` URL, `issue#N`, `issue #N`,
  bare `#N` (>=3 digits), and cross-repo `vllm issue#N` (fetched from
  `vllm-project/vllm`). Before scanning, a **fast-forward-only** `git pull`
  is attempted on the resolved vllm-omni checkout (`--no-repo-pull` skips
  it; failures never abort the report). Repo root resolution order: explicit
  `--omni-repo-root`, `$OMNI_REPO_ROOT`, `$REPO_ROOT`, the skill's
  containing checkout, then `~/vllm-omni`. Issue data is fetched via
  GitHub REST `GET /repos/{{owner}}/{{repo}}/issues/{{n}}` (per-issue; `requests`
  direct call so 404s do not retry); pass `GITHUB_TOKEN` for stable rate
  limits — without it, the table still lists every site with real file /
  test / reason / issue-number data and the three GitHub-sourced columns
  fall back to `—`. **Layout:** `Issue #` is the **first** column (followed by
  Issue Title / State / Updated, then Test File / Test / Skip Mark / Skip
  Reason) and rows are sorted by issue number, so the HTML post-processor
  `release_md_to_html._group_skip_monitor_table_by_issue` folds every site
  sharing an issue under **one collapsible group row** (click the row or its
  caret to expand; *Expand all* / *Collapse all* buttons sit above the table).
  Markdown output keeps the flat, Issue-#-first table.
- **Open issues (stats window):** Identical to `release`, REST pagination `GET /issues?state=open&labels=bug`,
  filter `created_at` UTC date falls in `{{stats_from}}`..`{{stats_to}}`. The table's last two
  columns (**Follow-up action** / **Remarks**) are **manual triage** cells: in HTML the first is
  a `<select>` (Fix in a later iteration / Blocked by dependency / Won't fix (evaluated)) and the
  second a click-to-edit note box; both persist in `localStorage` keyed by the row's issue number
  (`open-issue-followup:#N` / `open-issue-note:#N`), so the same issue keeps its triage across
  report regenerations and across the release / development variants. Markdown keeps `—`.
"""
        if args.format == "html":
            archive_name = out_path.with_suffix(".md").name
            out_path.write_text(
                convert_release_report_markdown(
                    md,
                    archive_download_name=archive_name,
                    l2_l3_row_ok=True,
                    l2_l3_row_detail="",
                    di_row_ok=True,
                    di_row_detail="(Development: Test conclusion omitted; DI still accessible via Metrics overview)",
                    critical_row_ok=True,
                    critical_row_detail="",
                    assignee_row_ok=(unassigned_n == 0),
                    assignee_row_detail=(
                        "(Development: no Unassigned open bugs at report time)"
                        if unassigned_n == 0
                        else f"(Development: {unassigned_n} open bug(s) lack an assignee — see Metrics overview table.)"
                    ),
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
                    di_row_detail="(Development: Test conclusion omitted; DI still accessible via Metrics overview)",
                    critical_row_ok=True,
                    critical_row_detail="",
                    assignee_row_ok=(unassigned_n == 0),
                    assignee_row_detail=(
                        "(Development: no Unassigned open bugs at report time)"
                        if unassigned_n == 0
                        else f"(Development: {unassigned_n} open bug(s) lack an assignee — see Metrics overview table.)"
                    ),
                ),
                encoding="utf-8",
            )
        print(f"Wrote {out_path}")
        return

    # ---- ``--kind release`` (default) live path ----

    build_no = latest_scheduled_nightly_number(token)
    build_url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds/{build_no}"
    build = http_json(build_url, token)
    assert isinstance(build, dict)

    jobs = build.get("jobs") or []
    reportable = [
        j
        for j in jobs
        if not UPLOAD_PIPELINE_RE.match((j.get("name") or "").strip())
        and (j.get("name") or "").strip().lower() not in _NON_REPORTABLE_BK_JOB_NAMES
    ]
    states = [(j.get("state") or "").lower() for j in reportable]
    passed = sum(1 for s in states if s == "passed")
    # Only `failed` (a real runtime failure) counts toward H100 failure totals.
    # `broken` (transient pipeline-execution state — e.g. ``:email: Nightly
    # Collection & Email`` when SMTP/kanban is down) and `skipped` (the job
    # never ran, e.g. an earlier step failed-fast) are explicitly NOT failures.
    failed = sum(1 for s in states if s == "failed")
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
    ci_md = replace_ut_coverage_with_manual_edit(extract_ci_markdown(stats_raw))

    gh_token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip() or None

    # Add the **CI issue detection rate** row to the Metrics overview table
    # (Release variant only — the Development variant does not show this
    # metric, its Metrics overview uses the dev-only rows). The rate is the
    # share of bugs created in the stats window that also carry the
    # ``ci-failure`` label, which measures how well the CI pipeline catches
    # user-reported issues.
    ci_md = append_ci_issue_detection_rate_row(ci_md, gh_token, stats_from, stats_to)

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

    failed_jobs_rows = []
    h100_failed_steps: list[tuple[str, str, str]] = []
    for j in reportable:
        st = (j.get("state") or "").lower()
        # Only jobs whose state is **explicitly** `failed` are real test failures.
        # `broken` (transient Buildkite pipeline state) and `skipped` / `not_run` /
        # `blocked` (the job never ran) are filtered out so the failure narrative
        # reflects runtime failures only — orchestrators like `:email: Nightly
        # Collection & Email` and any pre-failed-fast steps are excluded.
        if st != "failed":
            continue
        name_raw = (j.get("name") or "").replace("|", "/")
        if "nightly collection&email" in name_raw.lower() or "nightly collection and email" in name_raw.lower():
            continue
        jid = (j.get("id") or "").strip()
        link = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}#{jid}"
        failed_jobs_rows.append(
            [
                name_raw,
                st,
                "See step log",
                f"[open]({link})",
                jid,
                "Filed / Not an issue",
            ]
        )
        h100_failed_steps.append((name_raw, st, link))

    failed_section = (
        render_markdown_table(
            ["Step / Job", "State", "Notes", "Step link", "Submit Issue", "Status"],
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
        h100_passed=passed,
        h100_failed=failed,
        h100_skipped=skipped,
    )

    # Failure Analysis: top-level section, one collapsible subsection per
    # GPU (H200/H800/A100 from local logs; H100 from Buildkite).
    # Mirrors the development variant's Failure Analysis layout.
    failure_analysis = render_failure_analysis_section(
        log_h200=args.log_dir_h200,
        log_h800=args.log_dir_h800,
        log_a100=args.log_dir_a100,
        h100_build_no=build_no,
        h100_build_url=f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}",
        h100_failed_steps=h100_failed_steps,
        include_h100=True,
    )

    # Issue tracking section is intentionally omitted from the release
    # report (``compose_full_report.py --kind release``). The dev path
    # already drops it; the release path used to render a separate
    # ``## Issue tracking`` block, but that block is now folded into the
    # Open issues section and the per-job Failure Analysis, so the
    # standalone block would just duplicate information.
    # "Remaining DI < 30" — self-calculated from open `label:bug` issues whose
    # `created_at` is on or before ``stats_to``.  Issues created after the
    # stats window end are excluded so the DI reflects only the bug backlog
    # that existed during the release period.  The start date is unbounded —
    # bugs from any earlier date are included as long as they are still open.
    # (The Development variant uses ``legacy_open_bug_di_total`` which has no
    # date filter at all; the two reports intentionally differ here.)
    open_issues_block = render_open_issues_section(stats_from, stats_to, gh_token, all_open=False)
    di_total_tenths, di_detail = release_open_bug_di_total(gh_token, stats_to)
    # Threshold rule: cumulative Outstanding DI ≤ 30 ⇒ Pass, > 30 ⇒ Fail.
    # ``BUG_DI_THRESHOLD_TENTHS`` is the tenths representation of 30 (300),
    # so the comparison is ``total_tenths <= 300`` (= DI ≤ 30.0). On GitHub
    # fetch failure we conservatively mark the row as Fail (we can't verify
    # the threshold); never as ``None`` so the row stays auto/non-clickable
    # and the operator can't override the auto-judgement.
    if di_total_tenths is None:
        di_row_ok = False
        di_row_detail = f"{di_detail or 'Unable to fetch open bugs'} (auto-judge defaulted to Fail on fetch error)"
    else:
        di_row_ok = di_total_tenths <= BUG_DI_THRESHOLD_TENTHS
        di_row_detail = di_detail

    md = f"""# vLLM-Omni Test Report - Scheduled Nightly

{conclusion}{ci_md}

{test_result}

{failure_analysis}
{open_issues_block}
## Data source

- **Test conclusion (auto):** (1) Buildkite **ready** (non-main) and **merge** (main non-nightly/weekly)
  each latest **finished** build has no `failed`/`broken` job (Upload * Pipeline steps
  excluded); (2) self-calculated **Outstanding DI** = sum of priority-label weights
  (`critical=10` / `high priority=3` / `medium priority=1` / `low priority=0.1` /
  `invalid=0`) across open `label:bug` whose `created_at` ≤ `{stats_to}`
  (start date unbounded; issues created after the stats window are excluded);
  threshold < 30; (3) no open
  `label:bug` + `label:critical`; (4) `All remaining bugs have assignees` is a
  manual user-selectable row; (5) `UT coverage meets this iteration requirement
  (Guide), Performance regression < 5% (Guide)` is a manual user-selectable
  row that does **not** influence the final Go / Rejected verdict.
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-h200` / `--log-dir-h800` / `--log-dir-a100`; H100 = Buildkite scheduled nightly
  (this build #{build_no}: **Build** table link/branch/commit only + Summary + failed jobs)
- **Failure Analysis:** Per-GPU failure detail (H200/H800/A100 from local nightly logs; H100 from
  Buildkite `failed` steps — `broken` is treated as a transient state, not a failure).
  Interactive **Status** column (Filed / Not an issue) backed by `localStorage`.
- **Open issues:** REST `label:bug`, `created_at` UTC date in `{stats_from}`..`{stats_to}`. The
  standalone ``## Issue tracking`` block has been folded into this Open issues
  section.
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
