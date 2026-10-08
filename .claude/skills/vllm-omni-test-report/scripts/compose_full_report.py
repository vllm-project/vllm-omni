#!/usr/bin/env python3
"""
Compose a full test report (default **HTML**).

Agents and users should **emit HTML by default**.

Two report kinds (``--kind``):

  - ``release`` (default): full release layout.
      - Test conclusion: interactive checklist (HTML) / static MD; auto rows: "Latest GPU CI(L1-L5)…" = latest
        finished **ready** + **merge** (same buckets as metrics) have no failed/broken jobs;
        "critical issues…" = no open ``critical``;
        "Remaining DI…" = open ``bug`` in stats window weighted by priority labels < 30;
        "bug assignees…" = open ``bug`` all have assignee
      - Metrics overview: ``buildkite_build_stats.py --markdown``; only the **bugs (first
        response)** row is rendered — the CI-category buckets (``ready`` / `merge` /
        ``nightly`` / ``weekly``) and the ``ut`` / ``ut (exclude models)`` rows are dropped
        to keep the section focused on bug response times + the appended CI issue
        detection rate row.
      - Test Result: Common stack from ``references/local-test-matrix.md``; H200/H800/A100 from
        optional ``--log-dir-h*`` (nightly-style Summary); H100 = Buildkite scheduled nightly
      - Failure Analysis: top-level section with per-GPU (H200/H800/A100 from local logs;
        H100 from Buildkite) collapsible subsections; interactive **Status** column (Filed /
        Not an issue) backed by localStorage, mirroring the Development variant's layout
      - Open issues: GitHub open bugs (``label:bug``); filter ``created_at`` to
        ``--stats-from..--stats-to`` (UTC)

  - ``development``: same as ``release`` but **drops Test conclusion**
    and replaces **Metrics overview** with a Development-flavored block focused on:
      - Outstanding DI (cumulative DI = sum of priority weights for **all** open ``label:bug``)
      - Open Critical Issue (count of issues with label ``critical`` that are still open)
      - DI Top10 + open `label:bug`+`label:ci-failure` (top-10 DI issues + all open CI-failure issues)

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
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from md_table import render_markdown_table  # noqa: E402
from nightly_local_log_report import (  # noqa: E402
    _compute_issue_di_nightly,
    _fetch_bugfix_pr_links,
    _filter_dropped_di_issues,
    _format_di_value_nightly,
    markdown_local_summary_from_log_dir,
)
from release_md_to_html import (  # noqa: E402
    RELEASE_CONCLUSION_PLACEHOLDER,
    convert_release_report_markdown,
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

# All three report kinds (release / development / nightly) now share the
# SLO-escalating model: ``DI = base × �days_open / slo_days⌉`` where
# ``base`` and ``slo_days`` come from :data:`nightly_local_log_report.BUG_DI_TABLE`.
# ``BUG_DI_THRESHOLD_TENTHS`` stores the red-alert threshold (DI > 30 ⇒ Fail)
# in tenths so the integer comparison stays exact under float drift.
BUG_DI_THRESHOLD_TENTHS = 300

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

# Buildkite job-name prefixes that are pure orchestration / infra (image build,
# skip-ci resolve, pipeline upload, report collection). The B200 "Scheduled
# Release" build runs these on ``cpu_queue_premerge``; they must not count as
# B200 test jobs. Stricter than the legacy H100 filter (which only drops
# ``Upload * Pipeline``) so the B200 Total / Passed / Failed reflect the real
# ``b200-k8s`` test matrix.
_BK_ORCHESTRATION_PREFIXES: tuple[str, ...] = (
    ":pipeline:",
    ":docker:",
    ":buildkit:",
    ":github:",
    ":email:",
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


def _bug_di_priority_label(issue: dict) -> str:
    """Return the **highest-priority** DI label for an issue (``critical`` /
    ``high priority`` / ``medium priority`` / ``low priority`` / ``invalid`` /
    ``unclassified``).  Label-only callers (e.g. CI-failure rows) that don't
    need a numeric value use this.  Case-insensitive on label names to match
    :func:`_issue_label_names`.
    """
    labels = _issue_label_names(issue)
    if "invalid" in labels:
        return "invalid"
    for label in (
        "critical",
        "high priority",
        "medium priority",
        "low priority",
    ):
        if label in labels:
            return label
    return "unclassified"


def slo_open_bug_di_total(
    gh_token: str | None,
    *,
    stats_to: str | None = None,
    now: datetime | None = None,
) -> tuple[int | None, str]:
    """Compute the SLO-escalating **Outstanding DI** for the report's
    conclusion row (release variant "Remaining DI < 30") and the
    Open-issues section's per-table summary.

    Iterates **open** ``label:bug`` issues (PRs excluded) and sums the
    per-issue SLO DI returned by :func:`_compute_issue_di_nightly`.  When
    ``stats_to`` is provided, issues whose ``created_at`` UTC date is after
    ``stats_to`` are excluded — this preserves the release variant's
    stats-window semantics (start date unbounded; only the backlog that
    existed during the release window counts).  When ``stats_to`` is ``None``
    every open bug is included (Development variant snapshot behaviour).

    ``now`` is the reference UTC instant; callers should pass a value
    captured once at report-generation start so all SLO-based totals in a
    single report stay on the same time base.  Defaults to
    ``datetime.now(timezone.utc)``.

    Returns ``(total_tenths, detail_str)``.  ``total_tenths`` is an integer
    in tenths of a DI unit so the existing red-alert comparison
    (``<= BUG_DI_THRESHOLD_TENTHS``) stays exact; ``detail_str`` is a
    short human-readable note (``Auto Outstanding DI=…`` form) shown in the
    conclusion cell.  ``total_tenths`` is ``None`` and ``detail_str`` is a
    short error string when the GitHub fetch fails.
    """
    try:
        issues = _github_fetch_open_bug_issues(gh_token)
    except Exception as exc:
        return None, f"Unable to fetch open bugs ({exc})"
    if stats_to is not None:
        issues = [i for i in issues if (d := _issue_created_date_utc(i)) is not None and d <= stats_to]
    counts: dict[str, int] = {
        "critical": 0,
        "high priority": 0,
        "medium priority": 0,
        "low priority": 0,
        "invalid": 0,
        "unclassified": 0,
    }
    total = 0.0
    for issue in issues:
        di, priority, _days, *_rest = _compute_issue_di_nightly(issue, now=now)
        total += di
        counts[priority if priority in counts else "unclassified"] += 1
    total_tenths = int(round(total * 10))
    parts = [
        f"{label}={counts[label]}"
        for label in (
            "critical",
            "high priority",
            "medium priority",
            "low priority",
            "invalid",
        )
        if counts[label]
    ]
    if counts["unclassified"]:
        parts.append(f"unclassified={counts['unclassified']}")
    detail = ", ".join(parts) if parts else "no open bug"
    return total_tenths, (f"Auto Outstanding DI={_format_di_tenths(total_tenths)} ({len(issues)} open bugs; {detail})")


def slo_open_bug_di_conclusion(
    issues: list[dict],
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Compatibility wrapper for the Open-issues section's auto row:
    pass when SLO-escalating total DI ≤ 30 (== ``BUG_DI_THRESHOLD_TENTHS``),
    fail otherwise.  Returns ``(ok, detail)``.
    """
    counts: dict[str, int] = {
        "critical": 0,
        "high priority": 0,
        "medium priority": 0,
        "low priority": 0,
        "invalid": 0,
        "unclassified": 0,
    }
    total = 0.0
    for issue in issues:
        di, priority, _days, *_rest = _compute_issue_di_nightly(issue, now=now)
        total += di
        counts[priority if priority in counts else "unclassified"] += 1
    total_tenths = int(round(total * 10))
    parts = [
        f"{label}={counts[label]}"
        for label in (
            "critical",
            "high priority",
            "medium priority",
            "low priority",
            "invalid",
        )
        if counts[label]
    ]
    if counts["unclassified"]:
        parts.append(f"unclassified={counts['unclassified']}")
    detail = ", ".join(parts) if parts else "no open bug in stats window"
    return total_tenths <= BUG_DI_THRESHOLD_TENTHS, (f"Auto DI={_format_di_tenths(total_tenths)} ({detail})")


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


# ---------------------------------------------------------------------------
# Development report helpers (``compose_full_report.py --kind development``)
# ---------------------------------------------------------------------------


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


def _compute_di_top10_slo(
    gh_token: str | None,
    now: datetime | None = None,
) -> tuple[float, list[tuple[float, str, float, int, str, str, list[int]]]]:
    """Compute per-issue DI using the SLO-escalating model and return the top-10 contributors.

    Returns ``(total_di, per_issue_sorted)`` where each entry in per_issue_sorted is
    ``(di, priority, days_open, issue_number, title, assignee, linked_bugfix_prs)``
    sorted by DI descending. ``linked_bugfix_prs`` is the list of ``[Bugfix]``
    PR numbers that close the issue (empty when none / GitHub unavailable).

    ``now`` is the reference UTC instant for ``days_open``; callers should pass a
    value captured once at report generation start so multiple report kinds (e.g.
    development vs nightly) computed back-to-back stay on the same time base and
    don't drift across a SLO boundary. Defaults to ``datetime.now(timezone.utc)``.
    """
    from datetime import timezone

    issues = _github_fetch_open_bug_issues(gh_token)
    # Drop ``wontfix`` / ``won't fix`` / ``invalid`` issues from the Top DI
    # ranking (and from the Outstanding DI total when no Development report
    # cares about them). The filter helper is shared with the nightly report
    # via :mod:`nightly_local_log_report` so the two paths agree on the
    # dropped-label set.
    rankable_issues = _filter_dropped_di_issues(issues)
    if now is None:
        now = datetime.now(timezone.utc)
    per_issue_raw: list[tuple[float, str, float, int, str, str]] = []
    total = 0.0
    # Outstanding DI snapshot iterates ALL open bugs (incl. dropped) so the
    # number reflects real backlog; ranking below uses only rankable issues.
    for issue in issues:
        di, _, _, _, _, _ = _compute_issue_di_nightly(issue, now=now)
        total += di
    for issue in rankable_issues:
        di, priority, days_open, issue_number, title, assignee = _compute_issue_di_nightly(issue, now=now)
        per_issue_raw.append((di, priority, days_open, issue_number, title, assignee))
    per_issue_sorted = sorted(per_issue_raw, key=lambda x: -x[0])

    # Cross-reference the top contributors with linked bugfix PRs (cap at 25
    # for the search; the table itself only renders the top 10).
    top_issue_numbers = [t[3] for t in per_issue_sorted[:25] if t[3]]
    bugfix_links = _fetch_bugfix_pr_links(gh_token, top_issue_numbers)
    per_issue_with_bugfix: list[tuple[float, str, float, int, str, str, list[int]]] = []
    for di, priority, days_open, issue_number, title, assignee in per_issue_sorted:
        linked = bugfix_links.get(issue_number, [])
        per_issue_with_bugfix.append((di, priority, days_open, issue_number, title, assignee, linked))
    return total, per_issue_with_bugfix


def _fetch_open_ci_failure_issue_rows(gh_token: str | None) -> tuple[int, str]:
    """Markdown table of all **open** issues with labels ``bug`` **and** ``ci-failure``
    (no date filter — all open issues regardless of creation date).

    Returns ``(count, markdown_table)``. Count is 0 and table is empty when no
    matching issues exist or GitHub is unavailable.
    """
    try:
        issues = _github_fetch_open_issues_with_labels(gh_token, "bug", CI_FAILURE_LABEL)
    except Exception as exc:
        return (
            0,
            render_markdown_table(
                ["Issue", "Title", "Priority", "Assignee", "Status"],
                [["*—*", f"*GitHub unavailable ({exc})*", "*—*", "*—*", "*—*"]],
            ),
        )
    if not issues:
        return 0, ""
    issues.sort(key=lambda x: int(x.get("number") or 0), reverse=True)
    body_rows: list[list[str]] = []
    for i in issues:
        num = i["number"]
        title = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        link = f"[#{num}](https://github.com/vllm-project/vllm-omni/issues/{num})"
        di_label = _bug_di_priority_label(i)
        assignees = i.get("assignees") or []
        assignee_str = (
            ", ".join("@" + str(a.get("login", "")) for a in assignees if isinstance(a, dict) and a.get("login"))
            if assignees
            else ""
        )
        body_rows.append([link, title, di_label, assignee_str or "—", "open"])
    return len(issues), render_markdown_table(["Issue", "Title", "Priority", "Assignee", "Status"], body_rows)


NEXT_STEPS_OUTSTANDING_HEADERS: list[str] = [
    "Item",
    "Assignee",
    "Status",
]

NEXT_STEPS_OUTSTANDING_STATUS_OPTIONS: tuple[str, ...] = (
    "Open",
    "In Progress",
    "Blocked",
    "Won't fix",
    "Fixed",
)


def render_next_steps_section(
    gh_token: str | None = None,
) -> str:
    """Markdown for ``## Outstanding Items`` — a purely manual-entry
    action table with **Add Item** capability.

    The table has 3 columns: 事项 (Item) / 责任人 (Assignee) / 状态 (Status).
    All cells are editable in HTML. Users can add rows via the "Add Item" button
    and delete rows. Data is persisted via localStorage.

    In HTML, the cells are upgraded by
    ``release_md_to_html._upgrade_next_steps_outstanding_cells`` and persisted
    via ``_NEXT_STEPS_OUTSTANDING_SCRIPT`` (localStorage + ``data-ns-*``
    attributes for Save-As persistence).
    """
    return (
        "## Outstanding Items\n\n"
        "Manual-entry action table. Click **Add Item** to add a row. "
        "All cells are editable in HTML (click to edit; persisted locally).\n\n"
        + render_markdown_table(NEXT_STEPS_OUTSTANDING_HEADERS, [["—", "—", "—"]])
        + "\n"
    )


def render_quality_defense_section() -> str:
    """Markdown for ``## Quality Defense Radar`` — per-model 7-axis coverage.

    Emits the H2 plus a single placeholder marker
    (``@@QUALITY_DEFENSE_INSERTION_POINT@@``) that
    :func:`release_md_to_html._upgrade_quality_defense_block` replaces with the
    full 3-column grid of inline SVGs (ten flagship models, each with up to 13
    clickable segments across 7 axes). The SVG block carries its own
    descriptive intro paragraph, so no preamble is emitted here. **Release
    variant only** — the development and nightly variants intentionally omit it.
    """
    return (
        "## Quality Defense Radar\n\n"
        "@@QUALITY_DEFENSE_INSERTION_POINT@@\n"
    )


METRIC_ANALYSIS_INSERTION_MARKER = "@@METRIC_ANALYSIS_INSERTION_POINT@@"


def render_metric_analysis_section() -> str:
    """Markdown for ``## Metric analysis`` — an editable rich-text analysis box.

    Emits the H2 plus a single placeholder marker
    (``@@METRIC_ANALYSIS_INSERTION_POINT@@``) that
    :func:`release_md_to_html._upgrade_metric_analysis_block` replaces with a
    ``contenteditable`` rich-text editor (bold / italic / sub-heading / list
    toolbar) whose content persists in
    ``localStorage['metric-analysis:release']`` and mirrors to a
    ``data-ma-value`` attribute for Save-Page-As persistence. **Release variant
    only** — the section sits immediately below the Metrics overview so the
    operator can author an English metric analysis next to the numbers it
    interprets.
    """
    return (
        "## Metric analysis\n\n"
        f"{METRIC_ANALYSIS_INSERTION_MARKER}\n"
    )


RESOURCE_USAGE_INSERTION_MARKER = "@@RESOURCE_USAGE_INSERTION_POINT@@"


def render_resource_usage_section() -> str:
    """Markdown for the **Resource Usage Analysis** section.

    Renders an H2 section whose body is a multi-module editor (in HTML). The
    Markdown body is intentionally empty inside the section — the placeholder
    marker is substituted by ``release_md_to_html._upgrade_resource_usage_block``
    with an ``+ Add module`` toolbar plus a container that the JS handler
    fills with one card per module. Each module has its own editable title
    input and a **rich-text body editor** (contenteditable, with a
    Bold / Italic / H3 / H4 / bullet list / numbered list / Clear formatting
    toolbar); clicking the module title (or the caret toggle,
    or pressing Enter while the title is focused) collapses the body so the
    module can be used as a section heading.

    Persistence:

    * in-memory DOM attribute ``data-uri-value`` (so browser *Save As* captures
      the user's analysis into the saved HTML file);
    * ``localStorage['resource-usage-analysis']`` holding a JSON array of
      ``{id, title, body, collapsed}`` modules — ``body`` is **HTML** written
      by the contenteditable editor; plain-text bodies from the older
      textarea version are converted at render time (survives reload on
      ``http(s)://`` origins; degrades to in-memory only on ``file://`` where
      Chrome blocks localStorage).

    Legacy migration: if the localStorage key still holds a plain string
    (the old single-textarea format) the JS handler wraps it as one module so
    pre-existing analyses are not lost.

    The H2 is automatically wrapped in a collapsible ``<details>`` by
    ``release_md_to_html._fold_release_report_section_cards``, so clicking the
    section title opens the editor. Only available in the ``--kind
    development`` report (manual engineering artefact — release reports do
    not include it).
    """
    return (
        "## Resource Usage Analysis\n\n"
        "Click the section title to expand the editor. Use **+ Add module** to "
        "create a new observation block (CPU/GPU/memory/disk, peak vs. average, "
        "follow-up actions, …). Each module has its own editable **title** and "
        "**body**: type the title to rename, type the body for the analysis. "
        "Click the title (or press Enter while the title is focused, or click "
        "the caret) to **collapse** the body back to just the title — useful "
        "for keeping many modules scannable. Edits persist automatically via "
        "`localStorage` and survive a *Save Page As* download of this HTML.\n\n"
        f"{RESOURCE_USAGE_INSERTION_MARKER}\n"
    )


def _development_snapshot_table_md() -> str:
    """The Development **Metrics overview** snapshot table + detail bullets.

    Two operator-editable rows — **UT coverage** and **Device-Hours / Build**.
    Both cells ship as marker placeholders that the HTML conversion step
    (:func:`release_md_to_html.convert_release_report_markdown`) substitutes
    with click-to-edit inputs persisted via ``localStorage``:

    | Row | Control |
    |-----|---------|
    | UT coverage | ``@@UT_CELL_INSERTION_POINT@@`` → editable cell (value persists in ``localStorage``) |
    | Device-Hours / Build | ``@@DEVICE_HOURS_PER_BUILD_CELL@@`` → editable ``<input>`` (:func:`release_md_to_html._upgrade_device_hours_cell`; persists in ``localStorage['device-hours-per-build']`` — the same control as the release Metrics overview row) |

    Shared by the live path (:func:`render_development_metrics_overview`) and
    ``render_development_report_markdown_preview`` so the two never drift.
    """
    snapshot_header = ["Metric (Development)", "Result"]
    snapshot_rows = [
        [
            "**UT coverage** (Unit Test coverage; click the cell to edit & persist locally)",
            "@@UT_CELL_INSERTION_POINT@@",
        ],
        [
            "**Device-Hours / Build** (compute burn; click the cell to edit & persist locally)",
            DEVICE_HOURS_PER_BUILD_MARKER,
        ],
    ]
    snapshot_table = render_markdown_table(snapshot_header, snapshot_rows)

    detail_lines = [
        "- Both cells are operator-editable and persist locally: UT coverage via the "
        "editable cell, Device-Hours / Build via ``localStorage['device-hours-per-build']`` "
        "(the same editable control as the release Metrics overview row).",
        "- The former auto-fetched rows (Outstanding DI, Open Critical Issue) were "
        "removed per operator request; the DI ladder remains visible in the "
        "**nightly** report's Daily focus card and the issue-monitor report.",
    ]
    return (
        "**Quick Overview (Development report only — both cells are click-to-edit "
        "and persisted locally)**\n\n"
        f"{snapshot_table}\n\n" + "\n".join(detail_lines)
    )


def render_development_metrics_overview() -> str:
    """
    Markdown body for the **Development** report's ``## Metrics overview`` section.

    Layout (per spec for ``--kind development``): a key snapshot table with two
    operator-editable rows — **UT coverage** and **Device-Hours / Build**. No
    network calls are made from this section; both cells are marker placeholders
    upgraded by the HTML conversion into click-to-edit inputs that persist via
    ``localStorage``. (The former auto-fetched Outstanding DI / Open Critical
    Issue rows were removed per operator request; the DI ladder lives in the
    nightly report's Daily focus card and the issue-monitor report.)
    """
    snapshot_md = _development_snapshot_table_md()
    section_md = (
        f"## Metrics overview\n\n"
        f"Source: `scripts/compose_full_report.py --kind development`; "
        f"Based on the release report layout, the **Test conclusion** "
        f"section is removed, and this section carries the operator-editable "
        f"snapshot (**UT coverage** · **Device-Hours / Build**). Both cells "
        f"are click-to-edit and persist locally; no metric is auto-fetched "
        f"here — the DI ladder lives in the **nightly** report's Daily focus "
        f"card and the issue-monitor report.\n\n"
        f"{snapshot_md}\n"
    )
    return section_md


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
    # Preview snapshot mirrors the live development snapshot exactly (shared
    # builder) so the two never drift: UT coverage + Device-Hours / Build,
    # both marker placeholders upgraded by the HTML conversion step.
    metrics_block = (
        "## Metrics overview\n\n"
        "*This section uses **preview placeholder data**: `compose_full_report.py --kind development` "
        "was not run; values below are layout demos only.*\n\n"
        f"{_development_snapshot_table_md()}\n"
    )

    # 1) Test Result: Overall test execution summary table + per-GPU nightly
    #    summaries. H100 is intentionally excluded from the development variant
    #    (it lives in the Buildkite CI side, not the local nightly log roll-up)
    #    — the empty `h100_ci_markdown` drops the H100 panel entirely. A3
    #    follows the H200/H800/A100 pattern (no log dir in preview).
    preview_overall_table = render_overall_test_execution_summary_table(
        log_h200=None,
        log_h800=None,
        log_a100=None,
        log_a3=None,
    )
    test_result = render_test_result_section(
        skill_dir,
        log_h200=None,
        log_h800=None,
        log_a100=None,
        log_a3=None,
        h100_ci_markdown="",
        overall_summary_table_md=preview_overall_table,
    )

    # 2) Failure Analysis: top-level section, one collapsible sub-section per
    #    local GPU. H100 and B200 are dropped for the development variant.
    failure_analysis = render_failure_analysis_section(
        log_h200=None,
        log_h800=None,
        log_a100=None,
        log_a3=None,
        include_h100=False,
        include_b200=False,
    )

    # 3) Skip Test Case Monitoring: hardcoded preview rows (no git pull, no
    #    AST scan, no GitHub API call). Two rows share one issue number so the
    #    HTML per-issue collapsible grouping is visible in preview mode.
    skip_monitor_preview = render_skip_issue_monitor_preview_section()

    # Next Steps (Outstanding Items) preview: manual-entry action table
    next_steps_preview = (
        "## Outstanding Items\n\n"
        "Manual-entry action table. Click **Add Item** to add a row. "
        "All cells are editable in HTML (click to edit; persisted locally).\n\n"
        + render_markdown_table(
            NEXT_STEPS_OUTSTANDING_HEADERS,
            [["—", "—", "—"]],
        )
        + "\n"
    )

    # Resource Usage Analysis preview: same editor marker as the live path;
    # ``_upgrade_resource_usage_block`` substitutes it with an editable
    # <textarea> + Save / Reset toolbar.
    resource_usage_preview = render_resource_usage_section()

    return f"""# vLLM-Omni Test Report - Development (Preview)

{metrics_block}

{test_result}

{failure_analysis}

{skip_monitor_preview}

{resource_usage_preview}

{next_steps_preview}
## Data source

- **Mode:** `compose_full_report.py --preview --kind development` (sample tables only)
- **Test Result:** Overall test execution summary (Total / Passed / Failed across H200 /
   H800 / A100 / A3; Failed cell links to matching Failure Analysis subsection) + per-GPU
   nightly summaries. **The H100 / Buildkite scheduled nightly chapter is omitted**
   from the development variant.
- **Failure Analysis:** Top-level section; one collapsible subsection per local GPU
   (H200 / H800 / A100 / A3). Mirrors the original failure-analysis pattern (per-job
   Failures & errors table).
- **Skip Test Case Monitoring:** top-level section; one hardcoded 5-row preview
   (no AST scan, no `git pull`, no GitHub API call). Two preview rows share one
   issue number so the per-issue collapsible grouping is visible.
- **Metrics overview:** operator-editable 2-row snapshot (**UT coverage** ·
   **Device-Hours / Build**). Both cells are click-to-edit inputs persisted via
   `localStorage` (Device-Hours / Build reuses the release Metrics overview
   control, `localStorage['device-hours-per-build']`). No GitHub REST calls.
- **Outstanding Items:** Manual-entry action table with Add Item
   capability (HTML only; persisted via localStorage).
- Live report: `buildkite_build_stats`, GitHub REST
"""


#: Header row for the **Open issues (stats window)** table.
#:
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
OPEN_ISSUE_ACTION_CELLS: list[str] = ["\u2014", "\u2014"]

#: Priority labels eligible for the **release** Open issues table.
#: Issues whose highest-priority label (via :func:`_bug_di_priority_label`)
#: is not in this set \u2014 ``low priority``, ``invalid``, or
#: unlabelled-priority bugs \u2014 are dropped by
#: :func:`github_open_bug_rows_in_range` when ``priority_filter=`` is set.
OPEN_ISSUES_RELEASE_PRIORITIES: frozenset[str] = frozenset(
    {"critical", "high priority", "medium priority"}
)


def github_open_bug_rows_in_range(
    gh_token: str | None,
    date_from: str,
    date_to: str,
    *,
    now: datetime | None = None,
    priority_filter: frozenset[str] | None = None,
) -> tuple[int, int, str, list[dict]]:
    """
    Paginate **open** issues with label ``bug`` (PR entries excluded).

    Return ``(total_open_bug_fetched, count_in_created_range, markdown_table, issues_in_range)``.
    ``count_in_created_range`` = issues whose **UTC calendar date** of ``created_at``
    lies in ``[date_from, date_to]`` inclusive (``YYYY-MM-DD`` strings).  Each row's
    ``DI`` column uses the **SLO-escalating** model (per-issue ``DI = base × ⌈days_open
    / slo_days⌉``) so it sums exactly to the conclusion-row total — call :func:`slo_open_bug_di_total`
    with the same ``now`` to get the matching total.

    When ``priority_filter`` is supplied, rows whose highest-priority label
    (via :func:`_bug_di_priority_label`) is not in the set are dropped after
    the ``created_at`` window filter and before sorting/rendering. The
    release variant passes :data:`OPEN_ISSUES_RELEASE_PRIORITIES` to
    narrow the table to ``critical`` / ``high priority`` /
    ``medium priority`` issues; the conclusion-row total (computed by
    :func:`slo_open_bug_di_total`) is intentionally **not** narrowed.
    """
    all_items = _github_fetch_open_bug_issues(gh_token)

    in_range = [i for i in all_items if (d := _issue_created_date_utc(i)) is not None and date_from <= d <= date_to]
    if priority_filter is not None:
        in_range = [i for i in in_range if _bug_di_priority_label(i) in priority_filter]
    in_range.sort(key=lambda x: x["created_at"], reverse=True)
    row_cells: list[list[str]] = []
    for i in in_range:
        t = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        u = (i.get("user") or {}).get("login", "")
        di, di_label, *_ = _compute_issue_di_nightly(i, now=now)
        row_cells.append(
            [
                f"[#{i['number']}](https://github.com/vllm-project/vllm-omni/issues/{i['number']})",
                t,
                str(i["created_at"])[:10],
                di_label,
                _format_di_value_nightly(di),
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


def github_open_bug_issue_rows(
    issues: list[dict],
    *,
    now: datetime | None = None,
) -> str:
    """Render the per-issue Markdown table shared by all-open and stats-window variants.

    Each row's ``DI`` column uses the **SLO-escalating** model so per-row values
    sum exactly to the conclusion-row total — pass the same ``now`` to
    :func:`slo_open_bug_di_total`.
    """
    sorted_items = sorted(issues, key=lambda x: x.get("created_at") or "", reverse=True)
    row_cells: list[list[str]] = []
    for i in sorted_items:
        t = (i.get("title") or "").replace("|", "\\|").replace("\n", " ")
        u = (i.get("user") or {}).get("login", "")
        di, di_label, *_ = _compute_issue_di_nightly(i, now=now)
        row_cells.append(
            [
                f"[#{i['number']}](https://github.com/vllm-project/vllm-omni/issues/{i['number']})",
                t,
                str(i.get("created_at") or "")[:10],
                di_label,
                _format_di_value_nightly(di),
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
    now: datetime | None = None,
    priority_filter: frozenset[str] | None = None,
) -> tuple[str, bool | None, str]:
    """Markdown for ``## Open issues`` plus DI conclusion data when GitHub fetch succeeds.

    When ``all_open=True``, the rendered section lists **every** open ``label:bug``
    issue in the ``vllm-project/vllm-omni`` repository (no ``created_at`` window
    filter) — used by the Development variant of ``compose_full_report.py`` so
    the report owner sees the whole backlog, not just the month-to-date slice.
    When ``all_open=False`` (default; release variant), the table is restricted to
    issues whose ``created_at`` UTC date falls in ``stats_from``..``stats_to``.

    ``now`` is forwarded to the per-row DI and the conclusion helper so every
    DI value in this section sums exactly to the report-level conclusion row
    total (no drift across a SLO boundary mid-render).
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
            issue_rows = github_open_bug_issue_rows(issues_all, now=now)
        else:
            open_total, open_range_n, issue_rows, issues_in_range = github_open_bug_rows_in_range(
                gh_token, stats_from, stats_to, now=now,
                priority_filter=priority_filter,
            )
        di_row_ok, di_row_detail = slo_open_bug_di_conclusion(issues_in_range, now=now)
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
    now: datetime | None = None,
    priority_filter: frozenset[str] | None = None,
) -> str:
    """Markdown for ``## Open issues`` block (GitHub REST, open ``label:bug`` only)."""
    section, _, _ = render_open_issues_section_with_di(
        stats_from, stats_to, gh_token,
        all_open=all_open, now=now, priority_filter=priority_filter,
    )
    return section


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
            _manifest_driven_groups,
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

    # Manifest-driven job list: timing_summary.log is the source of truth;
    # .log files are only parsed for FAILED jobs' pytest counts/excerpts. Fall
    # back to plain discover_job_logs when no manifest is present.
    discovered = discover_job_logs(log_dir)
    groups = _manifest_driven_groups(log_dir, discovered)
    if groups is None:
        groups = discovered
    if not groups:
        return (
            f'<a id="failure-analysis-{gpu.lower()}"></a>\n'
            f"#### {gpu} failures\n\n"
            f"*No job logs found under `{log_dir}`.*\n"
        )

    # Forward ``log_dir`` so the rollup (``timing_summary.log``) is consulted
    # before ``parse_pytest_log``; without it, stale pytest footers in
    # concatenated run logs inflate the failure count and diverge from the
    # nightly wrapper's authoritative ``OK`` / ``FAILED (exit N)`` rollup.
    job_rows = _local_job_rows_with_info(groups, log_dir=log_dir)
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
    ]
    for job_name, paths, info in failed_rows:
        chunks.append(f"##### Local job: `{_md_cell(job_name)}`")
        chunks.append("")
        rel = ", ".join(f"`{p.name}`" for p in paths)
        chunks.append(f"- Log files: {rel}")
        chunks.append("")
        fail_rows: list[list[str]] = []
        for row_index, node in enumerate(info["failed_nodes"]):
            fail_rows.append(
                [
                    _md_cell(node),
                    _md_cell(info["failed_reasons"].get(node, "")),
                    _excerpt_md_cell(
                        info["failure_excerpts"].get(node, ""),
                        node=node,
                        row_index=row_index,
                        report_context=f"compose-failure-{gpu.lower()}-{job_name}",
                    ),
                    "Submit issue",
                    "Filed / Not an issue",
                ]
            )
        for row_index, node in enumerate(info["error_nodes"]):
            fail_rows.append(
                [
                    _md_cell(node) + " (ERROR)",
                    _md_cell(info["error_reasons"].get(node, "")),
                    _excerpt_md_cell(
                        info["error_excerpts"].get(node, ""),
                        node=node,
                        row_index=row_index,
                        report_context=f"compose-error-{gpu.lower()}-{job_name}",
                    ),
                    "Submit issue",
                    "Filed / Not an issue",
                ]
            )
        chunks.append("###### Failures & errors")
        chunks.append("")
        chunks.append(
            render_markdown_table(
                ["Test node", "Log reason", "Excerpt (truncated)", "Submit Issue", "Status"],
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
            _manifest_driven_groups,
            discover_job_logs,
        )
    except Exception:
        return 0, 0
    # Manifest-driven job list (timing_summary.log is the source of truth);
    # fall back to plain discover_job_logs when no manifest is present.
    discovered = discover_job_logs(Path(log_dir))
    groups = _manifest_driven_groups(Path(log_dir), discovered)
    if groups is None:
        groups = discovered
    if not groups:
        return 0, 0
    # ``log_dir`` is forwarded so ``_local_job_rows_with_info`` can read the
    # ``timing_summary.log`` rollup first; without this a stale pytest footer
    # in a concatenated .log body would inflate the per-job failure count and
    # diverge from the nightly wrapper's authoritative ``OK`` / ``FAILED (exit
    # N)`` rollup (see ``scripts/nightly_local_log_report.py`` for details).
    job_rows = _local_job_rows_with_info(groups, log_dir=Path(log_dir))
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


def _render_b200_step_failure_section(
    *,
    build_no: int | None,
    build_url: str | None,
    failed_steps: list[tuple[str, str, str]],
) -> str:
    """Render a compact B200 (Buildkite scheduled release) failed-step block for the Failure Analysis section.

    Mirrors :func:`_render_buildkite_step_failure_section` (H100). ``failed_steps``
    is a list of ``(name, state, step_link)`` tuples — only Buildkite steps in
    ``failed`` state (``broken`` is intentionally excluded; it's a transient
    pipeline-execution state, not a real test failure). Emits the
    ``failure-analysis-b200`` anchor so the *Failed* cell in the Test Result
    Overall summary table's B200 row jumps here. Step log parsing is left to
    the dedicated nightly Buildkite pipeline (read-only against cached build
    metadata, matching the H100 precedent).
    """
    if not failed_steps:
        return (
            '<a id="failure-analysis-b200"></a>\n'
            "#### B200 (CI — Buildkite scheduled release) failures\n\n"
            "*No failed Buildkite steps in the latest scheduled release B200 build.*\n"
        )
    chunks: list[str] = [
        '<a id="failure-analysis-b200"></a>',
        "#### B200 (CI — Buildkite scheduled release) failures",
        "",
    ]
    if build_no is not None and build_url:
        chunks.append(f"- Latest scheduled release B200 build: [{build_no}]({build_url}).")
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
    log_a3=None,
    h100_build_no=None,
    h100_build_url=None,
    h100_failed_steps=None,
    include_h100: bool = True,
    b200_build_no=None,
    b200_build_url=None,
    b200_failed_steps=None,
    include_b200: bool = True,
) -> str:
    """Per-GPU failure detail blocks (no top-level heading).

    Always emits an anchor for every local GPU (H200/H800/A100/A3) so the
    *Failed* column links in the Overall test execution summary table land
    on a real target. ``include_h100=False`` skips the Buildkite H100 block
    (used by the development variant, which has no H100 data).
    ``include_b200=True`` (default; release path) renders the B200 (Buildkite
    scheduled release) failed-steps block; ``include_b200=False`` (development
    variant) skips B200 entirely.
    """
    local_pairs = [
        ("H200", log_h200),
        ("H800", log_h800),
        ("A100", log_a100),
        ("A3", log_a3),
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
    if include_b200:
        gpu_blocks.append(
            _render_b200_step_failure_section(
                build_no=b200_build_no,
                build_url=b200_build_url,
                failed_steps=b200_failed_steps or [],
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


# Max pages to scan when resolving the B200 release build (100 builds/page).
# The B200 "Scheduled Release" build runs ~monthly, so a 75-day (15-page) window
# comfortably covers one cadence gap; the resolver stops at the first match.
_B200_MAX_PAGES = 15


def _b200_is_reportable_job(name: str) -> bool:
    """True iff a Buildkite job is a real B200 test job (not orchestration/infra).

    Stricter than the legacy H100 filter (which only drops ``Upload * Pipeline``):
    also excludes the ``:pipeline:`` / ``:docker:`` / ``:buildkit:`` / ``:github:``
    / ``:email:`` orchestration steps that run on ``cpu_queue_premerge``, so the
    B200 Total / Passed / Failed reflect the real ``b200-k8s`` test matrix.
    """
    n = (name or "").strip()
    if not n:
        return False
    if UPLOAD_PIPELINE_RE.match(n):
        return False
    low = n.lower()
    if low in _NON_REPORTABLE_BK_JOB_NAMES:
        return False
    if any(low.startswith(p) for p in _BK_ORCHESTRATION_PREFIXES):
        return False
    return True


def latest_scheduled_release_b200_number(token: str) -> int:
    """Return the latest ``Scheduled Release B200 build`` number on ``main``.

    The B200 release build runs infrequently (~monthly), so a single
    ``per_page=50`` page (as :func:`latest_scheduled_nightly_number` uses) is
    not enough — page through up to :data:`_B200_MAX_PAGES` pages of 100 builds
    (~75-day window) and return the first build whose message matches
    ``scheduled release b200``.
    """
    pattern = re.compile(r"scheduled\s+release\s+b200", re.I)
    for page in range(1, _B200_MAX_PAGES + 1):
        url = (
            f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}"
            f"/builds?branch={BRANCH}&per_page=100&page={page}"
        )
        builds = http_json(url, token)
        assert isinstance(builds, list)
        if not builds:
            break
        for b in builds:
            if pattern.search((b.get("message") or "")):
                return int(b["number"])
    raise RuntimeError(
        "No 'Scheduled Release B200 build' found on main within "
        f"{_B200_MAX_PAGES * 100} recent builds. Pin one explicitly with --b200-build."
    )


def fetch_b200_build_summary(token: str, build_no: int | None = None) -> dict:
    """Fetch the latest (or pinned) Scheduled Release B200 build + reportable-job summary.

    Single Buildkite API call (``GET .../builds/{n}`` with jobs embedded).
    Returns a dict: ``build_no``, ``branch``, ``commit``, ``state``,
    ``created_at``, ``passed``, ``failed``, ``broken``, ``failed_steps``
    (list of ``(name, state, link)``). ``broken`` is counted separately and
    excluded from ``failed`` (matches the H100 rule at the dev-path walk).
    """
    no = build_no if build_no else latest_scheduled_release_b200_number(token)
    url = f"https://api.buildkite.com/v2/organizations/{ORG}/pipelines/{PIPELINE}/builds/{no}"
    build = http_json(url, token)
    assert isinstance(build, dict)
    jobs = build.get("jobs") or []
    reportable = [j for j in jobs if _b200_is_reportable_job((j.get("name") or ""))]
    passed = sum(1 for j in reportable if (j.get("state") or "").lower() == "passed")
    failed = sum(1 for j in reportable if (j.get("state") or "").lower() == "failed")
    broken = sum(1 for j in reportable if (j.get("state") or "").lower() == "broken")
    failed_steps: list[tuple[str, str, str]] = []
    for j in reportable:
        if (j.get("state") or "").lower() == "failed":
            name = (j.get("name") or "").replace("|", "/")
            jid = j.get("id") or ""
            link = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{no}#{jid}"
            failed_steps.append((name, (j.get("state") or "").lower(), link))
    return {
        "build_no": int(no),
        "branch": build.get("branch") or BRANCH,
        "commit": build.get("commit") or "",
        "state": (build.get("state") or "").lower(),
        "created_at": build.get("created_at") or "",
        "passed": passed,
        "failed": failed,
        "broken": broken,
        "failed_steps": failed_steps,
    }


def build_b200_ci_markdown_body(
    *,
    build_no: int,
    branch: str,
    commit: str,
    state: str,
    created_at: str,
    passed: int,
    failed: int,
    broken: int,
) -> str:
    """Render the B200 (CI — Buildkite scheduled release) chapter body.

    Lightweight: build metadata + reportable-job summary. No per-job log
    download (matches the H100 chapter precedent). The failed-step detail
    lives in the **Failure Analysis** section (``_render_b200_step_failure_section``),
    not in this Test Result chapter — so only the Build + Summary blocks are
    emitted here.
    """
    short = commit[:7] if len(commit) >= 7 else commit
    build_link = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}"
    created_short = (created_at or "").replace("T", " ")[:19] or "—"
    total = passed + failed + broken
    commit_cell = (
        f"`{short}` ([full](https://github.com/vllm-project/vllm-omni/commit/{commit}))"
        if commit
        else "—"
    )
    build_table = render_markdown_table(
        ["Field", "Value"],
        [
            ["**Build**", f"[{build_no}]({build_link})"],
            ["**Branch**", branch],
            ["**Commit**", commit_cell],
            ["**State**", state or "—"],
            ["**Created (UTC)**", created_short],
        ],
    )
    summary_table = render_markdown_table(
        ["Total", "Passed", "Failed", "Broken"],
        [[str(total), str(passed), str(failed), str(broken)]],
    )
    parts = [
        f"#### Build\n\n{build_table}\n",
        f"#### Summary (reportable jobs)\n\n{summary_table}\n",
    ]
    return "\n".join(parts)


def render_overall_test_execution_summary_table(
    *,
    log_h200,
    log_h800,
    log_a100,
    log_a3=None,
    h100_passed: int | None = None,
    h100_failed: int | None = None,
    h100_skipped: int | None = None,
    b200_passed: int | None = None,
    b200_failed: int | None = None,
    b200_broken: int | None = None,
) -> str:
    """Emit the combined Total / Passed / Failed table at the top of Test Result.

    Includes one row per local GPU (H200, H800, A100, A3) plus a B200 row when
    ``b200_passed`` / ``b200_failed`` are supplied (Buildkite scheduled release
    counts) and an H100 row when ``h100_passed`` / ``h100_failed`` are supplied
    (Buildkite scheduled nightly counts; ``broken`` steps are excluded so the
    totals stay aligned with the per-GPU failure detail). H100 totals stay
    blank in the development variant — see ``buildkite_build_stats.py`` for the
    same rule.

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
        _row("A3", log_a3),
    ]
    if b200_passed is not None or b200_failed is not None:
        total = (b200_passed or 0) + (b200_failed or 0) + (b200_broken or 0)
        rows.append(
            [
                "B200",
                str(total),
                str(b200_passed or 0),
                f"[{b200_failed or 0}](#failure-analysis-b200)",
            ]
        )
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
    log_a3=None,
    h100_ci_markdown: str,
    h100_passed=None,
    h100_failed=None,
    h100_skipped=None,
    b200_ci_markdown: str = "",
    b200_passed=None,
    b200_failed=None,
    b200_broken=None,
    dev_perf_h200=None,
    dev_perf_h800=None,
    dev_perf_a100=None,
    dev_perf_a3=None,
    include_failure_summary: bool = False,
    h100_build_no=None,
    h100_build_url=None,
    h100_failed_steps=None,
    overall_summary_table_md: str | None = None,
) -> str:
    """Render the Test Result section.

    Layout (per spec): Common stack + Overall test execution summary table
    (combined Total / Passed / Failed across all GPUs) + per-GPU nightly
    summaries (### H200 / ### H800 / ### A100 / ### A3, plus optional
    ### H100 (CI — Buildkite scheduled nightly) when ``h100_ci_markdown``
    is non-empty). The Failed column in the combined summary links to the
    matching Failure Analysis subsection.

    The H100 (CI — Buildkite scheduled nightly) panel is rendered **only**
    when ``h100_ci_markdown`` is a non-empty string. The release path
    passes the Buildkite body; the development path passes ``""`` so the
    H100 panel is dropped entirely from the dev Test Result section.

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
            log_a3=log_a3,
            h100_passed=h100_passed,
            h100_failed=h100_failed,
            h100_skipped=h100_skipped,
            b200_passed=b200_passed,
            b200_failed=b200_failed,
            b200_broken=b200_broken,
        )

    chunks: list[str] = [
        "## Test Result",
        "",
        "### Overall test execution summary",
        "",
        "Combined Total / Passed / Failed across the local machine types "
        "(H200 / H800 / A100 / A3)"
        + (" **and the B200 Buildkite scheduled release build**" if b200_ci_markdown else "")
        + (" **and the H100 Buildkite scheduled nightly build**" if h100_ci_markdown else "")
        + (
            ". Buildkite-sourced counts exclude `Upload * Pipeline` and "
            "orchestration-only steps (e.g. `Nightly Collection&Email`) from "
            "both Total and Failed; the B200/H100 Failed cell links to the "
            "matching chapter below."
            if (b200_ci_markdown or h100_ci_markdown)
            else ". The Failed cell links to the matching subsection under the next Failure Analysis section."
        ),
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
    chunks.extend(["", "### A3", ""])
    chunks.append(markdown_local_summary_from_log_dir(log_a3) if log_a3 else _gpu_log_placeholder("--log-dir-a3"))
    if dev_perf_a3:
        chunks.extend(["", dev_perf_a3.rstrip(), ""])
    # B200 (CI — Buildkite scheduled release). Only emit the panel when the caller
    # passes a non-empty `b200_ci_markdown` body (release path). The B200 chapter
    # emits Build + Summary only; the failed-step detail and the
    # ``failure-analysis-b200`` anchor live in the **Failure Analysis** section
    # (``_render_b200_step_failure_section``) so the B200 Failed cell in the
    # Overall summary table links there.
    if b200_ci_markdown:
        chunks.extend(["", "### B200 (CI — Buildkite scheduled release)", ""])
        chunks.append(b200_ci_markdown.rstrip())
        chunks.append("")
    # H100 (CI — Buildkite scheduled nightly). Only emit the panel when the caller
    # actually passes a non-empty `h100_ci_markdown` body (release path). The
    # development path passes ``""`` so the entire H100 chapter is dropped from
    # the development Test Result section.
    if h100_ci_markdown:
        chunks.extend(["", "### H100 (CI — Buildkite scheduled nightly)", ""])
        chunks.append(h100_ci_markdown.rstrip())
        chunks.append("")
    return "\n".join(chunks)


def render_failure_analysis_section(
    *,
    log_h200,
    log_h800,
    log_a100,
    log_a3=None,
    h100_build_no=None,
    h100_build_url=None,
    h100_failed_steps=None,
    include_h100: bool = True,
    b200_build_no=None,
    b200_build_url=None,
    b200_failed_steps=None,
    include_b200: bool = True,
) -> str:
    """Emit a top-level ## Failure Analysis section.

    Each local GPU (H200/H800/A100/A3) gets its own collapsible sub-section.
    ``include_h100=True`` (default; release path) also renders the H100
    Buildkite failed-steps block; ``include_h100=False`` (development
    variant) skips H100 entirely. ``include_b200=True`` (default; release
    path) renders the B200 (Buildkite scheduled release) failed-steps block;
    ``include_b200=False`` (development variant) skips B200 entirely.

    Anchors named ``failure-analysis-hXXX`` (always emitted, even for
    placeholders) so the *Failed* cells in the
    **Test Result → Overall test execution summary** table jump here.
    """
    intro = (
        "## Failure Analysis\n\n"
        "Per-machine failure detail. Click the *Failed* cell in the "
        "Test Result summary table to jump to the matching subsection below."
        "\n\n"
    )
    return intro + _render_failure_summary_blocks(
        log_h200=log_h200,
        log_h800=log_h800,
        log_a100=log_a100,
        log_a3=log_a3,
        h100_build_no=h100_build_no,
        h100_build_url=h100_build_url,
        h100_failed_steps=h100_failed_steps,
        include_h100=include_h100,
        b200_build_no=b200_build_no,
        b200_build_url=b200_build_url,
        b200_failed_steps=b200_failed_steps,
        include_b200=include_b200,
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


def restructure_metrics_to_two_columns(ci_md: str) -> str:
    """Restructure the release Metrics overview table to two columns.

    The upstream ``buildkite_build_stats.py --markdown`` emits a 5-column table
    (``CI category | Success rate/UT coverage | Avg duration | Other finished
    count | Bug avg first response``). For the release audience we collapse it
    to a 2-column ``Indicator | Value`` shape so each metric reads as a single
    name + value pair.

    Mapping (first-column label → output value):

    * ``bugs (first response, ...)`` → Bug avg first response (last column)
    * ``**CI issue detection rate**`` → percentage cell
    * ``**Device-Hours / Build**`` → marker (later upgraded by
      ``release_md_to_html``)

    Separator rows are normalised to a 2-column dash row. Rows that already
    have only 2 columns are passed through unchanged.
    """
    lines = ci_md.splitlines()
    out: list[str] = []
    header_emitted = False
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            out.append(line)
            continue
        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if not cells:
            out.append(line)
            continue
        # Header row
        if "CI category" in cells[0]:
            out.append("| Indicator | Value |")
            out.append("| --- | --- |")
            header_emitted = True
            continue
        # Separator row → rewrite to 2-col dashes
        if all(bool(re.match(r"^:?-{3,}:?$", c)) for c in cells):
            if not header_emitted:
                # Separator before header — emit a 2-col header so the table stays well-formed.
                out.append("| Indicator | Value |")
                out.append("| --- | --- |")
                header_emitted = True
            continue
        # Already 2-col → pass through.
        if len(cells) <= 2:
            out.append(line)
            continue
        # Data row. Pick the canonical value cell based on the first-column label.
        first = cells[0]
        first_lower = first.lower()
        if "ci issue detection rate" in first_lower:
            value = cells[1] if len(cells) > 1 else "-"
        elif "device-hours" in first_lower and "build" in first_lower:
            value = cells[1] if len(cells) > 1 else "-"
        elif "first response" in first_lower:
            value = cells[-1] if len(cells) > 1 else "-"
        else:
            # Generic fallback: take the Bug avg first response column (last)
            # which is the most informative for the release audience.
            value = cells[-1] if len(cells) > 1 else "-"
        out.append(f"| {first} | {value} |")
        if not header_emitted:
            out.insert(-1, "| Indicator | Value |")
            header_emitted = True
    return "\n".join(out)


def replace_ut_coverage_with_manual_edit(ci_md: str) -> str:
    """Filter the release Metrics overview rows.

    The ``buildkite_build_stats.py --markdown`` table has these data rows:

    - ``ready`` (non-main branches) — **dropped**
    - ``merge`` (main, ordinary runs) — **dropped**
    - ``nightly`` (main, scheduled nightly) — **dropped**
    - ``weekly`` (main, scheduled weekly) — **dropped**
    - ``ut`` — **dropped**
    - ``ut (exclude models)`` — **dropped**
    - ``bugs (first response, ... )`` — **kept**

    Only the bugs row carries useful information for a release audience; the
    CI-category buckets (ready / merge / nightly / weekly) and the UT coverage
    rows were intentionally excluded so the section stays focused on bug
    response times (and the CI issue detection rate appended below).
    """
    # CI-category labels and UT labels we drop entirely. Matched case-insensitive
    # against the first cell of each data row.
    drop_prefixes = (
        "ready",
        "merge",
        "nightly",
        "weekly",
        "ut",
    )

    lines = ci_md.splitlines()
    out_lines: list[str] = []
    in_table = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
                out_lines.append(line)
                continue
            # Separator row: just pass through
            if all(bool(re.match(r"^:?-{3,}:?$", (c or "").strip())) for c in stripped.split("|")[1:-1]):
                out_lines.append(line)
                continue
            # Data row: check first column prefix against drop_prefixes
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            first_cell = cells[0] if cells else ""
            first_lower = first_cell.lower()
            if first_lower.startswith(drop_prefixes):
                continue  # drop ready/merge/nightly/weekly/ut/ut (exclude models)
            out_lines.append(line)
        else:
            if in_table:
                in_table = False
            out_lines.append(line)

    return "\n".join(out_lines)


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
    new_row = _append_device_hours_build_row(new_row)

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


#: Marker substituted by ``release_md_to_html._upgrade_device_hours_cell``
#: into the editable inline input. Lives in the **Success rate/UT coverage**
#: column of the Device-Hours / Build row. Markdown rendering keeps the
#: raw marker text; HTML conversion replaces it with a click-to-fill input
#: backed by ``localStorage['device-hours-per-build']``.
DEVICE_HOURS_PER_BUILD_MARKER = "@@DEVICE_HOURS_PER_BUILD_CELL@@"


def _append_device_hours_build_row(ci_row: str) -> str:
    """Append the manual **Device-Hours / Build** row beneath ``ci_row``.

    The release Metrics overview terminates with the CI issue detection rate
    row. Operators track compute burn via a separate spreadsheet; rather than
    wiring the source into the report (which would tie the script to that
    sheet), we append a stub row whose **Success rate/UT coverage** cell is
    a magic marker. The Markdown renderer keeps the marker verbatim so the
    release Markdown export round-trips, while :func:`_upgrade_device_hours_cell`
    in :mod:`release_md_to_html` swaps it for an editable input.
    """
    return (
        ci_row
        + "\n"
        + "| **Device-Hours / Build** | "
        + DEVICE_HOURS_PER_BUILD_MARKER
        + " | - | - | - |"
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
            ["CI category", "Success rate/UT coverage", "Avg duration", "Other finished count", "Bug avg first response"],
            [
                [
                    f"bugs (first response, {stats_from}..{stats_to})",
                    "-",
                    "-",
                    "-",
                    "6.2h",
                ],
                [
                    f"CI issue detection rate ({stats_from}..{stats_to})",
                    "60.0% (3/5)",
                    "-",
                    "-",
                    "-",
                ],
                [
                    "**Device-Hours / Build**",
                    DEVICE_HOURS_PER_BUILD_MARKER,
                    "-",
                    "-",
                    "-",
                ],
                [
                    "*Note*",
                    "*Remove `--preview` and configure tokens to replace with real `buildkite_build_stats.py` output.*",
                    "-",
                    "-",
                    "-",
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

    # B200 (CI — Buildkite scheduled release) preview body. Sample counts
    # mirror the latest real B200 build (#15964, failed) so the preview renders
    # the chapter without a Buildkite call. Build + Summary only (failed-step
    # detail lives in the Failure Analysis preview block below).
    b200_body = build_b200_ci_markdown_body(
        build_no=build_no,
        branch=BRANCH,
        commit="c0ffee1deadbeefcafe000000000000000000001",
        state="failed",
        created_at="2026-09-23T09:53:27Z",
        passed=42,
        failed=9,
        broken=0,
    )
    b200_failed_steps = [
        ("Omni · Function Test with H100 · 2-GPU", "failed", demo_link_a),
        ("TTS · Function Test with L4", "failed", demo_link_b),
    ]

    test_result = render_test_result_section(
        skill_dir,
        log_h200=None,
        log_h800=None,
        log_a100=None,
        log_a3=None,
        h100_ci_markdown=h100_body,
        b200_ci_markdown=b200_body,
        b200_passed=42,
        b200_failed=9,
        b200_broken=0,
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
        b200_build_no=build_no,
        b200_build_url=f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{build_no}",
        b200_failed_steps=b200_failed_steps,
        include_b200=True,
    )

    open_issues_block = (
        "## Open issues (stats window)\n\n"
        f"Open issues labeled **bug**, state **open**, excluding PRs, with `created_at` "
        f"(UTC date) in **{stats_from}** … **{stats_to}**, filtered to issues whose highest-priority "
        f"label is `critical` / `high priority` / `medium priority` (drops `low priority`, `invalid`, "
        f"and unlabelled-priority bugs). "
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
                # Note: a `low priority` row was previously hardcoded here to demonstrate
                # the full priority range; it is intentionally omitted now that the release
                # Open issues table is narrowed to ``critical`` / ``high priority`` /
                # ``medium priority`` via ``OPEN_ISSUES_RELEASE_PRIORITIES``.
                [
                    "[#10061](https://github.com/vllm-project/vllm-omni/issues/10061)",
                    "Tokenizer hangs on multi-byte UTF-8 input",
                    "2026-05-20",
                    "critical",
                    "10",
                    "open",
                    "@dave-preview",
                    *OPEN_ISSUE_ACTION_CELLS,
                ],
            ],
        )
        + "\n"
    )

    next_steps_block = render_next_steps_section()

    # Quality Defense Radar: per-model 7-axis coverage across 10 flagship
    # models (Qwen3-Omni, MiniCPM, Qwen-TTS, Qwen-Image, HunyuanImage,
    # HunyuanVideo, Wan, MinimaxH3, Cosmos, LingBot), up to 13 clickable
    # segments per model. Release variant only — the preview also emits it
    # so the layout matches the live report HTML exactly.
    quality_defense_block = render_quality_defense_section()

    metric_analysis_section = render_metric_analysis_section()

    return f"""# vLLM-Omni Test Report - Scheduled Nightly

{conclusion}{ci_md}

{metric_analysis_section}

{quality_defense_block}

{test_result}

{failure_analysis}
{open_issues_block}
{next_steps_block}
## Data source

- **Mode:** `compose_full_report.py --preview` (sample tables only)
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-*`; H100 is Buildkite block
- **Failure Analysis:** Per-GPU failure detail. Interactive **Status** column
  (Filed / Not an issue) backed by `localStorage`.
- **Open issues:** Preview narrows the table to `critical` / `high priority` / `medium priority`
  (matches live report via ``OPEN_ISSUES_RELEASE_PRIORITIES``); `low priority` / `invalid` / unlabelled
  rows are dropped.
- **Next Steps (Outstanding Items):** manual-entry action table — see H2 between Open issues and Data
  source. HTML upgrade adds **Add Item** button and `localStorage`-backed editing (same as Development).
- **Quality Defense Radar:** per-model 7-axis coverage radar across 10 flagship models
  (Qwen3-Omni, MiniCPM, Qwen-TTS, Qwen-Image, HunyuanImage, HunyuanVideo, Wan, MinimaxH3,
  Cosmos, LingBot). Six axes split into GPU/NPU halves of one circle; Reliability is a
  single circle (4 models); Cosmos renders all axes as single circles. Segments pre-lit
  from the vllm-omni repo code; click to override. State kept in `localStorage`
  (`quality-defense:<model>:<segment-id>`) and mirrored to a `data-quality-on` attribute.
  No token required.
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
            "Metrics overview (only the bugs-first-response row is shown — "
            "ready/merge/nightly/weekly/ut rows are dropped) + "
            "Failure Analysis (per-GPU with interactive Status column) + "
            "Open issues (stats window) + Next Steps (Outstanding Items) + "
            "Quality Defense Radar (per-model 7-axis coverage across 10 flagship models). "
            "``development`` — same Test Result layout as release, but **Test conclusion** and "
            "**Open issues** sections are omitted and **Metrics overview** is replaced with "
            "a Development-flavored 2-row operator-editable snapshot "
            "(UT coverage · Device-Hours / Build); both cells are click-to-edit "
            "and persisted via ``localStorage``."
        ),
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
            "(or vllm-omni-test-report-development-YYYY-MM-DD.html for --kind development)."
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
        "--log-dir-a3",
        type=Path,
        default=None,
        help="Optional. Log root for **Test Result → A3** (same layout as --log-dir-h200).",
    )
    parser.add_argument(
        "--b200-build",
        type=int,
        default=None,
        help=(
            "Optional. Pin a specific **Scheduled Release B200 build** number for "
            "the Test Result → B200 (CI — Buildkite scheduled release) chapter "
            "(release variant only). When omitted, the latest 'Scheduled Release "
            "B200 build' on main is resolved automatically by paging recent "
            "builds. Useful when the B200 cadence stretches beyond the resolver's "
            "~75-day scan window."
        ),
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

    # Resolve default output filename based on kind.
    def _default_output_path() -> Path:
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
        return skill_dir / base

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
        out_path.write_text(
            convert_release_report_markdown(
                md,
                l2_l3_row_ok=True,
                l2_l3_row_detail="",
                di_row_ok=True,
                di_row_detail="Auto DI=4.1 (high priority=1, medium priority=1, low priority=1)",
                critical_row_ok=True,
                critical_row_detail="",
            ),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")
        return

    # ---- Live (non-preview) path ----

    # Capture one reference instant so all SLO-escalating DI computations in
    # this run (development snapshot, nightly Daily focus when invoked from
    # this script, release conclusion row, etc.) use the same ``days_open``.
    # Without this, a single critical bug can tick over its 1-day SLO between
    # two calls in the same process and silently add 10 DI to one report but
    # not another, making the development vs nightly Top10 tables diverge.
    from datetime import timezone as _tz

    report_now = datetime.now(_tz.utc)

    if args.kind == "development":
        # Development report: shares Test Result layout with release, but
        # - skips Test conclusion
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

        # Test Result: Overall test execution summary table + per-GPU nightly
        # summaries. H100 is intentionally excluded from the development variant
        # (it lives in the Buildkite CI side, not the local nightly log roll-up)
        # — we pass an empty `h100_ci_markdown` so the H100 panel is dropped
        # entirely. A3 (the fourth local GPU) follows the H200/H800/A100 pattern.
        test_result = render_test_result_section(
            skill_dir,
            log_h200=args.log_dir_h200,
            log_h800=args.log_dir_h800,
            log_a100=args.log_dir_a100,
            log_a3=args.log_dir_a3,
            h100_ci_markdown="",
        )

        # Failure Analysis: top-level section, one collapsible subsection per
        # local GPU. H100 and B200 are dropped for the development variant.
        failure_analysis = render_failure_analysis_section(
            log_h200=args.log_dir_h200,
            log_h800=args.log_dir_h800,
            log_a100=args.log_dir_a100,
            log_a3=args.log_dir_a3,
            include_h100=False,
            include_b200=False,
        )

        dev_metrics_md = render_development_metrics_overview()
        # open_issues_block is intentionally omitted from the development variant
        # (Metrics overview is fully operator-editable; the DI ladder lives in
        # the nightly report's Daily focus card + the issue-monitor report).

        # Next Steps (Outstanding Items): manual-entry action table
        next_steps_block = render_next_steps_section()

        # Skip Test Case Monitoring: static scan of `tests/**` for pytest skips
        # whose reason references a GitHub issue, cross-referenced via the
        # GitHub REST API. The pull is fast-forward-only and non-fatal so a
        # dirty working tree never blocks report generation.
        skip_monitor_block = render_skip_issue_monitor_section(
            repo_root=args.omni_repo_root,
            gh_token=gh_token,
            pull=not args.no_repo_pull,
        )

        # Resource Usage Analysis: manual-entry editor block. Renders an H2
        # section whose body is an in-place editable text box in HTML (the
        # marker is replaced by the HTML upgrade step with a <textarea> +
        # Reset/Save controls, persisted via localStorage). Lives only in the
        # development variant — release reports do not include it.
        resource_usage_block = render_resource_usage_section()

        md = f"""# vLLM-Omni Test Report - Development

* **Report date (UTC):** {today_utc}

{dev_metrics_md}

{test_result}

{failure_analysis}

{skip_monitor_block}

{resource_usage_block}

{next_steps_block}
## Data source

- **Kind:** `compose_full_report.py --kind development`
- **Metrics overview (Development, 2 operator-editable rows):**
  - UT coverage — click-to-edit cell; value persists locally via `localStorage`.
  - Device-Hours / Build — click-to-edit input; value persists via
    `localStorage['device-hours-per-build']` (the same editable control as the
    release Metrics overview row). No metric is auto-fetched in this section.
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100/A3 via
  `--log-dir-h200` / `--log-dir-h800` / `--log-dir-a100` / `--log-dir-a3`. **The H100
  (CI — Buildkite scheduled nightly) chapter is intentionally omitted from the
  development variant** — the development audience consumes local nightly logs,
  not the Buildkite pipeline roll-up (this build #{build_no}; reportable jobs only —
  upload steps excluded).
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
- **Open issues:** This section is **omitted** from the development variant (the DI ladder
  lives in the nightly report's Daily focus card and the issue-monitor report).
"""
        out_path.write_text(
            convert_release_report_markdown(
                md,
                l2_l3_row_ok=True,
                l2_l3_row_detail="",
                di_row_ok=True,
                di_row_detail="(Development: Test conclusion omitted)",
                critical_row_ok=True,
                critical_row_detail="",
            ),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")
        return

    # ---- ``--kind release`` (default) live path ----

    # The Buildkite scheduled-nightly fetch + reportable-job walk that used to
    # live here has been removed: the operator requested dropping all Buildkite
    # CI roll-up content (H100 chapter in Test Result, H100 subsection in
    # Failure Analysis, Latest GPU CI auto-judge). The release report now only
    # consumes Buildkite data via ``buildkite_build_stats.py`` for the
    # Metrics-overview bugs row. ``token`` is still required by the metric
    # script so we don't drop the early-existence check.
    if not token:
        raise SystemExit(
            "BUILDKITE_API_TOKEN (or BUILDKITE_TOKEN) must be set in the "
            "environment for the release Metrics overview."
        )
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
    # user-reported issues. ``append_ci_issue_detection_rate_row`` also
    # bundles the operator-editable **Device-Hours / Build**
    # row beneath it via ``_append_device_hours_build_row``; the cell
    # carries a ``@@DEVICE_HOURS_PER_BUILD_CELL@@`` marker that
    # ``release_md_to_html`` converts to an inline editable input backed
    # by ``localStorage`` (Markdown export keeps the marker verbatim).
    ci_md = append_ci_issue_detection_rate_row(ci_md, gh_token, stats_from, stats_to)
    # Collapse the upstream 5-column metrics table to a 2-column Indicator / Value
    # shape so the release Metrics overview reads as a clean name+value list.
    ci_md = restructure_metrics_to_two_columns(ci_md)

    # Latest GPU CI(L1-L5) row is now **manual** (operator-selectable) — the
    # Buildkite gate is no longer auto-judged. Skip the l2_l3_ready_merge_gate
    # call entirely so the row defaults to user-selectable Pass in the widget.
    l2_l3_row_ok = None
    l2_l3_row_detail = ""

    critical_row_ok, critical_row_detail = no_open_critical_labeled_issues(gh_token)

    # H100 (Buildkite scheduled nightly) data is intentionally NOT fetched or
    # rendered in the release Test Result / Failure Analysis sections anymore.
    # Operators requested removing Buildkite CI roll-up content from the report
    # — only local-machine test execution (H200/H800/A100/A3) remains. The
    # Buildkite API call + reportable-job walk (previously used to populate
    # the H100 chapter + the H100 failed-step block) is therefore skipped.
    #
    # B200 (Buildkite scheduled release) is the one Buildkite CI chapter kept in
    # the release Test Result. Fetch the latest "Scheduled Release B200 build"
    # (or a pinned --b200-build), walk its reportable jobs, and build a
    # lightweight metadata + summary + failed-step body. Failures degrade
    # gracefully (empty body) so a Buildkite hiccup never aborts the report.
    b200_body = ""
    b200_passed: int | None = None
    b200_failed: int | None = None
    b200_broken: int | None = None
    b200_build_no: int | None = None
    b200_build_url: str | None = None
    b200_failed_steps: list[tuple[str, str, str]] | None = None
    try:
        b200 = fetch_b200_build_summary(token, build_no=args.b200_build)
        b200_passed = b200["passed"]
        b200_failed = b200["failed"]
        b200_broken = b200["broken"]
        b200_build_no = b200["build_no"]
        b200_build_url = f"https://buildkite.com/{ORG}/{PIPELINE}/builds/{b200_build_no}"
        b200_failed_steps = b200["failed_steps"]
        b200_body = build_b200_ci_markdown_body(
            build_no=b200["build_no"],
            branch=b200["branch"],
            commit=b200["commit"],
            state=b200["state"],
            created_at=b200["created_at"],
            passed=b200["passed"],
            failed=b200["failed"],
            broken=b200["broken"],
        )
        print(
            f"B200: build #{b200['build_no']} [{b200['state']}] "
            f"({(b200['created_at'] or '')[:10]}) — "
            f"{b200['passed']} passed / {b200['failed']} failed / {b200['broken']} broken",
            file=sys.stderr,
        )
    except Exception as e:  # noqa: BLE001 — graceful degradation
        print(f"B200: fetch failed — {e}; chapter omitted", file=sys.stderr)

    conclusion = render_test_conclusion_section()
    # H100 (Buildkite scheduled nightly) chapter is intentionally omitted from
    # the release Test Result — the operator requested dropping that Buildkite
    # CI roll-up; only local-machine execution (H200/H800/A100/A3) plus the
    # **B200 Buildkite scheduled release** chapter remain. ``h100_ci_markdown=""``
    # suppresses the H100 chapter and H100 row in the Overall summary table;
    # ``b200_*`` carries the fetched B200 build's body + counts.
    test_result = render_test_result_section(
        skill_dir,
        log_h200=args.log_dir_h200,
        log_h800=args.log_dir_h800,
        log_a100=args.log_dir_a100,
        log_a3=args.log_dir_a3,
        h100_ci_markdown="",
        h100_passed=None,
        h100_failed=None,
        h100_skipped=None,
        b200_ci_markdown=b200_body,
        b200_passed=b200_passed,
        b200_failed=b200_failed,
        b200_broken=b200_broken,
    )

    # Failure Analysis: top-level section, one collapsible subsection per
    # local GPU (H200/H800/A100 from local logs). H100 (Buildkite) is
    # intentionally omitted from the release report. B200 (Buildkite scheduled
    # release) failed steps are rendered here (the Test Result B200 chapter
    # carries only Build + Summary).
    failure_analysis = render_failure_analysis_section(
        log_h200=args.log_dir_h200,
        log_h800=args.log_dir_h800,
        log_a100=args.log_dir_a100,
        log_a3=args.log_dir_a3,
        h100_build_no=None,
        h100_build_url=None,
        h100_failed_steps=None,
        include_h100=False,
        b200_build_no=b200_build_no,
        b200_build_url=b200_build_url,
        b200_failed_steps=b200_failed_steps,
        include_b200=True,
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
    # Uses the SLO-escalating model (`DI = base × ⌈days_open / slo_days⌉`),
    # same as the Development variant and the nightly Daily focus card; the
    # shared ``report_now`` keeps this conclusion row, the Open issues table
    # and any companion Development / nightly report on the same time base.
    open_issues_block = render_open_issues_section(
        stats_from, stats_to, gh_token,
        all_open=False,
        now=report_now,
        priority_filter=OPEN_ISSUES_RELEASE_PRIORITIES,
    )

    # Next Steps (Outstanding Items): manual-entry action table, identical
    # to the Development variant. ``release_md_to_html`` already injects
    # ``_upgrade_next_steps_outstanding_cells`` +
    # ``_NEXT_STEPS_OUTSTANDING_SCRIPT`` unconditionally for both variants
    # and ``_release_section_theme`` auto-applies the ``--outstanding``
    # theme modifier (clipboard SVG + red accent) when the H2 contains
    # ``outstanding items``, so emitting the H2 in the release markdown is
    # sufficient — no upgrade-script changes required.
    next_steps_block = render_next_steps_section()

    # Quality Defense Radar: per-model 7-axis coverage across 10 flagship
    # models (Qwen3-Omni, MiniCPM, Qwen-TTS, Qwen-Image, HunyuanImage,
    # HunyuanVideo, Wan, MinimaxH3, Cosmos, LingBot), up to 13 clickable
    # segments per model (6 axes split into GPU/NPU halves + Reliability
    # single). Release variant only — the
    # post-processor replaces the marker with the inline SVG via
    # ``_upgrade_quality_defense_block`` and wires the click-toggle script
    # via ``_QUALITY_DEFENSE_SCRIPT``; state persists via localStorage
    # + ``data-quality-on`` attribute (mirrors the existing pattern used by
    # ``oi-followup`` / ``ns-outstanding`` / ``fail-status``).
    quality_defense_block = render_quality_defense_section()
    metric_analysis_section = render_metric_analysis_section()
    # "Remaining DI < 30" is a **manual** user-selectable Pass/Fail row (like
    # the NPU row) — the operator judges it against the SLO-escalating
    # Outstanding DI (sum of per-issue SLO DI across open `label:bug` whose
    # `created_at` ≤ stats_to; threshold ≤ 30). Passing ``di_row_ok=None``
    # renders the row as a clickable toggle (see ``release_conclusion_widget_html``:
    # ``is_auto = auto_ok is not None``) and never overrides the operator's
    # selection with an auto-judgement. The Open issues section below still
    # lists the bugs the operator would judge against
    # (``render_open_issues_section`` is unchanged) — only the conclusion-row
    # auto-judgement is dropped.

    md = f"""# vLLM-Omni Test Report - Scheduled Nightly

{conclusion}{ci_md}

{metric_analysis_section}

{quality_defense_block}

{test_result}

{failure_analysis}
{open_issues_block}
{next_steps_block}
## Data source

- **Test conclusion (auto):** (1) Buildkite **ready** (non-main) and **merge** (main non-nightly/weekly)
  each latest **finished** build has no `failed`/`broken` job (Upload * Pipeline steps
  excluded); (2) **Remaining DI < 30** is a **manual** user-selectable Pass/Fail row (like
  the NPU row) — the operator judges it against the SLO-escalating Outstanding DI = sum of
  per-issue SLO DI
  (`DI = base × ⌈days_open / slo_days⌉` with `critical=10/slo=1d`, `high priority=3/slo=5d`,
  `medium priority=1/slo=10d`, `low priority=0.1/slo=14d`, `invalid=0`) across open
  `label:bug` whose `created_at` ≤ `{stats_to}` (start date unbounded; issues created
  after the stats window are excluded); threshold ≤ 30 (Pass when total ≤ 30,
  Fail when > 30). No longer auto-calculated — the operator's selection is authoritative;
  (3) no open
  `label:bug` + `label:critical`; (4) `UT coverage meets this iteration requirement
  (Guide), Performance regression < 10% (Guide)` is a manual user-selectable
  row that does **not** influence the final Go / Rejected verdict.
- **Test Result:** Common stack from `references/local-test-matrix.md`; H200/H800/A100 via
  `--log-dir-h200` / `--log-dir-h800` / `--log-dir-a100`; H100 chapter intentionally omitted
  (Buildkite scheduled nightly roll-up no longer rendered in the release report).
- **Failure Analysis:** Per-GPU failure detail (H200/H800/A100 from local nightly logs; H100 from
  Buildkite `failed` steps — `broken` is treated as a transient state, not a failure).
  Interactive **Status** column (Filed / Not an issue) backed by `localStorage`.
- **Open issues:** REST `label:bug`, `created_at` UTC date in `{stats_from}`..`{stats_to}`,
  filtered to issues whose highest-priority label is `critical` / `high priority` /
  `medium priority` (drops `low priority`, `invalid`, and unlabelled-priority bugs).
- **Next Steps (Outstanding Items):** manual-entry action table (Item / Assignee /
  Status) with ``Add Item`` button; cells editable in HTML, persisted via
  `localStorage` (same implementation as the Development variant). H2 appears between
  Open issues and Data source.
- **Quality Defense Radar:** per-model 7-axis coverage radar across 10 flagship models
  (Qwen3-Omni, MiniCPM, Qwen-TTS, Qwen-Image, HunyuanImage, HunyuanVideo, Wan,
  MinimaxH3, Cosmos, LingBot). Six axes (Functionality / Performance / Documentation /
  Precision / Stability / Gate) split into GPU/NPU halves of one circle; Reliability is
  a single circle (4 models); Cosmos renders all axes as single circles. Segments are
  pre-lit from the vllm-omni repo code and remain click-to-override. State persisted via
  `localStorage` (key `quality-defense:<model-id>:<segment-id>`) and mirrored to a
  `data-quality-on` attribute on each `<g>` so `Ctrl+S` Save-Page-As preserves state
  across origins. Release variant only — no token required.
- Buildkite API: `{ORG}/{PIPELINE}` branch `main`
- `scripts/buildkite_build_stats.py --from {stats_from} --to {stats_to} --markdown` (**bugs (first response, …)** =
  GitHub `label:bug` issues with `created_at` UTC date in the same `--from`..`--to` window)
"""
    out_path.write_text(
        convert_release_report_markdown(
            md,
            l2_l3_row_ok=l2_l3_row_ok,
            l2_l3_row_detail=l2_l3_row_detail,
            # di_row_ok=None ⇒ the "Remaining DI < 30" conclusion row renders as a
            # manual Pass/Fail toggle (operator-selectable), not an auto-judgement.
            di_row_ok=None,
            di_row_detail="",
            critical_row_ok=critical_row_ok,
            critical_row_detail=critical_row_detail,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
