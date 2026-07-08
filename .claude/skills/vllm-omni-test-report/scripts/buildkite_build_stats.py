#!/usr/bin/env python3
"""
Fetch vllm-omni builds from the Buildkite REST API for a date range and compute:

  - Success rate: passed / (passed + failed)
  - Average duration: arithmetic mean wall time (finished_at - created_at) for
    passed/failed builds that have both created_at and finished_at
  - With `--markdown`: main table has **Success rate/UT coverage**, **Bug avg first response** (GitHub: open +
    closed `label=bug`, **`created_at` UTC date** in the same **`--from`..`--to`** window as Buildkite stats,
    mean time issue opened -> first comment), plus **ut** /
    **ut (exclude models)** from Simple Unit Test log on the **latest `main` build that is not** a
    scheduled nightly or scheduled weekly (same ``classify_build`` heuristic as the merge bucket;
    not the raw newest build on the pipeline).
    ``ut (exclude models)`` uses per-file lines minus any path with a directory segment named
    ``models``; ready/merge/nightly/weekly
    rows use CI build stats only. The emitted
    Markdown table lists percentages only, not those path rules.

Four CI buckets (display names in reports):

  1. ready — non-`main` branches
  2. merge — `main`, ordinary runs (e.g. merged PRs), not scheduled nightly/weekly
  3. nightly — `main`, scheduled nightly (or other scheduled runs on `main` that match the legacy
     nightly heuristic but **not** scheduled weekly)
  4. weekly — `main`, Buildkite **Scheduled weekly** (message matches ``Scheduled weekly`` / ``scheduled weekly``)

Usage:

  Set BUILDKITE_API_TOKEN (or BUILDKITE_TOKEN). Optional: GITHUB_TOKEN (or GH_TOKEN) for GitHub bug metrics.
  pip install requests  # if missing
  python scripts/buildkite_build_stats.py [--from YYYY-MM-DD --to YYYY-MM-DD] [--markdown]

If `--from` / `--to` are both omitted, the window is the current UTC calendar month through today
(month-start 00:00 UTC to today end UTC). If you pass one, pass both (UTC dates, inclusive).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from md_table import render_markdown_table  # noqa: E402

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

# API constants
BUILDKITE_API_BASE = "https://api.buildkite.com/v2"
ORG_SLUG = "vllm"
PIPELINE_SLUG = "vllm-omni"

# Only finished builds; passed/failed count toward success rate; canceled/blocked reported separately
FINISHED_STATES = {"passed", "failed", "canceled", "blocked", "skipped", "not_run"}
SUCCESS_STATE = "passed"
FAIL_STATE = "failed"
# Step/job states that count toward a build's "all-pass / some-fail" verdict.
# Only ``failed`` (a real runtime test failure) counts. ``broken`` is a
# transient Buildkite pipeline-execution state — e.g. ``:docker: Build image``
# when the Docker registry is unreachable, or ``:email: Nightly Collection &
# Email`` when SMTP/kanban is down — and should never be treated as a CI
# regression. ``skipped`` / ``not_run`` / ``blocked`` / ``canceled`` mean the
# job never ran and also must not count. Mirrors the Test Result H100 filter
# in ``compose_full_report.py``.
FAIL_JOB_STATES = {"failed"}


def default_created_range_utc() -> tuple[str, str]:
    """First day of current UTC month through today (UTC), as YYYY-MM-DD strings."""
    today = datetime.now(timezone.utc).date()
    start = today.replace(day=1)
    return start.isoformat(), today.isoformat()


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


def get_api_token() -> str | None:
    token = os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN")
    return token.strip() if token else None


def get_github_token() -> str | None:
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return t.strip() if t else None


def _github_tls_verify() -> bool:
    """
    TLS certificate verification for GitHub ``requests`` calls.

    Set ``GITHUB_INSECURE_SSL`` to ``1`` / ``true`` / ``yes`` / ``on`` to disable verification
    (not recommended; use only when the OS trust store cannot validate ``api.github.com``, e.g.
    some corporate proxies). Buildkite requests are always verified.
    """
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


# GitHub REST: vllm-project/vllm-omni issues (label bug) for --markdown bug first-response column
GITHUB_REPO_REST = "https://api.github.com/repos/vllm-project/vllm-omni"


def _github_headers(gh_token: str | None) -> dict[str, str]:
    h: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "vllm-omni-buildkite-build-stats",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if gh_token:
        h["Authorization"] = f"Bearer {gh_token}"
    return h


def _fetch_bug_issues_state_created_range(
    state: str,
    date_from: str,
    date_to: str,
    gh_token: str | None,
) -> list[dict]:
    """
    state is open or closed; label=bug; excludes PR entries.
    Only issues whose ``created_at`` UTC calendar date (``YYYY-MM-DD``) lies in
    ``date_from`` .. ``date_to`` inclusive. Uses sort=created desc and stops once a page
    contains issues older than ``date_from`` (avoids scanning all closed history).
    """
    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{GITHUB_REPO_REST}/issues",
            params={
                "state": state,
                "labels": "bug",
                "per_page": 100,
                "page": page,
                "sort": "created",
                "direction": "desc",
            },
            headers=_github_headers(gh_token),
            timeout=60,
            verify=_github_tls_verify(),
        )
        if r.status_code == 403:
            raise RuntimeError("GitHub API 403 (rate limit or forbidden). Set GITHUB_TOKEN for higher limits.")
        r.raise_for_status()
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        stop_paging = False
        for i in batch:
            if i.get("pull_request"):
                continue
            ca = str(i.get("created_at") or "")
            if len(ca) < 10:
                continue
            d = ca[0:10]
            if d > date_to:
                continue
            if d < date_from:
                stop_paging = True
                continue
            out.append(i)
        if stop_paging or len(batch) < 100:
            break
        page += 1
    return out


def _first_comment_delay_seconds(issue: dict, gh_token: str | None) -> float | None:
    """Seconds from issue created_at to first comment created_at; None if no comments."""
    num = issue.get("number")
    if num is None:
        return None
    r = requests.get(
        f"{GITHUB_REPO_REST}/issues/{num}/comments",
        params={"per_page": 1, "page": 1},
        headers=_github_headers(gh_token),
        timeout=30,
        verify=_github_tls_verify(),
    )
    r.raise_for_status()
    comments = r.json()
    if not isinstance(comments, list) or not comments:
        return None
    t0 = parse_buildkite_time(issue.get("created_at"))
    t1 = parse_buildkite_time(comments[0].get("created_at"))
    if t0 is None or t1 is None:
        return None
    return max(0.0, (t1 - t0).total_seconds())


def github_bug_avg_first_response_range_cell(
    date_from: str,
    date_to: str,
) -> str:
    """
    Mean time from issue open to first comment for issues with label bug and
    ``created_at`` UTC date in ``date_from`` .. ``date_to`` (inclusive), combining open
    and closed issue lists (de-duped by issue number).

    See:
    https://github.com/vllm-project/vllm-omni/issues?q=is%3Aissue+state%3Aopen+label%3Abug
    https://github.com/vllm-project/vllm-omni/issues?q=is%3Aissue+state%3Aclosed+label%3Abug
    """
    gh = get_github_token()
    by_num: dict[int, dict] = {}
    try:
        for state in ("open", "closed"):
            for i in _fetch_bug_issues_state_created_range(state, date_from, date_to, gh):
                n = i.get("number")
                if isinstance(n, int):
                    by_num[n] = i
    except requests.RequestException as e:
        return f"*N/A* (GitHub API error: {e})"
    except RuntimeError as e:
        return f"*N/A* ({e})"

    window_issues = list(by_num.values())
    if not window_issues:
        return f"*N/A* (no bug issues created in {date_from}..{date_to})"

    delays: list[float] = []
    for i in window_issues:
        try:
            d = _first_comment_delay_seconds(i, gh)
        except requests.RequestException:
            continue
        if d is not None:
            delays.append(d)

    if not delays:
        return f"*N/A* (0 issues with comments among {len(window_issues)} opened in {date_from}..{date_to})"

    avg = sum(delays) / len(delays)
    return f"**{format_duration(avg)}** (avg, n={len(delays)}/{len(window_issues)} issues)"


def parse_link_header(link: str | None) -> dict[str, str]:
    """Parse RFC 5988 Link header; return rel -> url."""
    if not link:
        return {}
    out = {}
    for part in link.split(","):
        part = part.strip()
        m = re.match(r'<([^>]+)>;\s*rel="([^"]+)"', part)
        if m:
            out[m.group(2).strip().lower()] = m.group(1).strip()
    return out


def fetch_builds(
    token: str,
    created_from: str,
    created_to: str,
    *,
    branch: str | None = None,
    per_page: int = 100,
) -> list[dict]:
    """Fetch all builds for the pipeline created in [created_from, created_to] (paginated)."""
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds"
    params: dict[str, str | int] = {
        "created_from": created_from,
        "created_to": created_to,
        "per_page": per_page,
    }
    if branch:
        params["branch"] = branch
    headers = {"Authorization": f"Bearer {token}"}
    all_builds: list[dict] = []

    while True:
        last_exc: Exception | None = None
        r: requests.Response | None = None
        for attempt in range(10):
            try:
                if params:
                    r = requests.get(url, params=params, headers=headers, timeout=180)
                else:
                    r = requests.get(url, headers=headers, timeout=180)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After", "60")
                    try:
                        wait_s = int(float(ra)) + 1
                    except ValueError:
                        wait_s = 61
                    time.sleep(min(180, max(1, wait_s)))
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last_exc = e
                if attempt < 9:
                    time.sleep(min(8, 2 ** min(attempt, 3)))
        else:
            assert last_exc is not None
            raise last_exc
        assert r is not None
        data = r.json()
        page = data if isinstance(data, list) else [data]
        all_builds.extend(page)

        link = r.headers.get("Link") or r.headers.get("link")
        links = parse_link_header(link)
        next_url = links.get("next")
        if not next_url:
            break
        # Follow next URL as-is; do not send params again
        url = next_url
        params = {}
        time.sleep(
            max(
                0.0,
                float(os.environ.get("BUILDKITE_BUILDS_PAGE_SLEEP", "0.12")),
            )
        )

    return all_builds


_WEEKLY_MSG = re.compile(r"scheduled\s+weekly", re.IGNORECASE)


def is_scheduled_weekly(build: dict) -> bool:
    """True if build message indicates Buildkite *Scheduled weekly* (main-only bucket uses branch check)."""
    return bool(_WEEKLY_MSG.search(build.get("message") or ""))


def is_nightly_bucket(build: dict) -> bool:
    """
    True if this main-branch build counts toward the **nightly** metrics row.

    Excludes **scheduled weekly** (handled in ``classify_build``). Includes other scheduled
    ``main`` runs and the legacy message heuristics (``nightly``, ``scheduled`` + ``build``).
    """
    if is_scheduled_weekly(build):
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
    """Return 'non_main' | 'main_non_nightly' | 'main_nightly' | 'main_weekly'."""
    branch = (build.get("branch") or "").strip()
    if branch != "main":
        return "non_main"
    if is_scheduled_weekly(build):
        return "main_weekly"
    if is_nightly_bucket(build):
        return "main_nightly"
    return "main_non_nightly"


@dataclass
class Bucket:
    passed: int = 0
    failed: int = 0
    other_finished: int = 0  # canceled, blocked, etc.
    # Wall-clock seconds for passed/failed builds with both created_at and finished_at
    duration_seconds: list[float] = field(default_factory=list)

    @property
    def total_for_success_rate(self) -> int:
        return self.passed + self.failed

    @property
    def success_rate(self) -> float | None:
        t = self.total_for_success_rate
        if t == 0:
            return None
        return self.passed / t

    @property
    def avg_duration_seconds(self) -> float | None:
        if not self.duration_seconds:
            return None
        return sum(self.duration_seconds) / len(self.duration_seconds)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m}m{s}s"


SIMPLE_UNIT_TEST_JOB_RE = re.compile(r"^simple\s+unit\s+test\s*$", re.IGNORECASE)

# Simple Unit Test: fetch the step log up to this many bytes (rolling **tail** if the log is larger).
# The coverage text report starts at pytest's ``tests coverage`` banner and can be many MiB; a small tail
# can omit the **beginning** of the per-file table while still containing ``TOTAL``, which skews sums.
UT_LOG_MAX_BYTES = 200_000_000


def read_log_tail(token: str, log_url: str, *, tail_bytes: int = 12_000_000) -> str:
    """Fetch step log; keep only the last tail_bytes to cap memory (TOTAL line is near the end)."""
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(log_url, headers=headers, stream=True, timeout=300)
    r.raise_for_status()
    buf = bytearray()
    for chunk in r.iter_content(chunk_size=65_536):
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > tail_bytes:
            buf = buf[-tail_bytes:]
    return buf.decode("utf-8", errors="replace")


def read_log_tail_capped(token: str, log_url: str, *, max_bytes: int) -> str:
    """Like :func:`read_log_tail` but the size parameter is a maximum log length (default for UT is large)."""
    return read_log_tail(token, log_url, tail_bytes=max_bytes)


def extract_pytest_coverage_text_report(log_text: str) -> str:
    """
    Keep pytest's text coverage report from the ``tests coverage`` banner through EOF.

    Pytest prints ``================ tests coverage ================`` (with optional CI timestamp prefix)
    before the ``Name / Stmts / Miss / ...`` table and the final ``TOTAL`` row. Parsing the whole step log
    can pick up unrelated lines; restricting to this section matches a full local ``coverage report``.

    Uses the **last** occurrence of the substring ``tests coverage`` if the log contains multiple runs.
    If the marker is missing, returns ``log_text`` unchanged.
    """
    key = "tests coverage"
    pos = log_text.rfind(key)
    if pos == -1:
        return log_text
    line_start = log_text.rfind("\n", 0, pos) + 1
    return log_text[line_start:]


def _strip_ci_log_prefix_and_ansi(line: str) -> str:
    """Strip Buildkite ``[ISO-timestamp] `` prefix and ANSI SGR codes so coverage rows match cleanly."""
    s = line.rstrip()
    s = re.sub(r"^\[[^\]]+\]\s*", "", s)
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)
    return s.strip()


def parse_coverage_total_row(
    log_text: str,
) -> tuple[str | None, int | None, int | None]:
    """
    Find the coverage.py text-report **TOTAL** row (last such row in ``log_text``).

    Requires the row to **start** with ``TOTAL`` as a word (after optional CI log prefix), so lines that
    merely *contain* ``TOTAL`` in a path or prose are ignored. Integers between ``TOTAL`` and the trailing
    ``NN%`` are parsed; the first two are **Stmts** and **Miss** (line or branch report; branch adds more
    columns before Cover ``%``).

    Returns ``(cover_pct_digits, stmts, miss)`` e.g. ``("18", 64377, 51384)``, or three ``None`` if absent.
    """
    for line in reversed(log_text.splitlines()):
        stripped = _strip_ci_log_prefix_and_ansi(line)
        if not stripped or not re.match(r"^TOTAL\b", stripped, re.I):
            continue
        m_pct = re.search(r"(\d+)%\s*$", stripped)
        if not m_pct:
            continue
        before_pct = stripped[: m_pct.start()].rstrip()
        rest = re.sub(r"^TOTAL\s+", "", before_pct, flags=re.I).strip()
        nums = [int(x) for x in re.findall(r"\d+", rest)]
        if len(nums) < 2:
            continue
        return m_pct.group(1), nums[0], nums[1]
    return None, None, None


def parse_total_line_coverage_pct(log_text: str) -> str | None:
    """Percentage string from coverage ``TOTAL`` row, e.g. ``18%``."""
    p, _, _ = parse_coverage_total_row(log_text)
    return f"{p}%" if p else None


def parse_total_line_stmts_miss(log_text: str) -> tuple[int, int] | None:
    """**Stmts** and **Miss** from the same ``TOTAL`` row as :func:`parse_total_line_coverage_pct`."""
    _, s, m = parse_coverage_total_row(log_text)
    if s is None or m is None:
        return None
    return s, m


def parse_pytest_session_footer(log_text: str) -> tuple[str | None, str | None]:
    """
    Parse pytest session summary from Simple Unit Test log (last matching line wins).

    Handles:
      - ``25 passed in 120.2s``
      - ``=== 865 passed, 1 skipped, 591 deselected, 48 warnings in 333.31s (0:05:33) ===``

    Returns (duration_str like ``333.31s``, cleaned_summary_without_leading/trailing ``=``) or (None, None).
    """
    best_dur: str | None = None
    best_line: str | None = None
    for line in log_text.splitlines():
        m = re.search(r"(\d+)\s+passed\s+in\s+([\d.]+)s\b", line, re.I)
        if m:
            best_dur = f"{m.group(2)}s"
            best_line = line.strip()
            continue
        if re.search(r"\bpassed\b", line, re.I) and re.search(r"\bin\s+[\d.]+\s*s\b", line, re.I):
            m2 = re.search(r"\bin\s+([\d.]+)s\b", line, re.I)
            if m2:
                best_dur = f"{m2.group(1)}s"
                best_line = line.strip()
    cleaned: str | None = None
    if best_line:
        cleaned = re.sub(r"^=+\s*|\s*=+$", "", best_line).strip()
        if len(cleaned) > 160:
            cleaned = cleaned[:157] + "..."
    return best_dur, cleaned


def parse_pytest_session_duration(log_text: str) -> str | None:
    """Backward-compatible: duration only."""
    d, _ = parse_pytest_session_footer(log_text)
    return d


def _path_has_models_directory_segment(path: str) -> bool:
    """
    True if any path segment (split on ``/``) equals ``models`` (case-insensitive).
    Does **not** match a file named ``models.py`` (that segment is ``models.py``, not ``models``).
    """
    p = path.replace("\\", "/").strip()
    if not p:
        return False
    for seg in p.split("/"):
        if seg.lower() == "models":
            return True
    return False


def parse_coverage_data_line(line: str) -> tuple[str, int, int] | None:
    """
    One row from coverage.py text report (line or branch style).
    Returns (path, stmts, miss) or None.

    The **Cover** column is a ``NN%`` token. When a row includes a **Missing** column after Cover, the line
    does **not** end with ``%``; we anchor on the **last** ``\\d+(?:\\.\\d+)?%`` in the line (Missing is line
    numbers, not percentages). CI log lines may be prefixed with ``[timestamp]``; that is stripped like the
    ``TOTAL`` row parser.
    """
    line = _strip_ci_log_prefix_and_ansi(line)
    line = line.rstrip()
    if not line or line.strip().startswith(("-", "=")):
        return None
    if re.match(r"^Name\b", line):
        return None
    pct_matches = list(re.finditer(r"(\d+(?:\.\d+)?)%", line))
    if not pct_matches:
        return None
    m = pct_matches[-1]
    before = line[: m.start()].rstrip()
    parts = before.split()
    if len(parts) < 3:
        return None
    nums: list[int] = []
    idx = len(parts) - 1
    while idx >= 0 and parts[idx].isdigit():
        nums.insert(0, int(parts[idx]))
        idx -= 1
    if idx < 0:
        return None
    path = " ".join(parts[: idx + 1]).strip()
    up = path.upper()
    if up == "TOTAL" or up.startswith("TOTAL "):
        return None
    if len(nums) == 2:
        return path, nums[0], nums[1]
    if len(nums) == 4:
        return path, nums[0], nums[1]
    if len(nums) == 3:
        return path, nums[0], nums[1]
    return None


def compute_line_coverage_excluding_models_dirs(
    log_text: str,
) -> tuple[str | None, int | None, int | None, int | None]:
    """
    Recompute line coverage as ``sum(stmts - miss) / sum(stmts)`` over per-file rows, excluding any row whose
    path has a directory segment named ``models`` (``_path_has_models_directory_segment``).

    Returns ``(pct_str, total_stmts, covered_stmts, total_miss)`` or ``(None, None, None, None)`` if N/A.

    If ``log_text`` omits part of the table (truncated step log or ``UT_LOG_MAX_BYTES`` cap), the sum is over
    a **partial** file set and can diverge from the **TOTAL** row.
    """
    total_stmts = 0
    total_miss = 0
    used = False
    for line in log_text.splitlines():
        row = parse_coverage_data_line(line)
        if not row:
            continue
        path, stmts, miss = row
        if _path_has_models_directory_segment(path):
            continue
        total_stmts += stmts
        total_miss += miss
        used = True
    if not used or total_stmts <= 0:
        return None, None, None, None
    covered = total_stmts - total_miss
    p = 100.0 * covered / total_stmts
    if abs(p - round(p)) < 0.05:
        pct = f"{int(round(p))}%"
    else:
        pct = f"{p:.1f}%"
    return pct, total_stmts, covered, total_miss


def sum_parsed_coverage_table_stmts_miss(
    log_text: str,
) -> tuple[int | None, int | None]:
    """Sum **Stmts** and **Miss** over every parsed per-file row (including ``**/models/**``)."""
    ts, ms = 0, 0
    used = False
    for line in log_text.splitlines():
        row = parse_coverage_data_line(line)
        if not row:
            continue
        ts += row[1]
        ms += row[2]
        used = True
    if not used:
        return None, None
    return ts, ms


def fetch_latest_main_non_nightly_build(token: str) -> dict | None:
    """
    Newest finished-or-not build on ``branch=main`` that is **not** classified as nightly or weekly
    (``classify_build`` -> ``main_non_nightly``), same idea as
    https://buildkite.com/vllm/vllm-omni/builds?branch=main but skipping Scheduled nightly / weekly rows.
    Scans up to 10 API pages (50 builds/page) newest-first.
    """
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds"
    per_page = 50
    for page in range(1, 11):
        last_exc: Exception | None = None
        r = None
        for attempt in range(3):
            try:
                r = requests.get(
                    url,
                    params={"branch": "main", "per_page": per_page, "page": page},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=180,
                )
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(min(8, 2**attempt))
        if r is None:
            assert last_exc is not None
            raise last_exc
        builds = r.json()
        if not isinstance(builds, list) or not builds:
            return None
        for cand in builds:
            if classify_build(cand) == "main_non_nightly":
                return cand
        if len(builds) < per_page:
            return None
    return None


def _bk_get_with_retries(
    url: str,
    token: str,
    *,
    params: dict[str, str | int] | None = None,
) -> requests.Response:
    last_exc: Exception | None = None
    r: requests.Response | None = None
    for attempt in range(10):
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
            if attempt < 9:
                time.sleep(min(8, 2 ** min(attempt, 3)))
    assert last_exc is not None
    raise last_exc


def fetch_build_with_jobs(token: str, build_number: int | str) -> dict:
    """GET a single build (includes ``jobs[]``)."""
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds/{build_number}"
    r = _bk_get_with_retries(url, token)
    out = r.json()
    if not isinstance(out, dict):
        raise ValueError("unexpected Buildkite JSON for single build")
    return out


def ensure_build_with_jobs(token: str, build: dict) -> dict:
    """Refetch by number when the list endpoint omitted ``jobs``."""
    if build.get("jobs"):
        return build
    num = build.get("number")
    if num is None:
        return build
    return fetch_build_with_jobs(token, num)


def fetch_latest_finished_non_main_build(token: str) -> dict | None:
    """
    Newest **finished** build with ``classify_build`` -> ``non_main`` (a.k.a. **ready** in metrics).

    Scans all branches, newest-first, up to 10 API pages (50 builds/page).
    """
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds"
    per_page = 50
    for page in range(1, 11):
        r = _bk_get_with_retries(url, token, params={"per_page": per_page, "page": page})
        builds = r.json()
        if not isinstance(builds, list) or not builds:
            return None
        for cand in builds:
            st = (cand.get("state") or "").strip().lower()
            if st not in FINISHED_STATES:
                continue
            if classify_build(cand) == "non_main":
                return cand
        if len(builds) < per_page:
            return None
    return None


def fetch_latest_finished_merge_build(token: str) -> dict | None:
    """
    Newest **finished** build on ``main`` with ``classify_build`` -> ``main_non_nightly``
    (**merge** in metrics: not scheduled nightly/weekly).
    """
    url = f"{BUILDKITE_API_BASE}/organizations/{ORG_SLUG}/pipelines/{PIPELINE_SLUG}/builds"
    per_page = 50
    for page in range(1, 11):
        r = _bk_get_with_retries(
            url,
            token,
            params={"branch": "main", "per_page": per_page, "page": page},
        )
        builds = r.json()
        if not isinstance(builds, list) or not builds:
            return None
        for cand in builds:
            st = (cand.get("state") or "").strip().lower()
            if st not in FINISHED_STATES:
                continue
            if classify_build(cand) == "main_non_nightly":
                return cand
        if len(builds) < per_page:
            return None
    return None


# Names of jobs that should be **excluded** from a build's all-pass / all-fail check (matches
# ``compose_full_report.py`` ``UPLOAD_PIPELINE_RE``): upload-only steps don't represent real
# test pass/fail for the Development report's CI-row verdict.
_UPLOAD_PIPELINE_RE = re.compile(r"^Upload .+ Pipeline$", re.IGNORECASE)


def failed_job_names_from_build(build: dict) -> list[str]:
    """Names of reportable jobs whose state is ``failed`` (a real runtime failure).

    ``Upload * Pipeline`` jobs (e.g. ``Upload Nightly Pipeline``) are excluded
    because they are pipeline-upload steps — not real tests — and the
    ``l2_l3_ready_merge_gate`` (Test conclusion "Latest L2&L3 pass rate is
    100%") was incorrectly counting them as failures when the Buildkite
    pipeline marked them as skipped/blocked in the finished build.

    ``broken`` is intentionally **not** counted: it's a transient Buildkite
    pipeline-execution state for orchestration steps like
    ``:docker: Build image`` (Docker registry hiccup) and
    ``:email: Nightly Collection & Email`` (SMTP / kanban down). Only an
    explicit ``failed`` state from the job's own runner is treated as a real
    CI failure.
    """
    names: list[str] = []
    for j in build.get("jobs") or []:
        st = (j.get("state") or "").lower()
        if st not in FAIL_JOB_STATES:
            continue
        nm = (j.get("name") or "(unnamed)").strip()
        if _UPLOAD_PIPELINE_RE.match(nm):
            continue
        names.append(nm.replace("|", "/"))
    return names


def _build_reportable_jobs(build: dict) -> list[dict]:
    """Jobs that count toward the build's **all-pass / some-fail** verdict (upload steps skipped)."""
    return [j for j in (build.get("jobs") or []) if not _UPLOAD_PIPELINE_RE.match((j.get("name") or "").strip())]


def summarize_build_all_pass(build: dict | None) -> tuple[bool, str, int, int]:
    """
    For a (possibly-``None``) build returned by ``fetch_latest_finished_*_build``:

      Returns ``(all_pass, status_label, passed, failed_reportable)``:

      - ``all_pass``: ``True`` iff every **reportable** (non-``Upload * Pipeline``) job's state is
        ``passed``. ``False`` if any failed/broken job, or the build is missing / has no reportable
        jobs. ``None``-build also returns ``False``.
      - ``status_label**: short English text for the Development report's
        "Latest merge CI / nightly CI result" row, e.g. ``"All pass (passed=12)"`` or
        ``"Failed (passed=10, failed=2)"``.
      - ``passed``/``failed_reportable``**: counts of reportable jobs in each terminal state.
    """
    if not build:
        return False, "No latest CI build found (no finished build)", 0, 0
    jobs = _build_reportable_jobs(build)
    if not jobs:
        return False, "No reportable jobs (no reportable jobs)", 0, 0
    states = [(j.get("state") or "").lower() for j in jobs]
    passed = sum(1 for s in states if s == SUCCESS_STATE)
    failed = sum(1 for s in states if s in FAIL_JOB_STATES)
    if failed == 0 and (passed + failed) == len(jobs):
        return True, f"All pass (passed={passed})", passed, failed
    if failed:
        return False, f"Failed (passed={passed}, failed={failed})", passed, failed
    # No failed jobs but some non-pass terminal states (canceled / blocked / skipped / etc.)
    other = sum(1 for s in states if s in FINISHED_STATES and s not in (SUCCESS_STATE, *FAIL_JOB_STATES))
    return False, f"Not all passing (passed={passed}, other_finished={other})", passed, failed


def l2_l3_ready_merge_gate(token: str) -> tuple[bool, str]:
    """
    **Test conclusion** — "Latest L2&L3 pass rate is 100%".

    Pass iff the latest **finished** **ready** (non-main) and **merge** (main non-nightly/weekly)
    builds each have **no** ``failed`` jobs (``broken`` / ``skipped`` /
    ``canceled`` / ``blocked`` / ``not_run`` are all excluded — same bucket
    names as metrics table).
    """
    try:
        rb = fetch_latest_finished_non_main_build(token)
        mb = fetch_latest_finished_merge_build(token)
    except Exception as exc:
        return False, f"Buildkite request failed ({exc})"

    if rb is None:
        return (
            False,
            "No finished ready CI build found (non-main, same bucket as Metrics 'ready')",
        )
    if mb is None:
        return (
            False,
            "No finished merge CI build found (main non-nightly/weekly, same bucket as Metrics 'merge')",
        )

    rb = ensure_build_with_jobs(token, rb)
    mb = ensure_build_with_jobs(token, mb)
    rf = failed_job_names_from_build(rb)
    mf = failed_job_names_from_build(mb)
    if not rf and not mf:
        return True, ""

    parts: list[str] = []
    if rf:
        rn = rb.get("number", "?")
        head = ", ".join(rf[:8])
        tail = f" ({len(rf)} total)" if len(rf) > 8 else ""
        parts.append(f"ready #{rn} has failed jobs: {head}{tail}")
    if mf:
        mn = mb.get("number", "?")
        head = ", ".join(mf[:8])
        tail = f" ({len(mf)} total)" if len(mf) > 8 else ""
        parts.append(f"merge #{mn} has failed jobs: {head}{tail}")
    return False, "; ".join(parts)


def fetch_ut_coverage_simple_unit_test(
    token: str,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]:
    """
    Latest **main** non-nightly build (see ``fetch_latest_main_non_nightly_build``), job 'Simple Unit Test'.

    Returns ``(build_number, web_url, cov_pct, cov_excl_pct, pytest_dur, pytest_detail,
    excl_stmts, excl_covered, excl_miss, total_line_stmts, total_line_miss,
    table_sum_stmts, table_sum_miss)``.
    ``total_line_*`` come from the coverage ``TOTAL`` row; ``excl_*`` from summing per-file rows after dropping
    ``models`` path segments; ``table_sum_*`` sum **all** parsed per-file rows (sanity vs ``TOTAL``).
    ``excl_*``, ``total_line_*``, and ``table_sum_*`` may be ``None`` if parsing failed.
    """
    none13 = (None,) * 13
    b = fetch_latest_main_non_nightly_build(token)
    if b is None:
        return none13  # type: ignore[return-value]
    num = b.get("number")
    if num is None:
        return none13  # type: ignore[return-value]
    web = (b.get("web_url") or "").strip() or (f"https://buildkite.com/{ORG_SLUG}/{PIPELINE_SLUG}/builds/{num}")
    job = None
    for j in b.get("jobs") or []:
        name = (j.get("name") or "").strip()
        if SIMPLE_UNIT_TEST_JOB_RE.match(name):
            job = j
            break
    if job is None:
        return str(num), web, None, None, None, None, None, None, None, None, None, None, None
    log_url = (job.get("raw_log_url") or job.get("log_url") or "").strip()
    if not log_url:
        return str(num), web, None, None, None, None, None, None, None, None, None, None, None
    try:
        raw_log = read_log_tail_capped(token, log_url, max_bytes=UT_LOG_MAX_BYTES)
    except requests.RequestException:
        return str(num), web, None, None, None, None, None, None, None, None, None, None, None
    cov_text = extract_pytest_coverage_text_report(raw_log)
    cov_pct, total_line_stmts, total_line_miss = parse_coverage_total_row(cov_text)
    pct = f"{cov_pct}%" if cov_pct else None
    dur, pytest_detail = parse_pytest_session_footer(raw_log)
    pct_excl, excl_stmts, excl_covered, excl_miss = compute_line_coverage_excluding_models_dirs(cov_text)
    table_sum_stmts, table_sum_miss = sum_parsed_coverage_table_stmts_miss(cov_text)
    return (
        str(num),
        web,
        pct,
        pct_excl,
        dur,
        pytest_detail,
        excl_stmts,
        excl_covered,
        excl_miss,
        total_line_stmts,
        total_line_miss,
        table_sum_stmts,
        table_sum_miss,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Buildkite vllm-omni builds; success rate and avg duration by category"
    )
    parser.add_argument(
        "--from",
        dest="created_from",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Start date for Buildkite created_at filter (UTC, inclusive). "
            "Omit both --from and --to to use current UTC month through today."
        ),
    )
    parser.add_argument(
        "--to",
        dest="created_to",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "End date for Buildkite created_at filter (UTC, inclusive). "
            "Omit both --from and --to to use current UTC month through today."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each build's category and state",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also emit a Markdown block for the report (Metrics overview section).",
    )
    args = parser.parse_args()

    if args.created_from is None and args.created_to is None:
        args.created_from, args.created_to = default_created_range_utc()
    elif args.created_from is None or args.created_to is None:
        print(
            "buildkite_build_stats.py: pass both --from and --to, or omit both "
            "(defaults to current UTC month through today).",
            file=sys.stderr,
        )
        return 2

    token = get_api_token()
    if not token:
        print(
            "BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is not set; cannot call the Buildkite API.",
            file=sys.stderr,
        )
        print("Set one in the environment and retry.", file=sys.stderr)
        return 1

    created_from = f"{args.created_from}T00:00:00Z"
    created_to = f"{args.created_to}T23:59:59Z"

    print(f"Fetching {ORG_SLUG}/{PIPELINE_SLUG} builds {args.created_from} ~ {args.created_to}...")
    try:
        builds = fetch_builds(token, created_from, created_to)
    except requests.RequestException as e:
        print(f"API request failed: {e}", file=sys.stderr)
        if hasattr(e, "response") and e.response is not None:
            print(f"HTTP status: {e.response.status_code}", file=sys.stderr)
            print(e.response.text[:500], file=sys.stderr)
        return 1

    print(f"Fetched {len(builds)} build(s).\n")

    buckets: dict[str, Bucket] = defaultdict(Bucket)
    for b in builds:
        state = (b.get("state") or "").strip().lower()
        kind = classify_build(b)
        if args.verbose:
            print(f"  #{b.get('number')} branch={b.get('branch')} state={state} -> {kind}")

        if state not in FINISHED_STATES:
            continue
        bucket = buckets[kind]
        if state == SUCCESS_STATE:
            bucket.passed += 1
        elif state == FAIL_STATE:
            bucket.failed += 1
        else:
            bucket.other_finished += 1

        if state in (SUCCESS_STATE, FAIL_STATE):
            c_at = parse_buildkite_time(b.get("created_at"))
            f_at = parse_buildkite_time(b.get("finished_at"))
            if c_at is not None and f_at is not None:
                delta = (f_at - c_at).total_seconds()
                if delta >= 0:
                    bucket.duration_seconds.append(delta)

    # Print four CI buckets: success rate and average duration
    labels = {
        "non_main": "ready",
        "main_non_nightly": "merge",
        "main_nightly": "nightly",
        "main_weekly": "weekly",
    }
    keys_order = ("non_main", "main_non_nightly", "main_nightly", "main_weekly")
    print("--- By category (success rate: passed/failed only; avg duration: mean over builds with both timestamps) ---")
    rows_md: list[tuple[str, str, str, str]] = []
    for key in keys_order:
        bucket = buckets[key]
        label = labels[key]
        total = bucket.total_for_success_rate
        if total == 0:
            rate_str = "N/A (no passed/failed builds)"
        else:
            rate = bucket.success_rate
            rate_str = f"{rate * 100:.1f}% ({bucket.passed}/{total})"
        other = bucket.other_finished
        extra = f" (+ {other} other finished: canceled/blocked/etc.)" if other else ""

        avg_sec = bucket.avg_duration_seconds
        if avg_sec is None:
            dur_str = "N/A (no valid created_at/finished_at)"
            dur_md = "N/A"
        else:
            n_timed = len(bucket.duration_seconds)
            dur_str = f"{format_duration(avg_sec)} ({n_timed} build(s) with duration)"
            dur_md = f"{format_duration(avg_sec)} ({n_timed} builds)"

        print(f"  {label}: success rate = {rate_str}{extra}")
        print(f"           avg duration = {dur_str}")
        rows_md.append((label, rate_str.replace("|", "/"), dur_md, str(other)))

    if args.markdown:
        (
            bnum,
            _burl,
            cov,
            cov_excl,
            ut_pytest_dur,
            _,
            *_,
        ) = fetch_ut_coverage_simple_unit_test(token)
        ut_rate = f"**{cov}**" if cov else "*N/A* (no `TOTAL` ... `NN%` in Simple Unit Test log)"
        ut_excl_rate = f"**{cov_excl}**" if cov_excl else "*N/A* (no per-file coverage table in log)"
        if ut_pytest_dur:
            ut_dur = ut_pytest_dur.strip()
        else:
            ut_dur = "*N/A* (no pytest summary line with `passed` and `in ...s` in log)"
        ut_other = "-"
        if bnum is None:
            ut_rate = (
                "*N/A* (no merge-style `main` build found for Simple Unit Test; excludes scheduled nightly/weekly)"
            )
            ut_excl_rate = (
                "*N/A* (no merge-style `main` build found for Simple Unit Test; excludes scheduled nightly/weekly)"
            )
            ut_dur = "*N/A*"
            ut_other = "-"

        bug_cell = github_bug_avg_first_response_range_cell(
            args.created_from,
            args.created_to,
        ).replace("|", "/")

        print()
        print("## Metrics overview")
        print()
        print(
            f"Source: `scripts/buildkite_build_stats.py`; "
            f"window (Buildkite `created_at`, UTC): `{args.created_from}` - `{args.created_to}`."
        )
        print()
        metrics_header = [
            "CI category",
            "Success rate/UT coverage",
            "Avg duration",
            "Other finished count",
            "Bug avg first response",
        ]
        metrics_rows: list[list[str]] = []
        for label, rate, dur, other in rows_md:
            metrics_rows.append([label, rate, dur, other, "-"])
        metrics_rows.append(["ut", ut_rate, ut_dur, ut_other, "-"])
        metrics_rows.append(["ut (exclude models)", ut_excl_rate, "-", ut_other, "-"])
        metrics_rows.append(
            [
                f"bugs (first response, {args.created_from}..{args.created_to})",
                "-",
                "-",
                "-",
                bug_cell,
            ]
        )
        print(render_markdown_table(metrics_header, metrics_rows))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
