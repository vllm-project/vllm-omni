"""Parse ``timing_summary.log`` rollups for selective stability-log pulls.

The nightly wrapper ``tools/nightly/run_nightly_jobs.sh`` writes one
``timing_summary.log`` per run directory (``nightly_stability_jobs_*``,
``nightly_jobs_local_*``, ``nightly_jobs_<YYYYMMDD-HHMMSS>``) listing every
job that ran plus its duration and a status string.

The selective-pull optimization uses the rollup to **skip** downloading the
``<job>.log`` for any job whose status is ``OK`` — long-stability logs are
typically tens to hundreds of megabytes, so pulling only the rollup (a few
hundred bytes) plus the ``jobs/*.sh`` scripts and then later fetching just the
failed-job logs saves a lot of bandwidth on the laptop → cluster sync.

Format (single-line per job, plus a few trailing summary lines):

    === Job timing summary ===
      <job_name>  <duration>  OK
      <job_name>  <duration>  FAILED (exit <n>)
      <job_name>  <duration>  TIMED OUT
    Total wall time: 24h 47m (1 jobs)
    Failed jobs: 1/3                          # only present if any failed
    Result: one or more jobs failed. See logs under <dir>.
    Result: all jobs finished OK. Logs: <dir>/*.log

The parser is **best-effort** — malformed lines are skipped silently so a
partially-truncated rollup (e.g. an in-progress run) still produces useful
output. The downstream ``nightly_local_log_report.py`` falls back to scanning
``.log`` files when no rollup is present, so this module is only consulted
when the agent explicitly uses selective pull (``--stability-manifest-only``
on the report CLI) and the run directory is missing the actual log files
for some jobs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# One line per job in the rollup:
#   ``  stability_wan22  24h 47m  OK``
#   ``  stability_qwen3_omni  1h 12m 34s  FAILED (exit 1)``
#   ``  stability_flux  5m  TIMED OUT``
_JOB_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][\w\-\.]*)\s+"
    r"(?P<duration>\d+h(?:\s+\d+m)?(?:\s+\d+s)?|\d+m(?:\s+\d+s)?|\d+s)\s+"
    r"(?P<status>OK|TIMED OUT|FAILED(?:\s+\(exit\s+\d+\))?)\s*$"
)

# Status string → canonical token used by the report.
_OK_STATUSES = frozenset({"OK"})
_FAILED_STATUSES = frozenset({"TIMED OUT"})
_FAIL_REASON_RE = re.compile(r"^FAILED\s+\(exit\s+(\d+)\)\s*$")


@dataclass(frozen=True)
class ManifestEntry:
    """One job entry from ``timing_summary.log``."""

    job_name: str
    duration: str
    status: str  # one of "ok", "fail"
    raw_status: str  # original "OK" / "TIMED OUT" / "FAILED (exit N)"
    exit_code: int | None = None  # set for FAILED (exit N); None otherwise


@dataclass(frozen=True)
class StabilityManifest:
    """Parsed rollup from a single run directory."""

    run_dir_name: str  # basename of the source directory
    entries: tuple[ManifestEntry, ...]
    total_wall_time: str  # raw ``Total wall time: …`` string, "" if missing
    failed_count: int  # from ``Failed jobs: N/M`` or computed
    job_count: int
    result_line: str  # last ``Result: …`` line, "" if missing
    ok_jobs: tuple[str, ...]
    failed_jobs: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """``True`` iff every recorded entry is ``OK`` and no failure listed."""
        return self.failed_count == 0 and all(e.status == "ok" for e in self.entries)


def parse_timing_summary(path: Path, *, run_dir_name: str = "") -> StabilityManifest:
    """Parse a ``timing_summary.log`` file.

    Returns a :class:`StabilityManifest`. Missing or empty files yield a
    manifest with no entries (callers should fall back to scanning ``.log``
    files in that case). Unknown status strings are recorded as ``status="fail"``
    with ``raw_status`` preserved so the report can surface them.
    """
    if not path.is_file():
        return StabilityManifest(
            run_dir_name=run_dir_name or path.parent.name,
            entries=(),
            total_wall_time="",
            failed_count=0,
            job_count=0,
            result_line="",
            ok_jobs=(),
            failed_jobs=(),
        )

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return StabilityManifest(
            run_dir_name=run_dir_name or path.parent.name,
            entries=(),
            total_wall_time="",
            failed_count=0,
            job_count=0,
            result_line="",
            ok_jobs=(),
            failed_jobs=(),
        )

    entries: list[ManifestEntry] = []
    total_wall_time = ""
    failed_count = 0
    job_count = 0
    result_line = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        m = _JOB_LINE_RE.match(line)
        if m:
            name = m.group("name")
            duration = m.group("duration")
            raw_status = m.group("status")
            if raw_status in _OK_STATUSES:
                status = "ok"
                exit_code = None
            elif raw_status in _FAILED_STATUSES:
                status = "fail"
                exit_code = 124  # standard `timeout` exit code
            else:
                fr = _FAIL_REASON_RE.match(raw_status)
                if fr:
                    status = "fail"
                    try:
                        exit_code = int(fr.group(1))
                    except ValueError:
                        exit_code = None
                else:
                    # Unknown status — treat conservatively as fail so the
                    # caller surfaces the unexpected token in the report.
                    status = "fail"
                    exit_code = None
            entries.append(
                ManifestEntry(
                    job_name=name,
                    duration=duration,
                    status=status,
                    raw_status=raw_status,
                    exit_code=exit_code,
                )
            )
            if status == "fail":
                failed_count += 1
            continue

        stripped = line.strip()
        if stripped.startswith("Total wall time:") or stripped.startswith("Total job time:"):
            total_wall_time = stripped
            m2 = re.search(r"\(\s*(\d+)\s*jobs?\s*\)", stripped)
            if m2:
                try:
                    job_count = int(m2.group(1))
                except ValueError:
                    pass
            continue

        if stripped.startswith("Failed jobs:"):
            m2 = re.search(r"Failed\s+jobs:\s*(\d+)\s*/\s*(\d+)", stripped)
            if m2:
                try:
                    failed_count = int(m2.group(1))
                    job_count = int(m2.group(2))
                except ValueError:
                    pass
            continue

        if stripped.startswith("Result:"):
            result_line = stripped
            continue

    if not job_count:
        job_count = len(entries)

    ok_jobs = tuple(e.job_name for e in entries if e.status == "ok")
    failed_jobs = tuple(e.job_name for e in entries if e.status == "fail")

    return StabilityManifest(
        run_dir_name=run_dir_name or path.parent.name,
        entries=tuple(entries),
        total_wall_time=total_wall_time,
        failed_count=failed_count,
        job_count=job_count,
        result_line=result_line,
        ok_jobs=ok_jobs,
        failed_jobs=failed_jobs,
    )


def discover_manifests(log_dir: Path) -> list[StabilityManifest]:
    """Find every ``timing_summary.log`` under ``log_dir`` and parse it.

    Searches recursively so the helper works against both flat and pillar/dim
    log trees; ignores nested ``timing_summary.log`` files under directories
    like ``logs/`` / ``nohup/`` that ``run_nightly_jobs.sh`` may write as
    sibling rollups — only the **direct parent** ``run_dir`` is treated as a
    stability run summary when its basename starts with one of the nightly
    family prefixes (``nightly_stability_jobs_*``, ``nightly_jobs_local_*``,
    ``nightly_jobs_<YYYYMMDD-HHMMSS>``).
    """

    results: list[StabilityManifest] = []
    if not log_dir.is_dir():
        return results

    accepted_prefixes = (
        "nightly_stability_jobs_",
        "nightly_jobs_local_",
        "nightly_jobs_",
    )

    for summary in sorted(log_dir.glob("**/timing_summary.log")):
        run_dir = summary.parent
        base = run_dir.name
        if not any(base.startswith(p) for p in accepted_prefixes):
            # Skip nested rollups that don't live directly under a nightly
            # run directory (e.g. under ``logs/`` nohup folders).
            continue
        results.append(parse_timing_summary(summary, run_dir_name=base))

    return results


def select_failed_jobs(manifest: StabilityManifest) -> tuple[str, ...]:
    """Return the list of jobs the caller must still pull.

    The selective-pull optimization pulls the rollup (and ``jobs/*.sh``)
    first, then asks this helper for the list of jobs whose ``.log`` file
    should be downloaded. Jobs whose status is anything other than ``OK``
    are returned.
    """
    return manifest.failed_jobs


def missing_job_logs(
    manifest: StabilityManifest,
    present_logs: Iterable[str],
) -> tuple[str, ...]:
    """Compute which job names from the manifest are missing their ``.log``.

    ``present_logs`` is the set of stem names (e.g. ``"stability_wan22"``)
    that exist on disk after the partial pull. The return value is the list
    of manifest jobs whose ``.log`` file is **expected** but not present —
    those are exactly the cases the report should treat as "manifest only"
    (status known, no failure-detail log available).
    """
    present = {name.strip() for name in present_logs if name.strip()}
    return tuple(e.job_name for e in manifest.entries if e.job_name not in present)


__all__ = [
    "ManifestEntry",
    "StabilityManifest",
    "discover_manifests",
    "missing_job_logs",
    "parse_timing_summary",
    "select_failed_jobs",
]
