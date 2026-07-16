"""Discover nightly job logs under ``logs/nightly_jobs`` (shared by report + kanban prep)."""

from __future__ import annotations

from pathlib import Path

LOG_SUFFIXES = (".log", ".out", ".txt")
_LOG_GLOBS = ("*.log", "*.out", "*.txt")

# Directories that ``run_nightly_jobs.sh`` lays out next to the actual job
# logs. None of these is itself a test job — they hold raw nohup logs,
# generated ``.sh`` scripts, or ``.json`` perf results — so discovering them
# as jobs would create bogus ``Other / logs`` rows in the Local Test summary.
_INFRASTRUCTURE_DIR_NAMES = frozenset(
    {
        "logs",
        "jobs",
        "perf_results",
        "perf-results",
        "results",
        "raw",
        "nohup",
        "tmp",
        "__pycache__",
    }
)

# Job names that are not real test jobs and must never appear in the
# Local Test summary table or be counted in any job stats. These are
# emitted by the nightly wrapper script itself — for example
# ``run_nightly_jobs.sh`` writes a ``timing_summary.log`` rollup at the
# end of a run that lists the duration / status of every actual job, so
# treating it as a job would both duplicate the real job rows and add an
# empty/blank row in development and release variants. Comparison is
# case-insensitive on the lowercased name.
#
# Recursive discovery flattens a nested path into one job name; a
# ``nightly_jobs_20260703-043849/timing_summary.log`` becomes
# ``nightly_jobs_20260703-043849_timing_summary``. The exact match set
# below catches that flattened name directly, but branch #2 of
# ``discover_job_logs`` additionally checks the suffix ``_timing_summary``
# so any ``<run_root>_timing_summary`` job is filtered even when the run
# root name isn't enumerated here.
_NON_TEST_JOB_NAMES = frozenset(
    {
        "timing_summary",
    }
)

# Job-name suffix patterns (lowercased) that disqualify a discovered job
# from the summary tables. The nightly wrapper writes a per-run
# ``<run_id>_timing_summary`` rollup that mirrors every actual job's
# duration/status — counting it would double-count the real jobs.
_NON_TEST_JOB_SUFFIXES = ("_timing_summary",)


def discover_job_logs(log_dir: Path) -> list[tuple[str, list[Path]]]:
    """Return ``(job_name, log_paths)`` using the same rules as ``nightly_local_log_report``.

    Sub-directories whose name is in :data:`_INFRASTRUCTURE_DIR_NAMES` are
    skipped so they don't surface as fake jobs (e.g. a stray ``logs/`` dir
    containing nohup output would otherwise appear as a job named ``logs``).

    Discovery **recurses** into nested sub-directories: a H200/H800/A100 log
    tree laid out as ``<gpu>/<pillar>/<dim>/<job>.log`` is flattened into one
    job per ``.log`` file, with the job name being the file's path relative
    to ``log_dir`` (slashes replaced with ``_``). Previously the top-level
    sub-dir was treated as a single job and its inner ``.log`` files were
    silently dropped — which made an H200/H800 dir with the conventional
    pillar/dim layout look like "the whole log file is one job".

    Job-name suffix patterns in :data:`_NON_TEST_JOB_SUFFIXES` (e.g.
    ``_timing_summary``) are filtered so the nightly wrapper's per-run
    rollup doesn't double-count every real job.
    """
    if not log_dir.is_dir():
        return []

    infra_names = _INFRASTRUCTURE_DIR_NAMES | _NON_TEST_JOB_NAMES
    log_root = log_dir.resolve()

    def _is_infra_dir(p: Path) -> bool:
        return p.is_dir() and p.name.lower() in infra_names

    def _job_name_for(rel: Path) -> str:
        # ``relative_to`` raises if rel is not under log_root; we always
        # pass rels derived from log_root.rglob, so this is safe.
        parts = list(rel.parts)
        last = parts[-1] if parts else ""
        if Path(last).suffix.lower() in LOG_SUFFIXES:
            parts[-1] = Path(last).stem
        return "_".join(parts)

    def _is_non_test_job(name: str) -> bool:
        lname = name.lower()
        if lname in _NON_TEST_JOB_NAMES:
            return True
        return any(lname.endswith(sfx) for sfx in _NON_TEST_JOB_SUFFIXES)

    merged: dict[str, list[Path]] = {}

    # 1) Direct top-level log files (flat layout) — keep the existing
    #    behaviour of using the file stem as the job name.
    for p in sorted(log_dir.iterdir(), key=lambda q: q.name):
        if p.name.startswith("."):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() not in LOG_SUFFIXES:
            continue
        if _is_non_test_job(p.stem):
            continue
        merged.setdefault(p.stem, []).append(p)

    # 2) Recurse into nested sub-directories (pillar/dim layout). Skip
    #    infrastructure dirs at any depth so an ``H200/logs/`` nohup folder
    #    doesn't pollute the result.
    for path in sorted(log_root.rglob("*"), key=lambda q: q.as_posix().lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in LOG_SUFFIXES:
            continue
        # Reject any file under a directory whose name is in the skip set,
        # but only consider ancestors INSIDE log_root (Path.parents also
        # includes the parents of log_root itself, which would otherwise
        # spuriously match ``/tmp`` for a log_root under ``/tmp/...``).
        rel = path.relative_to(log_root)
        if any(part.lower() in infra_names for part in rel.parts[:-1]):
            continue
        # Skip top-level files (handled by branch 1) so we don't double-count.
        if len(rel.parts) == 1:
            continue
        name = _job_name_for(rel)
        if _is_non_test_job(name):
            continue
        merged.setdefault(name, []).append(path)

    out: list[tuple[str, list[Path]]] = []
    for name in sorted(merged.keys()):
        paths = merged[name]
        if not paths:
            continue
        seen: set[str] = set()
        uniq: list[Path] = []
        for path in sorted(paths, key=lambda q: q.as_posix().lower()):
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                uniq.append(path)
        out.append((name, uniq))
    return out


def read_combined_job_logs(paths: list[Path], *, include_headers: bool = False) -> str:
    parts: list[str] = []
    for p in paths:
        if include_headers:
            parts.append(f"===== {p.name} =====\n")
        try:
            parts.append(p.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError as e:
            parts.append(f"<<< read error {p}: {e} >>>\n")
        if include_headers:
            parts.append("\n")
    return "".join(parts)
