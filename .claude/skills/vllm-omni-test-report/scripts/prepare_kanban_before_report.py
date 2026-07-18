#!/usr/bin/env python3
"""
Prepare the vllm-omni-kanban checkout before generating a nightly/release HTML report.

Workflow (run from ``skills/vllm-omni-test-report/`` after log sync):

1. ``git pull --rebase`` on the local https://github.com/hsliuustc0106/vllm-omni-kanban clone.
2. When ``$REPO_ROOT/logs/.kanban_perf_source`` (or local-only ``nightly_jobs_local_*`` sync)
   contains perf JSON, sync into ``data/local_nightly_raw/manual_YYYYMMDD/``
   (``YYYYMMDD`` from **``nightly_jobs_local_*``** only;
   **stability** ``nightly_stability_jobs_*`` (NOTE: order is ``stability_jobs`` not
   ``jobs_stability``) and **general** ``nightly_jobs_YYYYMMDD-*`` perf JSON are
   **not** copied to kanban):
   copy result JSON and
   ``logs/nightly_jobs/local_pytest_hunyuan_image.log`` (or ``test_hunyuan_image3.log``) as
   ``test_hunyuan_image3.log``.
3. ``mkdocs build`` in the kanban repo (``mkdocs_hooks`` → sync + ``generate_charts``)
   to refresh ``docs/assets/charts/*_history.json``.

Requires ``gh`` authenticated for git pull (same as ``push_report_to_kanban.py``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from kanban_local_nightly_raw import (
    HUNYUAN_MANUAL_DEST_LOG,
    HUNYUAN_NIGHTLY_SOURCE_LOG_CANDIDATES,
    LOCAL_NIGHTLY_RAW,
    clear_last_manual_marker,
    resolve_hunyuan_nightly_source_log,
    write_last_manual_marker,
)
from kanban_repo_config import KANBAN_REPO_URL
from laptop_path_defaults import (
    DEFAULT_KANBAN_REPO_ROOT_DISPLAY,
    DEFAULT_LAPTOP_REPO_ROOT_DISPLAY,
    resolve_kanban_repo_root,
    resolve_laptop_repo_root,
)
from local_perf_results import (
    KANBAN_PERF_SOURCE_DIRNAME,
    NIGHTLY_JOBS_SOURCE_MARKER,
    default_nightly_log_dir,
    infer_kanban_manual_date_yyyymmdd,
    local_perf_result_files,
    resolve_kanban_manual_log_dir,
)
from push_report_to_kanban import (
    _git_current_branch,
    _run_git,
    ensure_gh_authenticated,
)


@dataclass
class PrepareResult:
    kanban_repo: Path
    pulled: bool
    manual_dir: Path | None = None
    perf_files_copied: list[str] = field(default_factory=list)
    log_files_copied: list[str] = field(default_factory=list)
    mkdocs_ran: bool = False
    notes: list[str] = field(default_factory=list)


def _default_repo_root() -> Path:
    return resolve_laptop_repo_root()


def _default_kanban_repo() -> Path:
    return resolve_kanban_repo_root()


def pull_kanban_repo(
    kanban_repo: Path,
    *,
    remote: str = "origin",
    branch: str | None = None,
) -> str:
    kanban_repo = kanban_repo.resolve()
    if not (kanban_repo / ".git").is_dir():
        raise RuntimeError(f"Not a git repository: {kanban_repo}")
    ensure_gh_authenticated()
    branch = branch or _git_current_branch(kanban_repo)
    proc = _run_git(
        kanban_repo,
        "pull",
        "--rebase",
        remote,
        branch,
        check=False,
        gh_credential=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git pull --rebase {remote} {branch} failed: {detail}")
    return branch


def allocate_manual_dir(raw_root: Path, *, day_yyyymmdd: str | None = None) -> Path:
    """Return ``manual_YYYYMMDD`` using the synced nightly run date when available."""
    if not day_yyyymmdd:
        raise ValueError("day_yyyymmdd is required (infer from synced nightly_jobs_* before calling)")
    raw_root.mkdir(parents=True, exist_ok=True)
    return raw_root / f"manual_{day_yyyymmdd}"


def _reset_manual_dir(manual_dir: Path) -> None:
    """Ensure ``manual_dir`` exists empty (same-day re-sync replaces prior contents)."""
    if manual_dir.exists():
        shutil.rmtree(manual_dir)
    manual_dir.mkdir(parents=True, exist_ok=True)


def sync_local_nightly_raw(
    kanban_repo: Path,
    *,
    log_dir: Path,
    repo_root: Path,
) -> tuple[Path | None, list[str], list[str], list[str]]:
    """Copy perf JSON + job logs from ``nightly_jobs_local_*`` only into kanban ``manual_*``."""
    notes: list[str] = []
    log_dir = log_dir.resolve()
    repo_root = repo_root.resolve()
    kanban_log_dir = resolve_kanban_manual_log_dir(repo_root=repo_root, log_dir=log_dir)
    if kanban_log_dir is None:
        notes.append(
            f"No nightly_jobs_local_* perf JSON for kanban manual_* "
            f"(general nightly_jobs_* perf is excluded; expected {KANBAN_PERF_SOURCE_DIRNAME} "
            f"after log sync); skipped manual_* sync."
        )
        return None, [], [], notes

    perf_files = local_perf_result_files(kanban_log_dir)
    if not perf_files:
        notes.append(f"Kanban perf source has no JSON under {kanban_log_dir}; skipped manual_* sync.")
        return None, [], [], notes

    raw_root = (kanban_repo / LOCAL_NIGHTLY_RAW).resolve()
    run_day = infer_kanban_manual_date_yyyymmdd(repo_root=repo_root, log_dir=log_dir)
    if not run_day:
        notes.append(
            f"Could not infer manual_* date from nightly_jobs_local_* sources under {log_dir} "
            f"(expected {NIGHTLY_JOBS_SOURCE_MARKER} with nightly_jobs_local_* line); skipped manual_* sync."
        )
        return None, [], [], notes

    manual_dir = allocate_manual_dir(raw_root, day_yyyymmdd=run_day)
    notes.append(
        f"manual_* from nightly_jobs_local_* only: {run_day} → {manual_dir.name} (kanban perf root: {kanban_log_dir})"
    )
    _reset_manual_dir(manual_dir)

    perf_copied: list[str] = []
    used_names: set[str] = set()
    for src in perf_files:
        dest_name = src.name
        if dest_name in used_names:
            stem = src.stem
            suffix = src.suffix
            n = 2
            while True:
                alt = f"{stem}_{n}{suffix}"
                if alt not in used_names:
                    dest_name = alt
                    break
                n += 1
        used_names.add(dest_name)
        shutil.copy2(src, manual_dir / dest_name)
        perf_copied.append(dest_name)

    log_copied: list[str] = []
    src_log = resolve_hunyuan_nightly_source_log(kanban_log_dir)
    if src_log is not None:
        dest_log = HUNYUAN_MANUAL_DEST_LOG
        if dest_log in used_names:
            notes.append(f"Skipped {src_log.name}: {dest_log} already taken by a perf JSON basename.")
        else:
            shutil.copy2(src_log, manual_dir / dest_log)
            used_names.add(dest_log)
            log_copied.append(dest_log)
    else:
        tried = ", ".join(HUNYUAN_NIGHTLY_SOURCE_LOG_CANDIDATES)
        notes.append(
            f"Missing Hunyuan job log under {log_dir.resolve()} (tried: {tried}); manual dir contains perf JSON only."
        )

    write_last_manual_marker(kanban_repo, manual_dir)
    notes.append(f"Marked {manual_dir.name} for archive push (.last_manual_dir).")

    return manual_dir, perf_copied, log_copied, notes


def run_mkdocs_build(kanban_repo: Path) -> None:
    kanban_repo = kanban_repo.resolve()
    mkdocs_yml = kanban_repo / "mkdocs.yml"
    if not mkdocs_yml.is_file():
        raise RuntimeError(f"mkdocs.yml not found under kanban repo: {kanban_repo}")

    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build"],
        cwd=str(kanban_repo),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"mkdocs build failed (exit {proc.returncode}): {detail[:2000]}")


def prepare_kanban_before_report(
    kanban_repo: Path,
    *,
    repo_root: Path | None = None,
    log_dir: Path | None = None,
    remote: str = "origin",
    branch: str | None = None,
    skip_pull: bool = False,
    skip_manual_sync: bool = False,
    skip_mkdocs: bool = False,
) -> PrepareResult:
    kanban_repo = kanban_repo.resolve()
    notes: list[str] = []
    pulled = False
    clear_last_manual_marker(kanban_repo)

    if not skip_pull:
        branch = pull_kanban_repo(kanban_repo, remote=remote, branch=branch)
        pulled = True
        notes.append(f"Pulled latest {KANBAN_REPO_URL} ({remote}/{branch}).")
    else:
        notes.append("Skipped git pull (--skip-pull).")

    manual_dir: Path | None = None
    perf_copied: list[str] = []
    log_copied: list[str] = []

    if not skip_manual_sync and repo_root is not None:
        job_log_dir = (log_dir or default_nightly_log_dir(repo_root)).resolve()
        if job_log_dir.is_dir():
            manual_dir, perf_copied, log_copied, sync_notes = sync_local_nightly_raw(
                kanban_repo,
                log_dir=job_log_dir,
                repo_root=repo_root,
            )
            notes.extend(sync_notes)
            if manual_dir is not None:
                rel = manual_dir.relative_to(kanban_repo)
                notes.append(f"Created {rel} with {len(perf_copied)} perf JSON and {len(log_copied)} log file(s).")
        else:
            notes.append(f"Nightly log dir missing ({job_log_dir}); skipped manual_* sync.")
    elif skip_manual_sync:
        notes.append("Skipped manual_* sync (--skip-manual-sync).")
    else:
        notes.append(f"Repo root missing ({DEFAULT_LAPTOP_REPO_ROOT_DISPLAY}); skipped manual_* sync.")

    mkdocs_ran = False
    if not skip_mkdocs:
        run_mkdocs_build(kanban_repo)
        mkdocs_ran = True
        notes.append("Ran mkdocs build; refreshed docs/assets/charts/*_history.json.")

    return PrepareResult(
        kanban_repo=kanban_repo,
        pulled=pulled,
        manual_dir=manual_dir,
        perf_files_copied=perf_copied,
        log_files_copied=log_copied,
        mkdocs_ran=mkdocs_ran,
        notes=notes,
    )


def format_prepare_summary(result: PrepareResult) -> str:
    lines = [
        "Kanban pre-report preparation",
        f"  repo: {result.kanban_repo}",
    ]
    if result.manual_dir is not None:
        lines.append(f"  manual_dir: {result.manual_dir}")
        lines.append(f"  perf_json: {len(result.perf_files_copied)}")
        lines.append(f"  log_files: {len(result.log_files_copied)}")
    for note in result.notes:
        lines.append(f"  - {note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kanban-repo-root",
        type=Path,
        default=_default_kanban_repo(),
        help=(f"Local clone of {KANBAN_REPO_URL} (default: $KANBAN_REPO_ROOT or {DEFAULT_KANBAN_REPO_ROOT_DISPLAY})."),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_default_repo_root(),
        help=(
            "Local vLLM-Omni checkout with synced logs (incl. perf JSON under logs/nightly_jobs) "
            f"(default: $REPO_ROOT or {DEFAULT_LAPTOP_REPO_ROOT_DISPLAY})."
        ),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Override nightly log dir for job logs + perf JSON (default: <repo-root>/logs/nightly_jobs).",
    )
    parser.add_argument("--git-remote", default="origin", help="Git remote for pull (default: origin).")
    parser.add_argument("--git-branch", default=None, help="Git branch for pull (default: current branch).")
    parser.add_argument("--skip-pull", action="store_true", help="Do not git pull --rebase.")
    parser.add_argument(
        "--skip-manual-sync", action="store_true", help="Do not create manual_* under local_nightly_raw."
    )
    parser.add_argument("--skip-mkdocs", action="store_true", help="Do not run mkdocs build.")
    args = parser.parse_args()

    try:
        result = prepare_kanban_before_report(
            args.kanban_repo_root,
            repo_root=args.repo_root,
            log_dir=args.log_dir,
            remote=args.git_remote,
            branch=args.git_branch,
            skip_pull=args.skip_pull,
            skip_manual_sync=args.skip_manual_sync,
            skip_mkdocs=args.skip_mkdocs,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_prepare_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
