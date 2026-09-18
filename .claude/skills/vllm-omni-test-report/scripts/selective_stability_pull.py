#!/usr/bin/env python3
"""Selective stability-log pull: skip downloading ``OK``-status job logs.

The nightly wrapper writes one ``timing_summary.log`` per run directory with
per-job status (``OK`` / ``FAILED (exit N)`` / ``TIMED OUT``). Long-stability
logs are often tens to hundreds of megabytes, so downloading every job's
``.log`` to discover pass/fail is wasteful — the rollup is enough to know
which logs are worth pulling.

This tool performs a **two-phase pull** from the cluster/container:

1. **Phase 1 — manifest pull:** pack only ``timing_summary.log`` + ``jobs/``
   from every stability run directory matched by the current sync scope.
   Extract on the laptop and parse each rollup.
2. **Phase 2 — selective full-log pull:** for each ``FAILED`` /
   ``TIMED OUT`` job, pack **just that** ``<job>.log`` from the matching
   run directory. ``OK`` jobs are not pulled; their status still lands in
   the report via :mod:`stability_log_manifest`.

The output dir layout matches the regular log-sync flow: every pulled run
directory is placed under ``$REPO_ROOT/logs/<run_dir_basename>/`` so the
report's discovery code can read it via the same path conventions.

Usage (H200 — direct SSH):

    python3 scripts/selective_stability_pull.py \
        --ssh-host my_h200 \
        --repo-root ~/vllm-omni \
        --sync-scope stability

Usage (H800 — srun + docker exec):

    python3 scripts/selective_stability_pull.py \
        --ssh-host h800.example.com \
        --slurm-user fq9hpsacuser07 \
        --slurm-jobid 12345 \
        --container-name omni_wy_24g \
        --repo-root ~/vllm-omni \
        --sync-scope stability

By default the script only pulls stability runs (which is where the bulk
savings come from). Pass ``--include-local`` to also pull the regular
nightly and local runs in one tarball (no selective phase 2 applies to
them — they're typically much smaller).

The script writes a JSON sidecar to
``$REPO_ROOT/logs/.selective_pull_manifest.json`` so downstream tooling
(``nightly_local_log_report.py``) can detect the manifest-only mode without
having to re-parse every ``timing_summary.log``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

# Make sibling ``stability_log_manifest`` importable when this script is run
# directly (``python3 scripts/selective_stability_pull.py``).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from stability_log_manifest import (  # noqa: E402  (sys.path tweak above)
    StabilityManifest,
    parse_timing_summary,
    select_failed_jobs,
)

SCRIPT_BANNER = "selective_stability_pull"

# Remote globs — same names the regular log-fetch flow uses.
_STABILITY_GLOB = "nightly_stability_jobs_*"
_LOCAL_GLOB = "nightly_jobs_local_*"
_GENERAL_GLOB = "nightly_jobs_[0-9]*-[0-9]*"


@dataclass
class RemoteRunDir:
    """One remote run directory selected for pull."""

    basename: str  # e.g. ``nightly_stability_jobs_20260808-031552``
    glob_family: str  # ``"stability"`` / ``"local"`` / ``"general"``


@dataclass
class SelectivePullResult:
    """Summary of the two-phase pull, written to a JSON sidecar."""

    repo_root: str
    sync_scope: str
    include_local: bool
    runs: list[dict] = field(default_factory=list)
    total_pulled_bytes: int = 0
    total_skipped_bytes: int = 0

    def to_sidecar_json(self) -> dict:
        return {
            "tool": SCRIPT_BANNER,
            "repo_root": self.repo_root,
            "sync_scope": self.sync_scope,
            "include_local": self.include_local,
            "runs": self.runs,
            "total_pulled_bytes": self.total_pulled_bytes,
            "total_skipped_bytes": self.total_skipped_bytes,
        }


# ---------------------------------------------------------------------------
# Remote command construction
# ---------------------------------------------------------------------------


def _build_remote_prefix(
    *,
    ssh_host: str,
    slurm_jobid: str | None,
    slurm_user: str | None,
    container_name: str | None,
    cluster_repo_root: str,
) -> str:
    """Return the bash one-liner prefix that enters the right execution context.

    - **H200** (``ssh_host`` only, no slurm/container) — direct SSH.
    - **H800 with srun jobid** — wrap every command in
      ``srun --jobid=… --overlap docker exec <container>``.
    - **H800 with slurm_user but no jobid** — fetch the user-id first.
    """
    inner = f'export ROOT={shlex.quote(cluster_repo_root)}\n  export LOGS_ROOT="${{ROOT}}/logs"\n'

    if slurm_jobid and container_name:
        # H800: srun --overlap docker exec <container>
        return (
            "bash -lc '"
            "type module >/dev/null 2>&1 && module load slurm 2>/dev/null; "
            f"srun --jobid={shlex.quote(slurm_jobid)} --overlap "
            f"docker exec {shlex.quote(container_name)} bash -lc "
            f"{shlex.quote(inner)}"
            "'"
        )

    if slurm_user and not slurm_jobid:
        # H800 fallback: resolve jobid first, then wrap. We return a *marker*
        # string the caller recognizes and unwraps in two phases.
        return f"RESOLVE_JOBID::{shlex.quote(ssh_host)}::{shlex.quote(slurm_user)}::{shlex.quote(container_name or '')}"

    # H200 (default): plain SSH.
    return ""


def _ssh_run(ssh_host: str, remote_bash: str, stdin_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    """Run ``remote_bash`` on ``ssh_host`` and return the completed process.

    ``remote_bash`` is a *single* shell command string (the body of an
    ``ssh ... bash -lc '…'``). The caller is responsible for any quoting.
    """
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=120",
        ssh_host,
        remote_bash,
    ]
    return subprocess.run(
        cmd,
        input=stdin_bytes,
        capture_output=True,
        check=False,
    )


def _remote_resolve_latest_runs(
    *,
    ssh_host: str,
    cluster_repo_root: str,
    sync_scope: str,
    include_local: bool,
    slurm_user: str | None,
    slurm_jobid: str | None,
    container_name: str | None,
) -> list[RemoteRunDir]:
    """Return the list of remote run dirs selected for pull (basenames only)."""
    families: list[tuple[str, str]] = []
    if sync_scope in ("stability", "all"):
        families.append(("stability", _STABILITY_GLOB))
    if include_local or sync_scope == "all":
        families.append(("local", _LOCAL_GLOB))
    if sync_scope in ("default", "all"):
        families.append(("general", _GENERAL_GLOB))

    if not families:
        return []

    # Build a remote command that prints one basename per line, prefixed with
    # the family name (e.g. ``stability\tnightly_stability_jobs_…``). We use
    # ls -dt + head -1 to pick the latest directory matching each glob.
    family_lines = []
    for family, glob_pat in families:
        family_lines.append(
            f'shopt -s nullglob; _c=( "$LOGS_ROOT"/{glob_pat} ); '
            f"shopt -u nullglob; "
            f"if ((${{#_c[@]}}>0)); then "
            f"printf '%s\\t%s\\n' '{family}' \"$(ls -dt \"${{_c[@]}}\" | head -1 | xargs -n1 basename)\"; "
            f"fi"
        )
    body = "ROOT=" + shlex.quote(cluster_repo_root) + '\nLOGS_ROOT="$ROOT/logs"\n' + "\n".join(family_lines)

    remote_bash = _wrap_for_remote(body, ssh_host, slurm_user, slurm_jobid, container_name)
    proc = _ssh_run(ssh_host, remote_bash)
    if proc.returncode != 0:
        sys.stderr.write(
            f"[{SCRIPT_BANNER}] remote listing failed (exit={proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='replace')}\n"
        )
        return []

    runs: list[RemoteRunDir] = []
    for raw in proc.stdout.decode("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or "\t" not in line:
            continue
        family, basename = line.split("\t", 1)
        basename = basename.strip()
        if basename:
            runs.append(RemoteRunDir(basename=basename, glob_family=family))
    return runs


def _wrap_for_remote(
    body: str,
    ssh_host: str,
    slurm_user: str | None,
    slurm_jobid: str | None,
    container_name: str | None,
) -> str:
    """Wrap a remote shell snippet for SSH / srun / docker exec."""
    if slurm_jobid and container_name:
        # H800 — wrap in srun + docker exec.
        return (
            "bash -lc '"
            "type module >/dev/null 2>&1 && module load slurm 2>/dev/null; "
            f"srun --jobid={shlex.quote(slurm_jobid)} --overlap "
            f"docker exec {shlex.quote(container_name)} bash -lc "
            f"{shlex.quote(body)}"
            "'"
        )
    if slurm_user and not slurm_jobid:
        # Resolve jobid first; the caller must do this in two phases (we
        # don't try to embed it into a single remote call here).
        raise RuntimeError("slurm_user provided without slurm_jobid — call _resolve_slurm_jobid first")
    # H200 default — plain bash -lc.
    return f"bash -lc {shlex.quote(body)}"


def _resolve_slurm_jobid(ssh_host: str, slurm_user: str) -> str | None:
    """H800: find the user's first RUNNING jobid via squeue."""
    body = (
        "type module >/dev/null 2>&1 && module load slurm 2>/dev/null; "
        f"squeue -u {shlex.quote(slurm_user)} -t RUNNING -h -o %i | head -1"
    )
    proc = _ssh_run(ssh_host, f"bash -lc {shlex.quote(body)}")
    if proc.returncode != 0:
        return None
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    return out or None


# ---------------------------------------------------------------------------
# Phase 1 — manifest pull
# ---------------------------------------------------------------------------


def _phase1_pull_manifests(
    *,
    ssh_host: str,
    cluster_repo_root: str,
    runs: list[RemoteRunDir],
    repo_root: Path,
    slurm_jobid: str | None,
    container_name: str | None,
) -> list[Path]:
    """Pull only ``timing_summary.log`` + ``jobs/`` from each run dir.

    Returns the local paths of every extracted run directory.
    """
    if not runs:
        return []

    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Remote pack: for each run dir, pack only timing_summary.log + jobs/.
    pack_args = []
    for run in runs:
        pack_args.append(f"{run.basename}/timing_summary.log")
        pack_args.append(f"{run.basename}/jobs")

    body = (
        f"ROOT={shlex.quote(cluster_repo_root)}\n"
        f'LOGS_ROOT="$ROOT/logs"\n'
        f'cd "$LOGS_ROOT" || exit 1\n'
        f"tar czf - --ignore-failed-read {' '.join(shlex.quote(a) for a in pack_args)}\n"
    )
    remote_bash = _wrap_for_remote(body, ssh_host, None, slurm_jobid, container_name)
    proc = _ssh_run(ssh_host, remote_bash)
    if proc.returncode != 0:
        raise RuntimeError(
            f"phase-1 pack failed (exit={proc.returncode}): {proc.stderr.decode('utf-8', errors='replace')}"
        )

    manifest_tgz = logs_dir / ".selective_pull_phase1.tgz"
    manifest_tgz.write_bytes(proc.stdout)
    try:
        with tarfile.open(manifest_tgz, "r:gz") as tf:
            # Safety: only extract entries that live directly under a known
            # run-dir basename, so a malicious archive cannot escape.
            for member in tf.getmembers():
                if not member.isreg() and not member.isdir():
                    continue
                first = member.name.split("/", 1)[0]
                if first not in {r.basename for r in runs}:
                    continue
                target = logs_dir / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Drop any setuid / device bits — these are plain log files.
                member.mode = member.mode & 0o755
                tf.extract(member, logs_dir)
    finally:
        manifest_tgz.unlink(missing_ok=True)

    return [logs_dir / r.basename for r in runs]


# ---------------------------------------------------------------------------
# Phase 2 — selective full-log pull (failed jobs only)
# ---------------------------------------------------------------------------


def _phase2_pull_failed_logs(
    *,
    ssh_host: str,
    cluster_repo_root: str,
    run_dir: Path,
    failed_jobs: Iterable[str],
    slurm_jobid: str | None,
    container_name: str | None,
) -> int:
    """Pull the ``.log`` files for ``failed_jobs`` from one run dir.

    Returns the number of bytes downloaded. Skips jobs whose ``.log`` is
    already present locally.
    """
    # Only consider jobs whose log is missing locally — saves a round trip
    # on retries.
    jobs_to_pull: list[str] = []
    for job in failed_jobs:
        candidate = run_dir / f"{job}.log"
        if candidate.exists():
            continue
        jobs_to_pull.append(job)
    if not jobs_to_pull:
        return 0

    pack_args = [f"{run_dir.name}/{job}.log" for job in jobs_to_pull]
    body = (
        f"ROOT={shlex.quote(cluster_repo_root)}\n"
        f'LOGS_ROOT="$ROOT/logs"\n'
        f'cd "$LOGS_ROOT" || exit 1\n'
        f"tar czf - --ignore-failed-read {' '.join(shlex.quote(a) for a in pack_args)}\n"
    )
    remote_bash = _wrap_for_remote(body, ssh_host, None, slurm_jobid, container_name)
    proc = _ssh_run(ssh_host, remote_bash)
    if proc.returncode != 0:
        # Non-fatal — record the failure on stderr and continue so the user
        # can still produce a report from the manifest + any locally-cached
        # logs. The report will surface these as "manifest only — log not
        # pulled" rows.
        sys.stderr.write(
            f"[{SCRIPT_BANNER}] phase-2 pack failed for {run_dir.name} "
            f"(exit={proc.returncode}): {proc.stderr.decode('utf-8', errors='replace')}\n"
        )
        return 0

    payload = proc.stdout
    if not payload:
        return 0

    with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isreg():
                    continue
                first = member.name.split("/", 1)[0]
                if first != run_dir.name:
                    continue
                target = run_dir / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                member.mode = member.mode & 0o755
                tf.extract(member, run_dir.parent)
    finally:
        tmp_path.unlink(missing_ok=True)

    return len(payload)


# ---------------------------------------------------------------------------
# Sidecar + report-side hints
# ---------------------------------------------------------------------------


def _write_sidecar(repo_root: Path, result: SelectivePullResult) -> None:
    sidecar = repo_root / "logs" / ".selective_pull_manifest.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(result.to_sidecar_json(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=SCRIPT_BANNER,
        description="Two-phase stability-log pull (manifest first, then failed-job logs only).",
    )
    p.add_argument(
        "--ssh-host",
        required=True,
        help="SSH connection name (e.g. my_h200). Required for any pull.",
    )
    p.add_argument(
        "--cluster-repo-root",
        default="/rebase/vllm-omni",
        help="Path to the cluster checkout (default: /rebase/vllm-omni).",
    )
    p.add_argument(
        "--repo-root",
        default="~/vllm-omni",
        help="Local repo root (default: ~/vllm-omni).",
    )
    p.add_argument(
        "--sync-scope",
        choices=("stability", "default", "all"),
        default="stability",
        help="Which remote run families to pull (default: stability).",
    )
    p.add_argument(
        "--include-local",
        action="store_true",
        help="Also pull nightly_jobs_local_* (smaller, no selective phase-2).",
    )

    # H800 plumbing.
    p.add_argument("--slurm-user", default=None, help="Slurm username (H800 only).")
    p.add_argument("--slurm-jobid", default=None, help="Slurm jobid (H800 only).")
    p.add_argument(
        "--container-name",
        default=None,
        help="Docker container name (H800 only — logs are inside the workload container).",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without doing any SSH / tarball.",
    )
    p.add_argument(
        "--print-sidecar",
        action="store_true",
        help="After the pull, print the JSON sidecar to stdout (for piping).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    cluster_repo_root = args.cluster_repo_root

    # H800 fallback: resolve jobid from squeue if user gave --slurm-user but
    # no jobid yet.
    slurm_jobid = args.slurm_jobid
    if args.slurm_user and not slurm_jobid and args.container_name:
        resolved = _resolve_slurm_jobid(args.ssh_host, args.slurm_user)
        if not resolved:
            sys.stderr.write(
                f"[{SCRIPT_BANNER}] could not resolve a RUNNING jobid for user "
                f"{args.slurm_user!r}; pass --slurm-jobid explicitly.\n"
            )
            return 2
        slurm_jobid = resolved
        print(f"[{SCRIPT_BANNER}] resolved slurm jobid={slurm_jobid}")

    # Step 1 — figure out which run dirs to pull.
    if args.dry_run:
        runs = []
        # Synthesize a tiny demo run so the dry-run output is non-empty.
        print(
            f"[{SCRIPT_BANNER}] dry-run: would ssh into {args.ssh_host!r} and "
            f"list {cluster_repo_root}/logs/ for scope={args.sync_scope}"
        )
        return 0

    runs = _remote_resolve_latest_runs(
        ssh_host=args.ssh_host,
        cluster_repo_root=cluster_repo_root,
        sync_scope=args.sync_scope,
        include_local=args.include_local,
        slurm_user=args.slurm_user,
        slurm_jobid=slurm_jobid,
        container_name=args.container_name,
    )
    if not runs:
        print(f"[{SCRIPT_BANNER}] no matching run dirs under {cluster_repo_root}/logs")
        return 0

    print(f"[{SCRIPT_BANNER}] selected runs: {[r.basename for r in runs]}")

    # Phase 1 — manifest pull.
    extracted_runs = _phase1_pull_manifests(
        ssh_host=args.ssh_host,
        cluster_repo_root=cluster_repo_root,
        runs=runs,
        repo_root=repo_root,
        slurm_jobid=slurm_jobid,
        container_name=args.container_name,
    )

    # Parse each manifest.
    manifests: list[StabilityManifest] = []
    for local_run in extracted_runs:
        summary = local_run / "timing_summary.log"
        if not summary.is_file():
            continue
        m = parse_timing_summary(summary, run_dir_name=local_run.name)
        manifests.append(m)

    # Phase 2 — selective full-log pull (failed jobs only).
    result = SelectivePullResult(
        repo_root=str(repo_root),
        sync_scope=args.sync_scope,
        include_local=args.include_local,
    )
    for local_run, manifest in zip(extracted_runs, manifests):
        run_record = {
            "run_dir": local_run.name,
            "ok_jobs": list(manifest.ok_jobs),
            "failed_jobs": list(manifest.failed_jobs),
            "pulled_logs": [],
            "skipped_logs": [],
        }

        # Only do phase 2 for stability runs; local/general runs are tiny.
        is_stability = local_run.name.startswith("nightly_stability_jobs_")
        if not is_stability:
            run_record["note"] = "non-stability run — no phase-2 selective pull"
            result.runs.append(run_record)
            continue

        # Pull only failed-job logs.
        failed = select_failed_jobs(manifest)
        pulled_bytes = _phase2_pull_failed_logs(
            ssh_host=args.ssh_host,
            cluster_repo_root=cluster_repo_root,
            run_dir=local_run,
            failed_jobs=failed,
            slurm_jobid=slurm_jobid,
            container_name=args.container_name,
        )
        run_record["pulled_bytes"] = pulled_bytes

        for job in manifest.entries:
            target = local_run / f"{job.job_name}.log"
            if target.is_file():
                run_record["pulled_logs"].append(job.job_name)
                try:
                    result.total_pulled_bytes += target.stat().st_size
                except OSError:
                    pass
            else:
                run_record["skipped_logs"].append({"job": job.job_name, "status": job.raw_status})
                # Approximate skip size: try to stat the remote file via the
                # next phase-2 round; in practice we just sum what's on disk
                # minus what we pulled, so leave the estimate at 0 here.
        result.runs.append(run_record)

    _write_sidecar(repo_root, result)
    print(
        f"[{SCRIPT_BANNER}] done — pulled={result.total_pulled_bytes} bytes across "
        f"{sum(len(r['pulled_logs']) for r in result.runs)} logs; "
        f"sidecar: {repo_root / 'logs' / '.selective_pull_manifest.json'}"
    )

    if args.print_sidecar:
        print(json.dumps(result.to_sidecar_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
