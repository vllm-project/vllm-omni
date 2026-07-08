#!/usr/bin/env python3
"""
Archive a generated HTML test report into https://github.com/hsliuustc0106/vllm-omni-kanban and stage for push.

Copies the report into ``data/nightly_test_report/`` or ``data/release_test_report/``
(with kanban-expected filenames). **Do not commit** ``docs/assets/test_reports/`` — that
tree is gitignored in kanban; ``mkdocs build`` / ``mkdocs serve`` regenerates it from
``data/`` via ``scripts/mkdocs_hooks.py`` → ``sync_test_reports.py``.

Then ``git pull --rebase``, ``git add`` (data HTML only), and print a **push preview**.
This script does **not** commit or push. After reviewing the preview, run
``push_kanban_report.py`` separately (it prompts for confirmation before ``git push``).

Uses **GitHub CLI (``gh``)** credentials for pull (``gh auth git-credential``).
Requires ``gh`` installed and authenticated (``gh auth login`` or ``GH_TOKEN`` / ``GITHUB_TOKEN``).

Default remote repo: https://github.com/hsliuustc0106/vllm-omni-kanban

Run from the skill directory after generating a report::

  python scripts/push_report_to_kanban.py \\
    --report ./nightly-report-buildkite-latest-2026-06-22.html \\
    --kanban-repo-root /path/to/vllm-omni-kanban \\
    --kind nightly

  python scripts/push_kanban_report.py \\
    --kanban-repo-root /path/to/vllm-omni-kanban
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2

from kanban_local_nightly_raw import (
    HUNYUAN_MANUAL_DEST_LOG,
    HUNYUAN_NIGHTLY_SOURCE_LOG_CANDIDATES,
    LOCAL_NIGHTLY_RAW,
    manual_dir_rel,
    resolve_manual_dir_for_archive,
)
from kanban_repo_config import KANBAN_REPO_GIT_SUFFIX, KANBAN_REPO_URL
from laptop_path_defaults import (
    DEFAULT_KANBAN_REPO_ROOT_DISPLAY,
    resolve_kanban_repo_root,
)
from report_naming import utc_report_date_iso  # noqa: E402

GH_CLI_INSTALL_HINT = (
    "GitHub CLI (gh) is not installed. Archive push requires gh:\n"
    "  Windows: winget install --id GitHub.cli\n"
    "  macOS:   brew install gh\n"
    "  Linux:   https://github.com/cli/cli/blob/trunk/docs/install_linux.md\n"
    "After install, log in: gh auth login\n"
    "Docs: https://cli.github.com/"
)
GH_AUTH_HINT = (
    "gh is not logged in or the token is invalid. Run: gh auth login\n"
    "Or set GH_TOKEN / GITHUB_TOKEN in the environment (requires repo scope)."
)
GH_GIT_CREDENTIAL_HELPER = "!gh auth git-credential"

NIGHTLY_CANONICAL_RE = re.compile(
    r"^nightly-report-buildkite-latest-(?P<date>\d{4}-\d{2}-\d{2})\.html$",
    re.IGNORECASE,
)
RELEASE_CANONICAL_RE = re.compile(
    r"^vllm-omni-release-test-report-(?P<date>\d{4}-\d{2}-\d{2})\.html$",
    re.IGNORECASE,
)
RELEASE_LOOSE_RE = re.compile(
    r"^vllm-omni-test-report(?:-preview)?-(?P<date>\d{4}-\d{2}-\d{2})\.html$",
    re.IGNORECASE,
)
DEVELOPMENT_CANONICAL_RE = re.compile(
    r"^vllm-omni-test-report-development-(?P<date>\d{4}-\d{2}-\d{2})\.html$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArchivePlan:
    kind: str  # "nightly" | "release" | "development"
    report_date: str  # YYYY-MM-DD
    dest_name: str
    dest_rel: Path  # relative to kanban repo root
    local_nightly_manual_rel: Path | None = None  # e.g. data/local_nightly_raw/manual_YYYYMMDD


@dataclass(frozen=True)
class PushPreview:
    """Staged changes ready to commit and push."""

    plan: ArchivePlan
    kanban_repo: Path
    remote: str
    remote_url: str
    branch: str
    commit_message: str
    paths: list[str]
    name_status: str
    diff_stat: str
    file_details: list[str]
    staging_warnings: list[str] = field(default_factory=list)


class GhCliRequiredError(RuntimeError):
    """Raised when gh is missing or not authenticated for push."""


class PushCancelledError(RuntimeError):
    """User declined push after reviewing the preview."""


class PushConfirmationRequiredError(RuntimeError):
    """Non-interactive session: user must confirm after reviewing the embedded preview."""

    def __init__(self, preview: PushPreview) -> None:
        self.preview = preview
        super().__init__(format_push_confirmation_request(preview))


def _gh_cli_path() -> str | None:
    return shutil.which("gh")


def ensure_gh_cli() -> None:
    if _gh_cli_path() is None:
        raise GhCliRequiredError(GH_CLI_INSTALL_HINT)


def ensure_gh_authenticated() -> None:
    ensure_gh_cli()
    proc = _run_gh("auth", "status", check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        msg = GH_AUTH_HINT
        if detail:
            msg = f"{GH_AUTH_HINT}\n{detail}"
        raise GhCliRequiredError(msg)


def _run_git(
    repo: Path,
    *args: str,
    check: bool = True,
    gh_credential: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = ["git"]
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


def _run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    ensure_gh_cli()
    proc = subprocess.run(
        ["gh", *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"gh {' '.join(args)} failed: {detail}")
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
        args.extend(["-c", "user.name=vllm-omni-test-report"])
    if not email:
        args.extend(["-c", "user.email=vllm-omni-test-report@users.noreply.github.com"])
    return args


def _git_remote_url(repo: Path, remote: str) -> str:
    proc = _run_git(repo, "remote", "get-url", remote, check=False)
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def infer_report_kind_and_date(
    report_path: Path,
    *,
    kind: str | None = None,
    report_date: str | None = None,
) -> tuple[str, str]:
    name = report_path.name
    if kind is None:
        if NIGHTLY_CANONICAL_RE.match(name) or name.lower().startswith("nightly"):
            kind = "nightly"
        elif DEVELOPMENT_CANONICAL_RE.match(name) or "-development-" in name.lower():
            kind = "development"
        elif RELEASE_CANONICAL_RE.match(name) or RELEASE_LOOSE_RE.match(name):
            kind = "release"
        else:
            kind = "nightly" if "nightly" in name.lower() else "release"
    kind = kind.lower()
    if kind not in ("nightly", "release", "development"):
        raise ValueError(f"Unsupported report kind: {kind!r}")

    if report_date:
        return kind, report_date

    for pattern in (NIGHTLY_CANONICAL_RE, DEVELOPMENT_CANONICAL_RE, RELEASE_CANONICAL_RE, RELEASE_LOOSE_RE):
        match = pattern.match(name)
        if match:
            return kind, match.group("date")

    # Non-canonical filenames may embed log-sync timestamps; use generation date instead.
    return kind, utc_report_date_iso()


def build_archive_plan(
    report_path: Path,
    *,
    kind: str | None = None,
    report_date: str | None = None,
) -> ArchivePlan:
    resolved_kind, resolved_date = infer_report_kind_and_date(
        report_path,
        kind=kind,
        report_date=report_date,
    )
    if resolved_kind == "nightly":
        dest_name = f"nightly-report-buildkite-latest-{resolved_date}.html"
        dest_rel = Path("data") / "nightly_test_report" / dest_name
    elif resolved_kind == "development":
        dest_name = f"vllm-omni-test-report-development-{resolved_date}.html"
        dest_rel = Path("data") / "development_test_report" / dest_name
    else:
        dest_name = f"vllm-omni-release-test-report-{resolved_date}.html"
        dest_rel = Path("data") / "release_test_report" / dest_name
    return ArchivePlan(
        kind=resolved_kind,
        report_date=resolved_date,
        dest_name=dest_name,
        dest_rel=dest_rel,
    )


def archive_report_to_kanban(
    report_path: Path,
    kanban_repo: Path,
    *,
    kind: str | None = None,
    report_date: str | None = None,
) -> ArchivePlan:
    report_path = report_path.resolve()
    kanban_repo = kanban_repo.resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Report not found: {report_path}")
    if not kanban_repo.is_dir():
        raise NotADirectoryError(f"Kanban repo root not found: {kanban_repo}")

    plan = build_archive_plan(report_path, kind=kind, report_date=report_date)
    dest = kanban_repo / plan.dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy2(report_path, dest)
    return plan


def _git_paths_for_plan(plan: ArchivePlan) -> list[str]:
    """Tracked paths: report HTML under ``data/*_test_report/`` plus optional ``manual_*`` raw."""
    paths = [str(plan.dest_rel).replace("\\", "/")]
    if plan.local_nightly_manual_rel is not None:
        paths.append(str(plan.local_nightly_manual_rel).replace("\\", "/"))
    return paths


def _staged_rel_paths_scoped(kanban_repo: Path, scope_paths: list[str]) -> set[str]:
    if not scope_paths:
        return set()
    proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--name-only",
        "--",
        *scope_paths,
        check=False,
    )
    return {raw.strip().replace("\\", "/") for raw in (proc.stdout or "").splitlines() if raw.strip()}


def _git_add_archive_paths(kanban_repo: Path, plan: ArchivePlan) -> None:
    """Stage report HTML and ``manual_*`` (force-add raw dir so ``*.log`` is not skipped)."""
    paths = _git_paths_for_plan(plan)
    _run_git(kanban_repo, "add", paths[0])
    if plan.local_nightly_manual_rel is not None:
        manual = str(plan.local_nightly_manual_rel).replace("\\", "/")
        _run_git(kanban_repo, "add", "-f", manual)


def _manual_staging_warnings(
    kanban_repo: Path,
    plan: ArchivePlan,
    staged_rels: set[str],
) -> list[str]:
    if plan.local_nightly_manual_rel is None:
        return []
    manual_dir = kanban_repo / plan.local_nightly_manual_rel
    if not manual_dir.is_dir():
        return [
            f"Warning: {plan.local_nightly_manual_rel.as_posix()} not found on disk "
            "(run prepare_kanban_before_report.py first)."
        ]

    warnings: list[str] = []
    log_rel = f"{plan.local_nightly_manual_rel.as_posix()}/{HUNYUAN_MANUAL_DEST_LOG}"
    log_path = manual_dir / HUNYUAN_MANUAL_DEST_LOG
    if log_path.is_file():
        if log_rel not in staged_rels:
            warnings.append(
                f"Warning: {log_rel} is on disk but not staged "
                f"(kanban .gitignore may block *.log; archive uses git add -f)."
            )
    else:
        warnings.append(
            f"Warning: {log_rel} missing — prep should copy one of "
            f"logs/nightly_jobs/{{{', '.join(HUNYUAN_NIGHTLY_SOURCE_LOG_CANDIDATES)}}} "
            f"→ {HUNYUAN_MANUAL_DEST_LOG}."
        )

    for path in sorted(manual_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(kanban_repo).as_posix()
        if rel not in staged_rels and rel != log_rel:
            warnings.append(f"Warning: on disk but not staged: {rel}")
    return warnings


def _default_commit_message(plan: ArchivePlan) -> str:
    msg = f"chore(reports): archive {plan.kind} test report {plan.report_date}"
    if plan.local_nightly_manual_rel is not None:
        msg += f" + {plan.local_nightly_manual_rel.name}"
    return msg


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _staged_report_rel_paths(kanban_repo: Path) -> list[str]:
    proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--name-only",
        check=False,
    )
    paths: list[str] = []
    for raw in (proc.stdout or "").splitlines():
        rel = raw.strip().replace("\\", "/")
        if (
            rel.startswith("data/nightly_test_report/")
            or rel.startswith("data/release_test_report/")
            or rel.startswith("data/development_test_report/")
        ):
            paths.append(rel)
    return paths


def _infer_manual_rel_from_staged(staged: list[str]) -> Path | None:
    prefix = f"{LOCAL_NIGHTLY_RAW.as_posix()}/"
    for rel in staged:
        if rel.startswith(prefix) and rel.count("/") >= 2:
            parts = rel.split("/")
            if parts[2].startswith("manual_"):
                return Path("/".join(parts[:3]))
    return None


def infer_plan_from_staged_rel(rel: str) -> ArchivePlan:
    rel = rel.replace("\\", "/")
    name = Path(rel).name
    if "nightly_test_report" in rel:
        match = NIGHTLY_CANONICAL_RE.match(name)
        if match:
            report_date = match.group("date")
        else:
            report_date = utc_report_date_iso()
        return ArchivePlan(
            kind="nightly",
            report_date=report_date,
            dest_name=name,
            dest_rel=Path(rel),
        )

    if "development_test_report" in rel:
        match = DEVELOPMENT_CANONICAL_RE.match(name)
        if match:
            report_date = match.group("date")
        else:
            report_date = utc_report_date_iso()
        return ArchivePlan(
            kind="development",
            report_date=report_date,
            dest_name=name,
            dest_rel=Path(rel),
        )

    match = RELEASE_CANONICAL_RE.match(name) or RELEASE_LOOSE_RE.match(name)
    if match:
        report_date = match.group("date")
    else:
        report_date = utc_report_date_iso()
    return ArchivePlan(
        kind="release",
        report_date=report_date,
        dest_name=name,
        dest_rel=Path(rel),
    )


def build_push_preview_from_staged(
    kanban_repo: Path,
    *,
    remote: str = "origin",
    branch: str | None = None,
    commit_message: str | None = None,
) -> PushPreview | None:
    kanban_repo = kanban_repo.resolve()
    proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--name-only",
        check=False,
    )
    staged_all = [raw.strip().replace("\\", "/") for raw in (proc.stdout or "").splitlines() if raw.strip()]
    report_staged = _staged_report_rel_paths(kanban_repo)
    if not report_staged:
        return None

    plan_base = infer_plan_from_staged_rel(report_staged[0])
    manual_rel = _infer_manual_rel_from_staged(staged_all)
    plan = ArchivePlan(
        kind=plan_base.kind,
        report_date=plan_base.report_date,
        dest_name=plan_base.dest_name,
        dest_rel=plan_base.dest_rel,
        local_nightly_manual_rel=manual_rel,
    )
    branch = branch or _git_current_branch(kanban_repo)
    commit_message = commit_message or _default_commit_message(plan)
    paths = _git_paths_for_plan(plan)
    staged_rels = _staged_rel_paths_scoped(kanban_repo, paths)
    warnings = _manual_staging_warnings(kanban_repo, plan, staged_rels)
    return _build_push_preview(
        kanban_repo,
        plan,
        remote=remote,
        branch=branch,
        commit_message=commit_message,
        paths=paths,
        staging_warnings=warnings,
    )


def _staged_file_details(kanban_repo: Path, paths: list[str]) -> list[str]:
    proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--name-only",
        "--",
        *paths,
        check=False,
    )
    lines: list[str] = []
    for raw in (proc.stdout or "").splitlines():
        rel = raw.strip()
        if not rel:
            continue
        path = kanban_repo / rel
        if path.is_file():
            size = _format_bytes(path.stat().st_size)
            lines.append(f"  {rel}  ({size})")
        else:
            lines.append(f"  {rel}")
    return lines


def _build_push_preview(
    kanban_repo: Path,
    plan: ArchivePlan,
    *,
    remote: str,
    branch: str,
    commit_message: str,
    paths: list[str],
    staging_warnings: list[str] | None = None,
) -> PushPreview:
    name_status_proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--name-status",
        "--",
        *paths,
        check=False,
    )
    diff_stat_proc = _run_git(
        kanban_repo,
        "diff",
        "--cached",
        "--stat",
        "--",
        *paths,
        check=False,
    )
    return PushPreview(
        plan=plan,
        kanban_repo=kanban_repo,
        remote=remote,
        remote_url=_git_remote_url(kanban_repo, remote),
        branch=branch,
        commit_message=commit_message,
        paths=paths,
        name_status=(name_status_proc.stdout or "").strip() or "(no staged diff)",
        diff_stat=(diff_stat_proc.stdout or "").strip() or "(no diff stat)",
        file_details=_staged_file_details(kanban_repo, paths),
        staging_warnings=staging_warnings or [],
    )


def format_push_preview(preview: PushPreview) -> str:
    lines = [
        "",
        "=" * 60,
        "Kanban push preview",
        "=" * 60,
        f"Repository : {preview.kanban_repo}",
        f"Remote     : {preview.remote}" + (f" -> {preview.remote_url}" if preview.remote_url else ""),
        f"Branch     : {preview.branch}",
        f"Target     : {KANBAN_REPO_URL}",
        f"Report kind: {preview.plan.kind}",
        f"Report date: {preview.plan.report_date}",
        f"Archive    : {preview.plan.dest_rel.as_posix()}",
    ]
    if preview.plan.local_nightly_manual_rel is not None:
        lines.append(f"Local raw  : {preview.plan.local_nightly_manual_rel.as_posix()} (included in commit)")
    lines.extend(
        [
            "Note       : docs/assets/test_reports/ is gitignored; MkDocs syncs from data/ at build",
            f"Commit msg : {preview.commit_message}",
            "",
            "Staged files:",
        ]
    )
    if preview.file_details:
        lines.extend(preview.file_details)
    else:
        lines.append("  (none)")
    if preview.staging_warnings:
        lines.append("")
        lines.append("Staging warnings:")
        lines.extend(f"  {w}" for w in preview.staging_warnings)
    lines.extend(
        [
            "",
            "Name-status (staged):",
            preview.name_status,
            "",
            "Diff stat (staged):",
            preview.diff_stat,
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def format_staged_ready_message(preview: PushPreview) -> str:
    """Full message after archive+stage (step 1); includes the push preview inline."""
    return "\n".join(
        [
            format_push_preview(preview),
            "",
            "-" * 60,
            "Staged for commit (not pushed yet).",
            "Review the repository, branch, commit message, and staged diff above.",
            "Next step — push after you confirm:",
            f"  python scripts/push_kanban_report.py --kanban-repo-root {preview.kanban_repo}",
            "",
            "Agents: paste the entire block above (including Kanban push preview) "
            "to the user before asking whether to push.",
        ]
    )


def format_push_confirmation_request(preview: PushPreview) -> str:
    """Full message when push confirmation is required (step 2, non-interactive)."""
    return "\n".join(
        [
            format_push_preview(preview),
            "",
            "-" * 60,
            "Push confirmation required.",
            "Review the repository, branch, commit message, and staged diff above.",
            "Reply yes/confirm in chat to approve; then re-run:",
            f"  python scripts/push_kanban_report.py --kanban-repo-root {preview.kanban_repo} --yes",
            "",
            "Agents: paste the entire block above to the user; do not summarize as "
            "'ready to push' without showing these details.",
        ]
    )


def _unstage_paths(kanban_repo: Path, paths: list[str]) -> None:
    _run_git(kanban_repo, "reset", "HEAD", "--", *paths, check=False)


def _confirm_push(preview: PushPreview, *, assume_yes: bool) -> None:
    if assume_yes:
        print(format_push_preview(preview), flush=True)
        print("Push confirmed via --yes.", flush=True)
        return
    if sys.stdin.isatty():
        print(format_push_preview(preview), flush=True)
        try:
            answer = input("\nProceed with git commit and push to kanban? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("y", "yes"):
            return
        _unstage_paths(preview.kanban_repo, preview.paths)
        raise PushCancelledError("Push cancelled by user.")

    # Non-interactive: keep staged so a later --yes run can push.
    raise PushConfirmationRequiredError(preview)


def _prepare_staged_push(
    kanban_repo: Path,
    plan: ArchivePlan,
    *,
    remote: str,
    branch: str | None,
    commit_message: str | None,
    skip_pull: bool,
) -> PushPreview | None:
    kanban_repo = kanban_repo.resolve()
    if not (kanban_repo / ".git").exists():
        raise RuntimeError(f"Not a git repository: {kanban_repo}")

    remote_url = _git_remote_url(kanban_repo, remote)
    if remote_url and KANBAN_REPO_GIT_SUFFIX not in remote_url:
        print(
            f"Warning: remote {remote!r} URL {remote_url!r} does not look like vllm-omni-kanban.",
            file=sys.stderr,
        )

    branch = branch or _git_current_branch(kanban_repo)
    commit_message = commit_message or _default_commit_message(plan)
    paths = _git_paths_for_plan(plan)

    ensure_gh_authenticated()

    if not skip_pull:
        pull = _run_git(
            kanban_repo,
            "pull",
            "--rebase",
            remote,
            branch,
            check=False,
            gh_credential=True,
        )
        if pull.returncode != 0:
            detail = (pull.stderr or pull.stdout or "").strip()
            raise RuntimeError(f"git pull --rebase {remote} {branch} failed: {detail}")

    _git_add_archive_paths(kanban_repo, plan)
    status = _run_git(kanban_repo, "status", "--porcelain", *paths, check=False)
    if not (status.stdout or "").strip():
        return None

    staged_rels = _staged_rel_paths_scoped(kanban_repo, paths)
    warnings = _manual_staging_warnings(kanban_repo, plan, staged_rels)

    return _build_push_preview(
        kanban_repo,
        plan,
        remote=remote,
        branch=branch,
        commit_message=commit_message,
        paths=paths,
        staging_warnings=warnings,
    )


def commit_and_push_staged(
    preview: PushPreview,
    *,
    dry_run: bool = False,
) -> str:
    if dry_run:
        path_list = " ".join(preview.paths)
        return (
            f"[dry-run] would git commit -m {preview.commit_message!r} -- {path_list}; "
            f"git push {preview.remote} {preview.branch} (gh credential)"
        )

    ensure_gh_authenticated()
    commit_args = _git_commit_identity_args(preview.kanban_repo)
    commit_args.extend(["commit", "-m", preview.commit_message, "--", *preview.paths])
    _run_git(preview.kanban_repo, *commit_args)
    push = _run_git(
        preview.kanban_repo,
        "push",
        preview.remote,
        preview.branch,
        check=False,
        gh_credential=True,
    )
    if push.returncode != 0:
        detail = (push.stderr or push.stdout or "").strip()
        raise RuntimeError(f"git push {preview.remote} {preview.branch} failed (via gh credentials): {detail}")
    return (
        f"Pushed {preview.plan.kind} report {preview.plan.report_date} to "
        f"{preview.remote}/{preview.branch} ({KANBAN_REPO_URL}) using gh CLI credentials."
    )


def prepare_report_for_kanban(
    report_path: Path,
    kanban_repo: Path,
    *,
    kind: str | None = None,
    report_date: str | None = None,
    remote: str = "origin",
    branch: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    archive_only: bool = False,
    skip_pull: bool = False,
    local_nightly_manual_dir: Path | None = None,
    skip_local_nightly_raw: bool = False,
) -> tuple[ArchivePlan, str]:
    plan_base = archive_report_to_kanban(
        report_path,
        kanban_repo,
        kind=kind,
        report_date=report_date,
    )
    manual_dir = None
    if not skip_local_nightly_raw:
        manual_dir = resolve_manual_dir_for_archive(
            kanban_repo,
            local_nightly_manual_dir,
            auto=True,
        )
    manual_rel = Path(manual_dir_rel(manual_dir, kanban_repo)) if manual_dir is not None else None
    plan = ArchivePlan(
        kind=plan_base.kind,
        report_date=plan_base.report_date,
        dest_name=plan_base.dest_name,
        dest_rel=plan_base.dest_rel,
        local_nightly_manual_rel=manual_rel,
    )
    if archive_only:
        note = f"Archived to {kanban_repo / plan.dest_rel} (no git staging)."
        if manual_rel is not None:
            note += f" Local raw dir ready: {kanban_repo / manual_rel}"
        return plan, note

    paths = _git_paths_for_plan(plan)
    branch = branch or _git_current_branch(kanban_repo.resolve())
    commit_message = commit_message or _default_commit_message(plan)

    if dry_run:
        extra = ""
        if manual_rel is not None:
            extra = f"; git add {manual_rel.as_posix()}"
        return plan, (
            f"[dry-run] would verify gh CLI + gh auth status; "
            f"git -C {kanban_repo} pull --rebase {remote} {branch} (gh credential); "
            f"git add {' '.join(paths)}{extra}; show push preview; "
            f"then run push_kanban_report.py to confirm, commit, and push"
        )

    preview = _prepare_staged_push(
        kanban_repo,
        plan,
        remote=remote,
        branch=branch,
        commit_message=commit_message,
        skip_pull=skip_pull,
    )
    if preview is None:
        return plan, f"No git changes under {', '.join(paths)}; kanban already up to date."

    return plan, format_staged_ready_message(preview)


def _default_kanban_repo() -> Path:
    return resolve_kanban_repo_root()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Copy an HTML test report into vllm-omni-kanban ({KANBAN_REPO_URL}) data/, stage it, "
            "and print a push preview (does not commit or push)."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to the generated HTML report file.",
    )
    parser.add_argument(
        "--kanban-repo-root",
        type=Path,
        default=_default_kanban_repo(),
        help=(
            f"Local checkout of {KANBAN_REPO_URL}. Default: $KANBAN_REPO_ROOT or {DEFAULT_KANBAN_REPO_ROOT_DISPLAY}."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=("nightly", "release", "development"),
        default=None,
        help="Report type (default: infer from filename).",
    )
    parser.add_argument(
        "--date",
        dest="report_date",
        default=None,
        help="Report date YYYY-MM-DD (default: UTC today when filename is not canonical).",
    )
    parser.add_argument(
        "--remote",
        default="origin",
        help="Git remote name (default: origin).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Git branch to push (default: current branch).",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Custom git commit message.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned git actions without commit/push.",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Copy into kanban data/ only; do not stage or push.",
    )
    parser.add_argument(
        "--skip-pull",
        action="store_true",
        help="Skip git pull --rebase before staging (not recommended).",
    )
    parser.add_argument(
        "--local-nightly-manual-dir",
        type=Path,
        default=None,
        help=(
            "Explicit data/local_nightly_raw/manual_* directory to include in the push "
            "(default: .last_manual_dir marker written by prepare_kanban_before_report.py)."
        ),
    )
    parser.add_argument(
        "--skip-local-nightly-raw",
        action="store_true",
        help="Do not stage data/local_nightly_raw/manual_* (report HTML only).",
    )
    args = parser.parse_args()

    try:
        plan, note = prepare_report_for_kanban(
            args.report,
            args.kanban_repo_root,
            kind=args.kind,
            report_date=args.report_date,
            remote=args.remote,
            branch=args.branch,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
            archive_only=args.archive_only,
            skip_pull=args.skip_pull,
            local_nightly_manual_dir=args.local_nightly_manual_dir,
            skip_local_nightly_raw=args.skip_local_nightly_raw,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError, GhCliRequiredError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"Archived {args.report} -> {args.kanban_repo_root / plan.dest_rel}")
    if plan.local_nightly_manual_rel is not None:
        print(f"Local raw staged: {args.kanban_repo_root / plan.local_nightly_manual_rel}")
    print(note)


if __name__ == "__main__":
    main()
