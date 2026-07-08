#!/usr/bin/env python3
"""
Commit and push staged kanban test reports after user confirmation.

Run **after** ``push_report_to_kanban.py`` has copied the report and staged it.
This script is the only step that performs ``git commit`` and ``git push``; it
always prompts for confirmation (or requires ``--yes`` after explicit user approval
in non-interactive / agent sessions).

Example::

  python scripts/push_kanban_report.py \\
    --kanban-repo-root /path/to/vllm-omni-kanban
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from laptop_path_defaults import DEFAULT_KANBAN_REPO_ROOT_DISPLAY
from push_report_to_kanban import (
    KANBAN_REPO_URL,
    GhCliRequiredError,
    PushCancelledError,
    PushConfirmationRequiredError,
    _confirm_push,
    _default_kanban_repo,
    build_push_preview_from_staged,
    commit_and_push_staged,
    format_push_confirmation_request,
    format_push_preview,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Commit and push staged test reports in vllm-omni-kanban "
            f"({KANBAN_REPO_URL}) "
            "(requires prior push_report_to_kanban.py staging)."
        ),
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
        help="Custom git commit message (default: infer from staged report).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned git actions without commit/push.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Print the full push preview (staged files, diff stat) and exit. "
            "Use before asking the user to confirm push."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help=(
            "Skip interactive confirmation and push immediately after showing preview. "
            "Use only after the user explicitly confirmed push in chat."
        ),
    )
    args = parser.parse_args()

    kanban_repo = args.kanban_repo_root.resolve()
    if not (kanban_repo / ".git").exists():
        print(f"Not a git repository: {kanban_repo}", file=sys.stderr)
        sys.exit(1)

    try:
        preview = build_push_preview_from_staged(
            kanban_repo,
            remote=args.remote,
            branch=args.branch,
            commit_message=args.commit_message,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if preview is None:
        print(
            "No staged report under data/nightly_test_report/, "
            "data/release_test_report/, or data/development_test_report/.\n"
            "Run push_report_to_kanban.py first to copy and stage the report "
            "(and optional data/local_nightly_raw/manual_*).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.preview_only or args.dry_run:
        if args.preview_only:
            print(format_push_confirmation_request(preview), flush=True)
        else:
            print(format_push_preview(preview))
            print(
                commit_and_push_staged(preview, dry_run=True),
                flush=True,
            )
        return

    try:
        _confirm_push(preview, assume_yes=args.yes)
        note = commit_and_push_staged(preview)
    except PushCancelledError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(0)
    except PushConfirmationRequiredError as exc:
        print(str(exc), flush=True)
        sys.exit(3)
    except (RuntimeError, GhCliRequiredError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(note)


if __name__ == "__main__":
    main()
