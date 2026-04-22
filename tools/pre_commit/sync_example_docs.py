#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-commit hook to sync example documentation."""

import subprocess
import sys
from pathlib import Path

EXAMPLES_PREFIX = "examples/"
DOCS_EXAMPLES_PREFIX = "docs/user_guide/examples/"
NAV_FILE = "docs/.nav.yml"


def get_repo_root() -> Path:
    """Return the absolute path to the git repo root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_staged_files() -> list[str]:
    """Return a list of file paths currently staged in the git index."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def get_modified_files(repo_root: Path) -> list[str]:
    """Get files in docs/user_guide/examples/ and .nav.yml that differ
    from the index (i.e. were modified by the generator)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", DOCS_EXAMPLES_PREFIX, NAV_FILE],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def get_untracked_files(repo_root: Path) -> list[str]:
    """Get new untracked files in docs/user_guide/examples/."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            DOCS_EXAMPLES_PREFIX,
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def get_deleted_tracked_files(repo_root: Path) -> list[str]:
    """Get tracked files in docs/user_guide/examples/ that no longer exist
    on disk (stale files from deleted examples)."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--deleted",
            "--",
            DOCS_EXAMPLES_PREFIX,
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def run_generator(repo_root: Path) -> None:
    """Import and run the example doc generator."""
    hooks_dir = str(repo_root / "docs" / "mkdocs" / "hooks")
    sys.path.insert(0, hooks_dir)
    try:
        import generate_examples

        generate_examples.on_startup("build", False)
    finally:
        sys.path.pop(0)


def main() -> int:
    staged = get_staged_files()
    if not staged:
        return 0

    examples_staged = [f for f in staged if f.startswith(EXAMPLES_PREFIX)]
    docs_examples_staged = [f for f in staged if f.startswith(DOCS_EXAMPLES_PREFIX)]

    # Block direct edits to generated files
    if docs_examples_staged and not examples_staged:
        print(
            "\033[91merror:\033[0m Direct edits to "
            "docs/user_guide/examples/ are not allowed.\n"
            "These files are auto-generated from examples/.\n"
            "Edit the source files in examples/ instead, then "
            "the pre-commit hook will regenerate the docs.",
            file=sys.stderr,
        )
        return 1

    # Nothing to do if no examples were touched
    if not examples_staged:
        return 0

    repo_root = get_repo_root()

    try:
        run_generator(repo_root)
    except Exception as e:
        print(
            f"\033[91merror:\033[0m Failed to generate example documentation: {e}",
            file=sys.stderr,
        )
        return 1

    # Auto-stage the regenerated docs and nav file
    modified = get_modified_files(repo_root)
    untracked = get_untracked_files(repo_root)
    deleted = get_deleted_tracked_files(repo_root)

    out_of_sync = modified + untracked + deleted
    if not out_of_sync:
        return 0

    # Stage modified and new files
    if modified or untracked:
        subprocess.run(
            ["git", "add", DOCS_EXAMPLES_PREFIX, NAV_FILE],
            cwd=repo_root,
            check=True,
        )

    # Remove stale doc files for deleted examples
    if deleted:
        subprocess.run(
            ["git", "rm", "-f", *deleted],
            cwd=repo_root,
            check=True,
        )

    print("Auto-synced example documentation:")
    for f in modified:
        print(f"  modified:  {f}")
    for f in untracked:
        print(f"  new file:  {f}")
    for f in deleted:
        print(f"  deleted:   {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
