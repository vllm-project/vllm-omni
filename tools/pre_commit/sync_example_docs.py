#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep docs/user_guide/examples/ in sync with examples/ on commit.

The pages under docs/user_guide/examples/ are generated from the README files
in examples/. mkdocs regenerates them at build time and CI fails when they are
stale, so contributors have to remember to run a docs build before pushing.

This hook does the same regeneration against the staged tree, without starting
mkdocs. It reuses docs/mkdocs/hooks/generate_examples.py, which is also what
mkdocs.yml loads.

Behaviour:

* only docs/user_guide/examples/ or docs/.nav.yml staged: the edit is rejected,
  because those files are machine-written and any manual change is lost on the
  next generation;
* something under examples/ staged: the pages are regenerated and staged, so the
  commit carries the matching docs;
* nothing relevant staged: exit quietly.

Set SKIP=sync-example-docs to disable it, for example when rebasing a branch
that is already green.
"""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

EXAMPLES_PREFIX = "examples/"
GENERATED_PREFIXES = ("docs/user_guide/examples/", "docs/.nav.yml")
GENERATOR_HOOK_DIR = "docs/mkdocs/hooks"


def git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=cwd,
    )
    return result.stdout


def repo_root() -> Path:
    """Return the working tree root."""
    return Path(git("rev-parse", "--show-toplevel").strip())


def staged_files(root: Path) -> list[str]:
    """Return the paths staged for commit, relative to the repo root."""
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", cwd=root)
    return [line for line in out.splitlines() if line]


def diverging_generated_files(root: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (modified, new, deleted) generated files on disk vs the index."""
    pathspec = list(GENERATED_PREFIXES)
    modified = [line for line in git("diff", "--name-only", "--", *pathspec, cwd=root).splitlines() if line]
    new = [
        line
        for line in git("ls-files", "--others", "--exclude-standard", "--", *pathspec, cwd=root).splitlines()
        if line
    ]
    deleted = [line for line in git("ls-files", "--deleted", "--", *pathspec, cwd=root).splitlines() if line]
    return modified, new, deleted


@contextmanager
def generator_importable(root: Path):
    """Make the mkdocs hook importable, then put sys.path back."""
    hook_dir = str(root / GENERATOR_HOOK_DIR)
    sys.path.insert(0, hook_dir)
    try:
        yield
    finally:
        sys.path.pop(0)


def regenerate(root: Path) -> None:
    """Run the same generation mkdocs runs at build time.

    The generator imports ``regex`` and ``yaml``. If either is missing, the
    hook's ``additional_dependencies`` are incomplete — that is a bug in this
    hook's setup, not in the contributor's commit, so say so plainly instead of
    letting a bare ModuleNotFoundError surface.
    """
    try:
        with generator_importable(root):
            import generate_examples

            generate_examples.on_startup("build", False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"the mkdocs generator needs '{exc.name}', which this hook does not "
            f"install. Add it to additional_dependencies under the "
            f"'sync-example-docs' entry in .pre-commit-config.yaml."
        ) from exc


def main() -> int:
    root = repo_root()
    staged = staged_files(root)

    examples_staged = [f for f in staged if f.startswith(EXAMPLES_PREFIX)]
    generated_staged = [f for f in staged if f.startswith(GENERATED_PREFIXES)]

    if generated_staged and not examples_staged:
        print(
            "error: these files are generated, edit the sources in examples/ instead:",
            file=sys.stderr,
        )
        for path in generated_staged:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nThe next generation overwrites anything typed here. Change the "
            "README under examples/ and commit it; this hook updates the pages.",
            file=sys.stderr,
        )
        return 1

    if not examples_staged:
        return 0

    try:
        regenerate(root)
    except Exception as exc:  # noqa: BLE001 - surface any generator failure to the user
        print(
            f"error: generating the example pages failed: {exc}",
            file=sys.stderr,
        )
        return 1

    modified, new, deleted = diverging_generated_files(root)
    if not modified and not new and not deleted:
        return 0

    if modified or new:
        subprocess.run(
            ["git", "add", "--", *GENERATED_PREFIXES],
            cwd=root,
            check=True,
        )
    if deleted:
        subprocess.run(["git", "rm", "-f", "--", *deleted], cwd=root, check=True)

    print("Synced the example pages with examples/:")
    for path in modified:
        print(f"  modified: {path}")
    for path in new:
        print(f"  new:      {path}")
    for path in deleted:
        print(f"  removed:  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
