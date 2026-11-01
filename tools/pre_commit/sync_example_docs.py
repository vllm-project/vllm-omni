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

* docs/user_guide/examples/ staged without examples/: the edit is rejected,
  because those pages are machine-written and any manual change is lost on the
  next generation;
* docs/.nav.yml staged without examples/: rejected only when the edit reaches
  the User Guide -> Examples block, the one block the generator owns. A
  navigation entry for a hand-written page is kept by the generator and is
  allowed;
* something under examples/ staged: the pages are regenerated and staged, so the
  commit carries the matching docs;
* nothing relevant staged: exit quietly.

Set SKIP=sync-example-docs to disable it, for example when rebasing a branch
that is already green.
"""

import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

EXAMPLES_PREFIX = "examples/"
GENERATED_PAGE_PREFIX = "docs/user_guide/examples/"
NAV_FILE = "docs/.nav.yml"
GENERATED_PREFIXES = (GENERATED_PAGE_PREFIX, NAV_FILE)
GENERATOR_HOOK_DIR = "docs/mkdocs/hooks"
NAV_ROOT_SECTION = "User Guide"
NAV_GENERATED_SECTION = "Examples"


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


def nav_examples_section(text: str):
    """Return the ``User Guide -> Examples`` block of a nav document, or None.

    That block is the only part of docs/.nav.yml the generator rewrites
    (``generate_examples.update_nav_file``); every other section is preserved.
    """
    import yaml

    data = yaml.safe_load(text) or {}
    for item in data.get("nav") or []:
        if isinstance(item, dict) and NAV_ROOT_SECTION in item:
            for sub in item[NAV_ROOT_SECTION] or []:
                if isinstance(sub, dict) and NAV_GENERATED_SECTION in sub:
                    return sub[NAV_GENERATED_SECTION]
    return None


def nav_generated_block_changed(root: Path) -> bool:
    """True when the staged docs/.nav.yml edits the generated Examples block.

    Comparing the block rather than the whole file is what lets a contributor
    add a hand-written page and its nav entry: the generator preserves the
    sections it does not own, so an edit elsewhere is legitimate.
    """
    try:
        staged_text = git("show", f":{NAV_FILE}", cwd=root)
        head_text = git("show", f"HEAD:{NAV_FILE}", cwd=root)
    except subprocess.CalledProcessError:
        # New file, or no HEAD yet: nothing generated to compare against.
        return False
    return nav_examples_section(staged_text) != nav_examples_section(head_text)


def staged_files(root: Path) -> list[str]:
    """Return the paths staged for commit, relative to the repo root.

    Deletions are kept: a commit that only removes an example still has to
    regenerate the pages, and deleting a generated page is a violation like
    editing one. Both were missed while the filter dropped D.
    """
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMRD", cwd=root)
    return [line for line in out.splitlines() if line]


def index_paths(root: Path) -> list[str]:
    """Paths the generator reads that exist in the index rather than on disk."""
    out = git("ls-files", "-z", "--", EXAMPLES_PREFIX, NAV_FILE, cwd=root)
    return [path for path in out.split("\0") if path]


def export_index(root: Path, destination: Path) -> None:
    """Write the staged snapshot of the generator's inputs into ``destination``.

    Only the index is exported. An untracked README left in the working tree is
    not part of the commit being made, so the generator must not see it, and a
    staged deletion is already gone from the index.
    """
    destination.mkdir(parents=True, exist_ok=True)
    paths = index_paths(root)
    if not paths:
        return
    subprocess.run(
        ["git", "checkout-index", "-z", "--stdin", f"--prefix={destination.as_posix().rstrip('/')}/"],
        cwd=root,
        input="\0".join(paths) + "\0",
        text=True,
        check=True,
    )


def apply_snapshot(root: Path, snapshot: Path) -> tuple[list[str], list[str]]:
    """Write the snapshot's pages back and report (to_stage, to_remove).

    Every page the snapshot produced is returned for staging, even when the
    bytes already match on disk: the page can exist untracked (someone ran a
    docs build before staging the example) and comparing the working tree alone
    would leave the source committed without it. Redundant writes are still
    skipped, which is a separate concern from what reaches the index.

    A page the snapshot did not produce is removed only if git tracks it; a
    local untracked page is left alone.
    """
    produced_root = snapshot / GENERATED_PAGE_PREFIX
    target_root = root / GENERATED_PAGE_PREFIX
    to_stage: list[str] = []
    produced: set[str] = set()
    if produced_root.is_dir():
        for page in sorted(produced_root.rglob("*.md")):
            relative = page.relative_to(produced_root).as_posix()
            produced.add(relative)
            destination = target_root / relative
            text = page.read_text(encoding="utf-8")
            if not destination.exists() or destination.read_text(encoding="utf-8") != text:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
            to_stage.append(f"{GENERATED_PAGE_PREFIX}{relative}")

    nav_source = snapshot / NAV_FILE
    if nav_source.exists():
        nav_text = nav_source.read_text(encoding="utf-8")
        nav_target = root / NAV_FILE
        if not nav_target.exists() or nav_target.read_text(encoding="utf-8") != nav_text:
            nav_target.write_text(nav_text, encoding="utf-8")
        to_stage.append(NAV_FILE)

    tracked = {path for path in git("ls-files", "-z", "--", GENERATED_PAGE_PREFIX, cwd=root).split("\0") if path}
    to_remove: list[str] = []
    if target_root.is_dir():
        for page in sorted(target_root.rglob("*.md")):
            relative = page.relative_to(target_root).as_posix()
            path = f"{GENERATED_PAGE_PREFIX}{relative}"
            if relative not in produced and path in tracked:
                to_remove.append(path)
    return to_stage, to_remove


@contextmanager
def generator_importable(root: Path):
    """Make the mkdocs hook importable, then put sys.path back."""
    hook_dir = str(root / GENERATOR_HOOK_DIR)
    sys.path.insert(0, hook_dir)
    try:
        yield
    finally:
        sys.path.pop(0)


def regenerate(root: Path) -> tuple[list[str], list[str]]:
    """Regenerate the pages from the index and report what moved.

    The generator reads directories, so running it against the working tree
    would also pick up an untracked example README and publish it. It runs over
    an exported index snapshot instead, and only that snapshot's output is
    written back.

    The generator imports ``regex`` and ``yaml``. If either is missing, the
    hook's ``additional_dependencies`` are incomplete — that is a bug in this
    hook's setup, not in the contributor's commit, so say so plainly instead of
    letting a bare ModuleNotFoundError surface.
    """
    with tempfile.TemporaryDirectory(prefix="sync-example-docs-") as temporary:
        snapshot = Path(temporary)
        export_index(root, snapshot)
        try:
            with generator_importable(root):
                import generate_examples

                generate_examples.ROOT_DIR = snapshot
                generate_examples.EXAMPLE_DIR = snapshot / EXAMPLES_PREFIX
                generate_examples.EXAMPLE_DOC_DIR = snapshot / GENERATED_PAGE_PREFIX
                generate_examples.NAV_FILE = snapshot / NAV_FILE
                generate_examples.on_startup("build", False)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"the mkdocs generator needs '{exc.name}', which this hook does not "
                f"install. Add it to additional_dependencies under the "
                f"'sync-example-docs' entry in .pre-commit-config.yaml."
            ) from exc
        return apply_snapshot(root, snapshot)


def main() -> int:
    root = repo_root()
    staged = staged_files(root)

    examples_staged = [f for f in staged if f.startswith(EXAMPLES_PREFIX)]
    pages_staged = [f for f in staged if f.startswith(GENERATED_PAGE_PREFIX)]
    nav_staged = NAV_FILE in staged

    # The generated pages are always off-limits without an examples/ change.
    # docs/.nav.yml is only off-limits when the staged edit touches the block
    # the generator owns; a navigation entry for a hand-written page is fine.
    blocked = list(pages_staged)
    if nav_staged and not examples_staged and nav_generated_block_changed(root):
        blocked.append(NAV_FILE)

    if blocked and not examples_staged:
        print(
            "error: these files are generated, edit the sources in examples/ instead:",
            file=sys.stderr,
        )
        for path in blocked:
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
        to_stage, to_remove = regenerate(root)
    except Exception as exc:  # noqa: BLE001 - surface any generator failure to the user
        print(
            f"error: generating the example pages failed: {exc}",
            file=sys.stderr,
        )
        return 1

    if not to_stage and not to_remove:
        return 0

    if to_stage:
        subprocess.run(["git", "add", "--", *to_stage], cwd=root, check=True)
    if to_remove:
        subprocess.run(["git", "rm", "-f", "--", *to_remove], cwd=root, check=True)

    print("Synced the example pages with examples/:")
    for path in to_stage:
        print(f"  staged:  {path}")
    for path in to_remove:
        print(f"  removed: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
