# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Integration tests for the sync-example-docs pre-commit hook.

The unit tests in test_sync_example_docs.py call sync.main() directly, which
cannot see the boundary pre-commit puts in front of it: deleted paths never
reach a hook unless the entry declares always_run, so a commit that only
removes an example used to invoke nothing at all. These cases drive the real
pre-commit runner on a throwaway repo, with a copy of the real generator, so a
deletion-only commit has to do the regeneration for real.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

pytest.importorskip("pre_commit", reason="the integration cases drive the pre-commit runner")

ROOT_DIR = Path(__file__).parents[2]
HOOK_PATH = ROOT_DIR / "tools/pre_commit/sync_example_docs.py"
GENERATOR_PATH = ROOT_DIR / "docs/mkdocs/hooks/generate_examples.py"
CONFIG_PATH = ROOT_DIR / ".pre-commit-config.yaml"

EXAMPLE_DIR = "examples/offline_inference/text_to_image"
PAGE = "docs/user_guide/examples/offline_inference/text_to_image.md"
GENERATED_PAGE = "docs/user_guide/examples/offline_inference/image_to_image.md"

BASE_NAV = """\
nav:
  - Home: index.md
  - User Guide:
    - Serving:
      - Serving API: user_guide/serving/api.md
    - Examples:
      - examples/README.md
  - Developer Guide:
    - Contributing: contributing/index.md
"""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _hook_config() -> str:
    """The repository's own hook entry, wired to run here without an env build.

    Everything else is kept as configured — notably always_run — so a test repo
    behaves like the real one and dropping a key upstream surfaces here too.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for repository in config["repos"]:
        hooks = [hook for hook in repository.get("hooks", []) if hook.get("id") == "sync-example-docs"]
        if hooks:
            hook = dict(hooks[0])
            hook["entry"] = f'"{sys.executable}" "{HOOK_PATH.as_posix()}"'
            hook["language"] = "system"
            hook.pop("additional_dependencies", None)
            return yaml.safe_dump({"repos": [{"repo": "local", "hooks": [hook]}]}, sort_keys=False)
    raise AssertionError("sync-example-docs is missing from .pre-commit-config.yaml")


def _repo(tmp_path: Path) -> Path:
    """A throwaway repo wired to the real hook and the real generator."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs/mkdocs/hooks").mkdir(parents=True)
    shutil.copy(GENERATOR_PATH, repo / "docs/mkdocs/hooks/generate_examples.py")
    (repo / "examples").mkdir()
    shutil.copy(
        ROOT_DIR / "examples/model_display_names.yml",
        repo / "examples/model_display_names.yml",
    )
    (repo / "docs/.nav.yml").write_text(BASE_NAV, encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text(_hook_config(), encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def _pre_commit(repo: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PRE_COMMIT_HOME=str(tmp_path / "pre-commit-home"))
    return subprocess.run(
        [sys.executable, "-m", "pre_commit", "run"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _nav(repo: Path) -> str:
    return (repo / "docs/.nav.yml").read_text(encoding="utf-8")


def test_the_repository_config_runs_the_hook_for_deletion_only_commits():
    """Without always_run pre-commit skips the hook and the removal ships stale."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    entries = [
        hook for repository in config["repos"] for hook in repository["hooks"] if hook["id"] == "sync-example-docs"
    ]

    assert entries, "sync-example-docs is missing from .pre-commit-config.yaml"
    assert entries[0].get("always_run") is True


def test_deletion_only_commit_regenerates_the_pages(tmp_path):
    repo = _repo(tmp_path)
    readme = repo / EXAMPLE_DIR / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("# Text to image", encoding="utf-8")
    _git(repo, "add", "-A")

    assert _pre_commit(repo, tmp_path).returncode == 0
    _git(repo, "commit", "-qm", "add the example")
    assert (repo / PAGE).exists()
    assert "text_to_image" in _nav(repo)

    _git(repo, "rm", "-q", f"{EXAMPLE_DIR}/README.md")
    done = _pre_commit(repo, tmp_path)

    assert done.returncode == 0, done.stderr
    assert not (repo / PAGE).exists()
    assert "text_to_image" not in _nav(repo)


def test_a_script_left_behind_does_not_keep_the_page(tmp_path):
    """The generator only discovers a subdirectory through a .md file."""
    repo = _repo(tmp_path)
    readme = repo / EXAMPLE_DIR / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("# Text to image", encoding="utf-8")
    (repo / EXAMPLE_DIR / "run.py").write_text("print('hi')", encoding="utf-8")
    _git(repo, "add", "-A")

    assert _pre_commit(repo, tmp_path).returncode == 0
    _git(repo, "commit", "-qm", "example with a script")
    assert (repo / PAGE).exists()

    _git(repo, "rm", "-q", f"{EXAMPLE_DIR}/README.md")
    done = _pre_commit(repo, tmp_path)

    assert done.returncode == 0, done.stderr
    assert not (repo / PAGE).exists()
    assert "text_to_image" not in _nav(repo)


def test_deleting_a_generated_page_alone_is_rejected_through_the_hook(tmp_path):
    repo = _repo(tmp_path)
    page = repo / GENERATED_PAGE
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("# generated", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "the page")

    _git(repo, "rm", "-q", GENERATED_PAGE)
    done = _pre_commit(repo, tmp_path)

    assert done.returncode != 0
    assert "generated" in (done.stdout + done.stderr).lower()


def test_an_untracked_example_never_reaches_the_index(tmp_path):
    """Regeneration reads the index, so a local draft cannot ride along."""
    repo = _repo(tmp_path)
    readme = repo / EXAMPLE_DIR / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("# Text to image", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _pre_commit(repo, tmp_path).returncode == 0
    _git(repo, "commit", "-qm", "add the example")
    assert (repo / PAGE).exists()

    draft = repo / "examples/local_notes/README.md"
    draft.parent.mkdir(parents=True)
    draft.write_text("# Local draft, must stay out of the commit", encoding="utf-8")

    readme.write_text("# Text to image, reworded", encoding="utf-8")
    _git(repo, "add", f"{EXAMPLE_DIR}/README.md")

    done = _pre_commit(repo, tmp_path)

    assert done.returncode == 0, done.stderr
    staged = _git(repo, "diff", "--cached", "--name-only").stdout.split()
    assert PAGE in staged
    assert not any("local_notes" in path for path in staged)
    assert "local_notes" not in _nav(repo)
    assert draft.read_text(encoding="utf-8") == "# Local draft, must stay out of the commit"


def test_a_page_that_already_matches_on_disk_is_still_staged(tmp_path):
    """A page left by a docs build must reach the index with its source."""
    repo = _repo(tmp_path)
    readme = repo / EXAMPLE_DIR / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("# Text to image", encoding="utf-8")
    _git(repo, "add", "-A")
    assert _pre_commit(repo, tmp_path).returncode == 0
    _git(repo, "commit", "-qm", "the first example")

    other = "examples/offline_inference/image_to_image"
    other_page = "docs/user_guide/examples/offline_inference/image_to_image.md"
    other_readme = repo / other / "README.md"
    other_readme.parent.mkdir(parents=True)
    other_readme.write_text("# Image to image", encoding="utf-8")
    _git(repo, "add", f"{other}/README.md")
    assert _pre_commit(repo, tmp_path).returncode == 0
    assert (repo / other_page).exists()
    _git(repo, "commit", "-qm", "the second source only", "--no-verify", "--", f"{other}/README.md")

    # Reset the index to HEAD: the page is now untracked while sitting on disk
    # with exactly the bytes the generator produces — what a docs build leaves.
    # The navigation goes back to its committed bytes too, or pre-commit would
    # stash an unstaged nav edit and fail to restore it after the hook rewrites
    # the file.
    _git(repo, "reset", "-q")
    _git(repo, "checkout", "--", "docs/.nav.yml")
    assert other_page not in _git(repo, "diff", "--cached", "--name-only").stdout.split()
    assert (repo / other_page).exists()

    readme.write_text("# Text to image, reworded", encoding="utf-8")
    _git(repo, "add", f"{EXAMPLE_DIR}/README.md")

    done = _pre_commit(repo, tmp_path)

    assert done.returncode == 0, done.stderr
    staged_blob = _git(repo, "rev-parse", f":{other_page}", check=False)
    assert staged_blob.returncode == 0, "the generated page is missing from the index"
    assert staged_blob.stdout.strip() == _git(repo, "hash-object", other_page).stdout.strip()
