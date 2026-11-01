# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for tools/pre_commit/sync_example_docs.py.

The hook used to reject every staged docs/.nav.yml whenever no examples/ file
was staged, which blocked a hand-written page and its navigation entry. The
generator only rewrites the ``User Guide -> Examples`` block, so only that
block has to stay off limits.
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT_DIR = Path(__file__).parents[2]
HOOK_PATH = ROOT_DIR / "tools/pre_commit/sync_example_docs.py"
SPEC = importlib.util.spec_from_file_location("sync_example_docs", HOOK_PATH)
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)

NAV = "docs/.nav.yml"

BASE_NAV = """\
nav:
  - Home: index.md
  - User Guide:
    - Serving:
      - Serving API: user_guide/serving/api.md
    - Examples:
      - examples/README.md
      - Offline Inference:
        - Image-To-Image: user_guide/examples/offline_inference/image_to_image.md
  - Developer Guide:
    - Contributing: contributing/index.md
"""


def _repo(tmp_path):
    """A throwaway git repo with the nav file committed."""

    def run(*args):
        return subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )

    run("init", "-q")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "test")
    (tmp_path / "docs").mkdir()
    (tmp_path / NAV).write_text(BASE_NAV, encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "base")
    return run


def test_nav_entry_for_a_hand_written_page_can_be_committed(tmp_path, monkeypatch):
    """Adding a serving page and its nav entry must not be blocked."""
    run = _repo(tmp_path)
    edited = BASE_NAV.replace(
        "      - Serving API: user_guide/serving/api.md\n",
        "      - Serving API: user_guide/serving/api.md\n      - Batch API: user_guide/serving/batch.md\n",
    )
    (tmp_path / NAV).write_text(edited, encoding="utf-8")
    run("add", NAV)

    monkeypatch.chdir(tmp_path)
    assert sync.main() == 0


def test_edit_inside_the_generated_examples_block_is_still_rejected(tmp_path, monkeypatch):
    run = _repo(tmp_path)
    edited = BASE_NAV.replace(
        "      - examples/README.md\n",
        "      - examples/README.md\n      - Hand Written: user_guide/examples/hand.md\n",
    )
    (tmp_path / NAV).write_text(edited, encoding="utf-8")
    run("add", NAV)

    monkeypatch.chdir(tmp_path)
    assert sync.main() == 1


def test_generated_page_staged_without_examples_is_rejected(tmp_path, monkeypatch):
    run = _repo(tmp_path)
    page = tmp_path / "docs/user_guide/examples/offline_inference/image_to_image.md"
    page.parent.mkdir(parents=True)
    page.write_text("# typed by hand\n", encoding="utf-8")
    run("add", str(page.relative_to(tmp_path)))

    monkeypatch.chdir(tmp_path)
    assert sync.main() == 1


def test_nav_examples_section_extraction():
    section = sync.nav_examples_section(BASE_NAV)

    assert section[0] == "examples/README.md"
    assert "Offline Inference" in section[1]
    assert sync.nav_examples_section("nav:\n  - Home: index.md\n") is None


def _repo_with_generator(tmp_path):
    """A throwaway repo that also carries the real mkdocs generator.

    ``generate_examples`` derives its root from the location of its own file,
    so a copy under ``docs/mkdocs/hooks`` works on the temporary repo.
    """
    run = _repo(tmp_path)
    hooks = tmp_path / "docs/mkdocs/hooks"
    hooks.mkdir(parents=True)
    shutil.copy(
        ROOT_DIR / "docs/mkdocs/hooks/generate_examples.py",
        hooks / "generate_examples.py",
    )
    (tmp_path / "examples").mkdir(exist_ok=True)
    shutil.copy(
        ROOT_DIR / "examples/model_display_names.yml",
        tmp_path / "examples/model_display_names.yml",
    )
    return run


def test_removing_an_example_updates_the_generated_navigation(tmp_path, monkeypatch):
    """Deleting an example README has to take its page and nav entry with it."""
    run = _repo_with_generator(tmp_path)
    example = tmp_path / "examples/offline_inference/text_to_image/README.md"
    example.parent.mkdir(parents=True)
    example.write_text("# Text to image", encoding="utf-8")
    run("add", "-A")

    monkeypatch.chdir(tmp_path)
    assert sync.main() == 0
    run("commit", "-q", "-m", "add the example")

    page = tmp_path / "docs/user_guide/examples/offline_inference/text_to_image.md"
    assert page.exists()
    assert "text_to_image" in (tmp_path / NAV).read_text(encoding="utf-8")

    run("rm", "-q", "examples/offline_inference/text_to_image/README.md")

    assert sync.main() == 0
    assert not page.exists()
    assert "text_to_image" not in (tmp_path / NAV).read_text(encoding="utf-8")


def test_removing_a_generated_page_on_its_own_is_rejected(tmp_path, monkeypatch):
    """A deletion-only commit cannot slip past the generated-page guard."""
    run = _repo(tmp_path)
    page = tmp_path / "docs/user_guide/examples/offline_inference/image_to_image.md"
    page.parent.mkdir(parents=True)
    page.write_text("# generated", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "the page")

    run("rm", "-q", str(page.relative_to(tmp_path)))

    monkeypatch.chdir(tmp_path)
    assert sync.main() == 1
