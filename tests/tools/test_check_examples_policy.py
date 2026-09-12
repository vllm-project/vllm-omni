# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for check_examples_policy.py."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.pre_commit.check_examples_policy import (
    APPROVED_EXCEPTIONS,
    _get_added_paths,
    _get_merge_base,
    _load_baseline,
    main,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILE = REPO_ROOT / "tools" / "pre_commit" / "examples_policy_baseline.txt"


def test_baseline_file_exists():
    assert BASELINE_FILE.is_file(), "examples_policy_baseline.txt must exist"


def test_baseline_not_empty():
    baseline = _load_baseline()
    assert len(baseline) > 0, "baseline must not be empty"


def test_baseline_only_python_paths():
    baseline = _load_baseline()
    for path in baseline:
        assert path.endswith(".py"), f"non-Python path in baseline: {path}"
        assert path.startswith("examples/"), f"path outside examples/: {path}"


def test_baseline_no_duplicates():
    lines = BASELINE_FILE.read_text().splitlines()
    paths = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert len(paths) == len(set(paths)), "baseline contains duplicate entries"


def test_approved_exceptions_are_valid():
    for path in APPROVED_EXCEPTIONS:
        assert path.endswith(".py"), f"non-Python path in exceptions: {path}"
        assert path.startswith("examples/"), f"path outside examples/: {path}"


def test_no_new_paths_passes():
    # No new paths added - should pass
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_grandfathered_path_passes():
    # A path already in baseline should pass
    baseline = _load_baseline()
    existing = next(iter(baseline))
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[existing]):
        with patch("tools.pre_commit.check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_approved_exception_passes():
    # An approved exception should pass
    exception = next(iter(APPROVED_EXCEPTIONS))
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[exception]):
        with patch("tools.pre_commit.check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_new_model_specific_path_fails():
    # A brand new model-specific path should be blocked
    new_path = "examples/offline_inference/some_new_model/end2end.py"
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[new_path]):
        with patch("tools.pre_commit.check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 1


def test_modification_passes():
    # Modifications do not show up in ACR diff - get_added_paths returns nothing
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_deletion_passes():
    # Deletions do not show up in ACR diff - get_added_paths returns nothing
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_non_python_file_ignored():
    # Non-Python files under examples/ should not be flagged
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_multiple_new_paths_all_blocked():
    # Multiple new paths should all be reported
    new_paths = [
        "examples/offline_inference/model_a/end2end.py",
        "examples/offline_inference/model_b/end2end.py",
    ]
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=new_paths):
        with patch("tools.pre_commit.check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 1


def test_path_outside_examples_not_flagged():
    # Paths outside examples/ should not be flagged
    with patch("tools.pre_commit.check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_policy_rejects_committed_and_staged_new_example_paths(tmp_path):
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    # Build a repository with one commit to use as the merge base
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "copy_source.py").write_text("copy source\n")
    (examples / "rename_source.py").write_text("rename source\n")
    (examples / "modify.py").write_text("before\n")
    (examples / "delete.py").write_text("delete\n")
    git("add", "examples")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")

    (examples / "added.py").write_text("added\n")
    (examples / "copied.py").write_text("copy source\n")
    git("mv", "examples/rename_source.py", "examples/renamed.py")
    (examples / "modify.py").write_text("after\n")
    (examples / "delete.py").unlink()
    git("add", "examples")
    git("commit", "-qm", "changes")

    # Test staged but not committed case
    (examples / "staged.py").write_text("staged\n")
    git("add", "examples/staged.py")

    with patch("tools.pre_commit.check_examples_policy.REPO_ROOT", tmp_path):
        merge_base = _get_merge_base(base)
        added_paths = _get_added_paths(merge_base)
        policy_result = main(["--base-ref", base])

    # The committed and staged new paths are blocked
    assert merge_base == base
    assert policy_result == 1
    assert set(added_paths) == {
        "examples/added.py",
        "examples/copied.py",
        "examples/renamed.py",
        "examples/staged.py",
    }

    # A repository without a merge base must fail rather than skip the check.
    repo_without_merge_base = tmp_path / "repo_without_merge_base"
    repo_without_merge_base.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo_without_merge_base, check=True)
    with patch(
        "tools.pre_commit.check_examples_policy.REPO_ROOT",
        repo_without_merge_base,
    ):
        with pytest.raises(SystemExit):
            _get_merge_base("HEAD")
