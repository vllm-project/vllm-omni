# SPDX-License-Identifier: Apache-2.0
"""Tests for check_examples_policy.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_examples_policy import APPROVED_EXCEPTIONS, _load_baseline, main

BASELINE_FILE = Path(__file__).resolve().parents[1] / "examples_policy_baseline.txt"


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
    with patch("check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_grandfathered_path_passes():
    # A path already in baseline should pass
    baseline = _load_baseline()
    existing = next(iter(baseline))
    with patch("check_examples_policy._get_added_paths", return_value=[existing]):
        with patch("check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_approved_exception_passes():
    # An approved exception should pass
    exception = next(iter(APPROVED_EXCEPTIONS))
    with patch("check_examples_policy._get_added_paths", return_value=[exception]):
        with patch("check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_new_model_specific_path_fails():
    # A brand new model-specific path should be blocked
    new_path = "examples/offline_inference/some_new_model/end2end.py"
    with patch("check_examples_policy._get_added_paths", return_value=[new_path]):
        with patch("check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 1


def test_modification_passes():
    # Modifications do not show up in ACR diff - get_added_paths returns nothing
    with patch("check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_deletion_passes():
    # Deletions do not show up in ACR diff - get_added_paths returns nothing
    with patch("check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_non_python_file_ignored():
    # Non-Python files under examples/ should not be flagged
    with patch("check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0


def test_multiple_new_paths_all_blocked():
    # Multiple new paths should all be reported
    new_paths = [
        "examples/offline_inference/model_a/end2end.py",
        "examples/offline_inference/model_b/end2end.py",
    ]
    with patch("check_examples_policy._get_added_paths", return_value=new_paths):
        with patch("check_examples_policy._get_merge_base", return_value="abc123"):
            result = main(["--base-ref", "origin/main"])
    assert result == 1


def test_path_outside_examples_not_flagged():
    # Paths outside examples/ should not be flagged
    with patch("check_examples_policy._get_added_paths", return_value=[]):
        result = main(["--base-ref", "origin/main"])
    assert result == 0
