# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SELECTOR_PATH = Path(".buildkite/amd/scripts/select_test_suites.py")
WEEKLY_PIPELINE = Path(".buildkite/amd/test-amd-weekly.yml")
SPEC = importlib.util.spec_from_file_location("select_test_suites", SELECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


@pytest.mark.parametrize(
    ("branch", "labels", "expected"),
    [
        ("main", (), ("merge",)),
        ("release/v0.30.0", (), ("merge",)),
        ("release/v0.30.0", ("ready",), ("merge",)),
        ("0.30.0", (), ("ready",)),
        ("v0.30.0", (), ("ready",)),
        ("feature", ("ready",), ("ready",)),
        ("feature", ("merge-test",), ("merge",)),
        ("feature", ("merge-test", "ready"), ("ready", "merge")),
        ("feature", ("nightly-test",), ("nightly",)),
        ("feature", ("weekly-test",), ("weekly",)),
        ("feature", ("ready", "merge-test", "nightly-test"), ("ready", "merge", "nightly")),
        ("feature", ("amd-test",), ("ready",)),
        ("feature", ("amd-test", "merge-test"), ("merge",)),
        ("feature", ("amd-test", "ready", "merge-test"), ("ready", "merge")),
        ("feature", ("amd-test", "nightly-test"), ("nightly",)),
        ("feature", ("not-ready", "merge-test-extra"), ("ready",)),
    ],
)
def test_label_suite_selection(branch, labels, expected):
    assert SELECTOR.select_amd_test_suites(branch=branch, labels=labels) == expected


def test_debug_override_takes_precedence_and_normalizes_input():
    assert SELECTOR.select_amd_test_suites(
        branch="main",
        labels=("ready",),
        debug_test_yaml=" NIGHTLY, merge, ready,",
        nightly=False,
    ) == ("nightly", "merge", "ready")


def test_debug_weekly_selects_four_unconditional_jobs():
    assert SELECTOR.select_amd_test_suites(
        branch="feature",
        labels=(),
        debug_test_yaml="weekly",
    ) == ("weekly",)

    weekly = yaml.safe_load(WEEKLY_PIPELINE.read_text(encoding="utf-8"))
    group = weekly["steps"][0]
    assert "if" not in group
    assert len(group["steps"]) == 4
    assert all("if" not in step for step in group["steps"])


def test_scheduled_main_selects_only_nightly():
    assert SELECTOR.select_amd_test_suites(
        branch="main",
        labels=(),
        nightly=True,
    ) == ("nightly",)


def test_scheduled_main_selects_only_weekly():
    assert SELECTOR.select_amd_test_suites(
        branch="main",
        labels=(),
        weekly=True,
    ) == ("weekly",)


def test_scheduled_main_rejects_nightly_and_weekly_together():
    with pytest.raises(ValueError, match="nightly or weekly"):
        SELECTOR.select_amd_test_suites(
            branch="main",
            labels=(),
            nightly=True,
            weekly=True,
        )


def test_version_branch_pull_request_follows_labels():
    assert SELECTOR.select_amd_test_suites(
        branch="release/v0.30.0",
        labels=("ready",),
        pull_request="123",
    ) == ("ready",)
    assert SELECTOR.select_amd_test_suites(
        branch="release/v0.30.0",
        labels=("merge-test",),
        pull_request="123",
    ) == ("merge",)
    assert SELECTOR.select_amd_test_suites(
        branch="release/v0.30.0",
        labels=(),
        pull_request="123",
    ) == ("ready",)


def test_main_pull_request_still_selects_merge():
    assert SELECTOR.select_amd_test_suites(
        branch="main",
        labels=("ready",),
        pull_request="123",
    ) == ("merge",)


def test_scheduled_version_branch_selects_only_nightly():
    assert SELECTOR.select_amd_test_suites(
        branch="release/v0.30.0",
        labels=("ready",),
        nightly=True,
    ) == ("nightly",)


def test_scheduled_version_branch_selects_only_weekly():
    assert SELECTOR.select_amd_test_suites(
        branch="release/v0.30.0",
        labels=("ready",),
        weekly=True,
    ) == ("weekly",)


def test_empty_debug_override_uses_normal_selection():
    assert SELECTOR.select_amd_test_suites(
        branch="feature",
        labels=("merge-test",),
        debug_test_yaml="",
    ) == ("merge",)


@pytest.mark.parametrize("value", ["ready,ready", "nightly,nightly", ", ,", " \t"])
def test_invalid_debug_override(value):
    with pytest.raises(ValueError):
        SELECTOR.select_amd_test_suites(
            branch="feature",
            labels=(),
            debug_test_yaml=value,
        )
