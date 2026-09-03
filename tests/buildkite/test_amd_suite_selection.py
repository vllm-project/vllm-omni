# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SELECTOR_PATH = Path(".buildkite/amd/scripts/select_test_suites.py")
SPEC = importlib.util.spec_from_file_location("select_test_suites", SELECTOR_PATH)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


@pytest.mark.parametrize(
    ("branch", "labels", "expected"),
    [
        ("main", (), ("merge",)),
        ("feature", ("ready",), ("ready",)),
        ("feature", ("merge-test",), ("merge",)),
        ("feature", ("merge-test", "ready"), ("ready", "merge")),
        ("feature", ("amd-test",), ("ready",)),
        ("feature", ("nightly-test",), ("ready",)),
        ("feature", ("not-ready", "merge-test-extra"), ("ready",)),
    ],
)
def test_label_suite_selection(branch, labels, expected):
    assert SELECTOR.select_amd_test_suites(branch=branch, labels=labels) == expected


def test_debug_override_takes_precedence_and_normalizes_input():
    assert SELECTOR.select_amd_test_suites(
        branch="main",
        labels=("ready",),
        debug_test_yaml=" MERGE, ready,",
    ) == ("merge", "ready")


def test_empty_debug_override_uses_normal_selection():
    assert SELECTOR.select_amd_test_suites(
        branch="feature",
        labels=("merge-test",),
        debug_test_yaml="",
    ) == ("merge",)


@pytest.mark.parametrize("value", ["ready,ready", "nightly", ", ,"])
def test_invalid_debug_override(value):
    with pytest.raises(ValueError):
        SELECTOR.select_amd_test_suites(
            branch="feature",
            labels=(),
            debug_test_yaml=value,
        )
