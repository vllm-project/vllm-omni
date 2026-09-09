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
    ("branch", "labels", "nightly", "expected"),
    [
        ("main", (), False, ("merge",)),
        ("main", (), True, ("nightly",)),
        ("feature", ("ready",), False, ("ready",)),
        ("feature", ("merge-test",), False, ("merge",)),
        ("feature", ("ready", "merge-test"), False, ("ready", "merge")),
        ("feature", ("nightly-test",), False, ("nightly",)),
        ("feature", ("amd-test",), False, ("ready",)),
        ("feature", ("not-ready", "merge-test-extra"), False, ("ready",)),
    ],
)
def test_label_suite_selection(branch, labels, nightly, expected):
    assert (
        SELECTOR.select_amd_test_suites(
            branch=branch,
            labels=labels,
            nightly=nightly,
        )
        == expected
    )


def test_debug_override_takes_precedence() -> None:
    assert SELECTOR.select_amd_test_suites(
        branch="main",
        labels=("ready",),
        debug_test_yaml=" NIGHTLY, merge, ready,",
    ) == ("nightly", "merge", "ready")


@pytest.mark.parametrize("value", ["ready,ready", "weekly", ", ,"])
def test_invalid_debug_override(value: str) -> None:
    with pytest.raises(ValueError):
        SELECTOR.select_amd_test_suites(
            branch="feature",
            labels=(),
            debug_test_yaml=value,
        )
