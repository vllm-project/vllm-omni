#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select AMD test suites from labels and scheduled-build settings."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

SUITE_SPECS = {
    "ready": "READY_TESTS:test-amd-ready.yml",
    "merge": "MERGE_TESTS:test-amd-merge.yml",
    "nightly": "NIGHTLY_TESTS:test-amd-nightly.yml",
}


def _parse_debug_suites(value: str) -> tuple[str, ...]:
    suites = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not suites:
        raise ValueError("DEBUG_TEST_YAML did not contain a test suite")
    if len(set(suites)) != len(suites):
        raise ValueError("DEBUG_TEST_YAML contains a duplicate suite")
    unknown = [suite for suite in suites if suite not in SUITE_SPECS]
    if unknown:
        raise ValueError(f"unknown AMD test suite {unknown[0]!r}")
    return suites


def select_amd_test_suites(
    *,
    branch: str,
    labels: Iterable[str],
    debug_test_yaml: str = "",
    nightly: bool = False,
) -> tuple[str, ...]:
    """Return ordered AMD suites while preserving the legacy PR fallback."""
    if debug_test_yaml.strip():
        return _parse_debug_suites(debug_test_yaml)
    if branch == "main":
        return ("nightly",) if nightly else ("merge",)

    label_set = {label.strip() for label in labels if label.strip()}
    selected = tuple(
        suite
        for label, suite in (
            ("ready", "ready"),
            ("merge-test", "merge"),
            ("nightly-test", "nightly"),
        )
        if label in label_set
    )
    return selected or ("ready",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--debug-test-yaml", default="")
    parser.add_argument("--nightly", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    try:
        suites = select_amd_test_suites(
            branch=args.branch,
            labels=args.labels.splitlines(),
            debug_test_yaml=args.debug_test_yaml,
            nightly=bool(args.nightly),
        )
    except ValueError as exc:
        parser.error(str(exc))

    for suite in suites:
        print(SUITE_SPECS[suite])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
