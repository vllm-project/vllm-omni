#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select AMD L2/L3/L4 suites after Buildkite has created an AMD build.

This repository-side selector does not decide which GitHub label events start
the external ``vllm-omni-amd-ci`` pipeline. Its Buildkite PR build condition
must admit the relevant label event before this code can inspect the labels.
"""

from __future__ import annotations

import argparse
import sys
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

    seen: set[str] = set()
    for suite in suites:
        if suite not in SUITE_SPECS:
            raise ValueError(f"DEBUG_TEST_YAML entries must be 'merge', 'nightly', or 'ready', got {suite!r}")
        if suite in seen:
            raise ValueError(f"duplicate DEBUG_TEST_YAML suite {suite!r}")
        seen.add(suite)
    return suites


def select_amd_test_suites(
    *,
    branch: str,
    labels: Iterable[str],
    debug_test_yaml: str = "",
    nightly: bool = False,
) -> tuple[str, ...]:
    """Return ordered suite names while preserving legacy unlabeled PRs."""

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
    # AMD historically uploaded L2 for every PR build. Keep that behavior for
    # triggers such as amd-test while adding explicit L2/L3/L4 selection.
    return selected or ("ready",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True)
    parser.add_argument("--labels", default="")
    parser.add_argument("--debug-test-yaml", default="")
    parser.add_argument("--nightly", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    labels = tuple(args.labels.splitlines())
    try:
        suites = select_amd_test_suites(
            branch=args.branch,
            labels=labels,
            debug_test_yaml=args.debug_test_yaml,
            nightly=bool(args.nightly),
        )
    except ValueError as exc:
        parser.error(str(exc))

    label_set = {label.strip() for label in labels if label.strip()}
    if (
        args.branch != "main"
        and not args.debug_test_yaml.strip()
        and not ({"ready", "merge-test", "nightly-test"} & label_set)
    ):
        print(
            "No AMD L2/L3/L4 tier label found; preserving the legacy ready-suite fallback.",
            file=sys.stderr,
        )

    for suite in suites:
        print(SUITE_SPECS[suite])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
