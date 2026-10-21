#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Combine AMD test suites without rerunning identical pipeline steps."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

SUITE_GROUPS = frozenset({"READY_TESTS", "MERGE_TESTS", "NIGHTLY_TESTS", "WEEKLY_TESTS"})


def _parse_suite_spec(suite_spec: str) -> tuple[str, Path]:
    group_name, separator, input_path = suite_spec.partition(":")
    if not separator or not input_path:
        raise ValueError(f"Invalid AMD test suite spec {suite_spec!r}; expected GROUP:path")
    if group_name not in SUITE_GROUPS:
        raise ValueError(f"Unknown AMD test suite group {group_name!r}")
    return group_name, Path(input_path)


def _flatten_steps(
    suite: dict[str, Any],
    input_path: Path,
) -> list[dict[str, Any]]:
    entries = suite.get("steps")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"AMD test suite {input_path} has no steps")

    steps: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"AMD test suite {input_path} contains a non-mapping step")
        if "group" in entry:
            group_steps = entry.get("steps")
            if not isinstance(group_steps, list) or not group_steps:
                raise ValueError(f"AMD test suite {input_path} contains an empty step group")
            if not all(isinstance(step, dict) for step in group_steps):
                raise ValueError(f"AMD test suite {input_path} contains a non-mapping step")
            steps.extend(group_steps)
        else:
            steps.append(entry)
    return steps


def combine_test_suites(suite_specs: Sequence[str]) -> dict[str, Any]:
    """Combine selected suites and remove exact cross-suite duplicates.

    Args:
        suite_specs: Ordered ``GROUP:path`` suite specifications.

    Returns:
        A pipeline fragment grouped by suite.

    Raises:
        ValueError: If a suite specification or suite document is invalid.
    """
    if not suite_specs:
        raise ValueError("At least one AMD test suite spec is required")

    combined: dict[str, Any] = {"env": {}, "steps": []}
    prior_steps: list[dict[str, Any]] = []
    seen_groups: set[str] = set()

    for suite_spec in suite_specs:
        group_name, input_path = _parse_suite_spec(suite_spec)
        if group_name in seen_groups:
            raise ValueError(f"Duplicate AMD test suite group {group_name!r}")
        seen_groups.add(group_name)

        with input_path.open(encoding="utf-8") as test_file:
            suite = yaml.safe_load(test_file)
        if not isinstance(suite, dict):
            raise ValueError(f"AMD test suite {input_path} is not a mapping")

        suite_env = suite.get("env") or {}
        if not isinstance(suite_env, dict):
            raise ValueError(f"AMD test suite {input_path} env is not a mapping")
        for name, value in suite_env.items():
            previous = combined["env"].get(name, value)
            if previous != value:
                raise ValueError(f"Conflicting environment value for {name}: {previous!r} != {value!r}")
            combined["env"][name] = value

        suite_steps = _flatten_steps(suite, input_path)
        retained_steps = [step for step in suite_steps if step not in prior_steps]
        duplicate_steps = [step for step in suite_steps if step in prior_steps]
        if duplicate_steps:
            duplicate_jobs = sum(step.get("parallelism", 1) for step in duplicate_steps)
            print(
                f"Deduplicated {len(duplicate_steps)} identical {group_name} step definitions ({duplicate_jobs} jobs).",
                file=sys.stderr,
            )

        if retained_steps:
            combined["steps"].append({"group": group_name, "steps": retained_steps})
        prior_steps.extend(suite_steps)

    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument("suite_specs", nargs="+")
    args = parser.parse_args()

    combined = combine_test_suites(args.suite_specs)
    with Path(args.output_path).open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(combined, output_file, sort_keys=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
