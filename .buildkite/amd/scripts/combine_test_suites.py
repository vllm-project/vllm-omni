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


def _flatten_steps(suite: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for entry in suite.get("steps") or []:
        if "group" in entry:
            steps.extend(entry.get("steps") or [])
        else:
            steps.append(entry)
    return steps


def combine_test_suites(suite_specs: Sequence[str]) -> dict[str, Any]:
    """Group selected suites while dropping exact duplicates from later suites."""

    combined: dict[str, Any] = {"env": {}, "steps": []}
    prior_steps: list[dict[str, Any]] = []

    for suite_spec in suite_specs:
        group_name, input_path = suite_spec.split(":", 1)
        with Path(input_path).open(encoding="utf-8") as test_file:
            suite = yaml.safe_load(test_file)

        for name, value in (suite.get("env") or {}).items():
            previous = combined["env"].get(name, value)
            if previous != value:
                raise ValueError(f"Conflicting environment value for {name}: {previous!r} != {value!r}")
            combined["env"][name] = value

        suite_steps = _flatten_steps(suite)
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
