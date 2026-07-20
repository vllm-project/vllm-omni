#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a pre-commit hook that fails if test files are modified or added
that (probably) never run in the CI. For now, this means that every tests file
needs to have a CI level marker (e.g., core_model, advanced_model, full_model,
etc) and hardware mark / helper so that we ensure mutated tests will actually
be selected as long as there are pytest commands pointing at the right paths.
"""

import os
import re
import sys

# CI level markers
LEVEL_MARKERS = ("core_model", "advanced_model", "full_model", "slow")

# Hardware markers. These are platforms and resources that CI filters on.
# NOTE: If new hardware marks are added to pyproject.toml etc, we need
# to add them here as well.
HARDWARE_MARKERS = (
    "cpu",
    "gpu",
    "cuda",
    "rocm",
    "xpu",
    "npu",
    "musa",
    "H100",
    "L4",
    "MI325",
    "B60",
    "S5000",
    "A2",
    "A3",
)

# Helpers from tests/helpers/mark.py that auto-apply hardware marks.
HARDWARE_HELPERS = ("hardware_test", "hardware_marks")

# Match mark.X since we could also do `from pytest import mark`.
# \b prevents matching prefixes (e.g., mark.slow vs mark.slow_test).
LEVEL_RE = re.compile(r"mark\.(?:" + "|".join(LEVEL_MARKERS) + r")\b")
HARDWARE_RE = re.compile(r"mark\.(?:" + "|".join(HARDWARE_MARKERS) + r")\b")
HELPER_RE = re.compile(r"(?:" + "|".join(HARDWARE_HELPERS) + r")\s*\(")

MISSING_LEVEL_MARKER = "Level"
MISSING_HARDWARE_MARKER = "Hardware"

# Check if a file is located under tests/ and matches test_<something>.py
# or <something>_test.py, since pytest technically collects on both.
# Note that we use the former everywhere in this repo by convention.
TEST_FILE_RE = re.compile(r"^tests/(?:.*/)?(?:test_[^/]*\.py$|[^/]*_test\.py$)")


def is_test_file(path: str) -> bool:
    """Determine whether or not a path is pointing at a test file or not."""
    return bool(TEST_FILE_RE.match(path))


def read_test_file(path: str) -> str | None:
    """Read a test file's contents, or return None if it doesn't exist."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def has_level_marker(contents: str) -> bool:
    """Check if file contents contain at least one CI level marker."""
    return bool(LEVEL_RE.search(contents))


def has_hardware_marker(contents: str) -> bool:
    """Check if file contents contain any hardware marker or helpers."""
    return bool(HARDWARE_RE.search(contents) or HELPER_RE.search(contents))


def get_files_missing_markers(
    staged_files: list[str],
) -> dict[str, list[str]]:
    """Return a dict mapping file path to list of missing marker types."""
    results: dict[str, list[str]] = {}
    for path in staged_files:
        if is_test_file(path) and (contents := read_test_file(path)) is not None:
            missing = []
            if not has_level_marker(contents):
                missing.append(MISSING_LEVEL_MARKER)
            if not has_hardware_marker(contents):
                missing.append(MISSING_HARDWARE_MARKER)
            if missing:
                results[path] = missing
    return results


if __name__ == "__main__":
    missing = get_files_missing_markers(sys.argv[1:])

    if missing:
        file_lines = "\n".join(f"  - {path} [{' and '.join(problems)}]" for path, problems in missing.items())
        print(
            "\033[91merror:\033[0m test files are missing pytest marks "
            "required for Buildkite CI collection.\n\n"
            f"Level marks, e.g.: {', '.join(LEVEL_MARKERS[:4])}\n"
            f"Hardware marks, e.g.: {', '.join(HARDWARE_MARKERS[:4])}, ...\n"
            f"  or helpers: {', '.join(HARDWARE_HELPERS)}\n\n"
            "The following files are missing marks:\n"
            f"{file_lines}\n\n"
            "To skip: SKIP=check-test-ci-coverage git commit ..."
        )
        sys.exit(1)
