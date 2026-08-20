#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run mypy on changed files.

Ported from vllm/tools/pre_commit/mypy.py. Group files so tests can use
``follow_imports=skip`` without pulling that setting into library code.

Usage:
    python tools/pre_commit/mypy.py <python_version> <changed_files...>

Args:
    python_version: Python version to use (e.g. "3.10") or "local" to use
        the interpreter running this script.
    changed_files: List of changed files to check.
"""

from __future__ import annotations

import subprocess
import sys

import regex as re

# tests/ uses follow_imports=skip so third-party and library errors do not
# leak into test type-checking. After a directory is clean under skip, it
# can join the default group (follow_imports from pyproject.toml).
SEPARATE_GROUPS = [
    "tests",
]

# Model trees are large third-party ports. Drop a prefix from this list when
# that directory is ready to be type-checked (same ratchet as upstream vLLM).
EXCLUDE = [
    r"vllm_omni/model_executor/models/",
    r"vllm_omni/diffusion/models/",
]


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """Group changed files into different mypy calls."""
    exclude_pattern = re.compile(f"^{'|'.join(EXCLUDE)}.*")
    file_groups: dict[str, list[str]] = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    # Longest path first so a sub-directory is not shadowed by its parent.
    separate_groups = sorted(SEPARATE_GROUPS, key=len, reverse=True)
    for changed_file in changed_files:
        changed_file = _normalize_path(changed_file)
        if exclude_pattern.match(changed_file):
            continue
        for directory in separate_groups:
            if re.match(f"^{directory}.*", changed_file):
                file_groups[directory].append(changed_file)
                break
        else:
            if changed_file.startswith(("vllm_omni/", "tests/")):
                file_groups[""].append(changed_file)
    return file_groups


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
    file_group: str,
) -> int:
    """Run mypy on the given targets."""
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    print(f"$ {' '.join(args)} {file_group}")
    return subprocess.run(args + targets, check=False).returncode


def main() -> int:
    python_version = sys.argv[1]
    file_groups = group_files(sys.argv[2:])

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = None if file_group == "" else "skip"
        if changed_files:
            returncode |= mypy(changed_files, python_version, follow_imports, file_group)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
