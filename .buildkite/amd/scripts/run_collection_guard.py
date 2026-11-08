#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run an AMD step's exact pytest collection guard outside its GPU job.

The expected count and pytest arguments are read from the step, so this check
and the GPU job run the same guard on the same selection.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

GUARD = ".buildkite/amd/scripts/assert-pytest-collection.sh"
_SHELL_SYNTAX = frozenset("$`;&|<>")


def _iter_steps(steps: list[Any]) -> Iterator[dict[str, Any]]:
    for step in steps:
        if isinstance(step, dict):
            yield step
            yield from _iter_steps(step.get("steps") or [])


def find_step(steps: list[Any], label: str) -> dict[str, Any]:
    """Return the only step labeled ``label``, including steps inside groups."""
    matches = [step for step in _iter_steps(steps) if step.get("label") == label]
    if len(matches) != 1:
        raise ValueError(f"expected one step labeled {label!r}, found {len(matches)}")
    return matches[0]


def guard_arguments(step: dict[str, Any]) -> list[str]:
    """Return the expected count and pytest arguments of the step's guard call.

    Args:
        step: A Buildkite step whose commands call the guard exactly once.

    Returns:
        The guard arguments: the expected count followed by pytest arguments.

    Raises:
        ValueError: If the guard is not called exactly once with a positive
            count and literal pytest arguments.
    """
    calls: list[list[str]] = []
    for command in step.get("commands") or []:
        lines = command.splitlines()
        for start, line in enumerate(lines):
            if GUARD not in line or line.lstrip().startswith("#"):
                continue
            end = start
            while lines[end].rstrip().endswith("\\") and end + 1 < len(lines):
                end += 1
            words = shlex.split(" ".join(part.rstrip().removesuffix("\\") for part in lines[start : end + 1]))
            if GUARD in words:
                calls.append(words[words.index(GUARD) + 1 :])
    if len(calls) != 1:
        raise ValueError(f"expected one {GUARD} call, found {len(calls)}")
    arguments = calls[0]
    count = arguments[0] if arguments else ""
    if len(arguments) < 2 or not count.isdigit() or str(int(count)) != count or int(count) == 0:
        raise ValueError(f"{GUARD} needs a positive count and pytest arguments, got {arguments}")
    if any(_SHELL_SYNTAX.intersection(argument) for argument in arguments):
        raise ValueError(f"{GUARD} arguments must be literal, got {arguments}")
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path, help="AMD test suite YAML")
    parser.add_argument("label", help="exact label of the step that calls the guard")
    args = parser.parse_args(argv)

    doc = yaml.safe_load(args.pipeline.read_text(encoding="utf-8"))
    steps = doc.get("steps") if isinstance(doc, dict) else None
    try:
        arguments = guard_arguments(find_step(steps or [], args.label))
    except ValueError as error:
        parser.error(str(error))

    command = ["bash", GUARD, *arguments]
    print(f"+ {shlex.join(command)}", flush=True)
    status = subprocess.run(command, check=False).returncode
    if status:
        print(
            f"{args.label!r} in {args.pipeline} pins this collection. If this change adds, "
            "removes, or re-marks tests in it, update the count there and in its contract tests "
            "under tests/buildkite/.",
            file=sys.stderr,
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
