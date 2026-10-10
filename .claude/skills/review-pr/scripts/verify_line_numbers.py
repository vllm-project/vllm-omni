#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check review comment coordinates against the actual lines of a PR diff."""

import argparse
import ast
import json
import subprocess
import sys


def parse_hunk(text):
    """Parse a unified-diff hunk header without a runtime dependency."""
    parts = text.split(" ", 4)
    if len(parts) < 4 or parts[0] != "@@" or not parts[3].startswith("@@"):
        return None
    bounds = []
    for value, sign in zip(parts[1:3], ("-", "+")):
        if not value.startswith(sign):
            return None
        numbers = value[1:].split(",")
        if len(numbers) not in (1, 2) or not all(number.isdecimal() for number in numbers):
            return None
        bounds.extend((numbers[0], numbers[1] if len(numbers) == 2 else None))
    return tuple(bounds)


def diff_path(value):
    """Decode a git path, including C-quoted UTF-8 bytes."""
    if value.startswith('"'):
        value = ast.literal_eval(value).encode("latin1").decode("utf-8")
    return None if value == "/dev/null" else value[2:]


def diff_lines(diff):
    """Map each file/side to commentable lines, preserving rename/delete paths."""
    lines = {}
    old_path = new_path = None
    old_remaining = new_remaining = 0
    for text in diff.splitlines():
        if text.startswith("diff --git "):
            old_path = new_path = None
            old_remaining = new_remaining = 0
        elif hunk := parse_hunk(text):
            old_line, old_count, new_line, new_count = hunk
            old_line, new_line = int(old_line), int(new_line)
            old_remaining = int(old_count) if old_count is not None else 1
            new_remaining = int(new_count) if new_count is not None else 1
        elif old_remaining or new_remaining:
            # A hunk's content can itself start with --- or +++.
            path = new_path or old_path
            if text.startswith((" ", "-")):
                lines.setdefault((path, "LEFT"), set()).add(old_line)
                old_line += 1
                old_remaining -= 1
            if text.startswith((" ", "+")):
                lines.setdefault((path, "RIGHT"), set()).add(new_line)
                new_line += 1
                new_remaining -= 1
        elif text.startswith("--- "):
            old_path = diff_path(text[4:])
        elif text.startswith("+++ "):
            new_path = diff_path(text[4:])
    return lines


def validate_comments(comments, lines):
    """Validate single-line and same-side multiline review comments."""
    valid = True
    for comment in comments:
        path = comment["path"]
        side = comment.get("side", "RIGHT")
        line = comment["line"]
        start = comment.get("start_line", line)
        start_side = comment.get("start_side", side)
        available = lines.get((path, side), set())
        if (
            type(line) is not int
            or type(start) is not int
            or start < 1
            or line < start
            or start_side != side
            or line - start + 1 > len(available)
            or not all(number in available for number in range(start, line + 1))
        ):
            print(
                f"ERROR: {path}:{start}-{line} ({side}) is outside diff lines",
                file=sys.stderr,
            )
            valid = False
        else:
            print(f"OK: {path}:{line} ({side})", file=sys.stderr)
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_number", type=int)
    args = parser.parse_args()
    try:
        review = json.load(sys.stdin)
        comments = review["comments"]
        if not isinstance(comments, list):
            raise TypeError("comments must be an array")
        result = subprocess.run(
            [
                "gh",
                "pr",
                "diff",
                str(args.pr_number),
                "--repo",
                "vllm-project/vllm-omni",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return 0 if validate_comments(comments, diff_lines(result.stdout)) else 1
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr, end="")
        return 1
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
