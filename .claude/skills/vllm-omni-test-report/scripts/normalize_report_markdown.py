#!/usr/bin/env python3
"""Collapse blank lines inside GFM pipe tables and between consecutive list items."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def collapse_table_gaps(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(\|[^\n]*\|)\n\n(\|)", r"\1\n\2", s)
    return s


def collapse_list_gaps(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(^|\n)(- [^\n]+)\n\n(- )", r"\1\2\n\3", s)
    return s


def normalize_report(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    for _ in range(30):
        text = collapse_table_gaps(text)
    for _ in range(30):
        text = collapse_list_gaps(text)
    for _ in range(10):
        text = collapse_table_gaps(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    path = args.path
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(1)
    raw = path.read_text(encoding="utf-8")
    out = normalize_report(raw)
    path.write_text(out, encoding="utf-8")
    gaps = len(list(re.finditer(r"\|[^\n]+\|\n\n\|", out)))
    print(f"Wrote {path}; remaining |...|\\n\\n| patterns: {gaps}")


if __name__ == "__main__":
    main()
