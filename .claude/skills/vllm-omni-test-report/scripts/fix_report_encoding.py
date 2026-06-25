#!/usr/bin/env python3
"""Repair single-byte 0x9d placeholders (mojibake) in nightly report markdown → UTF-8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EM = "\u2014"  # em dash
MID = "\u00b7"  # middle dot
TIMES = "\u00d7"  # multiplication sign
ELL = "\u2026"  # ellipsis
APOS = "\u2019"  # right single quotation

# Order matters: more specific patterns first.
_REPLACEMENTS: list[tuple[bytes, str]] = [
    (b" (**bugs (first response, \x9d)**", f" (**bugs (first response, {ELL})**"),
    (b"reportable jobs \x9d ", f"reportable jobs {TIMES} "),
    (b"*\x9d*", f"*{EM}*"),
    (b"failed \x9d non-pytest", f"failed {EM} non-pytest"),
    (b"broken \x9d non-pytest", f"broken {EM} non-pytest"),
    (b"passed \x9d log fetch failed", f"passed {EM} log fetch failed"),
    (b"passed \x9d non-pytest", f"passed {EM} non-pytest"),
    (b"round\x9ds", f"round{APOS}s"),
    (b"(Buildkite \x9d Scheduled nightly)", f"(Buildkite {MID} Scheduled nightly)"),
    (b"**2026-04-01** \x9d **2026-05-07**", f"**2026-04-01** {ELL} **2026-05-07**"),
    (b"[issues \x9d bug + ci-failure]", f"[issues {MID} bug + ci-failure]"),
    (b" \x9d ", f" {MID} "),
]


def fix_raw(raw: bytes) -> bytes:
    for old, repl in _REPLACEMENTS:
        raw = raw.replace(old, repl.encode("utf-8"))
    if b"\x9d" in raw:
        n = raw.count(b"\x9d")
        raise ValueError(f"Unresolved 0x9d bytes remain ({n}); extend _REPLACEMENTS")
    return raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, nargs="?", help="Markdown file to fix in place")
    args = ap.parse_args()
    default = Path(__file__).resolve().parents[1] / "vllm-omni-test-report-2026-04-01_to_2026-05-07.md"
    path = args.path or default
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1
    raw = path.read_bytes()
    out = fix_raw(raw)
    out = out.replace(b"\r\n", b"\n")
    path.write_bytes(out)
    # strict UTF-8 round-trip check
    path.read_text(encoding="utf-8")
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
