#!/usr/bin/env python3
"""
Re-run only **### Analysis (CI Failure)** (GitHub Search: label:bug + label:ci-failure)
and splice into an existing report Markdown.

Requires GITHUB_TOKEN or GH_TOKEN for reliable Search API rate limits (optional but recommended).

Example:

  cd skills/vllm-omni-test-report
  python scripts/patch_report_ci_failure.py \\
    --report vllm-omni-test-report-2026-04-01_to_2026-05-07.md \\
    --stats-from 2026-04-01 --stats-to 2026-05-07
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import compose_full_report as cfr  # noqa: E402

DATA_SOURCE_LINE_RE = re.compile(
    r"^- GitHub Search:.*ci-github-ci-failure-issues\.md\s*\n",
    re.MULTILINE,
)
NEW_DATA_SOURCE_LINE = (
    "- GitHub Search: `label:bug` + `label:ci-failure`, `created` = "
    "`--stats-from`..`--stats-to` (UTC); see `references/ci-github-ci-failure-issues.md`\n"
)


def patch_markdown(path: Path, stats_from: str, stats_to: str) -> None:
    raw = path.read_text(encoding="utf-8")
    start = raw.find("### Analysis (CI Failure)")
    if start == -1:
        sys.exit(f"No '### Analysis (CI Failure)' heading in {path}")

    end = raw.find("\n## Open issues", start)
    if end == -1:
        sys.exit(f"No following '## Open issues' heading after CI Failure section in {path}")

    gh_token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip() or None

    new_section = cfr.render_ci_failure_section(stats_from, stats_to, gh_token)
    if not new_section.endswith("\n"):
        new_section += "\n"

    # Keep from "\n## Open issues" onward
    updated = raw[:start] + new_section + raw[end:]

    if DATA_SOURCE_LINE_RE.search(updated):
        updated = DATA_SOURCE_LINE_RE.sub(NEW_DATA_SOURCE_LINE, updated, count=1)
    path.write_text(updated, encoding="utf-8")
    print(f"Patched CI Failure section in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch Analysis (CI Failure) in an existing report.")
    ap.add_argument("--report", type=Path, required=True, help="Path to report .md")
    ap.add_argument("--stats-from", required=True, metavar="YYYY-MM-DD")
    ap.add_argument("--stats-to", required=True, metavar="YYYY-MM-DD")
    args = ap.parse_args()
    rpath = args.report
    if not rpath.is_file():
        sys.exit(f"Not a file: {rpath}")
    patch_markdown(rpath.resolve(), args.stats_from, args.stats_to)


if __name__ == "__main__":
    main()
