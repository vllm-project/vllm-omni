#!/usr/bin/env python3
"""
Re-fetch **## Open issues** (open ``label:bug``, ``created_at`` date in stats window)
and splice into an existing report. The refreshed table includes the DI priority and DI value
used by ``compose_full_report.py`` for the release conclusion row.

Requires GITHUB_TOKEN or GH_TOKEN for reliable pagination (optional but recommended).

Example:

  cd skills/vllm-omni-test-report
  python scripts/patch_report_open_issues.py \\
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

GITHUB_BULLET_RE = re.compile(
    r"^- GitHub: `GET /repos/vllm-project/vllm-omni/issues\?state=open&labels=bug`.*\n",
    re.MULTILINE,
)
NEW_GITHUB_BULLET = (
    "- GitHub: `GET /repos/vllm-project/vllm-omni/issues?state=open&labels=bug` (paginated); "
    "**Open issues** table = issues with `created_at` **UTC date** in `--stats-from`..`--stats-to`; "
    "DI = priority label weights (`critical` 10 / `high priority` 3 / `medium priority` 1 / "
    "`low priority` 0.1 / `invalid` 0)\n"
)


def patch_markdown(path: Path, stats_from: str, stats_to: str) -> None:
    raw = path.read_text(encoding="utf-8")
    start = raw.find("## Open issues")
    if start == -1:
        sys.exit(f"No '## Open issues' heading in {path}")

    end = raw.find("\n## Data source", start)
    if end == -1:
        sys.exit(f"No following '## Data source' after Open issues in {path}")

    gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    gh_token = gh_token.strip() if gh_token else None

    new_block = cfr.render_open_issues_section(stats_from, stats_to, gh_token)

    updated = raw[:start] + new_block + raw[end:]

    if GITHUB_BULLET_RE.search(updated):
        updated = GITHUB_BULLET_RE.sub(NEW_GITHUB_BULLET, updated, count=1)
    path.write_text(updated, encoding="utf-8")
    print(f"Patched Open issues section in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch Open issues section in an existing report.")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--stats-from", required=True, metavar="YYYY-MM-DD")
    ap.add_argument("--stats-to", required=True, metavar="YYYY-MM-DD")
    args = ap.parse_args()
    rp = args.report
    if not rp.is_file():
        sys.exit(f"Not a file: {rp}")
    patch_markdown(rp.resolve(), args.stats_from, args.stats_to)


if __name__ == "__main__":
    main()
