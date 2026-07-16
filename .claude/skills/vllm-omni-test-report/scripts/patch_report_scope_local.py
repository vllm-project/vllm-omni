#!/usr/bin/env python3
"""
Refresh **## Test content (job scope)** (from Buildkite build JSON + reference lookup) and/or
**## Test Result** (Common stack + H200/H800/A100 placeholders + preserved **H100** CI block) inside an existing report.

Uses the **Build** number already in the report (``| **Build** | [N](...) |``) unless ``--build``
is passed. Requires ``BUILDKITE_TOKEN`` or ``BUILDKITE_API_TOKEN`` when patching job scope.

Example (from skill directory)::

  python scripts/patch_report_scope_local.py --report vllm-omni-test-report-2026-05-07.md
  python scripts/patch_report_scope_local.py --report report.md --no-local
  python scripts/patch_report_scope_local.py --report report.md --build 9063 --no-job-scope

New reports use ``## Test Result`` … ``### H100（CI — Buildkite scheduled nightly）`` … ``## Issue tracking``.
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

BUILD_RE = re.compile(r"\| \*\*Build\*\* \| \[(\d+)\]")
JOB_SCOPE_DS_RE = re.compile(r"^- Job scope:.*$", re.MULTILINE)
H100_HEADING = "### H100（CI — Buildkite scheduled nightly）"


def _extract_h100_ci_block(raw: str) -> str | None:
    start = raw.find(H100_HEADING)
    if start == -1:
        return None
    body0 = raw.find("\n", start)
    if body0 == -1:
        return None
    body_start = body0 + 1
    end = raw.find("\n## Issue tracking", body_start)
    if end == -1:
        return None
    return raw[body_start:end].strip()


def _read_report(path: Path) -> str:
    """Read UTF-8. Avoid ``errors='replace'`` which can persist U+FFFD (�) in saved output."""
    return path.read_text(encoding="utf-8")


def patch_report(
    path: Path,
    *,
    build_no: int | None,
    do_job_scope: bool,
    do_local: bool,
) -> None:
    raw = _read_report(path)
    skill_dir = path.resolve().parent

    if do_job_scope:
        token = (os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN") or "").strip()
        if not token:
            sys.exit("BUILDKITE_API_TOKEN or BUILDKITE_TOKEN is required for --job-scope (default on).")
        bn = build_no
        if bn is None:
            m = BUILD_RE.search(raw)
            if not m:
                sys.exit("No | **Build** | [number] | in report; pass --build N.")
            bn = int(m.group(1))
        build_url = f"https://api.buildkite.com/v2/organizations/{cfr.ORG}/pipelines/{cfr.PIPELINE}/builds/{bn}"
        build = cfr.http_json(build_url, token)
        if not isinstance(build, dict):
            sys.exit(f"Unexpected Buildkite JSON for build {bn}")
        new_scope = cfr.render_job_scope_section(build, bn, skill_dir)
    else:
        bn = build_no or 0
        new_scope = None

    if new_scope is not None:
        start = raw.find("## Test content (job scope)")
        if start == -1:
            sys.exit("No '## Test content (job scope)' heading.")
        end = raw.find("\n## Test Result", start)
        if end == -1:
            sys.exit("No following '## Test Result'.")
        raw = raw[:start] + new_scope.rstrip() + "\n\n" + raw[end + 1 :]

    if do_local:
        h100 = _extract_h100_ci_block(raw)
        if not h100:
            h100 = (
                "*Could not parse H100 (CI) block from the report; run full `compose_full_report.py` to regenerate.*\n"
            )
        new_tr = cfr.render_test_result_section(
            skill_dir,
            log_h200=None,
            log_h800=None,
            log_a100=None,
            h100_ci_markdown=h100,
        )
        start = raw.find("## Test Result")
        if start == -1:
            sys.exit("No '## Test Result' heading.")
        end = raw.find("\n## Issue tracking", start)
        if end == -1:
            sys.exit("No following '## Issue tracking'.")
        raw = raw[:start] + new_tr.rstrip() + "\n\n" + raw[end + 1 :]

    if do_job_scope and bn:
        repl = (
            f"- Job scope: build **#{bn}** reportable jobs × `references/ci-job-test-scope.md` (Scope / intent lookup)"
        )
        if JOB_SCOPE_DS_RE.search(raw):
            raw = JOB_SCOPE_DS_RE.sub(repl, raw, count=1)
        # else: leave Data source unchanged

    path.write_text(raw, encoding="utf-8")
    print(f"Patched {path} (job_scope={do_job_scope}, local={do_local})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch job scope and/or local testing in a report.")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--build", type=int, default=None, help="Build number (default: from report)")
    ap.add_argument(
        "--no-job-scope",
        action="store_true",
        help="Skip Test content (job scope).",
    )
    ap.add_argument(
        "--no-local",
        action="store_true",
        help="Skip Test Result refresh (Common stack + GPU stubs; keeps existing H100 block when possible).",
    )
    args = ap.parse_args()
    rp = args.report
    if not rp.is_file():
        sys.exit(f"Not a file: {rp}")
    patch_report(
        rp.resolve(),
        build_no=args.build,
        do_job_scope=not args.no_job_scope,
        do_local=not args.no_local,
    )


if __name__ == "__main__":
    main()
