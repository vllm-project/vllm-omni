#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Apply shared diff-aware source filtering to selected AMD nightly steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_COMMON_SCRIPTS = Path(__file__).resolve().parents[2] / "common" / "scripts"
if str(_COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_COMMON_SCRIPTS))

from upload_pipeline import filter_pipeline_by_source_dependencies  # noqa: E402

AMD_SOURCE_FILTER_FALLBACKS = (
    ".buildkite/amd/scripts/bootstrap-amd-omni.sh",
    ".buildkite/amd/scripts/combine_test_suites.py",
    ".buildkite/amd/scripts/filter_test_suite.py",
    ".buildkite/amd/scripts/select_test_suites.py",
    ".buildkite/amd/test-template-amd-omni.j2",
    ".buildkite/common/ci_source_file_dependencies.yml",
    ".buildkite/common/scripts/upload_pipeline.py",
)


def should_filter_pr_nightly(
    *,
    pull_request: str,
    debug_test_yaml: str,
    scheduled_nightly: bool,
) -> bool:
    """Return whether a label-selected PR nightly should be diff-filtered."""
    return pull_request not in {"", "false"} and not debug_test_yaml and not scheduled_nightly


def filter_amd_nightly_suite(
    doc: dict[str, Any],
    changed_files: list[str],
    *,
    pipeline_path: Path,
    group_name: str | None = None,
) -> dict[str, Any]:
    """Filter a standalone nightly suite or its group in a combined suite."""
    if group_name is None:
        return filter_pipeline_by_source_dependencies(
            doc,
            changed_files,
            pipeline_path=pipeline_path,
            extra_fallback_prefixes=AMD_SOURCE_FILTER_FALLBACKS,
        )

    entries = doc.get("steps")
    if not isinstance(entries, list):
        raise ValueError("combined AMD test suite must contain a steps list")

    found = False
    filtered_entries: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("group") != group_name:
            filtered_entries.append(entry)
            continue
        if found:
            raise ValueError(f"combined AMD test suite contains duplicate {group_name!r} groups")
        found = True
        nested = entry.get("steps")
        if not isinstance(nested, list):
            raise ValueError(f"combined AMD test suite group {group_name!r} must contain a steps list")
        filtered = filter_pipeline_by_source_dependencies(
            {"steps": nested},
            changed_files,
            pipeline_path=pipeline_path,
            extra_fallback_prefixes=AMD_SOURCE_FILTER_FALLBACKS,
        )
        if filtered["steps"]:
            filtered_entries.append({**entry, "steps": filtered["steps"]})

    if not found:
        raise ValueError(f"combined AMD test suite does not contain group {group_name!r}")
    return {**doc, "steps": filtered_entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--changed-files", required=True)
    parser.add_argument("--pipeline-path", type=Path, required=True)
    parser.add_argument("--group-name")
    parser.add_argument("--pull-request", default="false")
    parser.add_argument("--debug-test-yaml", default="")
    parser.add_argument("--scheduled-nightly", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()

    if not should_filter_pr_nightly(
        pull_request=args.pull_request,
        debug_test_yaml=args.debug_test_yaml,
        scheduled_nightly=bool(args.scheduled_nightly),
    ):
        args.output_path.write_bytes(args.input_path.read_bytes())
        return 0

    doc = yaml.safe_load(args.input_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        parser.error(f"invalid AMD test suite YAML: {args.input_path}")
    changed_files = [path for line in args.changed_files.splitlines() if (path := line.strip())]

    try:
        filtered = filter_amd_nightly_suite(
            doc,
            changed_files,
            pipeline_path=args.pipeline_path,
            group_name=args.group_name,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    args.output_path.write_text(
        yaml.safe_dump(filtered, sort_keys=False, width=4096),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
