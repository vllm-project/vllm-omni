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


def _pipeline_relative_path(pipeline_path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    resolved = pipeline_path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return pipeline_path.as_posix().replace("\\", "/")


def _pipeline_inventory(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], list[Any], dict[str, dict[str, Any]]]:
    """Return shared metadata, group structure, and uniquely labelled leaves."""
    steps = doc.get("steps")
    if not isinstance(steps, list):
        raise ValueError("AMD nightly pipeline must contain a steps list")

    leaves: dict[str, dict[str, Any]] = {}

    def walk(entries: list[Any], parents: tuple[dict[str, Any], ...]) -> list[Any]:
        structure: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("AMD nightly pipeline steps must be mappings")
            if "steps" in entry:
                nested = entry["steps"]
                if not isinstance(nested, list):
                    raise ValueError("AMD nightly pipeline group steps must be a list")
                metadata = {key: value for key, value in entry.items() if key != "steps"}
                structure.append((metadata, walk(nested, (*parents, metadata))))
                continue

            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("AMD nightly leaf steps must have a non-empty label")
            if label in leaves:
                raise ValueError(f"AMD nightly pipeline contains duplicate leaf label {label!r}")
            leaves[label] = {"parents": parents, "step": entry}
        return structure

    structure = walk(steps, ())
    shared = {key: value for key, value in doc.items() if key != "steps"}
    return shared, structure, leaves


def _changed_pipeline_labels(
    current: dict[str, Any],
    base: dict[str, Any],
) -> set[str]:
    """Select changed/new leaves when shared and group structure is unchanged."""
    current_shared, current_structure, current_leaves = _pipeline_inventory(current)
    base_shared, base_structure, base_leaves = _pipeline_inventory(base)
    if current_shared != base_shared:
        raise ValueError("AMD nightly pipeline shared metadata changed")
    if current_structure != base_structure:
        raise ValueError("AMD nightly pipeline group structure changed")
    return {label for label, current_entry in current_leaves.items() if base_leaves.get(label) != current_entry}


def _prune_to_labels(doc: dict[str, Any], labels: set[str]) -> dict[str, Any]:
    """Keep selected leaves and their groups, stripping uploader-only metadata."""

    def walk(entries: list[Any]) -> list[Any]:
        selected: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("AMD nightly pipeline steps must be mappings")
            if "steps" in entry:
                nested = entry["steps"]
                if not isinstance(nested, list):
                    raise ValueError("AMD nightly pipeline group steps must be a list")
                kept = walk(nested)
                if kept:
                    group = {key: value for key, value in entry.items() if key != "source_file_dependencies"}
                    group["steps"] = kept
                    selected.append(group)
                continue
            if entry.get("label") in labels:
                selected.append({key: value for key, value in entry.items() if key != "source_file_dependencies"})
        return selected

    steps = doc.get("steps")
    if not isinstance(steps, list):
        raise ValueError("AMD nightly pipeline must contain a steps list")
    return {**doc, "steps": walk(steps)}


def _filter_nightly_steps(
    doc: dict[str, Any],
    changed_files: list[str],
    *,
    pipeline_path: Path,
    current_pipeline_doc: dict[str, Any] | None,
    base_pipeline_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    pipeline_rel = _pipeline_relative_path(pipeline_path)
    pipeline_changed = pipeline_rel in changed_files
    forced_labels: set[str] | None = None
    filter_files = changed_files

    if pipeline_changed and current_pipeline_doc is not None and base_pipeline_doc is not None:
        try:
            forced_labels = _changed_pipeline_labels(current_pipeline_doc, base_pipeline_doc)
        except (TypeError, ValueError) as exc:
            print(f"Keep all AMD nightly jobs: cannot safely compare pipeline revisions ({exc}).", file=sys.stderr)
        else:
            filter_files = [path for path in changed_files if path != pipeline_rel]
    elif pipeline_changed:
        print("Keep all AMD nightly jobs: base pipeline revision is unavailable.", file=sys.stderr)

    filtered = filter_pipeline_by_source_dependencies(
        doc,
        filter_files,
        pipeline_path=pipeline_path,
        extra_fallback_prefixes=AMD_SOURCE_FILTER_FALLBACKS,
    )
    if forced_labels is None:
        return filtered

    _, _, source_selected = _pipeline_inventory(filtered)
    selected_labels = set(source_selected) | forced_labels
    return _prune_to_labels(doc, selected_labels)


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
    current_pipeline_doc: dict[str, Any] | None = None,
    base_pipeline_doc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Filter a standalone nightly suite or its group in a combined suite."""
    if group_name is None:
        return _filter_nightly_steps(
            doc,
            changed_files,
            pipeline_path=pipeline_path,
            current_pipeline_doc=current_pipeline_doc or doc,
            base_pipeline_doc=base_pipeline_doc,
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
        filtered = _filter_nightly_steps(
            {"steps": nested},
            changed_files,
            pipeline_path=pipeline_path,
            current_pipeline_doc=current_pipeline_doc,
            base_pipeline_doc=base_pipeline_doc,
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
    parser.add_argument("--base-pipeline-path", type=Path)
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

    base_pipeline_doc = None
    current_pipeline_doc = doc if args.group_name is None else None
    if args.base_pipeline_path is not None:
        try:
            loaded_base = yaml.safe_load(args.base_pipeline_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_base, dict):
                raise ValueError(f"invalid base AMD nightly pipeline YAML: {args.base_pipeline_path}")
            base_pipeline_doc = loaded_base
            if args.group_name is not None:
                loaded_current = yaml.safe_load(args.pipeline_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_current, dict):
                    raise ValueError(f"invalid current AMD nightly pipeline YAML: {args.pipeline_path}")
                current_pipeline_doc = loaded_current
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            print(f"Keep all AMD nightly jobs: cannot load pipeline comparison ({exc}).", file=sys.stderr)

    try:
        filtered = filter_amd_nightly_suite(
            doc,
            changed_files,
            pipeline_path=args.pipeline_path,
            group_name=args.group_name,
            current_pipeline_doc=current_pipeline_doc,
            base_pipeline_doc=base_pipeline_doc,
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
