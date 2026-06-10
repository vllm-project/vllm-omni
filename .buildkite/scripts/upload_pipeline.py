#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render and optionally upload Buildkite pipeline YAML with diff-aware logic.

Bootstrap mode (pipeline.yml with __IMAGE_BUILD_IF__ placeholders):
  - Detect docs-only PR/main changes and substitute skip-ci ``if`` expressions.

Test pipeline mode (e.g. test-merge.yml):
  - Drop steps or groups whose ``source_file_dependencies`` do not match changed files.
  - ``source_file_dependencies`` may be set on a leaf step or on a group step.
  - Always strip ``source_file_dependencies`` from uploaded YAML (Buildkite does not
    recognize this uploader-only key).

Usage:
  python3 upload_pipeline.py [--upload] <pipeline.yml>

  # Bootstrap (replaces upload_pipeline_with_skip_ci.sh):
  python3 upload_pipeline.py --upload .buildkite/pipeline.yml

  # Test pipeline (replaces upload_test_pipeline_with_diff_skip.py):
  python3 upload_pipeline.py --upload .buildkite/test-merge.yml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "pyyaml"],
    )
    import yaml

LOG = "upload_pipeline"
ROOT = Path(__file__).resolve().parent.parent.parent
DOC_SEP = "\n---\n"
BOOTSTRAP_MARKER = "__IMAGE_BUILD_IF__"


def _log(message: str) -> None:
    print(f"{LOG}: {message}", file=sys.stderr)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def resolve_changed_files() -> list[str] | None:
    """Return changed file paths, or None when diff cannot be resolved."""
    is_pr = os.environ.get("BUILDKITE_PULL_REQUEST", "false") != "false" and os.environ.get(
        "BUILDKITE_PULL_REQUEST", ""
    )
    commit = os.environ.get("BUILDKITE_COMMIT", "")

    if is_pr:
        base_branch = os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main")
        base_ref = f"origin/{base_branch}"
        if _git("rev-parse", "--verify", base_ref).returncode != 0:
            _log(f"origin/{base_branch} not found locally; trying fetch")
            _git("fetch", "--depth=200", "origin", base_branch)
        if _git("rev-parse", "--verify", base_ref).returncode != 0:
            if _git("rev-parse", "--verify", base_branch).returncode == 0:
                base_ref = base_branch
            else:
                _log(f"cannot resolve PR base {base_branch}; using safe defaults")
                return None
        diff_range = f"{base_ref}...{commit}"
    elif os.environ.get("BUILDKITE_BRANCH", "") == "main":
        if _git("rev-parse", "--verify", f"{commit}^").returncode != 0:
            _log("main commit has no parent; using safe defaults")
            return None
        diff_range = f"{commit}^..{commit}"
    else:
        _log("not PR/main build; using safe defaults")
        return None

    result = _git("diff", "--name-only", diff_range)
    if result.returncode != 0:
        _log(f"git diff failed ({diff_range}); using safe defaults")
        return None

    files = [line for line in result.stdout.splitlines() if line.strip()]
    _log(f"{len(files)} changed file(s)")
    return files


def is_docs_only_change(changed_files: list[str]) -> bool:
    has_any = False
    for file_path in changed_files:
        if not file_path:
            continue
        has_any = True
        if file_path.startswith("docs/"):
            continue
        if file_path.endswith(".md"):
            continue
        if file_path == "mkdocs.yml":
            continue
        return False
    return has_any


def resolve_skip_ci(changed_files: list[str] | None) -> bool:
    if changed_files is None:
        _log("skip-ci=0 (could not resolve changed files)")
        return False
    if is_docs_only_change(changed_files):
        _log("docs-only change detected; skip-ci=1")
        return True
    _log("non-doc changes detected; skip-ci=0")
    return False


def render_bootstrap_pipeline(text: str, *, skip_ci: bool) -> str:
    if DOC_SEP in text:
        _, continuation = text.split(DOC_SEP, 1)
    else:
        continuation = text

    nightly_only = (
        '(build.pull_request.labels includes "nightly-test") || (build.branch == "main" && build.env("NIGHTLY") == "1")'
    )
    if skip_ci:
        image_if = f"'{nightly_only}'"
        ready_if = "'false'"
        merge_if = "'false'"
    else:
        image_if = "'true'"
        ready_if = '\'build.branch != "main" && build.pull_request.labels includes "ready"\''
        merge_if = (
            '\'(build.branch == "main" && build.env("NIGHTLY") != "1" '
            '&& build.env("WEEKLY") != "1") || '
            '(build.branch != "main" && build.pull_request.labels includes "merge-test")\''
        )

    return (
        continuation.replace("__IMAGE_BUILD_IF__", image_if)
        .replace("__UPLOAD_READY_IF__", ready_if)
        .replace("__UPLOAD_MERGE_IF__", merge_if)
    )


def _matches_dependencies(changed_files: list[str], prefixes: list[str]) -> bool:
    for path in changed_files:
        for prefix in prefixes:
            normalized = prefix.rstrip("/")
            if path == normalized or path.startswith(f"{normalized}/"):
                return True
    return False


def _strip_source_file_dependencies(step: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in step.items() if key != "source_file_dependencies"}


def _step_label(step: dict[str, Any]) -> str:
    return str(step.get("group") or step.get("label") or "<step>")


def _filter_steps(steps: list[Any], changed_files: list[str]) -> list[Any]:
    filtered: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            filtered.append(step)
            continue

        deps = step.get("source_file_dependencies")
        if deps is not None and not isinstance(deps, list):
            raise ValueError(
                f"source_file_dependencies must be a list in step {_step_label(step)!r}",
            )
        if deps is not None and not _matches_dependencies(changed_files, deps):
            _log(
                f"skip {_step_label(step)!r} (no changes under {deps})",
            )
            continue

        nested = step.get("steps")
        if nested is not None:
            kept_nested = _filter_steps(nested, changed_files)
            if not kept_nested:
                _log(f"omit empty group {_step_label(step)!r}")
                continue
            new_step = _strip_source_file_dependencies(step)
            new_step["steps"] = kept_nested
            filtered.append(new_step)
            continue

        if deps is not None:
            filtered.append(_strip_source_file_dependencies(step))
        else:
            filtered.append(step)

    return filtered


def _strip_uploader_metadata_from_steps(steps: list[Any]) -> list[Any]:
    """Remove uploader-only keys while keeping all steps (no diff filtering)."""
    stripped: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            stripped.append(step)
            continue

        deps = step.get("source_file_dependencies")
        if deps is not None and not isinstance(deps, list):
            raise ValueError(
                f"source_file_dependencies must be a list in step {_step_label(step)!r}",
            )

        nested = step.get("steps")
        new_step = _strip_source_file_dependencies(step)
        if nested is not None:
            new_step["steps"] = _strip_uploader_metadata_from_steps(nested)
        stripped.append(new_step)

    return stripped


def render_test_pipeline(
    doc: dict[str, Any],
    changed_files: list[str] | None,
) -> dict[str, Any]:
    steps = doc.get("steps")
    if not isinstance(steps, list):
        return doc
    if changed_files is not None:
        steps = _filter_steps(steps, changed_files)
    else:
        steps = _strip_uploader_metadata_from_steps(steps)
    return {**doc, "steps": steps}


def resolve_pipeline_path(arg: str) -> Path:
    path = Path(arg)
    if path.is_absolute():
        return path
    return ROOT / path


def render_pipeline(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    changed_files = resolve_changed_files()

    if BOOTSTRAP_MARKER in text:
        rendered = render_bootstrap_pipeline(text, skip_ci=resolve_skip_ci(changed_files))
        return rendered

    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(f"invalid pipeline YAML: {path}")

    doc = render_test_pipeline(doc, changed_files)

    return yaml.safe_dump(doc, sort_keys=False)


def upload_to_buildkite(content: str) -> None:
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=content,
        text=True,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pipeline",
        nargs="?",
        default=".buildkite/pipeline.yml",
        help="Pipeline YAML path (default: .buildkite/pipeline.yml)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Pipe rendered YAML to buildkite-agent pipeline upload",
    )
    args = parser.parse_args()

    path = resolve_pipeline_path(args.pipeline)
    if not path.is_file():
        _log(f"missing pipeline file: {path}")
        return 1

    rendered = render_pipeline(path)
    if args.upload:
        upload_to_buildkite(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
