#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render and optionally upload Buildkite pipeline YAML with diff-aware logic.

Bootstrap mode (pipeline.yml with __IMAGE_BUILD_IF__ placeholders):
  - Detect docs-only, pytest skip-mark-only, or combined docs+skip-mark PR/main changes and
    substitute skip-ci ``if`` expressions.

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
import re
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


def resolve_diff_range() -> str | None:
    """Return a git diff range for PR or main builds, or None when unavailable."""
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
        return f"{base_ref}...{commit}"

    if os.environ.get("BUILDKITE_BRANCH", "") == "main":
        if _git("rev-parse", "--verify", f"{commit}^").returncode != 0:
            _log("main commit has no parent; using safe defaults")
            return None
        return f"{commit}^..{commit}"

    _log("not PR/main build; using safe defaults")
    return None


def resolve_changed_files() -> list[str] | None:
    """Return changed file paths, or None when diff cannot be resolved."""
    diff_range = resolve_diff_range()
    if diff_range is None:
        return None

    result = _git("diff", "--name-only", diff_range)
    if result.returncode != 0:
        _log(f"git diff failed ({diff_range}); using safe defaults")
        return None

    files = [line for line in result.stdout.splitlines() if line.strip()]
    _log(f"{len(files)} changed file(s)")
    return files


def _is_doc_file(file_path: str) -> bool:
    if not file_path:
        return False
    if file_path.startswith("docs/"):
        return True
    if file_path.endswith(".md"):
        return True
    return file_path == "mkdocs.yml"


def is_docs_only_change(changed_files: list[str]) -> bool:
    has_any = False
    for file_path in changed_files:
        if not file_path:
            continue
        has_any = True
        if not _is_doc_file(file_path):
            return False
    return has_any


def _is_test_python_file(file_path: str) -> bool:
    path = Path(file_path)
    return path.suffix == ".py" and bool(path.parts) and path.parts[0] == "tests"


_SKIP_MARK_RE = re.compile(r"pytest\.mark\.skip(?:if)?\b|pytest\.skip\s*\(")
_PYTESTMARK_SKIP_RE = re.compile(r"pytest\.mark\.skip\b")
_PYTEST_MARK_ONLY_RE = re.compile(r"pytest\.mark\.\w+")


def _paren_balance(line: str) -> int:
    return line.count("(") - line.count(")")


def _is_skip_mark_related_content(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    return _SKIP_MARK_RE.search(stripped) is not None


def _is_pytestmark_adjacent_content(line: str) -> bool:
    """Allow pytestmark list refactors that only add skip/skipif alongside existing marks."""
    stripped = line.strip().rstrip(",")
    if not stripped:
        return True
    if stripped in {"[", "]"}:
        return True
    if stripped.startswith("#"):
        return True
    if stripped.startswith("pytestmark"):
        return True
    return _PYTEST_MARK_ONLY_RE.search(stripped) is not None


def diff_only_contains_skip_mark_changes(diff_text: str) -> bool:
    """True when diff only edits skip marks or reformats pytestmark to add them."""
    pending_depth = 0
    saw_change = False
    has_skip_mark_edit = False
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("@@"):
            pending_depth = 0
            continue
        if not (raw_line.startswith("+") or raw_line.startswith("-")):
            continue
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue

        saw_change = True
        content = raw_line[1:]

        if pending_depth > 0:
            pending_depth += _paren_balance(content)
            if pending_depth < 0:
                pending_depth = 0
            continue

        if _is_skip_mark_related_content(content):
            has_skip_mark_edit = True
            pending_depth = max(0, _paren_balance(content))
            continue

        if _is_pytestmark_adjacent_content(content):
            continue

        return False

    return saw_change and has_skip_mark_edit


def _git_file_diff(file_path: str, diff_range: str) -> str | None:
    result = _git("diff", diff_range, "--", file_path)
    if result.returncode != 0:
        _log(f"skip-mark-only: git diff failed for {file_path}")
        return None
    return result.stdout


def _is_new_file_diff(diff_text: str) -> bool:
    return "new file mode" in diff_text or "--- /dev/null" in diff_text


def _file_has_module_level_pytest_skip(content: str) -> bool:
    """True when ``pytestmark`` applies unconditional ``pytest.mark.skip`` to the module."""
    match = re.search(r"^pytestmark\s*=\s*\[(.*?)^\s*\]", content, re.MULTILINE | re.DOTALL)
    if match is not None:
        return _PYTESTMARK_SKIP_RE.search(match.group(1)) is not None
    match = re.search(r"^pytestmark\s*=\s*pytest\.mark\.skip\b", content, re.MULTILINE)
    return match is not None


def _file_content_at_commit(file_path: str, commit: str) -> str | None:
    result = _git("show", f"{commit}:{file_path}")
    if result.returncode != 0:
        _log(f"skip-mark-only: cannot read {file_path} at {commit}")
        return None
    return result.stdout


def _qualifies_as_skip_mark_test_change(file_path: str, diff_range: str) -> bool:
    diff_text = _git_file_diff(file_path, diff_range)
    if diff_text is None:
        return False
    if not diff_text.strip():
        _log(f"skip-mark-only: empty diff for {file_path}")
        return False

    if diff_only_contains_skip_mark_changes(diff_text):
        return True

    if not _is_new_file_diff(diff_text):
        _log(f"skip-mark-only: non-skip changes in {file_path}")
        return False

    commit = os.environ.get("BUILDKITE_COMMIT", "")
    if not commit:
        return False
    content = _file_content_at_commit(file_path, commit)
    if content is None:
        return False
    if _file_has_module_level_pytest_skip(content):
        _log(f"skip-mark-only: new test file with module-level pytest.mark.skip: {file_path}")
        return True

    _log(f"skip-mark-only: new test file without module-level pytest.mark.skip: {file_path}")
    return False


def is_skip_mark_only_change(changed_files: list[str], *, diff_range: str) -> bool:
    """True when every changed file is a tests/ module with only skip-mark edits."""
    has_any = False
    for file_path in changed_files:
        if not file_path:
            continue
        has_any = True
        if not _is_test_python_file(file_path):
            return False
        if not _qualifies_as_skip_mark_test_change(file_path, diff_range):
            return False
    return has_any


def is_docs_or_skip_mark_only_change(changed_files: list[str], *, diff_range: str) -> bool:
    """True when each changed file is a doc path or a qualifying skip-mark test change."""
    has_any = False
    for file_path in changed_files:
        if not file_path:
            continue
        has_any = True
        if _is_doc_file(file_path):
            continue
        if _is_test_python_file(file_path) and _qualifies_as_skip_mark_test_change(file_path, diff_range):
            continue
        _log(f"docs/skip-mark-only: rejecting {file_path}")
        return False
    return has_any


def resolve_skip_ci(changed_files: list[str] | None, *, diff_range: str | None = None) -> bool:
    if changed_files is None:
        _log("skip-ci=0 (could not resolve changed files)")
        return False

    if diff_range is not None and is_docs_or_skip_mark_only_change(changed_files, diff_range=diff_range):
        if is_docs_only_change(changed_files):
            _log("docs-only change detected; skip-ci=1")
        elif is_skip_mark_only_change(changed_files, diff_range=diff_range):
            _log("pytest skip-mark-only change detected; skip-ci=1")
        else:
            _log("docs + pytest skip-mark-only change detected; skip-ci=1")
        return True

    if diff_range is None and is_docs_only_change(changed_files):
        _log("docs-only change detected; skip-ci=1")
        return True

    _log("non-doc/non-skip-mark changes detected; skip-ci=0")
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
    diff_range = resolve_diff_range()
    changed_files = resolve_changed_files()

    if BOOTSTRAP_MARKER in text:
        rendered = render_bootstrap_pipeline(
            text,
            skip_ci=resolve_skip_ci(changed_files, diff_range=diff_range),
        )
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
