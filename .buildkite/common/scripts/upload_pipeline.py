#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render and optionally upload Buildkite pipeline YAML with diff-aware logic.

Bootstrap mode (cuda/pipeline.yml, npu/pipeline-npu.yml with ``vllm-omni:placeholder:*`` if sentinels):
  - Detect docs-only, pytest skip-mark-only, or combined skip-ci from git diff.
  - When only CI level YAML changes, enable **L2/L3** upload steps for affected levels only.

Test pipeline mode (e.g. test-merge.yml):
  - Drop steps whose ``source_file_dependencies`` do not match changed files.
  - Expand uploader-only ``mirror_hardwares: l4_1`` into ``agents`` (+ optional ``image``
    for NPU) + ``plugins`` (see ci_mirror_hardwares.yml).

Usage:
  python3 upload_pipeline.py [--upload] [--all | --e2e] <pipeline.yml>
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skip_ci import (
    ROOT,
    resolve_ci_context_from_git,
)

# --- Constants ---

LOG = "upload_pipeline"
DOC_SEP = "\n---\n"
# Valid Buildkite ``if`` placeholders (always false on hook upload); replaced by upload_pipeline.py.
PLACEHOLDER_PREFIX = "vllm-omni:placeholder:"
PLACEHOLDER_IMAGE_BUILD_IF = f'build.message == "{PLACEHOLDER_PREFIX}image-build"'
PLACEHOLDER_UPLOAD_READY_IF = f'build.message == "{PLACEHOLDER_PREFIX}upload-ready"'
PLACEHOLDER_UPLOAD_MERGE_IF = f'build.message == "{PLACEHOLDER_PREFIX}upload-merge"'
PLACEHOLDER_UPLOAD_NIGHTLY_IF = f'build.message == "{PLACEHOLDER_PREFIX}upload-nightly"'
PLACEHOLDER_UPLOAD_WEEKLY_IF = f'build.message == "{PLACEHOLDER_PREFIX}upload-weekly"'
BOOTSTRAP_MARKER = PLACEHOLDER_IMAGE_BUILD_IF
BOOTSTRAP_PLACEHOLDERS = (
    PLACEHOLDER_IMAGE_BUILD_IF,
    PLACEHOLDER_UPLOAD_READY_IF,
    PLACEHOLDER_UPLOAD_MERGE_IF,
    PLACEHOLDER_UPLOAD_NIGHTLY_IF,
    PLACEHOLDER_UPLOAD_WEEKLY_IF,
)
E2E_GROUP_MARKER = "E2E Test"
CI_MIRROR_HARDWARES_PATH = ROOT / ".buildkite/common/ci_mirror_hardwares.yml"

CUDA_NIGHTLY_ONLY = (
    '(build.pull_request.labels includes "nightly-test") || (build.branch == "main" && build.env("NIGHTLY") == "1")'
)
NPU_NIGHTLY_ONLY = (
    '(build.branch == "main" && build.env("NIGHTLY") == "1") || '
    '(build.branch != "main" && ('
    'build.pull_request.labels includes "nightly-test" || '
    'build.pull_request.labels includes "omni-test" || '
    'build.pull_request.labels includes "tts-test" || '
    'build.pull_request.labels includes "diffusion-x2iat-test" || '
    'build.pull_request.labels includes "diffusion-x2v-test"'
    "))"
)


# --- Logging ---


def _log(message: str) -> None:
    print(f"{LOG}: {message}", file=sys.stderr)


# --- Bootstrap pipeline (cuda/pipeline.yml, npu/pipeline-npu.yml) ---


def _get_bootstrap_platform(path: Path) -> str:
    parts = path.as_posix().split("/")
    return "npu" if "npu" in parts else "cuda"


def _is_bootstrap_pipeline(text: str) -> bool:
    return PLACEHOLDER_PREFIX in text


def _render_bootstrap_pipeline(
    text: str,
    *,
    decision,
    path: Path,
) -> str:
    """Replace bootstrap ``if`` placeholders from skip-ci decision (document 2 after ``---``)."""
    platform = _get_bootstrap_platform(path)

    def quoted_if(expr: str) -> str:
        return f"'({expr})'"

    disabled = "'false'"

    _, continuation = _split_pipeline_documents(text)
    placeholders = tuple(name for name in BOOTSTRAP_PLACEHOLDERS if name in continuation)

    nightly_only = NPU_NIGHTLY_ONLY if platform == "npu" else CUDA_NIGHTLY_ONLY
    nightly_main = 'build.branch == "main" && build.env("NIGHTLY") == "1"'

    if platform == "npu":
        ready_pr = (
            'build.branch != "main" && ('
            'build.pull_request.labels includes "npu-test" || '
            'build.pull_request.labels includes "ready"'
            ")"
        )
        merge_main = ""
        merge_pr = ""
        nightly_label_if = nightly_only
        weekly_label_if = ""
    else:
        ready_pr = 'build.branch != "main" && build.pull_request.labels includes "ready"'
        merge_main = 'build.branch == "main" && build.env("NIGHTLY") != "1" && build.env("WEEKLY") != "1"'
        merge_pr = 'build.branch != "main" && build.pull_request.labels includes "merge-test"'
        nightly_label_if = (
            '(build.branch == "main" && build.env("NIGHTLY") == "1") || '
            '(build.branch != "main" && ('
            'build.pull_request.labels includes "nightly-test" || '
            'build.pull_request.labels includes "omni-test" || '
            'build.pull_request.labels includes "tts-test" || '
            'build.pull_request.labels includes "diffusion-x2iat-test" || '
            'build.pull_request.labels includes "diffusion-x2v-test"'
            "))"
        )
        weekly_label_if = (
            '(build.branch == "main" && build.env("WEEKLY") == "1") || '
            '(build.branch != "main" && build.pull_request.labels includes "weekly-test")'
        )

    ready_base = f"({nightly_main}) || ({ready_pr})"
    merge_base = f"({nightly_main}) || (({merge_main}) || ({merge_pr}))" if platform == "cuda" else ""

    if decision.skip_all:
        image_if = quoted_if(nightly_only)
        ready_if = quoted_if(nightly_main)
        merge_if = quoted_if(nightly_main) if platform == "cuda" else disabled
        nightly_if = quoted_if(nightly_label_if)
        weekly_if = disabled
    elif decision.skip_l2_l3:
        l2_enabled = decision.is_run("npu", "l2") if platform == "npu" else decision.is_run("cuda", "l2")
        l3_enabled = platform == "cuda" and decision.is_run("cuda", "l3")

        ready_if = quoted_if(ready_base) if l2_enabled else disabled
        merge_if = quoted_if(merge_base) if l3_enabled else disabled
        nightly_if = quoted_if(nightly_label_if)
        weekly_if = quoted_if(weekly_label_if) if platform == "cuda" else disabled

        image_parts = [f"({nightly_label_if})"]
        if platform == "cuda":
            image_parts.append(f"({weekly_label_if})")
        if l2_enabled:
            image_parts.insert(0, f"({ready_base})")
        if l3_enabled:
            image_parts.insert(1 if l2_enabled else 0, f"({merge_base})")
        image_if = quoted_if(" || ".join(image_parts))
    else:
        image_if = "'true'"
        ready_if = quoted_if(ready_base)
        merge_if = quoted_if(merge_base) if platform == "cuda" else disabled
        nightly_if = quoted_if(nightly_label_if)
        weekly_if = quoted_if(weekly_label_if) if platform == "cuda" else disabled

    replacement_pairs = (
        (PLACEHOLDER_IMAGE_BUILD_IF, image_if),
        (PLACEHOLDER_UPLOAD_READY_IF, ready_if),
        (PLACEHOLDER_UPLOAD_MERGE_IF, merge_if),
        (PLACEHOLDER_UPLOAD_NIGHTLY_IF, nightly_if),
        (PLACEHOLDER_UPLOAD_WEEKLY_IF, weekly_if),
    )
    rendered = continuation
    for name, value in sorted(replacement_pairs, key=lambda item: len(item[0]), reverse=True):
        if name in rendered:
            rendered = rendered.replace(name, value)
    _assert_no_bootstrap_placeholders(rendered, placeholders)
    return rendered


def _assert_no_bootstrap_placeholders(content: str, placeholders: tuple[str, ...]) -> None:
    leftover = [name for name in placeholders if name in content]
    if leftover:
        raise ValueError(
            "unreplaced bootstrap placeholders in rendered pipeline: "
            f"{', '.join(leftover)}. Ensure bootstrap upload runs "
            ".buildkite/common/scripts/upload_pipeline.py --upload on the platform pipeline file.",
        )


def _split_pipeline_documents(text: str) -> tuple[str, str]:
    for separator in (DOC_SEP, "\r\n---\r\n", "\n---\r\n", "\r\n---\n"):
        if separator in text:
            head, tail = text.split(separator, 1)
            return head, tail
    return "", text


# --- Test pipeline (test-ready.yml, test-merge.yml) ---


@lru_cache(maxsize=1)
def _load_mirror_hardwares() -> dict[str, dict[str, Any]]:
    if not CI_MIRROR_HARDWARES_PATH.is_file():
        raise FileNotFoundError(f"missing CI mirror_hardwares registry: {CI_MIRROR_HARDWARES_PATH}")
    doc = yaml.safe_load(CI_MIRROR_HARDWARES_PATH.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"invalid CI mirror_hardwares registry: {CI_MIRROR_HARDWARES_PATH}")
    presets = doc.get("mirror_hardwares")
    if not isinstance(presets, dict):
        raise ValueError(f"mirror_hardwares must be a mapping in {CI_MIRROR_HARDWARES_PATH}")
    return presets


def _expand_mirror_hardwares(step: dict[str, Any]) -> dict[str, Any]:
    """Replace uploader-only ``mirror_hardwares`` with preset fields from ci_mirror_hardwares.yml."""
    hardware = step.get("mirror_hardwares")
    if hardware is None:
        return step

    if not isinstance(hardware, str) or not hardware.strip():
        raise ValueError(
            f"mirror_hardwares must be a non-empty string in step {_get_step_label(step)!r}",
        )

    preset = _load_mirror_hardwares().get(hardware)
    if preset is None:
        known = ", ".join(sorted(_load_mirror_hardwares()))
        raise ValueError(
            f"unknown mirror_hardwares {hardware!r} in step {_get_step_label(step)!r}; known: {known}",
        )

    if step.get("agents") is not None or step.get("plugins") is not None or step.get("image") is not None:
        raise ValueError(
            f"step {_get_step_label(step)!r} sets mirror_hardwares together with agents/plugins/image; "
            "use mirror_hardwares only",
        )

    expanded = copy.deepcopy(preset)
    return {key: value for key, value in step.items() if key != "mirror_hardwares"} | expanded


def _match_source_file(changed_files: list[str], prefixes: list[str]) -> bool:
    for path in changed_files:
        for prefix in prefixes:
            normalized = prefix.rstrip("/")
            if path == normalized or path.startswith(f"{normalized}/"):
                return True
    return False


def _get_step_label(step: dict[str, Any]) -> str:
    return str(step.get("group") or step.get("label") or "<step>")


def _process_test_steps(
    steps: list[Any],
    changed_files: list[str] | None,
) -> list[Any]:
    """Drop steps by ``source_file_dependencies`` when *changed_files* is set; always strip that field."""
    processed: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            processed.append(step)
            continue

        deps = step.get("source_file_dependencies")
        if deps is not None and not isinstance(deps, list):
            raise ValueError(
                f"source_file_dependencies must be a list in step {_get_step_label(step)!r}",
            )
        if changed_files is not None and deps is not None and not _match_source_file(changed_files, deps):
            _log(f"skip {_get_step_label(step)!r} (no changes under {deps})")
            continue

        nested = step.get("steps")
        if nested is not None:
            kept_nested = _process_test_steps(nested, changed_files)
            if changed_files is not None and not kept_nested:
                _log(f"omit empty group {_get_step_label(step)!r}")
                continue
            new_step = {key: value for key, value in step.items() if key != "source_file_dependencies"}
            new_step["steps"] = kept_nested
            processed.append(new_step)
            continue

        if deps is not None:
            processed.append(
                _expand_mirror_hardwares(
                    {key: value for key, value in step.items() if key != "source_file_dependencies"},
                ),
            )
        else:
            processed.append(_expand_mirror_hardwares(step))

    return processed


def _select_e2e_group_steps(steps: list[Any]) -> list[Any]:
    """Keep only top-level groups whose name contains ``E2E_GROUP_MARKER``."""
    selected = [
        step
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("group"), str) and E2E_GROUP_MARKER in step["group"]
    ]
    if not selected:
        _log(f"no group matching {E2E_GROUP_MARKER!r} found")
    else:
        _log(f"keep {len(selected)} group(s) matching {E2E_GROUP_MARKER!r}")
    return selected


def _render_test_pipeline(
    doc: dict[str, Any],
    changed_files: list[str] | None,
    *,
    e2e_only: bool = False,
) -> dict[str, Any]:
    """Filter steps by PR diff and strip uploader-only ``source_file_dependencies`` metadata."""
    steps = doc.get("steps")
    if not isinstance(steps, list):
        return doc
    if e2e_only:
        steps = _select_e2e_group_steps(steps)
    steps = _process_test_steps(steps, changed_files)
    return {**doc, "steps": steps}


# --- Entry (read file → bootstrap or test render → YAML string) ---


def _render_pipeline(
    path: Path,
    *,
    force_all: bool = False,
    e2e_only: bool = False,
) -> str:
    text = path.read_text(encoding="utf-8")
    ctx = resolve_ci_context_from_git()
    decision = ctx.decision
    if _is_bootstrap_pipeline(text) or force_all or e2e_only:
        changed_files = None
    else:
        changed_files = ctx.changed_files

    if _is_bootstrap_pipeline(text):
        return _render_bootstrap_pipeline(
            text,
            decision=decision,
            path=path,
        )

    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(f"invalid pipeline YAML: {path}")

    doc = _render_test_pipeline(doc, changed_files, e2e_only=e2e_only)
    return yaml.safe_dump(doc, sort_keys=False)


def _upload_to_buildkite(content: str) -> None:
    subprocess.run(
        ["buildkite-agent", "pipeline", "upload"],
        input=content,
        text=True,
        check=True,
    )


# --- CLI ---


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pipeline",
        nargs="?",
        default=".buildkite/cuda/pipeline.yml",
        help="Pipeline YAML path (default: .buildkite/cuda/pipeline.yml)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Pipe rendered YAML to buildkite-agent pipeline upload",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--all",
        action="store_true",
        help="Keep all steps (disable diff-aware skipping)",
    )
    mode.add_argument(
        "--e2e",
        action="store_true",
        help="Keep only the E2E Test group",
    )
    args = parser.parse_args()

    path = Path(args.pipeline)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        _log(f"missing pipeline file: {path}")
        return 1

    rendered = _render_pipeline(path, force_all=args.all, e2e_only=args.e2e)
    if args.upload:
        _upload_to_buildkite(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
