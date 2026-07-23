#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render and optionally upload Buildkite pipeline YAML with diff-aware logic.

Bootstrap mode (``bootstrap-upload-steps.yml``):
  - Hook uploads the entry YAML (``pipeline.yml`` / ``pipeline-npu.yml``) with one step that runs
    ``upload_pipeline.py --upload <platform>/bootstrap-upload-steps.yml``.
  - Injects ``if`` by step ``key`` from skip-ci and uploads child steps (image build, L2–L5 upload).
  - Detect docs-only, pytest skip-mark-only, or combined skip-ci from git diff.
  - When only CI level YAML changes, enable **L2/L3** upload steps for affected levels only.

Test pipeline mode (e.g. test-merge.yml):
  - Drop steps whose ``source_file_dependencies`` do not match changed files.
  - Expand uploader-only ``mirror_hardwares: l4_1`` into ``agents`` (+ optional ``image``
    for NPU) + ``plugins`` (see ci_mirror_hardwares.yml).

Usage:
  python3 upload_pipeline.py [--upload] [--all | --e2e] <pipeline.yml>

Requires PyYAML (``pip install pyyaml``); installs it automatically when missing.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pyyaml"],
        check=True,
    )
    import yaml

from skip_ci import (
    ROOT,
    resolve_ci_context_from_git,
)

# --- Constants ---

LOG = "upload_pipeline"
BOOTSTRAP_STEPS_FILENAME = "bootstrap-upload-steps.yml"
BOOTSTRAP_IMAGE_BUILD_KEYS = frozenset({"image-build", "image-build-a2", "image-build-a3"})
BOOTSTRAP_UPLOAD_IF_KEYS = {
    "upload-ready-pipeline": "ready",
    "upload-merge-pipeline": "merge",
    "upload-nightly-pipeline": "nightly",
    "upload-weekly-pipeline": "weekly",
}
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


# --- Bootstrap pipeline (bootstrap-upload-steps.yml) ---


def _get_bootstrap_platform(path: Path) -> str:
    parts = path.as_posix().split("/")
    return "npu" if "npu" in parts else "cuda"


def _load_bootstrap_steps(path: Path) -> str:
    if path.name != BOOTSTRAP_STEPS_FILENAME:
        raise ValueError(f"expected {BOOTSTRAP_STEPS_FILENAME}, got {path.name}")
    return path.read_text(encoding="utf-8")


def _format_bootstrap_if(expr: str) -> str:
    """Return a Buildkite ``if`` string. Buildkite rejects YAML bool ``if`` values."""
    if expr in ("true", "false"):
        return expr
    return f"({expr})"


def _compute_bootstrap_if_exprs(*, decision, platform: str) -> dict[str, str]:
    disabled = "false"
    nightly_main = 'build.branch == "main" && build.env("NIGHTLY") == "1"'
    nightly_only = NPU_NIGHTLY_ONLY if platform == "npu" else CUDA_NIGHTLY_ONLY

    if platform == "npu":
        ready_pr = (
            'build.branch != "main" && ('
            'build.pull_request.labels includes "npu-test" || '
            'build.pull_request.labels includes "ready"'
            ")"
        )
        nightly_label_if = nightly_only
        weekly_label_if = disabled
        merge_base = disabled
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
        merge_base = f"({nightly_main}) || (({merge_main}) || ({merge_pr}))"

    ready_base = f"({nightly_main}) || ({ready_pr})"
    weekly_main = 'build.branch == "main" && build.env("WEEKLY") == "1"'

    if decision.skip_all:
        # Docs / skip-mark only: no PR-label escape hatch. Main scheduled
        # NIGHTLY=1 still runs L4 + L2/L3 (--e2e); WEEKLY=1 still runs L5.
        image_expr = f"({nightly_main}) || ({weekly_main})" if platform == "cuda" else nightly_main
        ready_expr = nightly_main
        merge_expr = nightly_main if platform == "cuda" else disabled
        nightly_expr = nightly_main
        weekly_expr = weekly_main if platform == "cuda" else disabled
    elif decision.skip_l2_l3:
        l2_enabled = decision.is_run("npu", "l2") if platform == "npu" else decision.is_run("cuda", "l2")
        l3_enabled = platform == "cuda" and decision.is_run("cuda", "l3")

        ready_expr = ready_base if l2_enabled else disabled
        merge_expr = merge_base if l3_enabled else disabled
        nightly_expr = nightly_label_if
        weekly_expr = weekly_label_if if platform == "cuda" else disabled

        image_parts = [f"({nightly_label_if})"]
        if platform == "cuda":
            image_parts.append(f"({weekly_label_if})")
        if l2_enabled:
            image_parts.insert(0, f"({ready_base})")
        if l3_enabled:
            image_parts.insert(1 if l2_enabled else 0, f"({merge_base})")
        image_expr = " || ".join(image_parts)
    else:
        image_expr = "true"
        ready_expr = ready_base
        merge_expr = merge_base if platform == "cuda" else disabled
        nightly_expr = nightly_label_if
        weekly_expr = weekly_label_if if platform == "cuda" else disabled

    return {
        "image": _format_bootstrap_if(image_expr),
        "ready": _format_bootstrap_if(ready_expr),
        "merge": _format_bootstrap_if(merge_expr),
        "nightly": _format_bootstrap_if(nightly_expr),
        "weekly": _format_bootstrap_if(weekly_expr),
    }


def _apply_bootstrap_if(steps: list[Any], if_exprs: dict[str, str]) -> list[Any]:
    """Inject ``if`` by step key; drop steps that are unconditionally disabled."""
    kept: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            kept.append(step)
            continue
        nested = step.get("steps")
        if isinstance(nested, list):
            nested_kept = _apply_bootstrap_if(nested, if_exprs)
            if nested_kept:
                kept.append({**step, "steps": nested_kept})
            continue
        key = step.get("key")
        if key in BOOTSTRAP_IMAGE_BUILD_KEYS:
            step["if"] = if_exprs["image"]
        elif key in BOOTSTRAP_UPLOAD_IF_KEYS:
            step["if"] = if_exprs[BOOTSTRAP_UPLOAD_IF_KEYS[key]]
        if_expr = step.get("if")
        if if_expr == "false":
            _log(f"omit disabled bootstrap step {key!r}")
            continue
        if if_expr == "true":
            # Unconditional step: omit ``if`` (Buildkite requires string ``if``, not YAML bool).
            step.pop("if", None)
        kept.append(step)
    return kept


def _render_bootstrap_pipeline(
    steps_yaml: str,
    *,
    decision,
    path: Path,
) -> str:
    """Load bootstrap steps YAML and inject ``if`` expressions from skip-ci decision."""
    doc = yaml.safe_load(steps_yaml)
    if not isinstance(doc, dict):
        raise ValueError(f"invalid bootstrap steps YAML for {path}")
    steps = doc.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"bootstrap steps YAML must contain steps: list in {path}")

    platform = _get_bootstrap_platform(path)
    if_exprs = _compute_bootstrap_if_exprs(decision=decision, platform=platform)
    doc["steps"] = _apply_bootstrap_if(steps, if_exprs)
    return yaml.safe_dump(doc, sort_keys=False)


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
    if path.name == BOOTSTRAP_STEPS_FILENAME:
        ctx = resolve_ci_context_from_git()
        continuation = _load_bootstrap_steps(path)
        return _render_bootstrap_pipeline(
            continuation,
            decision=ctx.decision,
            path=path,
        )

    text = path.read_text(encoding="utf-8")
    ctx = resolve_ci_context_from_git()
    if force_all or e2e_only:
        changed_files = None
    else:
        changed_files = ctx.changed_files

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
        default=".buildkite/cuda/bootstrap-upload-steps.yml",
        help="Pipeline YAML path (default: .buildkite/cuda/bootstrap-upload-steps.yml)",
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
