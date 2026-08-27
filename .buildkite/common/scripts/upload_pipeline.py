#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render and optionally upload Buildkite pipeline YAML with diff-aware logic.

Bootstrap mode (``bootstrap-upload-steps.yml``):
  - Hook uploads the entry YAML (``pipeline.yml`` / ``pipeline-npu.yml``) with one step that runs
    ``upload_pipeline.py --upload <platform>/bootstrap-upload-steps.yml``.
  - Injects ``if`` by step ``key`` from skip-ci and uploads child steps (image build, L2–L5 upload).
  - Detect docs-only, pytest skip-mark-only, or combined skip-ci from git diff.
  - When only CI level YAML changes, enable **L2/L3** upload steps for affected levels only.

Test pipeline mode (e.g. test-level3.yml):
  - Drop steps whose ``source_file_dependencies`` do not match changed files.
  - Expand uploader-only ``mirror_hardwares`` into ``agents`` (+ optional ``image``
    for NPU) + ``plugins`` (see ci_mirror_hardwares.yml).
  - ``mirror_hardwares`` may be a GPU count (``2``) or a preset string (``l4_1``)::

      mirror_hardwares: 2
      # or
      mirror_hardwares: h100_2

    Count form composes ``{chip}_{count}`` from pytest ``-m`` SKU markers
    (``H100``, ``L4``, ``B200``). One SKU → that chip (``MIRROR_HW`` is ignored).
    Several SKUs → unset ``MIRROR_HW`` picks L4 then H100; ``MIRROR_HW=b200``
    matches ``B200`` in ``-m`` (otherwise the step is skipped). ``-m`` is not
    rewritten: B200 collection must already be in the YAML and on the tests.
    ``MIRROR_HW`` must be empty or ``b200`` (case-insensitive); unknown values
    (e.g. ``b20o``) fail the upload. A CUDA preset string such as ``h100_4``
    is omitted when ``MIRROR_HW=b200``.

Usage:
  python3 upload_pipeline.py [--upload] [--all | --e2e] <pipeline.yml>

Requires PyYAML (``pip install pyyaml``); installs it automatically when missing.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
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
    "upload-level2-pipeline": "level2",
    "upload-level3-pipeline": "level3",
    "upload-level4-pipeline": "level4",
    "upload-level5-pipeline": "level5",
}
E2E_GROUP_MARKER = "E2E Test"
CI_MIRROR_HARDWARES_PATH = ROOT / ".buildkite/common/ci_mirror_hardwares.yml"

# Bootstrap Buildkite ``if`` expressions.
# ``*_MAIN_IF``: main + env schedule. ``*_LABEL_IF``: PR label (and/or composed with MAIN).
# ``*_UPLOAD_IF``: full gate for uploading that child pipeline.
NIGHTLY_MAIN_IF = 'build.branch == "main" && build.env("NIGHTLY") == "1"'
NIGHTLY_LABEL_IF = (
    f"({NIGHTLY_MAIN_IF}) || "
    '(build.branch != "main" && ('
    'build.pull_request.labels includes "nightly-test" || '
    'build.pull_request.labels includes "omni-test" || '
    'build.pull_request.labels includes "tts-test" || '
    'build.pull_request.labels includes "diffusion-x2iat-test" || '
    'build.pull_request.labels includes "diffusion-x2v-test"'
    "))"
)
WEEKLY_E2E_IF = 'build.branch == "main" && build.env("WEEKLY") == "1"'
WEEKLY_MAIN_IF = 'build.branch == "main" && (build.env("WEEKLY") == "1" || build.env("NON_CRITICAL") == "1")'
WEEKLY_LABEL_IF = f'({WEEKLY_MAIN_IF}) || (build.branch != "main" && build.pull_request.labels includes "weekly-test")'
LEVEL2_LABEL_IF = 'build.branch != "main" && build.pull_request.labels includes "ready"'
LEVEL3_LABEL_IF = 'build.branch != "main" && build.pull_request.labels includes "merge-test"'
LEVEL3_MAIN_IF = (
    'build.branch == "main" && build.env("NIGHTLY") != "1" && '
    'build.env("WEEKLY") != "1" && build.env("NON_CRITICAL") != "1"'
)
LEVEL2_UPLOAD_IF = f"({WEEKLY_E2E_IF}) || ({LEVEL2_LABEL_IF})"
LEVEL3_UPLOAD_IF = f"({WEEKLY_E2E_IF}) || (({LEVEL3_MAIN_IF}) || ({LEVEL3_LABEL_IF}))"
BOOTSTRAP_DISABLED_IF = "false"
BOOTSTRAP_ENABLED_IF = "true"


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
    if platform == "npu":
        level2_upload = LEVEL2_LABEL_IF
        level3_upload = BOOTSTRAP_DISABLED_IF
        level5_label_if = BOOTSTRAP_DISABLED_IF
    else:
        level2_upload = LEVEL2_UPLOAD_IF
        level3_upload = LEVEL3_UPLOAD_IF
        level5_label_if = WEEKLY_LABEL_IF

    if decision.skip_all:
        # Docs / skip-mark only: no PR-label escape hatch. Main scheduled
        # NIGHTLY=1 still runs L4; WEEKLY=1 / NON_CRITICAL=1 still run L5.
        # main+WEEKLY=1 also uploads L2/L3 (those steps then pass --e2e).
        image_expr = f"({NIGHTLY_MAIN_IF}) || ({WEEKLY_MAIN_IF})" if platform == "cuda" else NIGHTLY_MAIN_IF
        level2_expr = WEEKLY_E2E_IF if platform == "cuda" else BOOTSTRAP_DISABLED_IF
        level3_expr = WEEKLY_E2E_IF if platform == "cuda" else BOOTSTRAP_DISABLED_IF
        level4_expr = NIGHTLY_MAIN_IF
        level5_expr = WEEKLY_MAIN_IF if platform == "cuda" else BOOTSTRAP_DISABLED_IF
    elif decision.skip_l2_l3:
        l2_enabled = decision.is_run("npu", "l2") if platform == "npu" else decision.is_run("cuda", "l2")
        l3_enabled = platform == "cuda" and decision.is_run("cuda", "l3")

        level2_expr = level2_upload if l2_enabled else BOOTSTRAP_DISABLED_IF
        level3_expr = level3_upload if l3_enabled else BOOTSTRAP_DISABLED_IF
        level4_expr = NIGHTLY_LABEL_IF
        level5_expr = level5_label_if if platform == "cuda" else BOOTSTRAP_DISABLED_IF

        image_parts = [f"({NIGHTLY_LABEL_IF})"]
        if platform == "cuda":
            image_parts.append(f"({level5_label_if})")
        if l2_enabled:
            image_parts.insert(0, f"({level2_upload})")
        if l3_enabled:
            image_parts.insert(1 if l2_enabled else 0, f"({level3_upload})")
        image_expr = " || ".join(image_parts)
    else:
        image_expr = BOOTSTRAP_ENABLED_IF
        level2_expr = level2_upload
        level3_expr = level3_upload if platform == "cuda" else BOOTSTRAP_DISABLED_IF
        level4_expr = NIGHTLY_LABEL_IF
        level5_expr = level5_label_if if platform == "cuda" else BOOTSTRAP_DISABLED_IF

    return {
        "image": _format_bootstrap_if(image_expr),
        "level2": _format_bootstrap_if(level2_expr),
        "level3": _format_bootstrap_if(level3_expr),
        "level4": _format_bootstrap_if(level4_expr),
        "level5": _format_bootstrap_if(level5_expr),
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


# --- Test pipeline (test-level2.yml, test-level3.yml) ---


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


# Longer tokens first so ``h100`` is not matched as a prefix of ``h200``.
_CUDA_MIRROR_CHIPS = ("b200", "h200", "h100", "l4")
_SUPPORTED_MIRROR_HW_SELECTORS = frozenset({"b200"})
# Unset MIRROR_HW + both H100 and L4 markers: prefer L4, then H100.
_COUNT_UNSET_CHIP_PREFERENCE = ("l4", "h100")
_PYTEST_MARKER_ARG = re.compile(r"-m\s+(?:\"([^\"]*)\"|'([^']*)'|(\S+))")
_COUNT_PRESET_RANGE = range(1, 5)


def _english_or(names: Any) -> str:
    items = [str(name) for name in names]
    if len(items) <= 2:
        return " or ".join(items)
    return f"{', '.join(items[:-1])}, or {items[-1]}"


def _get_mirror_hw_selector() -> str:
    """Return lowercase ``MIRROR_HW``, or empty to keep string presets / marker chips.

    A set value must be ``b200`` (case-insensitive). Unknown values fail closed
    so a typo cannot silently drop the CUDA pipeline.
    """
    return "b200"  # TODO: delete this line after B200 CI debug
    selector = os.environ.get("MIRROR_HW", "").strip().lower()
    if not selector:
        return ""
    if selector not in _SUPPORTED_MIRROR_HW_SELECTORS:
        raise ValueError(
            f"unsupported MIRROR_HW={selector!r}; expected empty or "
            f"{_english_or(f'{s!r}' for s in sorted(_SUPPORTED_MIRROR_HW_SELECTORS))}"
        )
    return selector


def _pytest_marker_chips(commands: Any) -> set[str]:
    """Positive CUDA SKU tokens in pytest ``-m`` (``H100`` from ``h100`` in ``_CUDA_MIRROR_CHIPS``)."""

    def _command_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "\n".join(_command_text(part) for part in value)
        return str(value)

    not_sku = re.compile(
        r"\bnot\s+(" + "|".join(re.escape(chip.upper()) for chip in _CUDA_MIRROR_CHIPS) + r")\b",
    )
    chips: set[str] = set()
    for match in _PYTEST_MARKER_ARG.finditer(_command_text(commands)):
        expr = match.group(1) or match.group(2) or match.group(3) or ""
        stripped = not_sku.sub(" ", expr)
        for chip in _CUDA_MIRROR_CHIPS:
            if re.search(rf"\b{re.escape(chip.upper())}\b", stripped):
                chips.add(chip)
    return chips


def _resolve_mirror_hardware_name(hardware: Any, *, step: dict[str, Any], step_label: str) -> str | None:
    """Map ``mirror_hardwares`` to a ``ci_mirror_hardwares.yml`` preset, or None to skip.

    Count (``2`` / ``"2"``): compose ``{chip}_{count}`` from pytest ``-m`` SKUs.
    One SKU uses that chip (``MIRROR_HW`` ignored). Several SKUs: unset prefers
    L4 then H100; ``b200`` must appear in ``-m``. String (``h100_2``): omitted
    when ``MIRROR_HW`` names a different CUDA chip. NPU strings ignore it.
    """
    # bool is a subclass of int; reject YAML ``true`` before treating it as a count.
    if isinstance(hardware, bool):
        raise ValueError(f"mirror_hardwares must not be a boolean in step {step_label!r}")

    count: int | None
    if isinstance(hardware, int):
        count = hardware
    elif isinstance(hardware, str) and hardware.strip().isdigit():
        count = int(hardware.strip())
    else:
        count = None

    if count is not None:
        if count not in _COUNT_PRESET_RANGE:
            raise ValueError(
                f"mirror_hardwares GPU count in step {step_label!r} must be 1–4, got {count}",
            )
        chips = _pytest_marker_chips(step.get("commands"))
        if not chips:
            raise ValueError(
                f"mirror_hardwares: {count} in step {step_label!r} needs a pytest -m SKU marker "
                f"({_english_or(chip.upper() for chip in _CUDA_MIRROR_CHIPS)}), "
                f"or an explicit preset string",
            )
        known = _load_mirror_hardwares()
        selector = _get_mirror_hw_selector()

        def _require_preset(chip: str) -> str:
            chosen = f"{chip}_{count}"
            if chosen not in known:
                raise ValueError(
                    f"mirror_hardwares: {count} in step {step_label!r} has no preset {chosen!r}",
                )
            return chosen

        if len(chips) == 1:
            chip = next(iter(chips))
            chosen = _require_preset(chip)
            extra = f" (ignored MIRROR_HW={selector!r})" if selector and selector != chip else ""
            _log(f"{step_label}: mirror_hardwares count {count} → single marker {chip}{extra} → {chosen!r}")
            return chosen

        if selector:
            if selector not in chips:
                _log(
                    f"skip {step_label}: MIRROR_HW={selector!r} is not among -m SKUs {sorted(chips)}",
                )
                return None
            chosen = _require_preset(selector)
            _log(f"{step_label}: mirror_hardwares count {count} → MIRROR_HW={selector!r} → {chosen!r}")
            return chosen

        tried: list[str] = []
        for chip in _COUNT_UNSET_CHIP_PREFERENCE:
            if chip not in chips:
                continue
            chosen = f"{chip}_{count}"
            tried.append(chosen)
            if chosen not in known:
                continue
            _log(
                f"{step_label}: mirror_hardwares count {count} → marker {chip} "
                f"(multiple SKU markers; preferred "
                f"{' then '.join(c.upper() for c in _COUNT_UNSET_CHIP_PREFERENCE)}) → {chosen!r}",
            )
            return chosen
        raise ValueError(
            f"mirror_hardwares: {count} in step {step_label!r} found SKU markers {sorted(chips)} "
            f"but no {'/'.join(c.upper() for c in _COUNT_UNSET_CHIP_PREFERENCE)} preset "
            f"among {tried or list(_COUNT_UNSET_CHIP_PREFERENCE)}",
        )

    if isinstance(hardware, str):
        name = hardware.strip()
        if not name:
            raise ValueError(f"mirror_hardwares must be a non-empty string in step {step_label!r}")
        selector = _get_mirror_hw_selector()
        token = name.lower()
        chip = next(
            (c for c in _CUDA_MIRROR_CHIPS if token == c or token.startswith(f"{c}_")),
            None,
        )
        if selector and chip is not None and chip != selector:
            _log(
                f"skip {step_label}: preset {name!r} is {chip} hardware; MIRROR_HW={selector!r}",
            )
            return None
        return name

    raise ValueError(
        f"mirror_hardwares must be a GPU count or a preset string in step {step_label!r}",
    )


def _expand_mirror_hardwares(step: dict[str, Any]) -> dict[str, Any] | None:
    """Replace uploader-only ``mirror_hardwares`` with preset fields from ci_mirror_hardwares.yml.

    Returns None when the current ``MIRROR_HW`` selector does not apply to this step.
    """
    hardware = step.get("mirror_hardwares")
    if hardware is None:
        return step

    step_label = _get_step_label(step)
    if step.get("agents") is not None or step.get("plugins") is not None or step.get("image") is not None:
        raise ValueError(
            f"step {step_label!r} sets mirror_hardwares together with agents/plugins/image; use mirror_hardwares only",
        )

    preset_name = _resolve_mirror_hardware_name(hardware, step=step, step_label=step_label)
    if preset_name is None:
        return None

    preset = _load_mirror_hardwares().get(preset_name)
    if preset is None:
        known = ", ".join(sorted(_load_mirror_hardwares()))
        raise ValueError(
            f"unknown mirror_hardwares {preset_name!r} in step {step_label!r}; known: {known}",
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
            if not kept_nested:
                _log(f"omit empty group {_get_step_label(step)!r}")
                continue
            new_step = {key: value for key, value in step.items() if key != "source_file_dependencies"}
            new_step["steps"] = kept_nested
            processed.append(new_step)
            continue

        leaf = {key: value for key, value in step.items() if key != "source_file_dependencies"}
        expanded = _expand_mirror_hardwares(leaf)
        if expanded is None:
            continue
        processed.append(expanded)

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
    # Validate MIRROR_HW once up front. Per-step skip must not run for typos
    # (e.g. b20o), or a pipeline can shrink to leftover CPU-only steps and
    # still upload successfully.
    _get_mirror_hw_selector()
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
    changed_files = None if force_all or e2e_only else ctx.changed_files

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
