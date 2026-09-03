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

Test pipeline mode (e.g. test-merge.yml):
  - Drop steps whose ``source_file_dependencies`` do not match changed files.
  - Expand uploader-only ``mirror_hardwares`` into ``agents`` (+ optional ``image``
    for NPU) + ``plugins`` (see ci_mirror_hardwares.yml).
  - Omit ``mirror_hardwares`` to compose ``{chip}_{n}`` from pytest ``-m`` SKU
    markers plus ``cards_n`` (max of several positives; ``not cards_1`` with no
    positive ``cards_*`` uses that chip's highest existing preset). An explicit
    preset string (``h100_2``) is used as-is (marks are ignored). Names that
    are not keys in ``ci_mirror_hardwares.yml`` fail the upload.

    Inferred chip: unset ``MIRROR_HW`` matches ``H100`` or ``L4`` in ``-m``
    (both → H100); no match skips the step. ``MIRROR_HW=b200`` matches
    ``B200`` in ``-m`` (otherwise skipped). ``-m`` is not rewritten.
    ``MIRROR_HW`` must be empty or ``b200``; unknown values fail the upload.
    A CUDA preset string such as ``h100_4`` is omitted when ``MIRROR_HW=b200``.

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
from typing import Any, Literal

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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers.mark import (  # noqa: E402
    get_skus_for_platform,
    get_supported_card_counts,
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
READY_LABEL_IF = 'build.branch != "main" && build.pull_request.labels includes "ready"'
MERGE_LABEL_IF = 'build.branch != "main" && build.pull_request.labels includes "merge-test"'
MERGE_MAIN_IF = (
    'build.branch == "main" && build.env("NIGHTLY") != "1" && '
    'build.env("WEEKLY") != "1" && build.env("NON_CRITICAL") != "1"'
)
READY_UPLOAD_IF = f"({WEEKLY_E2E_IF}) || ({READY_LABEL_IF})"
MERGE_UPLOAD_IF = f"({WEEKLY_E2E_IF}) || (({MERGE_MAIN_IF}) || ({MERGE_LABEL_IF}))"
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
        ready_upload = READY_LABEL_IF
        merge_upload = BOOTSTRAP_DISABLED_IF
        weekly_label_if = BOOTSTRAP_DISABLED_IF
    else:
        ready_upload = READY_UPLOAD_IF
        merge_upload = MERGE_UPLOAD_IF
        weekly_label_if = WEEKLY_LABEL_IF

    if decision.skip_all:
        # Docs / skip-mark only: no PR-label escape hatch. Main scheduled
        # NIGHTLY=1 still runs L4; WEEKLY=1 / NON_CRITICAL=1 still run L5.
        # main+WEEKLY=1 also uploads L2/L3 (those steps then pass --e2e).
        image_expr = f"({NIGHTLY_MAIN_IF}) || ({WEEKLY_MAIN_IF})" if platform == "cuda" else NIGHTLY_MAIN_IF
        ready_expr = WEEKLY_E2E_IF if platform == "cuda" else BOOTSTRAP_DISABLED_IF
        merge_expr = WEEKLY_E2E_IF if platform == "cuda" else BOOTSTRAP_DISABLED_IF
        nightly_expr = NIGHTLY_MAIN_IF
        weekly_expr = WEEKLY_MAIN_IF if platform == "cuda" else BOOTSTRAP_DISABLED_IF
    elif decision.skip_l2_l3:
        l2_enabled = decision.is_run("npu", "l2") if platform == "npu" else decision.is_run("cuda", "l2")
        l3_enabled = platform == "cuda" and decision.is_run("cuda", "l3")

        ready_expr = ready_upload if l2_enabled else BOOTSTRAP_DISABLED_IF
        merge_expr = merge_upload if l3_enabled else BOOTSTRAP_DISABLED_IF
        nightly_expr = NIGHTLY_LABEL_IF
        weekly_expr = weekly_label_if if platform == "cuda" else BOOTSTRAP_DISABLED_IF

        image_parts = [f"({NIGHTLY_LABEL_IF})"]
        if platform == "cuda":
            image_parts.append(f"({weekly_label_if})")
        if l2_enabled:
            image_parts.insert(0, f"({ready_upload})")
        if l3_enabled:
            image_parts.insert(1 if l2_enabled else 0, f"({merge_upload})")
        image_expr = " || ".join(image_parts)
    else:
        image_expr = BOOTSTRAP_ENABLED_IF
        ready_expr = ready_upload
        merge_expr = merge_upload if platform == "cuda" else BOOTSTRAP_DISABLED_IF
        nightly_expr = NIGHTLY_LABEL_IF
        weekly_expr = weekly_label_if if platform == "cuda" else BOOTSTRAP_DISABLED_IF

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


@lru_cache(maxsize=1)
def _cuda_mirror_chips() -> tuple[str, ...]:
    """Lowercase CUDA SKUs that have at least one ``{chip}_{n}`` preset.

    Longer tokens first so ``h100`` is not matched as a prefix of ``h200``.
    """
    skus = get_skus_for_platform("cuda")
    known = _load_mirror_hardwares()
    chips = {
        sku.lower() for sku in skus if any(name == sku.lower() or name.startswith(f"{sku.lower()}_") for name in known)
    }
    return tuple(sorted(chips, key=lambda chip: (-len(chip), chip)))


def _read_cards_marks(expr: str) -> tuple[set[int], bool]:
    """Parse registered ``cards_n`` / ``not cards_n`` from a pytest ``-m`` expr.

    Returns positive card counts and whether any ``not cards_*`` is present.
    """
    counts = get_supported_card_counts()
    alt = "|".join(str(n) for n in sorted(counts, key=lambda n: (-len(str(n)), -n))) if counts else "(?!)"
    positives: set[int] = set()
    has_not_cards = False
    for match in re.finditer(rf"\b(?:(not)\s+)?cards_({alt})\b", expr):
        if match.group(1):
            has_not_cards = True
        else:
            positives.add(int(match.group(2)))
    return positives, has_not_cards


def _read_hardware_marks(expr: str) -> set[str]:
    """Return lowercase positive CUDA SKUs that have a mirror preset.

    ``not H100`` is ignored. SKUs with no ``{chip}_*`` preset (e.g. H200) are dropped.
    """
    skus = sorted(get_skus_for_platform("cuda"), key=lambda s: (-len(s), s))
    if not skus:
        return set()
    alt = "|".join(re.escape(sku) for sku in skus)
    mirror = set(_cuda_mirror_chips())
    chips: set[str] = set()
    for match in re.finditer(rf"\b(?:(not)\s+)?({alt})\b", expr):
        if match.group(1):
            continue
        chip = match.group(2).lower()
        if chip in mirror:
            chips.add(chip)
    return chips


# CI policy: not pytest-mark registry.
_SUPPORTED_MIRROR_HW_SELECTORS = frozenset({"b200"})
# Unset MIRROR_HW: match H100 then L4 in -m (skip if neither).
_DEFAULT_INFER_CHIPS = ("h100", "l4")
_PYTEST_MARKER_ARG = re.compile(r"-m\s+(?:\"([^\"]*)\"|'([^']*)'|(\S+))")
_CARDS_CHIP_MAX: Literal["max"] = "max"


def _get_mirror_hw_selector() -> str:
    """Return lowercase ``MIRROR_HW``, or empty to keep string presets / marker chips.

    A set value must be ``b200`` (case-insensitive). Unknown values fail closed
    so a typo cannot silently drop the CUDA pipeline.
    """
    selector = os.environ.get("MIRROR_HW", "").strip().lower()
    if not selector:
        return ""
    if selector not in _SUPPORTED_MIRROR_HW_SELECTORS:
        allowed = " or ".join(repr(s) for s in sorted(_SUPPORTED_MIRROR_HW_SELECTORS))
        raise ValueError(f"unsupported MIRROR_HW={selector!r}; expected empty or {allowed}")
    return selector


def _read_pytest_marks(commands: Any) -> tuple[set[str], int | Literal["max"] | None]:
    """Read pytest ``-m`` from step commands: CUDA SKU chips and cards count.

    *chips* are lowercase SKUs from positive markers (``not H100`` is ignored).
    *cards* is the max positive ``cards_n``, ``"max"`` when only ``not cards_*``
    is present, or ``None``.
    """
    chunks: list[str] = []

    def _collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, list):
            for part in value:
                _collect(part)
        else:
            chunks.append(str(value))

    _collect(commands)
    text = "\n".join(chunks)

    chips: set[str] = set()
    positives: set[int] = set()
    has_not_cards = False
    for match in _PYTEST_MARKER_ARG.finditer(text):
        expr = match.group(1) or match.group(2) or match.group(3) or ""
        chips |= _read_hardware_marks(expr)
        card_counts, not_cards = _read_cards_marks(expr)
        positives.update(card_counts)
        has_not_cards = has_not_cards or not_cards

    if positives:
        cards: int | Literal["max"] | None = max(positives)
    elif has_not_cards:
        cards = _CARDS_CHIP_MAX
    else:
        cards = None
    return chips, cards


def _compose_mirror_hardware_name(
    chips: set[str],
    cards: int | Literal["max"] | None,
    *,
    step_label: str,
) -> str | None:
    """Assemble ``{chip}_{n}`` from pytest ``-m`` chips/cards, or None to skip.

    Unset ``MIRROR_HW``: ``H100`` if in ``-m``, else ``L4``, else skip.
    Set ``MIRROR_HW``: that chip if in ``-m``, else skip.
    ``cards`` is a registered ``cards_n``, ``"max"`` for that chip's highest
    existing preset, or None (error after a chip matched — no ``cards_*``).
    """
    selector = _get_mirror_hw_selector()
    if selector:
        chip = selector if selector in chips else None
        skip_reason = f"MIRROR_HW={selector!r} not in -m SKUs {sorted(chips)}"
    else:
        chip = next((c for c in _DEFAULT_INFER_CHIPS if c in chips), None)
        skip_reason = f"no H100 or L4 in -m SKUs {sorted(chips)}"
    if chip is None:
        _log(f"skip {step_label}: {skip_reason}")
        return None
    if cards is None:
        raise ValueError(
            f"step {step_label!r} has a pytest -m SKU marker but no cards_* "
            f"(or not cards_*); add cards_n or set mirror_hardwares to force a preset",
        )

    count_source = "not cards_*" if cards == _CARDS_CHIP_MAX else f"cards_{cards}"
    count = None if cards == _CARDS_CHIP_MAX else cards
    registered = get_supported_card_counts()
    if count is not None and count not in registered:
        supported = ", ".join(f"cards_{n}" for n in sorted(registered)) or "(none)"
        raise ValueError(
            f"{count_source} in step {step_label!r} is not a registered cards_* mark; supported: {supported}",
        )
    known = _load_mirror_hardwares()
    if count is None:
        available = [n for n in sorted(registered) if f"{chip}_{n}" in known]
        if not available:
            raise ValueError(
                f"{count_source} in step {step_label!r} has no CUDA count presets for {chip}",
            )
        count = max(available)
    chosen = f"{chip}_{count}"
    if chosen not in known:
        raise ValueError(
            f"{count_source} in step {step_label!r} has no preset {chosen!r}",
        )
    _log(f"{step_label}: {count_source} → {chosen!r}")
    return chosen


def _resolve_mirror_hardware_name(hardware: Any, *, step_label: str) -> str | None:
    """Map ``mirror_hardwares`` to a ``ci_mirror_hardwares.yml`` preset, or None to skip.

    The value is a preset name (``h100_2``). Unknown names fail later when the
    registry is loaded. CUDA names are omitted when ``MIRROR_HW`` names a
    different chip; NPU strings ignore ``MIRROR_HW``.
    """
    if isinstance(hardware, bool):
        raise ValueError(f"mirror_hardwares must not be a boolean in step {step_label!r}")
    if isinstance(hardware, str):
        name = hardware.strip()
    elif isinstance(hardware, int):
        name = str(hardware)
    else:
        raise ValueError(
            f"mirror_hardwares must be a preset string in step {step_label!r}",
        )
    if not name:
        raise ValueError(f"mirror_hardwares must be a non-empty string in step {step_label!r}")

    selector = _get_mirror_hw_selector()
    token = name.lower()
    chip = next(
        (c for c in _cuda_mirror_chips() if token == c or token.startswith(f"{c}_")),
        None,
    )
    if selector and chip is not None and chip != selector:
        _log(
            f"skip {step_label}: preset {name!r} is {chip} hardware; MIRROR_HW={selector!r}",
        )
        return None
    return name


def _expand_mirror_hardwares(step: dict[str, Any]) -> dict[str, Any] | None:
    """Replace ``mirror_hardwares`` (or infer it from ``-m``) with preset agents/plugins.

    Explicit preset string: used as-is; pytest marks are ignored.
    Omitted key: match H100/L4 (or ``MIRROR_HW``) in ``-m``, then compose
    ``{chip}_{n}`` from ``cards_n``. No match skips the step. Steps with no
    SKU/cards and no ``mirror_hardwares`` are left unchanged (CPU jobs).

    Returns None when the current ``MIRROR_HW`` selector does not apply.
    """
    step_label = _get_step_label(step)
    has_pool = step.get("agents") is not None or step.get("plugins") is not None or step.get("image") is not None

    if "mirror_hardwares" not in step:
        if has_pool:
            return step
        chips, cards = _read_pytest_marks(step.get("commands"))
        if not chips and cards is None:
            return step
        preset_name = _compose_mirror_hardware_name(chips, cards, step_label=step_label)
    else:
        if has_pool:
            raise ValueError(
                f"step {step_label!r} sets mirror_hardwares together with agents/plugins/image; "
                f"use mirror_hardwares only",
            )
        preset_name = _resolve_mirror_hardware_name(
            step.get("mirror_hardwares"),
            step_label=step_label,
        )

    if preset_name is None:
        return None

    preset = _load_mirror_hardwares().get(preset_name)
    if preset is None:
        known = ", ".join(sorted(_load_mirror_hardwares()))
        raise ValueError(
            f"unknown mirror_hardwares {preset_name!r} in step {step_label!r}; known: {known}",
        )

    expanded = copy.deepcopy(preset)
    merged = {key: value for key, value in step.items() if key != "mirror_hardwares"} | expanded
    # Preset retry (K8S_RETRY on l4_*) must not clobber a step that opted out.
    if "retry" in step:
        merged["retry"] = step["retry"]
    return merged


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
