# SPDX-License-Identifier: Apache-2.0
"""Ratchet on the TTS adapter migration (RFC #4327, #4855).

Per-model behaviour is being moved out of ``serving_speech.py`` and into
``entrypoints/openai/tts_adapters/``. The migration has stalled before, because
nothing stopped a new model from adding one more ``self._tts_model_type == ...``
branch faster than the migration removed them.

This gate makes the two debt counters monotonically non-increasing:

1. ``self._tts_model_type`` comparisons in ``serving_speech.py``
2. entries in ``tts_adapters.LEGACY_TTS_DETECTORS`` (models with no adapter)

Lowering a budget after removing branches is expected and encouraged — the check
insists on it, so the recorded numbers cannot silently drift upward. Raising one
requires editing this file, which makes it a reviewable decision instead of an
invisible one.

Source-only (``ast``), so it runs in pre-commit without importing vllm.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_SPEECH = Path("vllm_omni/entrypoints/openai/serving_speech.py")
ADAPTERS_INIT = Path("vllm_omni/entrypoints/openai/tts_adapters/__init__.py")

# Budgets. These may only go DOWN. See the module docstring.
MAX_MODEL_TYPE_BRANCHES = 27
MAX_LEGACY_DETECTORS = 0

_GUIDANCE = """
Move the per-model behaviour into its adapter under
``vllm_omni/entrypoints/openai/tts_adapters/`` instead of branching in the
shared serving module. ``docs/contributing/model/adding_tts_model.md`` walks
through it. If a branch genuinely cannot be expressed as an adapter hook, raise
the budget in this file and say why in the PR description.
""".strip()


_ATTR = "_tts_model_type"


def _reads_model_type(node: ast.AST) -> bool:
    """Whether ``node`` evaluates to the model-type discriminator."""
    if isinstance(node, ast.Attribute) and node.attr == _ATTR:
        return True
    # getattr(self, "_tts_model_type") / getattr(self, "_tts_model_type", None)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == _ATTR
    )


def _local_aliases(tree: ast.AST) -> set[str]:
    """Local names bound directly to the model type, e.g. ``kind = self._tts_model_type``.

    Shallow on purpose: one hop is what an author reaches for when a branch gets
    long, which is the drift this ratchet exists to catch.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        value = None
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            value = node.value
        if value is None or not _reads_model_type(value):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                aliases.add(target.id)
    return aliases


def _model_type_branches(tree: ast.AST) -> list[tuple[int, str]]:
    """Every branch keyed on the model-type discriminator.

    Counts direct comparisons, ``getattr`` reads, ``match`` subjects (one per
    ``case``), and comparisons against a local alias of the attribute. This is a
    ratchet against drift, not an adversarial sandbox: dispatch tables and other
    indirections still slip past, and are a review question.
    """
    aliases = _local_aliases(tree)

    def _is_subject(node: ast.AST) -> bool:
        return _reads_model_type(node) or (isinstance(node, ast.Name) and node.id in aliases)

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            if any(_is_subject(op) for op in [node.left, *node.comparators]):
                found.append((node.lineno, ast.unparse(node)))
        elif isinstance(node, ast.Match) and _is_subject(node.subject):
            subject = ast.unparse(node.subject)
            for case in node.cases:
                found.append((case.pattern.lineno, f"match {subject}: case {ast.unparse(case.pattern)}"))
    found.sort()
    return found


def _legacy_detectors(tree: ast.AST) -> list[tuple[int, str]]:
    """Every ``LegacyDetector(...)`` construction."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "LegacyDetector":
            name = "?"
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = str(kw.value.value)
            found.append((node.lineno, name))
    found.sort()
    return found


def _check(
    label: str,
    path: Path,
    actual: int,
    budget: int,
    sample: list[str],
    budget_name: str,
    notices: list[str],
) -> list[str]:
    if actual > budget:
        listing = "\n".join(f"      {line}" for line in sample)
        return [
            f"{path}: {actual} {label} exceeds the budget of {budget}.\n{listing}\n\n{_GUIDANCE}\n",
        ]
    if actual < budget:
        # Progress is never a build failure. Removing branches is the goal, so a
        # PR that does it must not hand-edit this file to stay green -- nor break
        # `main` for everyone else when it merges from a stale base.
        notices.append(
            f"{path}: {actual} {label}, below the budget of {budget}. "
            f"Lower {budget_name} to {actual} in "
            f"{Path(__file__).relative_to(REPO_ROOT)} to hold the new ground."
        )
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # pre-commit passes the staged filenames; we always audit the same two files,
    # so they are accepted and ignored.
    parser.add_argument("filenames", nargs="*")
    parser.parse_args(argv)

    serving_path = REPO_ROOT / SERVING_SPEECH
    adapters_path = REPO_ROOT / ADAPTERS_INIT
    for path in (serving_path, adapters_path):
        if not path.is_file():
            print(f"check_tts_adapter: expected file not found: {path}", file=sys.stderr)
            return 1

    notices: list[str] = []
    branches = _model_type_branches(ast.parse(serving_path.read_text()))
    detectors = _legacy_detectors(ast.parse(adapters_path.read_text()))

    errors = _check(
        "`_tts_model_type` comparisons",
        SERVING_SPEECH,
        len(branches),
        MAX_MODEL_TYPE_BRANCHES,
        [f"line {lineno}: {src}" for lineno, src in branches],
        "MAX_MODEL_TYPE_BRANCHES",
        notices,
    )
    errors += _check(
        "unmigrated model types in LEGACY_TTS_DETECTORS",
        ADAPTERS_INIT,
        len(detectors),
        MAX_LEGACY_DETECTORS,
        [f"line {lineno}: {name}" for lineno, name in detectors],
        "MAX_LEGACY_DETECTORS",
        notices,
    )

    for notice in notices:
        print(f"check_tts_adapter: {notice}", file=sys.stderr)
    for error in errors:
        print(f"check_tts_adapter: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
