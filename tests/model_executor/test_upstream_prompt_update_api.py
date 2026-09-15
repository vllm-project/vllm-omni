# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guard that Omni only calls prompt-update helpers upstream still provides.

vLLM 0.29 removed ``PromptUpdateDetails.select_text`` in favour of the
token-oriented ``select_token_id`` / ``select_token_ids``. Three MiniCPM-o call
sites survived the rebase and only failed at engine-core startup, where the
traceback surfaced as an opaque "Engine core initialization failed" across eight
CI jobs.

These helpers are referenced from replacement closures that no unit test drives,
so a static check is what actually catches the next removal. It resolves the
attribute against the installed class rather than a hard-coded list, so it
follows upstream automatically.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from vllm.multimodal.processing.processor import PromptUpdateDetails

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_OMNI_ROOT = Path(__file__).resolve().parents[2] / "vllm_omni"
_GUARDED = "PromptUpdateDetails"


def _referenced_attributes() -> dict[str, set[str]]:
    """Map ``PromptUpdateDetails.<attr>`` -> source files referencing it."""
    found: dict[str, set[str]] = {}
    for path in _OMNI_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == _GUARDED:
                found.setdefault(node.attr, set()).add(str(path.relative_to(_OMNI_ROOT.parent)))
    return found


def test_omni_only_uses_prompt_update_helpers_that_exist_upstream() -> None:
    referenced = _referenced_attributes()
    assert referenced, "expected Omni to reference PromptUpdateDetails somewhere"

    missing = {attr: sorted(files) for attr, files in referenced.items() if not hasattr(PromptUpdateDetails, attr)}

    assert not missing, (
        "Omni calls PromptUpdateDetails helpers that the installed vLLM does not "
        f"provide: {missing}. Upstream moved prompt updates to token ids; use "
        "select_token_id/select_token_ids with the encoded text."
    )


def test_token_oriented_helpers_are_the_ones_upstream_exposes() -> None:
    """Pin the replacements, so their removal fails here and not at startup."""
    assert hasattr(PromptUpdateDetails, "select_token_id")
    assert hasattr(PromptUpdateDetails, "select_token_ids")
    assert not hasattr(PromptUpdateDetails, "select_text"), (
        "select_text is back upstream; re-check the MiniCPM-o replacements."
    )
