# SPDX-License-Identifier: Apache-2.0
"""Tests for the TTS adapter migration ratchet.

The gate only earns its place if it actually catches the ways a per-model branch
gets reintroduced. These pin what it detects — and, just as importantly, the one
shape it is known to miss, so nobody mistakes it for an adversarial sandbox.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "pre_commit"))

from check_tts_adapter import (  # noqa: E402
    MAX_LEGACY_DETECTORS,
    MAX_MODEL_TYPE_BRANCHES,
    _legacy_detectors,
    _model_type_branches,
    main,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _count(src: str) -> int:
    return len(_model_type_branches(ast.parse(src)))


@pytest.mark.parametrize(
    ("label", "src", "expected"),
    [
        ("direct comparison", 'if self._tts_model_type == "foo": pass', 1),
        ("negated comparison", 'if self._tts_model_type != "foo": pass', 1),
        ("membership test", 'if self._tts_model_type in ("a", "b"): pass', 1),
        ("local alias", 'kind = self._tts_model_type\nif kind == "foo": pass', 1),
        ("getattr read", 'if getattr(self, "_tts_model_type") == "foo": pass', 1),
        ("getattr with default", 'if getattr(self, "_tts_model_type", None) == "foo": pass', 1),
        ("match statement counts each case", 'match self._tts_model_type:\n case "a": pass\n case _: pass', 2),
        ("alias then match", 'k = getattr(self, "_tts_model_type")\nmatch k:\n case "a": pass', 1),
        ("unrelated attribute is ignored", 'if self._other == "foo": pass', 0),
        ("unrelated local is ignored", 'kind = self._backend\nif kind == "foo": pass', 0),
    ],
)
def test_branch_detection(label, src, expected):
    assert _count(src) == expected, label


def test_dict_dispatch_is_a_known_gap():
    """Documented limitation, asserted so it is a decision and not a surprise.

    Table dispatch keyed on the model type reads as zero branches. The ratchet
    guards against drift in the shape people actually write; catching every
    indirection is a review question, not a lint one.
    """
    assert _count('TABLE = {"foo": 1}\nvalue = TABLE[self._tts_model_type]') == 0


def test_legacy_detector_counting():
    src = 'X = (LegacyDetector(name="a", stage_keys=frozenset({"s"})), LegacyDetector(name="b"))'
    found = _legacy_detectors(ast.parse(src))
    assert [name for _, name in found] == ["a", "b"]


def test_gate_passes_on_current_tree():
    """The recorded budgets must match reality, or the ratchet is decorative."""
    assert main([]) == 0


def test_budgets_are_the_real_counts():
    root = Path(__file__).resolve().parents[2]
    serving = root / "vllm_omni" / "entrypoints" / "openai" / "serving_speech.py"
    adapters = root / "vllm_omni" / "entrypoints" / "openai" / "tts_adapters" / "__init__.py"
    assert len(_model_type_branches(ast.parse(serving.read_text()))) == MAX_MODEL_TYPE_BRANCHES
    assert len(_legacy_detectors(ast.parse(adapters.read_text()))) == MAX_LEGACY_DETECTORS
