# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault-injection reliability tests must be selectable on MI300 weekly CI."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RELIABILITY_DIR = Path("tests/dfx/reliability")
NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
FAULT_CARD_COUNTS = {
    "test_reliability_qwen3_omni.py": 2,
    "test_reliability_wan22.py": 1,
    "test_reliability_hunyuan_image.py": 4,
}


def _decorator_name(decorator: ast.expr) -> str | None:
    func = decorator.func if isinstance(decorator, ast.Call) else decorator
    return getattr(func, "id", None) or getattr(func, "attr", None)


@pytest.mark.parametrize(("filename", "num_cards"), FAULT_CARD_COUNTS.items())
def test_fault_injection_tests_declare_mi325(filename: str, num_cards: int) -> None:
    path = RELIABILITY_DIR / filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    tests = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]

    assert tests
    for node in tests:
        calls = [dec for dec in node.decorator_list if _decorator_name(dec) == "hardware_test"]
        assert len(calls) == 1, node.name
        rendered = ast.get_source_segment(source, calls[0])
        assert rendered is not None
        assert '"rocm": "MI325"' in rendered
        assert f"num_cards={num_cards}" in rendered
        names = [_decorator_name(dec) for dec in node.decorator_list]
        if "parametrize" in names:
            assert names.index("hardware_test") < names.index("parametrize")


def test_nightly_pipeline_does_not_own_fault_injection_files() -> None:
    nightly = NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    for filename in FAULT_CARD_COUNTS:
        assert filename not in nightly
    assert "tests/dfx/reliability/" not in nightly
