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
QWEN_HARD_OOM_TESTS = {
    "test_reliability_fault_gpu_oom_error_contract_consistent_chat_speech",
    "test_reliability_fault_gpu_oom_large_request_failure",
    "test_reliability_fault_gpu_oom_concurrent_pressure_failure",
}
QWEN_RECOVERY_OOM_TEST = "test_reliability_fault_gpu_oom_recovers_after_fault_removed"


def _decorator_name(decorator: ast.expr) -> str | None:
    func = decorator.func if isinstance(decorator, ast.Call) else decorator
    return getattr(func, "id", None) or getattr(func, "attr", None)


def _string_tuple_assignment(tree: ast.Module, name: str) -> tuple[str, ...]:
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    value = ast.literal_eval(assignment.value)
    assert isinstance(value, tuple)
    assert all(isinstance(item, str) for item in value)
    return value


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


@pytest.mark.parametrize("filename", FAULT_CARD_COUNTS)
def test_gpu_oom_fault_contract_accepts_connection_loss(filename: str) -> None:
    tree = ast.parse((RELIABILITY_DIR / filename).read_text(encoding="utf-8"))
    assert "connection" in _string_tuple_assignment(tree, "FAULT_ERROR_KEYWORDS")


def _inject_gpu_oom_calls(node: ast.FunctionDef) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "inject_gpu_oom"
    ]


def test_qwen_hard_oom_tests_use_absolute_free_memory_ceiling() -> None:
    path = RELIABILITY_DIR / "test_reliability_qwen3_omni.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }

    for test_name in QWEN_HARD_OOM_TESTS:
        calls = _inject_gpu_oom_calls(functions[test_name])
        assert len(calls) == 1, test_name
        keyword = next((item for item in calls[0].keywords if item.arg == "max_free_bytes"), None)
        assert keyword is not None, test_name
        assert ast.get_source_segment(source, keyword.value) == 'OOM_INJECTION_CONFIG["max_free_bytes"]'

    recovery_calls = _inject_gpu_oom_calls(functions[QWEN_RECOVERY_OOM_TEST])
    assert len(recovery_calls) == 1
    assert all(item.arg != "max_free_bytes" for item in recovery_calls[0].keywords)


def test_qwen_hard_oom_tests_use_the_hip_eager_server_params() -> None:
    path = RELIABILITY_DIR / "test_reliability_qwen3_omni.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }

    for test_name in QWEN_HARD_OOM_TESTS:
        parametrizations = [
            decorator
            for decorator in functions[test_name].decorator_list
            if isinstance(decorator, ast.Call) and _decorator_name(decorator) == "parametrize"
        ]
        server_parametrize = next(
            decorator
            for decorator in parametrizations
            if decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and decorator.args[0].value == "omni_server_function"
        )
        assert isinstance(server_parametrize.args[1], ast.Name)
        assert server_parametrize.args[1].id == "QWEN_OOM_PARAMS"


def test_nightly_pipeline_does_not_own_fault_injection_files() -> None:
    nightly = NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    for filename in FAULT_CARD_COUNTS:
        assert filename not in nightly
    assert "tests/dfx/reliability/" not in nightly
