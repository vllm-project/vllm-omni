# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault-injection reliability cases must stay selectable on MI300 weekly CI."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RELIABILITY_DIR = Path("tests/dfx/reliability")
NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
CASE_CARD_COUNTS = {
    "test_qwen3_omni.py": 2,
    "test_wan22.py": 1,
    "test_hunyuan_image.py": 4,
}
FAULT_FILES = (
    "tests/dfx/reliability/test_qwen3_omni.py",
    "tests/dfx/reliability/test_wan22.py",
    "tests/dfx/reliability/test_hunyuan_image.py",
    "tests/dfx/reliability/test_minicpmo_4_5.py",
)
QWEN_HARD_OOM_TESTS = {
    "test_reliability_fault_gpu_oom_error_contract_consistent_chat_speech",
    "test_reliability_fault_gpu_oom_large_request_failure",
    "test_reliability_fault_gpu_oom_concurrent_pressure_failure",
}
QWEN_RECOVERY_OOM_TEST = "test_reliability_fault_gpu_oom_recovers_after_fault_removed"


def _assignment_value(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            assert node.value is not None
            return node.value
    raise AssertionError(f"missing assignment {name}")


@pytest.mark.parametrize(("filename", "num_cards"), CASE_CARD_COUNTS.items())
def test_fault_injection_cases_declare_mi325(filename: str, num_cards: int) -> None:
    source = (RELIABILITY_DIR / filename).read_text(encoding="utf-8")
    assert '"rocm": "MI325"' in source
    assert f"num_cards = {num_cards}" in source
    tree = ast.parse(source)
    keywords = ast.literal_eval(_assignment_value(tree, "FAULT_ERROR_KEYWORDS"))
    assert "connection" in keywords


def test_qwen_hard_oom_uses_free_memory_ceiling_and_eager_params() -> None:
    case_source = (RELIABILITY_DIR / "test_qwen3_omni.py").read_text(encoding="utf-8")
    case_tree = ast.parse(case_source)
    hard_config = next(
        node for node in case_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_hard_oom_config"
    )
    assert any(isinstance(node, ast.Constant) and node.value == "max_free_bytes" for node in ast.walk(hard_config))
    recover = _assignment_value(case_tree, "OOM_RECOVER_INJECTION_CONFIG")
    assert isinstance(recover, ast.Dict)
    assert all(not (isinstance(key, ast.Constant) and key.value == "max_free_bytes") for key in recover.keys)
    assert "reliability_scenarios_for_eager_oom" in (RELIABILITY_DIR / "helpers.py").read_text(encoding="utf-8")
    assert "eager_oom_stage_ids = (0, 1)" in case_source

    qwen_source = (RELIABILITY_DIR / "test_qwen3_omni.py").read_text(encoding="utf-8")
    for test_name in QWEN_HARD_OOM_TESTS | {QWEN_RECOVERY_OOM_TEST}:
        assert f"def {test_name}(" in qwen_source

    oom_source = (RELIABILITY_DIR / "scenarios" / "oom.py").read_text(encoding="utf-8")
    oom_tree = ast.parse(oom_source)
    classes = {node.name: node for node in oom_tree.body if isinstance(node, ast.ClassDef)}

    def _server_set(class_name: str) -> str:
        for statement in classes[class_name].body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "server_set" for target in statement.targets
            ):
                assert isinstance(statement.value, ast.Constant)
                assert isinstance(statement.value.value, str)
                return statement.value.value
        raise AssertionError(f"{class_name} has no server_set")

    for class_name in ("GpuOomErrorContract", "GpuOomLarge", "GpuOomConcurrent"):
        assert _server_set(class_name) == "oom"
    assert _server_set("GpuOomRecovery") == "default"

    functions = {node.name: node for node in oom_tree.body if isinstance(node, ast.FunctionDef)}
    starter = functions["_start_hard_oom"]
    assert any(isinstance(node, ast.Constant) and node.value == "max_free_bytes" for node in ast.walk(starter))
    recovery_fn = next(
        node for node in case_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_oom_recovery"
    )
    inject_calls = [
        node
        for node in ast.walk(recovery_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "inject_gpu_oom"
    ]
    assert len(inject_calls) == 1
    assert all(keyword.arg != "max_free_bytes" for keyword in inject_calls[0].keywords)


def test_nightly_pipeline_does_not_own_fault_injection_files() -> None:
    nightly = NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    for filename in FAULT_FILES:
        assert filename not in nightly
    assert "tests/dfx/reliability/" not in nightly
