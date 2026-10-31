# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault-injection reliability cases must stay selectable on MI300 weekly CI."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

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


def test_qwen_oom_uses_free_memory_ceiling_and_eager_params() -> None:
    case_source = (RELIABILITY_DIR / "test_qwen3_omni.py").read_text(encoding="utf-8")
    case_tree = ast.parse(case_source)
    case_functions = {node.name: node for node in case_tree.body if isinstance(node, ast.FunctionDef)}
    hard_config = next(
        node for node in case_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_hard_oom_config"
    )
    assert any(isinstance(node, ast.Constant) and node.value == "max_free_bytes" for node in ast.walk(hard_config))
    recover = _assignment_value(case_tree, "OOM_RECOVER_INJECTION_CONFIG")
    assert isinstance(recover, ast.Dict)
    recover_items = {
        key.value: value
        for key, value in zip(recover.keys, recover.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    recover_ceiling = recover_items["max_free_bytes"]
    assert isinstance(recover_ceiling, ast.Call)
    assert isinstance(recover_ceiling.func, ast.Name)
    assert recover_ceiling.func.id == "_recover_oom_max_free_bytes"

    ceiling_helper = case_functions["_recover_oom_max_free_bytes"]
    return_node = next(node for node in ast.walk(ceiling_helper) if isinstance(node, ast.Return))
    assert isinstance(return_node.value, ast.IfExp)
    assert ast.unparse(return_node.value.body) == "32 * 1024 * 1024"
    assert isinstance(return_node.value.orelse, ast.Constant)
    assert return_node.value.orelse.value is None
    assert "torch.version" in ast.unparse(return_node.value.test)
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
    assert _server_set("GpuOomRecovery") == "oom"

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
    max_free_keyword = next(keyword for keyword in inject_calls[0].keywords if keyword.arg == "max_free_bytes")
    assert isinstance(max_free_keyword.value, ast.Subscript)
    assert isinstance(max_free_keyword.value.value, ast.Name)
    assert max_free_keyword.value.value.id == "OOM_RECOVER_INJECTION_CONFIG"
    assert isinstance(max_free_keyword.value.slice, ast.Constant)
    assert max_free_keyword.value.slice.value == "max_free_bytes"


def _class_assignment(path: Path, class_name: str, name: str) -> ast.AST | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for statement in node.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in statement.targets
            ):
                return statement.value
    return None


def _literal_class_assignment(path: Path, class_name: str, name: str) -> object:
    value = _class_assignment(path, class_name, name)
    assert value is not None, f"{class_name}.{name} is missing"
    return ast.literal_eval(value)


def test_qwen_oom_recovery_forces_hip_eager_and_leaves_cuda_and_wan_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import yaml

    from tests.dfx.reliability import helpers
    from tests.helpers.stage_config import modify_stage_config

    qwen_path = RELIABILITY_DIR / "test_qwen3_omni.py"
    qwen_tree = ast.parse(qwen_path.read_text(encoding="utf-8"))
    scenarios = _literal_class_assignment(qwen_path, "Qwen3Omni", "scenarios")
    stage_ids = _literal_class_assignment(qwen_path, "Qwen3Omni", "eager_oom_stage_ids")
    recover = _assignment_value(qwen_tree, "OOM_RECOVER_INJECTION_CONFIG")
    assert isinstance(recover, ast.Dict)
    recover_items = {
        key.value: value
        for key, value in zip(recover.keys, recover.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    oom_source = RELIABILITY_DIR / "scenarios" / "oom.py"
    server_set = _literal_class_assignment(oom_source, "GpuOomRecovery", "server_set")
    assert isinstance(scenarios, list)
    assert isinstance(server_set, str)
    assert stage_ids == (0, 1)
    assert isinstance(stage_ids, tuple)
    assert isinstance(recover_items["max_free_bytes"], ast.Call)
    ratio = recover_items["target_mem_ratio"]
    assert isinstance(ratio, ast.Constant)
    assert ratio.value == 0.95
    qwen_scenarios = cast(list[dict[str, Any]], scenarios)
    qwen_stage_ids = cast(tuple[int, ...], stage_ids)
    assert {item["test_name"] for item in qwen_scenarios} == {
        "qwen3_omni_reliability_async_chunk",
        "qwen3_omni_reliability_default",
    }

    model = helpers.FaultModel()
    model.model_id = "qwen3_omni"
    model.scenarios = qwen_scenarios
    model.eager_oom_stage_ids = qwen_stage_ids
    deploy = Path("vllm_omni/deploy/qwen3_omni_moe.yaml")

    monkeypatch.setattr(torch.version, "hip", "6.0.0")
    hip_scenarios = model.scenarios_for_server_set(server_set)
    assert [item["test_name"] for item in hip_scenarios] == [item["test_name"] for item in qwen_scenarios]
    for scenario in hip_scenarios:
        rendered = modify_stage_config(str(deploy), scenario["server_params"]["update"], None)
        loaded = yaml.safe_load(Path(rendered).read_text(encoding="utf-8"))
        eager = {int(stage["stage_id"]): stage.get("enforce_eager") for stage in loaded["stages"]}
        assert eager[0] is True
        assert eager[1] is True
        assert eager[2] is False

    monkeypatch.setattr(torch.version, "hip", None)
    cuda_scenarios = model.scenarios_for_server_set(server_set)
    assert cuda_scenarios == qwen_scenarios
    assert all("update" not in item["server_params"] for item in cuda_scenarios)

    wan_path = RELIABILITY_DIR / "test_wan22.py"
    assert _class_assignment(wan_path, "Wan22", "eager_oom_stage_ids") is None
    wan = helpers.FaultModel()
    wan.model_id = "wan22"
    wan_scenarios = _literal_class_assignment(wan_path, "Wan22", "scenarios")
    assert isinstance(wan_scenarios, list)
    wan.scenarios = cast(list[dict[str, Any]], wan_scenarios)
    assert wan.scenarios_for_server_set("oom") == wan.scenarios_for_server_set("default")


def test_nightly_pipeline_does_not_own_fault_injection_files() -> None:
    nightly = NIGHTLY_PIPELINE.read_text(encoding="utf-8")
    for filename in FAULT_FILES:
        assert filename not in nightly
    assert "tests/dfx/reliability/" not in nightly
