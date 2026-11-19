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
    "test_wan22.py": 2,
    "test_hunyuan_image.py": 4,
}
FAULT_FILES = (
    "tests/dfx/reliability/test_qwen3_omni.py",
    "tests/dfx/reliability/test_wan22.py",
    "tests/dfx/reliability/test_hunyuan_image.py",
    "tests/dfx/reliability/test_minicpmo_4_5.py",
)
INVALID_PARAMETER_ROCM_MARKS = {
    "test_invalid_video_generation.py": {'hardware_marks(res={"cuda": "H100", "rocm": "MI325"})': 1},
    "test_invalid_server_control.py": {'hardware_marks(res={"cuda": "H100", "rocm": "MI325"})': 1},
    "test_invalid_image_generation.py": {'hardware_marks(res={"cuda": "H100", "rocm": "MI325"})': 1},
    "test_invalid_image_editing.py": {'hardware_marks(res={"cuda": "H100", "rocm": "MI325"})': 1},
    "test_invalid_audio_diffusion.py": {'hardware_marks(res={"cuda": "L4", "rocm": "MI325"})': 1},
    "test_invalid_audio_speech.py": {
        'hardware_marks(res={"cuda": "L4", "rocm": "MI325"})': 1,
        'hardware_marks(res={"cuda": "H100", "rocm": "MI325"})': 1,
    },
}


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


def test_minicpmo_fault_injection_declares_mi325() -> None:
    source = (RELIABILITY_DIR / "test_minicpmo_4_5.py").read_text(encoding="utf-8")
    assert 'res = {"cuda": "H100", "rocm": "MI325"}' in source
    assert "num_cards = 1" in source


@pytest.mark.parametrize(("filename", "expected_markers"), INVALID_PARAMETER_ROCM_MARKS.items())
def test_invalid_parameter_scopes_declare_mi325(filename: str, expected_markers: dict[str, int]) -> None:
    source = (RELIABILITY_DIR / "invalid_param_test" / filename).read_text(encoding="utf-8")
    for marker, expected_count in expected_markers.items():
        assert source.count(marker) == expected_count


def test_qwen_oom_uses_free_memory_ceiling_and_eager_params() -> None:
    case_source = (RELIABILITY_DIR / "test_qwen3_omni.py").read_text(encoding="utf-8")
    case_tree = ast.parse(case_source)
    functions = {node.name: node for node in case_tree.body if isinstance(node, ast.FunctionDef)}
    hard_return = next(node for node in ast.walk(functions["_hard_oom_config"]) if isinstance(node, ast.Return))
    assert isinstance(hard_return.value, ast.Dict)
    config = {
        key.value: value
        for key, value in zip(hard_return.value.keys, hard_return.value.values)
        if isinstance(key, ast.Constant)
    }
    ceiling = config["max_free_bytes"]
    assert isinstance(ceiling, ast.IfExp)
    assert ast.unparse(ceiling.body) == "32 * 1024 * 1024"
    assert ast.unparse(ceiling.orelse) == "256 * 1024 * 1024"
    assert "torch.version" in ast.unparse(ceiling.test)
    assert "eager_oom_stage_ids = (0, 1)" in case_source
    assert "reliability_scenarios_for_eager_oom" in (RELIABILITY_DIR / "helpers.py").read_text(encoding="utf-8")
    case = functions["test_gpu_oom_at_load"]
    assert any("GpuOomUnderLoad" in ast.unparse(decorator) for decorator in case.decorator_list)
    assert "GpuOomUnderLoad().run" in ast.unparse(case)
    assert "def oom_under_load_request" in case_source
    assert "def oom_preflight_request" in case_source
    shared_tree = ast.parse((RELIABILITY_DIR / "scenarios" / "oom.py").read_text(encoding="utf-8"))
    flow = next(
        node for node in shared_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_oom_under_load"
    )
    injections = [
        node
        for node in ast.walk(flow)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "inject_gpu_oom"
    ]
    assert len(injections) == 1
    assert "model.hard_oom_config" in ast.unparse(flow)
    assert any(keyword.arg == "max_free_bytes" for keyword in injections[0].keywords)
    assert "_run_oom_recovery" not in functions


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
    scenarios = _literal_class_assignment(qwen_path, "Qwen3Omni", "scenarios")
    stage_ids = _literal_class_assignment(qwen_path, "Qwen3Omni", "eager_oom_stage_ids")
    oom_source = RELIABILITY_DIR / "scenarios" / "oom.py"
    server_set = _literal_class_assignment(oom_source, "GpuOomUnderLoad", "server_set")
    assert isinstance(scenarios, list)
    assert isinstance(server_set, str)
    assert stage_ids == (0, 1)
    assert isinstance(stage_ids, tuple)
    qwen_scenarios = cast(list[dict[str, Any]], scenarios)
    qwen_stage_ids = cast(tuple[int, ...], stage_ids)
    assert {item["test_name"] for item in qwen_scenarios} == {
        "qwen3_omni_reliability_async_chunk",
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


@pytest.mark.parametrize(
    "filename,class_name",
    [("test_qwen3_omni_replica.py", "Qwen3OmniReplica"), ("test_minicpmo_4_5_replica.py", "MiniCPMO45Replica")],
)
def test_replica_coverage_is_intentionally_cuda_only(filename: str, class_name: str) -> None:
    resource = _class_assignment(RELIABILITY_DIR / filename, class_name, "res")
    assert resource is not None
    assert ast.literal_eval(resource) == {"cuda": "H100"}


@pytest.mark.parametrize("key", ["diffusion_wan22_reliability", "diffusion_hunyuan_image3_reliability"])
def test_diffusion_reliability_tracks_shared_stage_config_helper(key: str) -> None:
    import yaml

    registry = yaml.safe_load(Path(".buildkite/common/ci_source_file_dependencies.yml").read_text(encoding="utf-8"))
    assert "tests/helpers/stage_config.py" in registry["source_file_dependencies"][key]


def test_wan_reliability_uses_cli_deploy_without_stage_config_directory() -> None:
    path = RELIABILITY_DIR / "test_wan22.py"
    assert _class_assignment(path, "Wan22", "deploy_dir") is None
    scenarios = _literal_class_assignment(path, "Wan22", "scenarios")
    assert isinstance(scenarios, list) and scenarios
    for scenario in scenarios:
        params = scenario["server_params"]
        assert "stage_config_name" not in params
        assert "--usp" in params["server_args"]
