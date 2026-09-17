# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
STAGE_CONFIG_HELPER = Path("tests/helpers/stage_config.py")
FUNCTION_TEST = Path("tests/e2e/online_serving/test_qwen3_omni_expansion.py")
ASSERTIONS = Path("tests/helpers/assertions.py")
ACCURACY_DRIVER = Path("tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py")


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline["steps"])
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


def _load_stage_config_helper(monkeypatch: pytest.MonkeyPatch):
    stub = types.ModuleType("vllm_omni.config.stage_config")
    setattr(stub, "load_deploy_config", lambda config_path: config_path)
    monkeypatch.setitem(sys.modules, "vllm_omni.config.stage_config", stub)
    spec = importlib.util.spec_from_file_location("nightly_stage_config", STAGE_CONFIG_HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qwen3_ci_overlay_pins_only_the_talker_sampling_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_config_helper(monkeypatch)
    generated = Path(module.get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
    overlay = yaml.safe_load(generated.read_text(encoding="utf-8"))
    talker = next(stage for stage in overlay["stages"] if stage["stage_id"] == 1)

    assert talker["default_sampling_params"] == {"max_tokens": 1000, "seed": 42}
    production = yaml.safe_load(Path("vllm_omni/deploy/qwen3_omni_moe.yaml").read_text(encoding="utf-8"))
    production_talker = next(stage for stage in production["stages"] if stage["stage_id"] == 1)
    assert "seed" not in production_talker["default_sampling_params"]


def test_qwen3_ci_overlay_retains_long_output_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_config_helper(monkeypatch)
    generated = Path(module.get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
    overlay = yaml.safe_load(generated.read_text(encoding="utf-8"))
    thinker = next(stage for stage in overlay["stages"] if stage["stage_id"] == 0)
    function_source = FUNCTION_TEST.read_text(encoding="utf-8")

    assert thinker["default_sampling_params"]["max_tokens"] == 512
    assert "assert word_count >= 200" in function_source


def test_long_output_request_expands_only_its_downstream_audio_budgets() -> None:
    function_source = FUNCTION_TEST.read_text(encoding="utf-8")

    assert '"sampling_params_list": LONG_OUTPUT_SAMPLING_PARAMS' in function_source
    assert '"max_tokens": 3072' in function_source
    assert '"max_tokens": 6144' in function_source
    assert "assert word_count >= 200" in function_source


def test_qwen3_ci_overlay_reserves_rocm_thinker_encoder_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_stage_config_helper(monkeypatch)
    generated = Path(module.get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
    overlay = yaml.safe_load(generated.read_text(encoding="utf-8"))
    rocm_thinker = next(stage for stage in overlay["platforms"]["rocm"]["stages"] if stage["stage_id"] == 0)

    assert rocm_thinker == {
        "stage_id": 0,
        "gpu_memory_utilization": None,
        "kv_cache_memory_bytes": 80 * 1024**3,
    }
    production = yaml.safe_load(Path("vllm_omni/deploy/qwen3_omni_moe.yaml").read_text(encoding="utf-8"))
    production_thinker = next(stage for stage in production["stages"] if stage["stage_id"] == 0)
    assert production_thinker["gpu_memory_utilization"] == 0.9
    assert "kv_cache_memory_bytes" not in production_thinker


def test_function_expansion_uses_the_seeded_ci_overlay() -> None:
    source = FUNCTION_TEST.read_text(encoding="utf-8")
    assert 'get_deploy_config_path("ci/qwen3_omni_moe.yaml")' in source
    assert 'get_deploy_config_path("qwen3_omni_moe.yaml")' not in source


@pytest.mark.parametrize(
    ("label", "artifact_dir"),
    [
        ("Qwen3-Omni Function Expansion", "artifacts/rocm-qwen3-omni-function"),
        ("Qwen3-Omni Accuracy", "artifacts/rocm-qwen3-omni-accuracy"),
    ],
)
def test_unstable_jobs_pin_seed_and_retain_diagnostics(label: str, artifact_dir: str) -> None:
    step = _find_step(label)
    commands = step["commands"]
    pytest_commands = [command for command in commands if "pytest " in command]

    assert step["grade"] == "NonBlocking"
    assert len(pytest_commands) == 1
    assert "export VLLM_CI_QWEN3_OMNI_SEED=42" in commands
    assert "export PYTHONHASHSEED=42" in commands
    assert any(path == f"{artifact_dir}/**/*" for path in step["artifact_paths"])
    assert "--junitxml=" in pytest_commands[0]
    assert "pytest-summary.txt" in "\n".join(commands)
    assert "reproducibility.txt" in "\n".join(commands)


def test_function_and_accuracy_selection_are_unchanged() -> None:
    function = next(
        command for command in _find_step("Qwen3-Omni Function Expansion")["commands"] if "pytest " in command
    )
    accuracy = next(command for command in _find_step("Qwen3-Omni Accuracy")["commands"] if "pytest " in command)
    function_argv = split(function)
    accuracy_argv = split(accuracy)

    assert "tests/e2e/online_serving/test_qwen3_omni_expansion.py" in function_argv
    assert function_argv[function_argv.index("-m") + 1] == "full_model and rocm and MI325 and cards_2"
    assert "tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py" in accuracy_argv
    assert accuracy_argv[accuracy_argv.index("-m") + 1] == "full_model and rocm and MI325 and cards_2"


def test_quality_thresholds_are_not_weakened() -> None:
    assertions_source = ASSERTIONS.read_text(encoding="utf-8")
    accuracy_source = ACCURACY_DRIVER.read_text(encoding="utf-8")
    assert 'request_config.get("similarity_threshold", 0.8)' in assertions_source
    assert "default=0.35" in accuracy_source
