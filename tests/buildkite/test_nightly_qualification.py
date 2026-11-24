# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run the actual DFX guard functions without importing GPU runtimes."""

import ast
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RUNNER = Path("tests/dfx/perf/scripts/run_benchmark.py")
VALIDATION = Path("tests/e2e/accuracy/omni_duplex_eval/validation.py")
ACCURACY_CONFIG = VALIDATION.with_name("omni_duplex_eval_ci_config.json")


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=[*nodes], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("device_env", ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"])
@pytest.mark.parametrize("benchmark_fails", [False, True])
def test_judge_cleanup_leaves_final_memory_wait_to_primary(monkeypatch, device_env, benchmark_fails):
    monkeypatch.setenv(device_env, "4,7")
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms",
        SimpleNamespace(current_omni_platform=SimpleNamespace(device_control_env_var=device_env)),
    )
    alive: list[str] = []
    events = []

    @contextmanager
    def server(model, args, *, wait_for_device_memory=True, **kwargs):
        # A global memory wait cannot finish while the primary model is alive.
        assert not (wait_for_device_memory and alive)
        events.append(("enter", model, wait_for_device_memory))
        if model == "judge":
            assert kwargs["env_dict"] == {device_env: "7"}
        alive.append(model)
        try:
            yield SimpleNamespace(host="localhost", port=8001)
        finally:
            alive.remove(model)
            assert not (wait_for_device_memory and alive)
            events.append(("exit", model, wait_for_device_memory))

    ns = _functions(
        RUNNER,
        ["_resolve_judge_visible_devices", "_start_judge_server", "_start_omni_server"],
        {
            "os": os,
            "contextmanager": contextmanager,
            "OmniServer": server,
            "_resolve_offline_model": lambda model, **_: model,
            "_cli_flag_value": lambda *_: None,
            "_merge_omni_default_server_args": lambda *_, **__: [],
            "_preflight_server_vad": lambda *_: None,
            "_judge_server_params_for_test": lambda _: {"model": "judge", "device_index": "1"},
            "_omni_server_env": lambda: {},
        },
    )
    try:
        with ns["_start_omni_server"](("test", "primary", None, None, (), True)):
            assert alive == ["primary", "judge"]
            if benchmark_fails:
                raise RuntimeError("benchmark failed")
    except RuntimeError as exc:
        assert benchmark_fails and str(exc) == "benchmark failed"
    assert alive == []
    assert events == [
        ("enter", "primary", True),
        ("enter", "judge", False),
        ("exit", "judge", False),
        ("exit", "primary", True),
    ]


@pytest.mark.parametrize("baseline", [{}, {"H100": {"floor": 1}}, {"MI300": {}}, {"MI300": {"floor": None}}])
def test_required_accuracy_baseline_rejects_record_only_fallback(baseline):
    select = _functions(VALIDATION, ["select_duplex_eval_baseline"], {})["select_duplex_eval_baseline"]
    assert select({"baseline": baseline}, "MI300") == {}
    with pytest.raises(ValueError, match="No duplex-eval score floors"):
        select({"baseline": baseline}, "MI300", required=True)


def test_accuracy_uses_recorded_hardware_without_mi325_alias():
    select = _functions(VALIDATION, ["select_duplex_eval_baseline"], {})["select_duplex_eval_baseline"]
    config = json.loads(ACCURACY_CONFIG.read_text())
    assert select(config, "MI300", required=True) == config["baseline"]["MI300"]
    with pytest.raises(ValueError, match="MI325"):
        select(config, "MI325", required=True)


def test_amd_accuracy_option_rejects_uncalibrated_hardware_before_benchmark(monkeypatch):
    config = json.loads(ACCURACY_CONFIG.read_text())
    select = _functions(VALIDATION, ["select_duplex_eval_baseline"], {})["select_duplex_eval_baseline"]
    monkeypatch.setitem(
        sys.modules,
        "tests.e2e.accuracy.omni_duplex_eval.validation",
        SimpleNamespace(load_accuracy_config=lambda *_: config, select_duplex_eval_baseline=select),
    )
    function = next(
        n
        for n in ast.parse(RUNNER.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "test_performance_benchmark"
    )
    function.decorator_list = []
    namespace = {"pytest": pytest, "get_runtime_resource_label": lambda: "MI325"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"), namespace)
    with pytest.raises(ValueError, match="No duplex-eval score floors.*MI325"):
        namespace["test_performance_benchmark"](
            None,
            {"test_name": "test", "params": {"duplex_eval_accuracy_config": str(ACCURACY_CONFIG)}},
            SimpleNamespace(getoption=lambda name: name == "require_duplex_eval_score_floors"),
        )


@pytest.mark.parametrize(
    ("section", "metric", "floor"),
    [
        ("rtd", "mean_content_score", "min_rtd_mean_content_score"),
        ("rtd", "mean_avg_temporal_score", "min_rtd_mean_temporal_score"),
        ("pr", "mean_all_success", "min_pr_mean_all_success"),
    ],
)
def test_mi300_accuracy_rejects_scores_below_each_floor(tmp_path, section, metric, floor):
    config = json.loads(ACCURACY_CONFIG.read_text())
    config["baseline"] = config["baseline"]["MI300"]
    for split, ids in config["sample_ids_per_split"].items():
        for sample_id in ids:
            path = tmp_path / split / f"{sample_id}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(
                json.dumps({"protocol_pin": config["protocol_pin"], "response_meta": {"mix": config["mix"]}})
            )
    count = sum(map(len, config["sample_ids_per_split"].values()))
    summary = {
        "status": "completed",
        "total": count,
        "evaluated": count,
        "samples": count,
        "failed": 0,
        "skipped": 0,
        "errors": [],
        "protocol_pin": config["protocol_pin"],
        "score_root": str(tmp_path),
        "rtd": {
            "mean_content_score": 3.0,
            "mean_avg_temporal_score": 3.0,
            "by_task": dict.fromkeys(split for split in config["sample_ids_per_split"] if split.startswith("RTD_")),
        },
        "pr": {
            "mean_all_success": 1.0,
            "by_task": dict.fromkeys(["correction", "proactive_reminder", "post_event_reminder"]),
        },
    }
    result = {
        "completed": count,
        "failed": 0,
        "omni_duplex_eval": {"published": count, "artifact_errors": [], "accuracy": summary},
    }
    validate = _functions(
        VALIDATION,
        ["assert_accuracy_result"],
        {"json": json, "math": math, "Path": Path, "PROTOCOL_PIN": config["protocol_pin"], "MIX_QUESTION": "question"},
    )["assert_accuracy_result"]
    summary[section][metric] = config["baseline"][floor]
    validate(result, config)
    summary[section][metric] -= 0.001
    with pytest.raises(AssertionError, match="outside"):
        validate(result, config)
