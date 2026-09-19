# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
import os
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pytest

from tests.dfx.conftest import (
    create_paired_omni_benchmark_pytest_params,
    create_test_parameter_mapping,
    get_benchmark_params_for_server,
    get_runtime_resource_label,
    is_diffusion_perf_config,
    load_benchmark_configs,
    run_benchmark,
)
from tests.helpers.runtime import OmniServer

# Optional JSON field ``mark`` is applied as pytest marks via
# ``create_paired_omni_benchmark_pytest_params`` (e.g. ``"mark": [{"hardware_marks":
# {"res": {"cuda": "H100"}, "num_cards": 2}}, "full_model", "omni"]``).


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _get_config_file_from_argv() -> str | None:
    """Read ``--test-config-file`` from ``sys.argv`` at import time so parametrization can use it."""
    import sys

    for i, arg in enumerate(sys.argv):
        if arg == "--test-config-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--test-config-file="):
            return arg.split("=", 1)[1]
    return None


_PERF_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

CONFIG_FILE_PATH = _get_config_file_from_argv()
if CONFIG_FILE_PATH is None:
    _all_configs = load_benchmark_configs(config_dir=_PERF_TESTS_DIR)
    BENCHMARK_CONFIGS = [cfg for cfg in _all_configs if not is_diffusion_perf_config(cfg)]
    print(
        f"No --test-config-file: loaded {len(BENCHMARK_CONFIGS)} omni/tts case(s) from "
        f"{_PERF_TESTS_DIR}/*.json (skipped {len(_all_configs) - len(BENCHMARK_CONFIGS)} diffusion; "
        f"use -m to filter, e.g. -m tts)"
    )
else:
    BENCHMARK_CONFIGS = load_benchmark_configs(CONFIG_FILE_PATH)

DEPLOY_CONFIGS_DIR = Path(__file__).parent.parent / "deploy"
server_to_benchmark_mapping = create_test_parameter_mapping(BENCHMARK_CONFIGS)
paired_benchmark_params = create_paired_omni_benchmark_pytest_params(BENCHMARK_CONFIGS, DEPLOY_CONFIGS_DIR)

_omni_server_lock = threading.Lock()


class _SingleActiveContext:
    """Reuse one active context while its configuration key is unchanged."""

    def __init__(self) -> None:
        self._key: Any = None
        self._stack: ExitStack | None = None
        self._value: Any = None

    def acquire(self, key: Any, factory: Callable[[], AbstractContextManager[Any]]) -> Any:
        if self._stack is not None and key == self._key:
            return self._value

        self.close()
        stack = ExitStack()
        value = stack.enter_context(factory())
        self._key = key
        self._stack = stack
        self._value = value
        return value

    def close(self) -> None:
        stack = self._stack
        self._key = None
        self._stack = None
        self._value = None
        if stack is not None:
            stack.close()


def _config_for_test(test_name: str) -> dict[str, object] | None:
    for config in BENCHMARK_CONFIGS:
        if isinstance(config, dict) and config.get("test_name") == test_name:
            return config
    return None


def _judge_server_params_for_test(test_name: str) -> dict[str, object] | None:
    config = _config_for_test(test_name)
    if config is None:
        return None
    raw = config.get("judge_server_params")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"judge_server_params for {test_name} must be an object")
    return raw


def _resolve_judge_cuda_visible_devices(configured: object) -> str:
    """Map a judge card index onto the process-visible CUDA device list.

    ``judge_server_params.cuda_visible_devices`` is an index into
    ``CUDA_VISIBLE_DEVICES`` when that env is set (``1`` with
    ``CUDA_VISIBLE_DEVICES=2,3`` selects physical GPU 3). Without the env it
    is used as a raw device list.
    """
    if not isinstance(configured, str) or not configured:
        raise ValueError("judge_server_params.cuda_visible_devices must be a non-empty string")
    parent = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent is None or not parent.strip():
        return configured
    visible = [part.strip() for part in parent.split(",") if part.strip()]
    if configured.isdigit():
        index = int(configured)
        if index < 0 or index >= len(visible):
            raise ValueError(
                f"judge_server_params.cuda_visible_devices={configured!r} is outside CUDA_VISIBLE_DEVICES={parent!r}"
            )
        return visible[index]
    return configured


@contextmanager
def _start_judge_server(judge_params: dict[str, object]):
    model = judge_params.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("judge_server_params.model must be a non-empty string")
    extra = judge_params.get("extra_cli_args") or ()
    if not isinstance(extra, list | tuple):
        raise TypeError("judge_server_params.extra_cli_args must be a list")
    visible = _resolve_judge_cuda_visible_devices(judge_params.get("cuda_visible_devices", "1"))
    print(f"Starting OmniInteract judge with model: {model} on CUDA_VISIBLE_DEVICES={visible}")
    with OmniServer(
        model,
        [str(item) for item in extra],
        use_omni=bool(judge_params.get("use_omni", False)),
        env_dict={"CUDA_VISIBLE_DEVICES": visible},
    ) as judge:
        print(f"OmniInteract judge started on {judge.host}:{judge.port}")
        yield judge
        print("OmniInteract judge stopping...")
    print("OmniInteract judge stopped")


@contextmanager
def _start_omni_server(server_param):
    test_name, model, stage_config_path, stage_overrides, extra_cli_args, use_omni = server_param

    print(f"Starting OmniServer with test: {test_name}, model: {model}")

    server_args: list[str] = []
    if use_omni:
        server_args += ["--stage-init-timeout", "600", "--init-timeout", "900"]
    # --deploy-config and --stage-overrides compose at the CLI (see vllm_omni/entrypoints/utils.py):
    # deploy-config sets the base; stage-overrides are applied on top. Both can be set.
    if stage_config_path:
        server_args = ["--deploy-config", stage_config_path] + server_args
    if stage_overrides:
        server_args = ["--stage-overrides", stage_overrides] + server_args
    if extra_cli_args:
        server_args = list(extra_cli_args) + server_args
    judge_params = _judge_server_params_for_test(test_name)
    with OmniServer(model, server_args, use_omni=use_omni) as server:
        server.test_name = test_name
        server.judge_base_url = None
        print("OmniServer started successfully")
        if judge_params is None:
            yield server
        else:
            with _start_judge_server(judge_params) as judge:
                server.judge_base_url = f"http://{judge.host}:{judge.port}"
                yield server
        print("OmniServer stopping...")

    print("OmniServer stopped")


@pytest.fixture(scope="module")
def omni_server_context():
    """Start vLLM-Omni server as a subprocess with actual model weights.
    Reuse it for adjacent benchmark cases with the same server configuration.
    Multi-stage initialization can take 10-20+ minutes.
    """
    with _omni_server_lock:
        active_context = _SingleActiveContext()
        try:
            yield active_context
        finally:
            active_context.close()


@pytest.fixture
def omni_server(request, omni_server_context):
    return omni_server_context.acquire(request.param, lambda: _start_omni_server(request.param))


@pytest.fixture
def benchmark_params(request):
    """Benchmark parameters fixture; paired with ``omni_server`` via parametrization."""
    test_name, param_index = request.param

    all_params = get_benchmark_params_for_server(test_name, server_to_benchmark_mapping)

    if not all_params:
        raise ValueError(f"No benchmark parameters found for test: {test_name}")

    if param_index >= len(all_params):
        raise ValueError(f"No benchmark parameters found for index {param_index} in test: {test_name}")

    current = param_index + 1
    total = len(all_params)
    print(f"\n  Running benchmark {current}/{total} for {test_name}")

    return {
        "test_name": test_name,
        "params": all_params[param_index],
    }


def _resolve_num_warmups(params: dict[str, Any], *, default: int) -> int:
    value = params.get("num_warmups")
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("num_warmups must be a non-negative integer")
    return value


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


_OMNIINTERACT_AGGREGATE_MIN_IA_QTF1 = "omniinteract_aggregate_min_ia_qtf1"
_OMNIINTERACT_AGGREGATE_SUBSETS = "omniinteract_aggregate_subsets"
_OMNIINTERACT_AGGREGATE_GROUP = "omniinteract_aggregate_group"
_OMNIINTERACT_AGGREGATE_COUNTS: dict[tuple[str, str], tuple[float, float, float]] = {}
_OMNIINTERACT_MIN_EVALUATED_CASES = 1


def _reset_omniinteract_aggregate_counts() -> None:
    _OMNIINTERACT_AGGREGATE_COUNTS.clear()


def _ia_qtf1_from_counts(tp: float, fp: float, fn: float) -> float:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return (2.0 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def _assert_subset_evaluated(accuracy: dict[str, object]) -> None:
    """Reject a subset whose clipped or cancelled cases left nothing to score.

    Individual ineligible cases may be skipped. A subset with fewer than
    ``_OMNIINTERACT_MIN_EVALUATED_CASES`` evaluated cases must not pass, or its
    zero TP/FP/FN would be pooled as a successful accuracy result.
    """

    evaluated = accuracy.get("evaluated")
    if not isinstance(evaluated, int) or isinstance(evaluated, bool):
        raise AssertionError("OmniInteract accuracy evaluated count is missing")
    if evaluated < _OMNIINTERACT_MIN_EVALUATED_CASES:
        raise AssertionError(
            f"OmniInteract subset evaluated {evaluated} cases (skipped={accuracy.get('skipped')}); "
            f"at least {_OMNIINTERACT_MIN_EVALUATED_CASES} evaluated case is required"
        )


def _finite_accuracy_count(value: object, name: str) -> float:
    assert _is_finite_number(value), f"OmniInteract accuracy summary {name} is missing"
    return float(value)


def _aggregate_accuracy_params_for_test(test_name: str) -> dict[str, object]:
    config = _config_for_test(test_name)
    if config is None:
        return {}
    min_ia_qtf1 = config.get(_OMNIINTERACT_AGGREGATE_MIN_IA_QTF1)
    if min_ia_qtf1 is None:
        return {}
    if not _is_finite_number(min_ia_qtf1):
        raise ValueError("omniinteract_aggregate_min_ia_qtf1 must be a finite number")
    subsets: list[str] = []
    raw_params = config.get("benchmark_params") or []
    if not isinstance(raw_params, list):
        raise TypeError(f"benchmark_params for {test_name} must be a list")
    for item in raw_params:
        if not isinstance(item, dict) or not item.get("omniinteract_evaluate"):
            continue
        subset = item.get("omniinteract_subsets")
        if not isinstance(subset, str) or not subset:
            raise ValueError("omniinteract_aggregate_min_ia_qtf1 requires omniinteract_subsets on each evaluated case")
        subsets.append(subset)
    if not subsets:
        raise ValueError("omniinteract_aggregate_min_ia_qtf1 requires at least one evaluated OmniInteract subset")
    if len(set(subsets)) != len(subsets):
        raise ValueError("omniinteract_aggregate_subsets must not contain duplicates")
    return {
        _OMNIINTERACT_AGGREGATE_MIN_IA_QTF1: float(min_ia_qtf1),
        _OMNIINTERACT_AGGREGATE_SUBSETS: subsets,
        _OMNIINTERACT_AGGREGATE_GROUP: test_name,
    }


def _maybe_assert_aggregate_ia_qtf1(params: dict[str, object], acc_summary: dict[str, object]) -> None:
    min_ia_qtf1 = params.get(_OMNIINTERACT_AGGREGATE_MIN_IA_QTF1)
    if min_ia_qtf1 is None:
        return
    assert _is_finite_number(min_ia_qtf1), "omniinteract_aggregate_min_ia_qtf1 must be a finite number"
    subsets = params.get(_OMNIINTERACT_AGGREGATE_SUBSETS)
    if not isinstance(subsets, list | tuple) or not subsets:
        raise ValueError("omniinteract_aggregate_min_ia_qtf1 requires omniinteract_aggregate_subsets")
    if len(set(subsets)) != len(subsets):
        raise ValueError("omniinteract_aggregate_subsets must not contain duplicates")
    if not all(isinstance(item, str) and item for item in subsets):
        raise ValueError("omniinteract_aggregate_subsets must be non-empty strings")
    subset = params.get("omniinteract_subsets")
    if not isinstance(subset, str) or not subset:
        raise ValueError("omniinteract_aggregate_min_ia_qtf1 requires omniinteract_subsets")
    if subset not in subsets:
        raise ValueError(f"omniinteract_subsets {subset!r} is not in omniinteract_aggregate_subsets")
    group = params.get(_OMNIINTERACT_AGGREGATE_GROUP)
    if not isinstance(group, str) or not group:
        raise ValueError("omniinteract_aggregate_min_ia_qtf1 requires omniinteract_aggregate_group")
    tp = _finite_accuracy_count(acc_summary.get("Global_TP"), "Global_TP")
    fp = _finite_accuracy_count(acc_summary.get("Global_FP"), "Global_FP")
    fn = _finite_accuracy_count(acc_summary.get("Global_FN"), "Global_FN")
    _OMNIINTERACT_AGGREGATE_COUNTS[(group, subset)] = (tp, fp, fn)
    recorded = [_OMNIINTERACT_AGGREGATE_COUNTS.get((group, name)) for name in subsets]
    present = [counts for counts in recorded if counts is not None]
    if len(present) != len(subsets):
        return
    total_tp = sum(counts[0] for counts in present)
    total_fp = sum(counts[1] for counts in present)
    total_fn = sum(counts[2] for counts in present)
    ia_qtf1 = _ia_qtf1_from_counts(total_tp, total_fp, total_fn)
    print(
        f"OmniInteract aggregate All Global IA-QTF1: {ia_qtf1:.6f} "
        f"(TP={total_tp:.6f} / FP={total_fp:.6f} / FN={total_fn:.6f})"
    )
    assert ia_qtf1 >= float(min_ia_qtf1), f"OmniInteract aggregate All Global IA-QTF1 {ia_qtf1} is below {min_ia_qtf1}"


def assert_result(result, params, num_prompt) -> None:
    assert result["completed"] == num_prompt, "Request failures exist"
    if params.get("dataset_name") == "omniinteract":
        summary = result.get("omniinteract")
        assert isinstance(summary, dict), "OmniInteract summary is missing"
        assert (summary.get("total"), summary.get("success"), summary.get("failed")) == (
            num_prompt,
            num_prompt,
            0,
        ), "OmniInteract requests did not all succeed"
        assert summary.get("artifacts_complete") is True, "OmniInteract artifacts are incomplete"
        if params.get("omniinteract_evaluate"):
            accuracy = summary.get("accuracy")
            assert isinstance(accuracy, dict), "OmniInteract accuracy is missing"
            assert accuracy.get("status") == "ok", "OmniInteract accuracy did not complete"
            assert accuracy.get("failed") == 0, "OmniInteract accuracy reported failed cases"
            _assert_subset_evaluated(accuracy)
            acc_summary = accuracy.get("summary")
            assert isinstance(acc_summary, dict), "OmniInteract accuracy summary is missing"
            ia_qtf1 = acc_summary.get("IA_QTF1")
            assert _is_finite_number(ia_qtf1), "OmniInteract All Global IA-QTF1 is missing"
            min_ia_qtf1 = params.get("omniinteract_min_ia_qtf1")
            if min_ia_qtf1 is not None:
                assert _is_finite_number(min_ia_qtf1), "omniinteract_min_ia_qtf1 must be a finite number"
                assert float(ia_qtf1) >= float(min_ia_qtf1), (
                    f"OmniInteract All Global IA-QTF1 {ia_qtf1} is below {min_ia_qtf1}"
                )
            _maybe_assert_aggregate_ia_qtf1(params, acc_summary)
    baseline = params.get("baseline")
    hardware = result.get("Hardware")
    hardware_baseline = baseline.get(hardware) if isinstance(baseline, dict) and isinstance(hardware, str) else None
    if isinstance(hardware_baseline, dict) and "mean_tpot_ms" in hardware_baseline:
        num_tpot_samples = result.get("num_tpot_samples")
        mean_tpot_ms = result.get("mean_tpot_ms")
        assert isinstance(num_tpot_samples, int) and not isinstance(num_tpot_samples, bool) and num_tpot_samples > 0, (
            "TPOT baseline is configured, but no measurable TPOT samples were produced"
        )
        assert (
            isinstance(mean_tpot_ms, int | float) and not isinstance(mean_tpot_ms, bool) and math.isfinite(mean_tpot_ms)
        ), "TPOT baseline is configured, but mean_tpot_ms is not finite"
    expected_audio_turns = params.get("expected_duplex_audio_turns_per_session")
    if expected_audio_turns is not None:
        session_metrics = result.get("duplex_session_metrics")
        assert isinstance(session_metrics, list), "Duplex session metrics are missing"
        assert len(session_metrics) == num_prompt, (
            f"Expected {num_prompt} duplex session metric rows, got {len(session_metrics)}"
        )
        assert all(
            isinstance(metric, dict) and metric.get("audio_turn_count") == expected_audio_turns
            for metric in session_metrics
        ), f"Not every duplex session emitted {expected_audio_turns} audio turns"


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "omni_server,benchmark_params",
    paired_benchmark_params,
    indirect=["omni_server", "benchmark_params"],
)
def test_performance_benchmark(omni_server, benchmark_params):
    test_name = benchmark_params["test_name"]
    params = dict(benchmark_params["params"])
    dataset_name = params.get("dataset_name", "")
    if params.get("omniinteract_evaluate"):
        judge_base_url = getattr(omni_server, "judge_base_url", None)
        if not isinstance(judge_base_url, str) or not judge_base_url:
            raise ValueError("omniinteract_evaluate requires top-level judge_server_params to start a judge")
        params["omniinteract_judge_base_url"] = judge_base_url

    host = omni_server.host
    port = omni_server.port
    model = omni_server.model

    print(f"Running benchmark for model: {model}")
    print(f"Benchmark parameters: {benchmark_params}")

    resource_label = get_runtime_resource_label()

    def to_list(value, default=None):
        if value is None:
            return [] if default is None else [default]
        return [value] if not isinstance(value, (list, tuple)) else list(value)

    qps_list = to_list(params.get("request_rate"))
    num_prompt_list = to_list(params.get("num_prompts"))
    max_concurrency_list = to_list(params.get("max_concurrency"))

    max_len = max(len(qps_list), len(max_concurrency_list))
    if len(num_prompt_list) == 1 and max_len > 1:
        num_prompt_list = num_prompt_list * max_len
    elif max_len == 1 and len(num_prompt_list) > 1:
        if len(qps_list) == 1:
            qps_list = qps_list * len(num_prompt_list)
        if len(max_concurrency_list) == 1:
            max_concurrency_list = max_concurrency_list * len(num_prompt_list)
        max_len = max(len(qps_list), len(max_concurrency_list))
    elif len(num_prompt_list) != max_len and max_len > 0:
        raise ValueError("The number of prompts does not match the QPS or max_concurrency")

    args = ["--host", host, "--port", str(port)]
    exclude_keys = {
        "request_rate",
        "baseline",
        "num_prompts",
        "max_concurrency",
        "num_warmups",
        "task",
        "enabled",
        "eval_phase",
        "trust_remote_code",
        "expected_duplex_audio_turns_per_session",
        "omniinteract_min_ia_qtf1",
        _OMNIINTERACT_AGGREGATE_MIN_IA_QTF1,
        _OMNIINTERACT_AGGREGATE_SUBSETS,
        _OMNIINTERACT_AGGREGATE_GROUP,
        "judge_server_params",
    }

    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue

        arg_name = f"--{key.replace('_', '-')}"

        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            json_str = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            args.extend([arg_name, json_str])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])

    for config in BENCHMARK_CONFIGS:
        if config.get("test_name") != test_name:
            continue
        server_params = config.get("server_params") or {}
        if server_params.get("trust_remote_code") or params.get("trust_remote_code"):
            args.append("--trust-remote-code")
        break

    # QPS / request-rate sweep
    for sweep_index, (qps, num_prompt) in enumerate(zip(qps_list, num_prompt_list)):
        args = args + ["--request-rate", str(qps), "--num-prompts", str(num_prompt)]
        result = run_benchmark(
            args=args,
            test_name=test_name,
            flow=qps,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            baseline_config=params.get("baseline"),
            sweep_index=sweep_index,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
            resource_label=resource_label,
            num_warmups=_resolve_num_warmups(params, default=2),
        )
        assert_result(result, {**params, **_aggregate_accuracy_params_for_test(test_name)}, num_prompt)

    # concurrency test
    for sweep_index, (concurrency, num_prompt) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        args = args + ["--max-concurrency", str(concurrency), "--num-prompts", str(num_prompt), "--request-rate", "inf"]
        result = run_benchmark(
            args=args,
            test_name=test_name,
            flow=concurrency,
            dataset_name=dataset_name,
            num_prompt=num_prompt,
            baseline_config=params.get("baseline"),
            sweep_index=sweep_index,
            random_input_len=params.get("random_input_len"),
            random_output_len=params.get("random_output_len"),
            resource_label=resource_label,
            num_warmups=_resolve_num_warmups(params, default=max(2, int(concurrency))),
        )
        assert_result(result, {**params, **_aggregate_accuracy_params_for_test(test_name)}, num_prompt)
