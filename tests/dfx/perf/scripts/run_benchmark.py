# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager
from functools import partial
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
    with OmniServer(model, server_args, use_omni=use_omni) as server:
        server.test_name = test_name
        print("OmniServer started successfully")
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


# ---------------------------------------------------------------------------
# Omni-DuplexEval three-phase orchestration (design v2 §3.1 / §5.7 / §8.4)
#
# Phase 1 (generate, timed, peak 1 card) stays on the existing
# ``omni_server`` + ``conftest.run_benchmark`` path. Phase 2 (judge) and
# Phase 3 (summarize + merge) run after the timed server has exited, in
# separate sub-processes, so the judge never shares a card or a time slice
# with the benchmark window.
# ---------------------------------------------------------------------------

DUPLEX_EVAL_DATASET_NAME = "omni-duplex-eval"

#: ``benchmark_params`` keys that are *not* ``vllm bench serve`` CLI flags.
#: Every other key is rendered as ``--<key with underscores -> dashes>``; an
#: unknown flag is a hard argparse error, so this allow-list must stay in sync
#: with the perf JSON (see ``tests/dfx/perf/tests/test_runner_metadata.py``).
_BENCHMARK_PARAM_EXCLUDE_KEYS = frozenset(
    {
        "request_rate",
        "baseline",
        "num_prompts",
        "max_concurrency",
        "num_warmups",
        "task",
        "name",
        "enabled",
        "eval_phase",
        "trust_remote_code",
        "expected_duplex_audio_turns_per_session",
        # Omni-DuplexEval judge / summarizer knobs: Phase 2/3 only (design §8.3).
        "duplex_eval_judge_video_mode",
        "duplex_eval_judge_fps",
        "duplex_eval_judge_modalities",
        "duplex_eval_judge_model",
        "duplex_eval_content_frame_limit",
        "duplex_eval_eval_workers",
        "duplex_eval_exclude_ids_pending",
        # Optional Gate knob: minimum number of successfully scored samples.
        "duplex_eval_min_scored",
    }
)

_JUDGE_HEALTH_RETRIES = 60
_JUDGE_HEALTH_INTERVAL_S = 2.0
_JUDGE_HEALTH_TIMEOUT_S = 5.0
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")

_duplex_eval_lock = threading.Lock()
_judge_server_lock = threading.Lock()
#: Phase-1 results that Phase 2/3 must judge and merge (design §5.7 registry).
_DUPLEX_EVAL_CASES: list[dict[str, Any]] = []
#: test_names whose Phase-1 parametrization was actually *collected* (even if it
#: failed before registering a result). Distinguishes "deselected by -m" from
#: "generation broke", so the Phase-2 guard fails loudly only in the latter.
_DUPLEX_EVAL_ATTEMPTED: set[str] = set()


def _duplex_eval_configs() -> list[dict[str, Any]]:
    """Collected perf configs that declare at least one Omni-DuplexEval case."""
    return [
        cfg
        for cfg in BENCHMARK_CONFIGS
        if any(
            (entry or {}).get("dataset_name") == DUPLEX_EVAL_DATASET_NAME
            for entry in (cfg.get("benchmark_params") or [])
        )
    ]


def _judge_server_params_by_test_name() -> dict[str, dict[str, Any]]:
    return {str(cfg["test_name"]): dict(cfg.get("judge_server_params") or {}) for cfg in _duplex_eval_configs()}


def _expand_env_path(value: str) -> str:
    """Expand ``${VAR}`` / ``$VAR`` / ``~`` so perf JSON never embeds host paths."""

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        resolved = os.environ.get(name)
        if not resolved:
            raise ValueError(
                f"Omni-DuplexEval perf config references unset environment variable {name!r}; "
                "export it (e.g. BENCHMARK_ASSETS=<mirror root>) or point dataset_path at a "
                "pinned Hugging Face id"
            )
        return resolved

    return os.path.expanduser(_ENV_VAR_RE.sub(_replace, value))


def _resolve_duplex_eval_params(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve env-var/``~`` path parameters and fail loudly on a missing mirror.

    Design §8.6 requires the cropped local mirror to live outside the repo, so
    the JSON references it through ``BENCHMARK_ASSETS`` rather than a
    machine-specific absolute path.
    """
    resolved = dict(params)
    for key in ("dataset_path", "duplex_eval_media_root", "duplex_eval_ref_audio"):
        value = resolved.get(key)
        if isinstance(value, str) and ("$" in value or value.startswith("~")):
            try:
                resolved[key] = _expand_env_path(value)
            except ValueError as exc:
                pytest.fail(str(exc), pytrace=False)
    dataset_path = resolved.get("dataset_path")
    if isinstance(dataset_path, str) and dataset_path.startswith(("/", "~")) and not Path(dataset_path).exists():
        pytest.fail(
            f"Omni-DuplexEval dataset_path {dataset_path!r} does not exist. Build the cropped local "
            "mirror (design §8.6, tools/benchmarks/build_omni_duplex_eval_mirror.py) and export "
            "BENCHMARK_ASSETS, or pin a Hugging Face dataset id.",
            pytrace=False,
        )
    return resolved


def _register_duplex_eval_case(
    *,
    test_name: str,
    params: dict[str, Any],
    result_path: str,
    num_prompt: int,
    selected_ids: list[str],
) -> None:
    with _duplex_eval_lock:
        _DUPLEX_EVAL_CASES.append(
            {
                "test_name": test_name,
                "params": dict(params),
                "result_path": result_path,
                "num_prompt": int(num_prompt),
                "selected_ids": list(selected_ids),
            }
        )


def _register_phase_one_result(
    *,
    result: dict[str, Any],
    params: dict[str, Any],
    test_name: str,
    flow: Any,
    num_prompt: int,
    since: float,
) -> None:
    """Locate and record the Phase-1 perf result so Phase 2/3 can merge into it."""
    if params.get("dataset_name") != DUPLEX_EVAL_DATASET_NAME:
        return
    from vllm_omni.benchmarks.duplex_eval import _locate_result_file

    result_path = _locate_result_file(
        bench_dir=os.environ.get("BENCHMARK_DIR", "tests"),
        test_name=test_name,
        dataset_name=DUPLEX_EVAL_DATASET_NAME,
        flow=flow,
        num_prompt=num_prompt,
        since=since,
    )
    summary = result.get("duplex_eval") or {}
    _register_duplex_eval_case(
        test_name=test_name,
        params=params,
        result_path=str(result_path),
        num_prompt=num_prompt,
        selected_ids=[str(item) for item in (summary.get("selected_ids") or [])],
    )


def _assert_duplex_eval_result(result: dict[str, Any], params: dict[str, Any], num_prompt: int) -> None:
    """Gate G1-G6 (design §7.2), tolerant of the pre-merge (generate-only) phase."""
    summary = result.get("duplex_eval")
    assert isinstance(summary, dict), "Omni-DuplexEval summary is missing"
    assert summary.get("total") == num_prompt, (
        f"Omni-DuplexEval selected {summary.get('total')} samples, expected {num_prompt}"
    )
    assert summary.get("generated") == summary.get("total"), (
        "Omni-DuplexEval generation incomplete: "
        f"generated={summary.get('generated')} total={summary.get('total')} ids={summary.get('failure_ids')}"
    )
    assert summary.get("artifacts_complete") is True, "Omni-DuplexEval artifacts are incomplete"
    assert summary.get("clock") == "media", f"Omni-DuplexEval clock must be 'media', got {summary.get('clock')!r}"
    assert not summary.get("clock_mismatch_ids"), (
        f"Omni-DuplexEval clock mismatch ids={summary.get('clock_mismatch_ids')}"
    )
    exclude = {str(item) for item in (params.get("duplex_eval_exclude_ids") or [])}
    assert {str(item) for item in (summary.get("exclude_ids") or [])} == exclude, (
        f"Omni-DuplexEval exclude-id echo mismatch: result={summary.get('exclude_ids')} params={sorted(exclude)}"
    )
    # G6: `exclude_ids` may be bare ids ("565") or `split/id`, while
    # `selected_ids` is always `split/id`; compare on both forms so a bare id
    # exclusion is not silently vacuous (design §7.2 leaves this implicit).
    overlap = sorted(
        str(key)
        for key in (summary.get("selected_ids") or [])
        if str(key) in exclude or str(key).rsplit("/", 1)[-1] in exclude
    )
    assert not overlap, f"Omni-DuplexEval excluded samples were still selected: {overlap}"

    if not summary.get("judge_enabled"):
        # Phase 1 (generate-only) or a degraded judge run: G4/G5 do not apply.
        return
    assert summary.get("phase") == "generate+evaluate+summarize", (
        f"Omni-DuplexEval judge phase did not complete: phase={summary.get('phase')!r}"
    )
    score_summary = summary.get("score_summary")
    assert isinstance(score_summary, dict), "Omni-DuplexEval score_summary is missing"
    assert score_summary.get("protocol_pin"), "Omni-DuplexEval score_summary is missing protocol_pin"
    assert summary.get("scored") == score_summary.get("samples"), (
        "Omni-DuplexEval scored count disagrees with score_summary.samples"
    )
    min_scored = params.get("duplex_eval_min_scored")
    if min_scored is not None:
        assert int(summary.get("scored") or 0) >= int(min_scored), (
            f"Omni-DuplexEval judge coverage too low: scored={summary.get('scored')} "
            f"min_scored={min_scored} pending={summary.get('pending_ids')}"
        )


def _resolve_num_warmups(params: dict[str, Any], *, default: int) -> int:
    value = params.get("num_warmups")
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("num_warmups must be a non-negative integer")
    return value


def assert_result(result, params, num_prompt) -> None:
    assert result["completed"] == num_prompt, "Request failures exist"
    if params.get("dataset_name") == DUPLEX_EVAL_DATASET_NAME:
        _assert_duplex_eval_result(result, params, num_prompt)
    if params.get("dataset_name") == "omniinteract":
        summary = result.get("omniinteract")
        assert isinstance(summary, dict), "OmniInteract summary is missing"
        assert (summary.get("total"), summary.get("success"), summary.get("failed")) == (
            num_prompt,
            num_prompt,
            0,
        ), "OmniInteract requests did not all succeed"
        assert summary.get("artifacts_complete") is True, "OmniInteract artifacts are incomplete"
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
    params = benchmark_params["params"]
    dataset_name = params.get("dataset_name", "")
    if dataset_name == DUPLEX_EVAL_DATASET_NAME:
        # Resolve ${VAR}/~ mirror references before the generic CLI rendering
        # below turns every key into a ``vllm bench serve`` flag.
        params = _resolve_duplex_eval_params(params)
        with _duplex_eval_lock:
            _DUPLEX_EVAL_ATTEMPTED.add(test_name)

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
    exclude_keys = _BENCHMARK_PARAM_EXCLUDE_KEYS

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
        phase_one_started = time.time()
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
        assert_result(result, params, num_prompt)
        _register_phase_one_result(
            result=result,
            params=params,
            test_name=test_name,
            flow=qps,
            num_prompt=num_prompt,
            since=phase_one_started,
        )

    # concurrency test
    for sweep_index, (concurrency, num_prompt) in enumerate(zip(max_concurrency_list, num_prompt_list)):
        phase_one_started = time.time()
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
        assert_result(result, params, num_prompt)
        _register_phase_one_result(
            result=result,
            params=params,
            test_name=test_name,
            flow=concurrency,
            num_prompt=num_prompt,
            since=phase_one_started,
        )


# ---------------------------------------------------------------------------
# Phase 2/3/4: judge server, evaluate, summarize, merge, Gate (design §5.7)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def judge_server_context():
    """Lazily start/stop the Qwen3-Omni judge server (separate lock from omni_server)."""
    with _judge_server_lock:
        active_context = _SingleActiveContext()
        try:
            yield active_context
        finally:
            active_context.close()


def _omni_bench_cli() -> list[str]:
    return [sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "bench"]


def _judge_health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/models"


def _wait_for_judge_health(host: str, port: int, *, required: bool) -> bool:
    """Poll the judge's OpenAI endpoint; hard-fail unless ``required`` is false."""
    url = _judge_health_url(host, port)
    last_error: str | None = None
    for _attempt in range(_JUDGE_HEALTH_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=_JUDGE_HEALTH_TIMEOUT_S) as response:
                if response.status == 200:
                    return True
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(_JUDGE_HEALTH_INTERVAL_S)
    message = f"Omni-DuplexEval judge unreachable at {url} after {_JUDGE_HEALTH_RETRIES} attempts: {last_error}"
    if required:
        pytest.fail(message, pytrace=False)
    print(f"{message} (judge_server_params.required=false -> degrading)")
    return False


def _judge_server_param(test_name: str, judge_cfg: dict[str, Any]) -> tuple:
    """Build the ``_start_omni_server`` tuple for the judge (reuses its construction)."""
    model = str(judge_cfg.get("model") or "").strip()
    if not model:
        pytest.fail(f"{test_name}: judge_server_params.model is required", pytrace=False)
    extra_cli_args = tuple(str(item) for item in (judge_cfg.get("extra_cli_args") or ()))
    use_omni = bool(judge_cfg.get("use_omni", True))
    return (f"{test_name}::judge", model, None, None, extra_cli_args, use_omni)


def _run_or_fail(command: list[str], *, phase: str) -> None:
    print(f"[duplex-eval] {phase}: {' '.join(command)}")
    process = subprocess.run(command, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    if process.stdout:
        print(process.stdout)
    if process.returncode != 0:
        tail = "\n".join((process.stderr or "").strip().splitlines()[-50:])
        pytest.fail(
            f"Omni-DuplexEval {phase} failed with exit code {process.returncode}\n"
            f"command: {' '.join(command)}\nstderr tail:\n{tail}",
            pytrace=False,
        )


def _build_evaluate_command(case: dict[str, Any], *, judge_url: str, judge_model: str) -> list[str]:
    """Translate the serve-side case parameters into the standalone evaluate CLI (U7)."""
    params = case["params"]
    command = _omni_bench_cli() + ["omni-duplex-eval", "--omni", "evaluate"]
    command += ["--dataset", str(params["dataset_path"]), "--split", str(params.get("duplex_eval_split", "all"))]
    command += ["--response-root", str(params["duplex_eval_output_dir"])]
    command += ["--score-root", str(params["duplex_eval_score_dir"])]
    if params.get("duplex_eval_media_root"):
        command += ["--media-root", str(params["duplex_eval_media_root"])]
    command += ["--judge-base-url", judge_url, "--judge-model", judge_model]
    command += ["--judge-video-mode", str(params.get("duplex_eval_judge_video_mode", "frame-sample"))]
    command += ["--judge-fps", str(int(params.get("duplex_eval_judge_fps", 2)))]
    command += ["--content-frame-limit", str(int(params.get("duplex_eval_content_frame_limit", 16)))]
    command += ["--eval-workers", str(int(params.get("duplex_eval_eval_workers", 1)))]
    if params.get("duplex_eval_allow_invalid_clock"):
        command.append("--allow-invalid-clock")
    modalities = params.get("duplex_eval_judge_modalities")
    if modalities:
        command += ["--judge-modalities", *[str(item) for item in modalities]]
    # Phase-1 selected_ids are ``split/id``; ``load_samples`` matches bare ids and
    # ``--split`` already pins the split, so strip the prefix (design §5.7 P2).
    selected_ids = [str(item).rsplit("/", 1)[-1] for item in (case.get("selected_ids") or [])]
    command += ["--ids", *selected_ids]
    exclude_ids = [str(item) for item in (params.get("duplex_eval_exclude_ids") or [])]
    if exclude_ids:
        command += ["--exclude-ids", *exclude_ids]
    return command


def _build_summarize_command(score_dir: Path) -> list[str]:
    return _omni_bench_cli() + [
        "omni-duplex-eval",
        "--omni",
        "summarize",
        "--score-root",
        str(score_dir),
        "--output-json",
        str(score_dir / "score_summary.json"),
    ]


def _judge_metadata(params: dict[str, Any], *, judge_url: str, judge_model: str, protocol_pin: str) -> dict[str, Any]:
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import CONTENT_FRAME_LIMIT

    return {
        "model": judge_model,
        "base_url": judge_url,
        "video_mode": str(params.get("duplex_eval_judge_video_mode", "frame-sample")),
        "fps": int(params.get("duplex_eval_judge_fps", 2)),
        "modalities": [str(item) for item in (params.get("duplex_eval_judge_modalities") or [])],
        "content_frame_limit": int(params.get("duplex_eval_content_frame_limit", CONTENT_FRAME_LIMIT)),
        "eval_workers": int(params.get("duplex_eval_eval_workers", 1)),
        "protocol_pin": protocol_pin,
    }


def _record_judge_skipped(result_path: Path, reason: str) -> None:
    """Degraded path (``judge_server_params.required=false``): mark and keep G1-G3."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    duplex = payload.get("duplex_eval")
    if not isinstance(duplex, dict):
        raise AssertionError(f"duplex_eval side-channel missing in {result_path}")
    if duplex.get("phase") != "generate":
        raise AssertionError(f"refusing to mark judge skipped: duplex_eval.phase={duplex.get('phase')!r}")
    duplex["judge_enabled"] = False
    duplex["judge_skipped_reason"] = reason
    tmp = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, result_path)


def _run_duplex_eval_phase2_and_merge(
    case: dict[str, Any],
    *,
    judge_server: Any,
    judge_cfg: dict[str, Any],
) -> None:
    from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import PROTOCOL_PIN
    from vllm_omni.benchmarks.duplex_eval import merge_duplex_eval_into_result

    params = case["params"]
    result_path = Path(str(case["result_path"]))
    score_dir = Path(str(params["duplex_eval_score_dir"]))
    judge_url = f"http://{judge_server.host}:{judge_server.port}"
    judge_model = str(judge_cfg.get("model") or "")
    if not case.get("selected_ids"):
        pytest.fail(f"Omni-DuplexEval Phase 1 recorded no selected_ids for {result_path.name}", pytrace=False)

    # Guard O-6: the judge must score exactly the Phase-1 selection set.
    _run_or_fail(
        _build_evaluate_command(case, judge_url=judge_url, judge_model=judge_model),
        phase="evaluate",
    )
    _run_or_fail(_build_summarize_command(score_dir), phase="summarize")

    summary_path = score_dir / "score_summary.json"
    if not summary_path.exists():
        pytest.fail(f"Omni-DuplexEval summarize produced no {summary_path}", pytrace=False)
    score_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if score_summary.get("protocol_pin") != PROTOCOL_PIN:
        pytest.fail(
            f"Omni-DuplexEval protocol pin mismatch: {score_summary.get('protocol_pin')!r} != {PROTOCOL_PIN!r}",
            pytrace=False,
        )

    judge_meta = _judge_metadata(params, judge_url=judge_url, judge_model=judge_model, protocol_pin=PROTOCOL_PIN)
    merged = merge_duplex_eval_into_result(
        result_path,
        score_summary=score_summary,
        judge_meta=judge_meta,
        phases=["generate", "evaluate", "summarize"],
    )
    # Gate G4-G6 on the merged result (Phase 1 already ran G1-G3).
    assert_result(merged, params, int(case["num_prompt"]))


@pytest.mark.benchmark
@pytest.mark.omni
@pytest.mark.local_model
def test_duplex_eval_judge_and_merge(omni_server_context, judge_server_context):
    """Phase 2/3/4: judge with a separate 2-card server, summarize, merge, Gate.

    Defined *after* ``test_performance_benchmark`` so pytest runs it once the
    Phase-1 parametrization has finished and every timed duplex server has been
    stopped (design §5.7 方案 A). Its ``omni`` + ``local_model`` marks mirror the
    documented local command (``-m "omni and local_model"``); an extra mark
    clause (e.g. ``cards_2``) would deselect this phase, so keep the documented
    expression.
    """
    if not _DUPLEX_EVAL_CASES:
        with _duplex_eval_lock:
            attempted = sorted(_DUPLEX_EVAL_ATTEMPTED)
        if attempted:
            pytest.fail(
                "Omni-DuplexEval generation phase produced no result; judge phase cannot run "
                f"(attempted test_names: {attempted})",
                pytrace=False,
            )
        pytest.skip("no Omni-DuplexEval case collected in this run")

    # Release the timed duplex server's cards before the judge starts (顺序独占).
    omni_server_context.close()

    judge_cfg_by_name = _judge_server_params_by_test_name()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in list(_DUPLEX_EVAL_CASES):
        grouped.setdefault(case["test_name"], []).append(case)

    for test_name, cases in grouped.items():
        judge_cfg = judge_cfg_by_name.get(test_name) or {}
        required = bool(judge_cfg.get("required", True))
        judge_param = _judge_server_param(test_name, judge_cfg)
        judge_server = judge_server_context.acquire(judge_param, partial(_start_omni_server, judge_param))
        try:
            healthy = _wait_for_judge_health(judge_server.host, judge_server.port, required=required)
            for case in cases:
                if not healthy:
                    _record_judge_skipped(
                        Path(str(case["result_path"])),
                        "judge unreachable (judge_server_params.required=false)",
                    )
                    continue
                _run_duplex_eval_phase2_and_merge(case, judge_server=judge_server, judge_cfg=judge_cfg)
        finally:
            judge_server_context.close()
