# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for pinned datasets on the shared DFX execution path."""

import hashlib
import io
import json
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.dfx import conftest as dfx
from tests.dfx.perf.scripts import run_benchmark as runner
from tests.e2e.accuracy.helpers import dataset_accuracy_errors
from vllm_omni.benchmarks.data_modules import DatasetSource
from vllm_omni.benchmarks.validation import metric_coverage_errors

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    def make(dataset="seed-tts", *, accuracy=True, count=None):
        root = tmp_path / dataset
        root.mkdir(exist_ok=True)
        if dataset == "seed-tts":
            (root / "en/prompt_wavs").mkdir(parents=True, exist_ok=True)
            (root / "en/prompt_wavs/ref.wav").write_bytes(b"reference")
            manifest = root / "en/meta.lst"
            # Keep an invalid source row: the loader must not replace it with a later row.
            manifest.write_text(
                "invalid-row\n" + "\n".join(f"id{i}|ref|prompt_wavs/ref.wav|text{i}" for i in range(4)) + "\n"
            )
        else:
            (root / "Videos").mkdir(exist_ok=True)
            manifest = root / "qa.json"
            manifest.write_text(json.dumps([{"question": f"q{i}"} for i in range(5)]))
        monkeypatch.setenv("TEST_DATASET_ROOT", str(root))
        expected = 5 if count is None else count
        limits = (
            {"max_seed_tts_mean_wer": 0.05, "min_seed_tts_mean_sim": 0.75}
            if dataset == "seed-tts"
            else {"min_daily_omni_accuracy": 0.69}
        )
        return {
            "dataset_name": dataset,
            "backend": "openai-chat-omni",
            "endpoint": "/v1/chat/completions",
            "num_prompts": [expected],
            "max_concurrency": [1],
            "extra_body": {"modalities": ["text", "audio"]},
            "dataset_source": {
                "repo": "test/dataset",
                "revision": "pinned-revision",
                "manifest": str(manifest.relative_to(root)),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "count": 5,
                "root_env": "TEST_DATASET_ROOT",
            },
            "result_validation": {
                "accuracy": accuracy,
                "require_audio": True,
                "metrics": {"mean_audio_rtf": "num_audio_rtf_samples"},
                "accuracy_baseline": limits if accuracy else {},
            },
            "result_metadata": {"benchmark_suite": "shared-dataset-test"},
            "benchmark_env": {
                "SEED_TTS_WER_EVAL": str(int(accuracy and dataset == "seed-tts")),
                "SEED_TTS_SIM_EVAL": str(int(accuracy and dataset == "seed-tts")),
                "SEED_TTS_UTMOS_EVAL": "0",
            },
        }

    return make


def _result(params):
    count = params["num_prompts"][0]
    result = {
        "request_count": count,
        "completed": count,
        "failed": 0,
        "successful_audio_requests": count,
        "mean_audio_rtf": 0.8,
        "num_audio_rtf_samples": count,
        "num_tpot_samples": 0,
        "mean_tpot_ms": 0.0,
    }
    if params["result_validation"]["accuracy"]:
        if params["dataset_name"] == "daily-omni":
            result.update(daily_omni_evaluated_ok=count, daily_omni_accuracy=0.8)
        else:
            result.update(
                seed_tts_session_count=count,
                seed_tts_request_failed=0,
                seed_tts_content_evaluated=count,
                seed_tts_content_error_mean=0.0,
                seed_tts_sim_evaluated=count,
                seed_tts_sim_mean=0.8,
            )
    return result


@pytest.mark.parametrize("model", ["Qwen/Qwen3-Omni", "openbmb/MiniCPM-o"])
@pytest.mark.parametrize("dataset", ["daily-omni", "seed-tts"])
@pytest.mark.parametrize("accuracy", [False, True])
def test_models_share_fixture_and_execution_path(tmp_path, monkeypatch, scenario, model, dataset, accuracy):
    params = scenario(dataset, accuracy=accuracy, count=5 if accuracy else 2)
    if dataset == "daily-omni":
        params["daily_omni_pack_mode"] = "minicpm-interleave" if model.startswith("openbmb/") else "qwen"
    else:
        params["seed_tts_system_prompt"] = f"Read the user text aloud for {model}."
    monkeypatch.setattr(runner, "server_to_benchmark_mapping", {"test_shared": {"benchmark_params": [params]}})
    monkeypatch.setattr(runner, "get_runtime_resource_label", lambda: "test-hardware")
    monkeypatch.setattr(dfx.subprocess, "check_output", lambda *args, **kwargs: "test-head\n")
    monkeypatch.setenv("SEED_TTS_UTMOS_EVAL", "1")
    perf = tmp_path / "results"
    perf.mkdir()
    monkeypatch.setenv("BENCHMARK_DIR", str(perf))
    payload = _result(params)
    commands = []

    class BenchmarkProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)
            assert dfx.os.environ["SEED_TTS_UTMOS_EVAL"] == "0"
            assert dfx.os.environ["SEED_TTS_WER_EVAL"] == str(int(accuracy and dataset == "seed-tts"))
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            path = Path(command[command.index("--result-dir") + 1]) / command[command.index("--result-filename") + 1]
            path.write_text(json.dumps(payload))

        def wait(self):
            return 0

    @dataclass
    class Request:
        param: tuple[str, int] = ("test_shared", 0)

    @dataclass
    class Server:
        model: str
        host: str = "localhost"
        port: int = 8000

    class Options:
        @staticmethod
        def getoption(name):
            assert name in ("assert_baseline", "require_baseline")
            return False

    monkeypatch.setattr(dfx.subprocess, "Popen", BenchmarkProcess)
    with contextmanager(runner.benchmark_params.__wrapped__)(Request()) as prepared:
        runner.test_performance_benchmark(Server(model), prepared, Options())
        subset = Path(
            prepared["params"]["dataset_path"] if dataset == "seed-tts" else prepared["params"]["daily_omni_qa_json"]
        )
        assert subset.exists()
    assert not subset.exists()
    assert dfx.os.environ["SEED_TTS_UTMOS_EVAL"] == "1"
    assert len(commands) == 1
    assert commands[0][commands[0].index("--model") + 1] == model
    option = "daily_omni_pack_mode" if dataset == "daily-omni" else "seed_tts_system_prompt"
    assert commands[0][commands[0].index(f"--{option.replace('_', '-')}") + 1] == params[option]
    saved = json.loads(next(perf.glob("result_*.json")).read_text())
    assert saved["validation_errors"] == []
    assert saved["mean_audio_rtf"] == payload["mean_audio_rtf"]
    assert saved["mean_tpot_ms"] is None
    assert saved["accuracy_evaluated"] is accuracy
    assert saved["qualification_scope"] == ("full-dataset" if accuracy else "performance-subset")


def test_seed_prefix_is_materialized_before_loader_filtering(scenario):
    params = scenario(accuracy=False, count=2)
    with ExitStack() as stack:
        prepared = dfx.prepare_dataset_benchmark(params, stack)
        path = prepared["seed_tts_root"]
        assert isinstance(path, str)
        root = Path(path)
        assert (root / "en/meta.lst").read_text().splitlines() == ["invalid-row", "id0|ref|prompt_wavs/ref.wav|text0"]
        assert (root / "en/prompt_wavs/ref.wav").read_bytes() == b"reference"


def test_accuracy_requires_full_pinned_population_except_smoke(scenario):
    params = scenario(count=2)
    with ExitStack() as stack, pytest.raises(ValueError, match="complete pinned dataset"):
        dfx.prepare_dataset_benchmark(params, stack)
    params["result_validation"]["smoke"] = True
    with ExitStack() as stack:
        dfx.prepare_dataset_benchmark(params, stack)


@pytest.mark.parametrize("indices", [[0, 0], [0, 5], [0, True], [0], "0,4"])
def test_daily_smoke_requires_distinct_matching_indices(scenario, indices):
    params = scenario("daily-omni", count=2)
    params["result_validation"]["smoke"] = True
    params["dataset_source"]["sample_indices"] = indices
    with ExitStack() as stack, pytest.raises(ValueError, match="indices"):
        dfx.prepare_dataset_benchmark(params, stack)


def test_manifest_mismatch_fails_before_loading(scenario):
    params = scenario()
    source = DatasetSource.from_config(params["dataset_source"])
    root = source.resolve(["en/**"])
    (root / "en/meta.lst").write_text("changed\n")
    with pytest.raises(ValueError, match="manifest differs"):
        source.resolve(["en/**"])


@pytest.mark.parametrize("dataset", ["daily-omni", "seed-tts"])
def test_incomplete_scores_fail_even_with_good_means(scenario, dataset):
    params = scenario(dataset)
    result = _result(params)
    key = "daily_omni_evaluated_ok" if dataset == "daily-omni" else "seed_tts_sim_evaluated"
    result[key] -= 1
    errors = dataset_accuracy_errors(result, dataset, 5, params["result_validation"]["accuracy_baseline"])
    assert any(key in error for error in errors)


@pytest.mark.parametrize("bad", [None, True, 0.0, float("nan"), float("inf")])
def test_metric_validity_is_required_without_performance_baselines(bad):
    result = {"mean_audio_rtf": bad, "num_audio_rtf_samples": 5}
    assert metric_coverage_errors(result, 5, {"mean_audio_rtf": "num_audio_rtf_samples"})


def test_finite_mean_does_not_hide_missing_measurements():
    result = {"mean_audio_rtf": 0.5, "num_audio_rtf_samples": 4}
    assert metric_coverage_errors(result, 5, {"mean_audio_rtf": "num_audio_rtf_samples"})


@pytest.mark.parametrize(
    "dataset,metric",
    [
        ("daily-omni", "daily_omni_accuracy"),
        ("seed-tts", "seed_tts_content_error_mean"),
        ("seed-tts", "seed_tts_sim_mean"),
    ],
)
def test_accuracy_scores_must_be_finite(scenario, dataset, metric):
    params = scenario(dataset)
    result = _result(params)
    result[metric] = float("nan")
    assert any(
        metric in error
        for error in dataset_accuracy_errors(result, dataset, 5, params["result_validation"]["accuracy_baseline"])
    )


def test_partial_result_diagnostics_are_saved_before_assertion(tmp_path, monkeypatch, scenario):
    params = scenario()
    result = _result(params)
    result["seed_tts_sim_evaluated"] = 4
    monkeypatch.setattr(dfx.subprocess, "check_output", lambda *args, **kwargs: "test-head\n")
    monkeypatch.setenv("BENCHMARK_DIR", str(tmp_path))

    class BenchmarkProcess:
        def __init__(self, command, **kwargs):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            path = Path(command[command.index("--result-dir") + 1]) / command[command.index("--result-filename") + 1]
            path.write_text(json.dumps(result))

        def wait(self):
            return 0

    monkeypatch.setattr(dfx.subprocess, "Popen", BenchmarkProcess)
    saved = dfx.run_benchmark(
        [],
        "test_partial",
        1,
        "seed-tts",
        5,
        resource_label="test-hardware",
        result_transform=lambda result, code: dfx.annotate_dataset_result(
            result, code, params, check_baseline=False, require_baseline=False
        ),
    )
    persisted = json.loads(next(tmp_path.glob("result_*.json")).read_text())
    assert persisted["validation_errors"] == saved["validation_errors"]
    assert any("seed_tts_sim_evaluated" in error for error in saved["validation_errors"])
    with pytest.raises(AssertionError, match="seed_tts_sim_evaluated"):
        runner.assert_result(saved, params, 5)
