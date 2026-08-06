"""Tests for DFX runner metadata field exclusion."""

import json

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_task_excluded_from_cli_args():
    """'task' field must not become --task CLI arg."""
    params = {
        "task": "voice_clone",
        "dataset_name": "seed-tts",
        "backend": "openai-audio-speech",
        "endpoint": "/v1/audio/speech",
        "percentile-metrics": "audio_rtf,audio_ttfp",
        "baseline": {"mean_audio_rtf": [0.5]},
    }
    exclude_keys = {
        "request_rate",
        "baseline",
        "num_prompts",
        "max_concurrency",
        "task",
        "enabled",
        "eval_phase",
        "trust_remote_code",
    }
    args = []
    for key, value in params.items():
        if key in exclude_keys or value is None:
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool) and value:
            args.append(arg_name)
        elif isinstance(value, dict):
            args.extend([arg_name, json.dumps(value)])
        elif not isinstance(value, bool):
            args.extend([arg_name, str(value)])
    assert "--task" not in args
    assert "--enabled" not in args
    assert "--dataset-name" in args


def test_enabled_false_entry_is_skipped():
    """benchmark_params entry with enabled=false should be skipped."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from tests.dfx.conftest import create_test_parameter_mapping

    configs = [
        {
            "test_name": "test_model",
            "server_params": {"model": "some/model"},
            "benchmark_params": [
                {
                    "task": "voice_clone",
                    "enabled": True,
                    "dataset_name": "seed-tts",
                    "backend": "openai-audio-speech",
                    "endpoint": "/v1/audio/speech",
                    "num_prompts": [10],
                    "max_concurrency": [1],
                    "percentile-metrics": "audio_rtf",
                    "baseline": {},
                },
                {
                    "task": "voice_design",
                    "enabled": False,
                    "dataset_name": "seed-tts-design",
                    "backend": "openai-audio-speech",
                    "endpoint": "/v1/audio/speech",
                    "num_prompts": [5],
                    "max_concurrency": [1],
                    "percentile-metrics": "audio_rtf",
                    "baseline": {},
                },
            ],
        }
    ]
    mapping = create_test_parameter_mapping(configs)
    params = mapping["test_model"]["benchmark_params"]
    # Only the enabled=True entry should appear
    assert len(params) == 1
    assert params[0].get("task") == "voice_clone"


def test_resolve_pytest_marks_hardware_dict_with_extra():
    from tests.dfx.conftest import resolve_pytest_marks

    marks = resolve_pytest_marks(
        [
            {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 2}},
            "full_model",
            "diffusion",
            "local_model",
        ]
    )
    names = {m.name for m in marks}
    assert "H100" in names
    assert "cuda" in names
    assert "gpu" in names
    assert "distributed_cuda" in names
    assert "full_model" in names
    assert "diffusion" in names
    assert "local_model" in names


def test_resolve_pytest_marks_rejects_legacy_object_format():
    from tests.dfx.conftest import resolve_pytest_marks

    with pytest.raises(ValueError, match="mark must be a list"):
        resolve_pytest_marks(
            {
                "hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1},
                "marks": ["full_model"],
            }
        )


def test_extract_mark_resource_label():
    from tests.dfx.conftest import extract_mark_resource_label

    assert extract_mark_resource_label(None) == "na"
    assert (
        extract_mark_resource_label(
            [
                {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}},
                "full_model",
            ]
        )
        == "H100"
    )
    assert (
        extract_mark_resource_label(
            [
                {
                    "hardware_marks": {
                        "res": {"cuda": "H100", "rocm": "MI325"},
                        "num_cards": 2,
                    }
                },
                "diffusion",
            ]
        )
        == "H100-MI325"
    )


def test_resource_label_for_filename():
    from tests.dfx.conftest import resource_label_for_filename

    assert resource_label_for_filename("H100") == ""
    assert resource_label_for_filename("L4") == "L4"
    assert resource_label_for_filename("910B") == "910B"
    assert resource_label_for_filename("na") == "na"


def test_hardware_json_value():
    from tests.dfx.conftest import hardware_json_value

    assert hardware_json_value("H100") == "H100"
    assert hardware_json_value("na") == ""
    assert hardware_json_value(None) == ""


def test_extract_configs_resource_label(monkeypatch):
    from tests.dfx.conftest import extract_configs_resource_label, get_runtime_resource_label

    monkeypatch.setattr(
        "tests.dfx.conftest._read_runtime_device_name",
        lambda *, device_id=0: "NVIDIA H100 80GB HBM3",
    )
    get_runtime_resource_label(refresh=True)
    assert extract_configs_resource_label([]) == "H100"
    get_runtime_resource_label(refresh=True)
    monkeypatch.setattr(
        "tests.dfx.conftest._read_runtime_device_name",
        lambda *, device_id=0: "Ascend910B2",
    )
    assert get_runtime_resource_label(refresh=True) == "910B"


def test_load_benchmark_configs_from_dir(tmp_path):
    from tests.dfx.conftest import load_benchmark_configs

    (tmp_path / "a.json").write_text(
        json.dumps([{"test_name": "test_a", "server_params": {"model": "m/a"}, "benchmark_params": []}]),
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps([{"test_name": "test_b", "server_params": {"model": "m/b"}, "benchmark_params": []}]),
        encoding="utf-8",
    )
    configs = load_benchmark_configs(config_dir=tmp_path)
    assert [c["test_name"] for c in configs] == ["test_a", "test_b"]


def test_create_unique_server_pytest_params_applies_marks(tmp_path):
    from tests.dfx.conftest import create_unique_server_pytest_params

    configs = [
        {
            "test_name": "test_with_mark",
            "mark": [
                {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}},
                "full_model",
            ],
            "server_params": {"model": "some/model"},
            "benchmark_params": [{"name": "p0", "num_prompts": 1}],
        },
        {
            "test_name": "test_without_mark",
            "server_params": {"model": "other/model"},
            "benchmark_params": [{"name": "p0", "num_prompts": 1}],
        },
    ]
    params = create_unique_server_pytest_params(configs, tmp_path)
    by_id = {p.id: p for p in params}
    assert len(by_id["test_with_mark"].values) == 1
    assert isinstance(by_id["test_with_mark"].values[0], tuple)
    assert any(m.name == "H100" for m in by_id["test_with_mark"].marks)
    assert not any(m.name == "H100" for m in by_id["test_without_mark"].marks)


def test_is_diffusion_perf_config():
    from tests.dfx.conftest import is_diffusion_perf_config

    assert not is_diffusion_perf_config(
        {"test_name": "omni_a", "mark": [{"hardware_marks": {"res": {"cuda": "H100"}}}, "omni"]}
    )
    assert is_diffusion_perf_config(
        {
            "test_name": "diff_a",
            "server_type": "vllm-omni",
            "mark": [{"hardware_marks": {"res": {"cuda": "H100"}}}, "diffusion"],
        }
    )


def test_benchmark_param_id_suffix_from_task_eval_phase():
    from tests.dfx.conftest import _unique_benchmark_param_id_suffixes

    params = [
        {"task": "default_voice", "eval_phase": "latency"},
        {"task": "default_voice", "eval_phase": "throughput"},
    ]
    assert _unique_benchmark_param_id_suffixes(params) == [
        "default_voice_latency",
        "default_voice_throughput",
    ]


def test_create_paired_omni_benchmark_pytest_params(tmp_path):
    from tests.dfx.conftest import create_paired_omni_benchmark_pytest_params

    configs = [
        {
            "test_name": "test_omni",
            "mark": [
                {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}},
                "omni",
            ],
            "server_params": {"model": "m/omni"},
            "benchmark_params": [{"name": "p0", "num_prompts": 1}, {"name": "p1", "num_prompts": 2}],
        },
        {
            "test_name": "test_tts",
            "mark": [
                {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}},
                "tts",
            ],
            "server_params": {"model": "m/tts"},
            "benchmark_params": [{"name": "p0", "num_prompts": 1}],
        },
    ]
    params = create_paired_omni_benchmark_pytest_params(configs, tmp_path)
    by_id = {p.id: p for p in params}
    assert set(by_id) == {"test_omni-p0", "test_omni-p1", "test_tts-p0"}
    omni_row, bench_row = by_id["test_tts-p0"].values
    assert omni_row[0] == "test_tts"
    assert bench_row == ("test_tts", 0)
    assert any(m.name == "tts" for m in by_id["test_tts-p0"].marks)
    assert not any(m.name == "tts" for m in by_id["test_omni-p0"].marks)


@pytest.mark.parametrize(
    ("metric", "current", "baseline"),
    [
        ("request_throughput", 1.0, 1.0),
        ("mean_ttft_ms", 100.0, 100.0),
    ],
)
def test_omni_baseline_accepts_values_at_threshold(metric, current, baseline):
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    assert_result(
        {"completed": 1, metric: current},
        {"baseline": {"H100": {metric: baseline}}, "baseline_hardware": "H100"},
        1,
        assert_baseline=True,
    )


@pytest.mark.parametrize(
    ("metric", "current", "baseline", "message"),
    [
        ("request_throughput", 0.9, 1.0, "below baseline"),
        ("mean_ttft_ms", 101.0, 100.0, "exceeded baseline"),
    ],
)
def test_omni_baseline_rejects_regressions(metric, current, baseline, message):
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    with pytest.raises(AssertionError, match=message):
        assert_result(
            {"completed": 1, metric: current},
            {"baseline": {"H100": {metric: baseline}}, "baseline_hardware": "H100"},
            1,
            assert_baseline=True,
        )


def test_omni_duplex_expected_audio_turns_accepts_complete_session():
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    assert_result(
        {
            "completed": 1,
            "duplex_session_metrics": [{"audio_turn_count": 4}],
        },
        {"expected_duplex_audio_turns_per_session": 4},
        1,
        assert_baseline=False,
    )


def test_omni_duplex_expected_audio_turns_rejects_incomplete_session():
    from tests.dfx.perf.scripts.run_benchmark import assert_result

    with pytest.raises(AssertionError, match="emitted 4 audio turns"):
        assert_result(
            {
                "completed": 1,
                "duplex_session_metrics": [{"audio_turn_count": 3}],
            },
            {"expected_duplex_audio_turns_per_session": 4},
            1,
            assert_baseline=False,
        )


def test_is_hardware_nested_baseline():
    from tests.dfx.conftest import is_hardware_nested_baseline

    assert is_hardware_nested_baseline(
        {
            "H100": {"mean_ttft_ms": [1.0, 2.0], "mean_e2el_ms": [10.0, 20.0]},
            "B200": {"mean_ttft_ms": [0.9, 1.8], "mean_e2el_ms": [9.0, 18.0]},
        }
    )
    assert not is_hardware_nested_baseline({"mean_ttft_ms": [1.0, 2.0], "mean_e2el_ms": [10.0, 20.0]})
    assert not is_hardware_nested_baseline({"mean_ttft_ms": {"1": 1.0, "32": 2.0}})
    assert not is_hardware_nested_baseline({})


def test_resolve_baseline_for_sweep_keeps_all_hardware_for_one_concurrency():
    from tests.dfx.conftest import resolve_baseline_for_sweep

    baseline = {
        "H100": {
            "mean_ttft_ms": [96.4, 140.8, 271.9, 362.3, 507.8],
            "mean_e2el_ms": [18507.0, 28365.0, 31907.0, 48161.0, 72630.0],
        },
        "B200": {
            "mean_ttft_ms": [90.0, 130.0, 250.0, 340.0, 480.0],
            "mean_e2el_ms": [17000.0, 26000.0, 30000.0, 45000.0, 70000.0],
        },
    }
    # max_concurrency=[1,4,8,16,32] -> index 4 is concurrency 32
    got = resolve_baseline_for_sweep(baseline, sweep_index=4)
    assert got == {
        "H100": {"mean_ttft_ms": 507.8, "mean_e2el_ms": 72630.0},
        "B200": {"mean_ttft_ms": 480.0, "mean_e2el_ms": 70000.0},
    }
    # First sweep step keeps both hardware buckets too.
    got0 = resolve_baseline_for_sweep(baseline, sweep_index=0)
    assert set(got0) == {"H100", "B200"}
    assert got0["H100"]["mean_ttft_ms"] == 96.4
    assert got0["B200"]["mean_ttft_ms"] == 90.0


def test_resolve_baseline_for_sweep_rejects_flat_baseline():
    from tests.dfx.conftest import resolve_baseline_for_sweep

    with pytest.raises(ValueError, match="hardware-nested"):
        resolve_baseline_for_sweep(
            {"throughput_qps": [0.4, 0.6], "latency_mean": [1.0, 2.0]},
            sweep_index=1,
        )


def test_resolve_baseline_for_sweep_supports_list_and_scalar_under_hardware():
    from tests.dfx.conftest import resolve_baseline_for_sweep

    # Sweep-aligned lists under each hardware bucket (canonical form).
    listed = {
        "H100": {"throughput_qps": [0.4, 0.6, 0.8], "latency_mean": [1.0, 2.0, 3.0]},
        "B200": {"throughput_qps": [0.5, 0.7, 0.9], "latency_mean": [0.9, 1.8, 2.7]},
    }
    assert resolve_baseline_for_sweep(listed, sweep_index=1) == {
        "H100": {"throughput_qps": 0.6, "latency_mean": 2.0},
        "B200": {"throughput_qps": 0.7, "latency_mean": 1.8},
    }

    # Scalars under hardware stay as-is (single-concurrency cases).
    scalar = {"H100": {"throughput_qps": 0.5}}
    assert resolve_baseline_for_sweep(scalar, sweep_index=0) == {"H100": {"throughput_qps": 0.5}}
    assert resolve_baseline_for_sweep(None) == {}
    assert resolve_baseline_for_sweep({}) == {}


def test_resolve_baseline_value_errors():
    from tests.dfx.conftest import resolve_baseline_value

    with pytest.raises(ValueError, match="sweep_index"):
        resolve_baseline_value([1.0, 2.0], sweep_index=None)
    with pytest.raises(IndexError):
        resolve_baseline_value([1.0], sweep_index=1)
    with pytest.raises(TypeError, match="not supported"):
        resolve_baseline_value({"1": 0.4}, sweep_index=0)


def test_diffusion_build_run_params_resolves_baseline_per_sweep(tmp_path, monkeypatch):
    """``_build_run_params`` / ``_iter_sweep_runs`` narrow list baselines per concurrency."""
    import importlib
    import sys

    cfg = tmp_path / "mini_diffusion_perf.json"
    cfg.write_text(
        json.dumps(
            [
                {
                    "test_name": "test_mini",
                    "server_type": "vllm-omni",
                    "mark": [
                        {"hardware_marks": {"res": {"cuda": "H100"}, "num_cards": 1}},
                        "diffusion",
                        "full_model",
                    ],
                    "server_params": {"model": "m/mini"},
                    "benchmark_params": [{"name": "p0", "num-prompts": 1, "max-concurrency": 1}],
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["pytest", "--test-config-file", str(cfg)])
    sys.modules.pop("tests.dfx.perf.scripts.run_diffusion_benchmark", None)
    from tests.dfx.perf.scripts import run_diffusion_benchmark as rdb

    importlib.reload(rdb)

    params = {
        "name": "c_sweep",
        "dataset": "random",
        "task": "t2i",
        "num-prompts": [8, 16, 32],
        "max-concurrency": [1, 8, 32],
        "baseline": {
            "H100": {"throughput_qps": [0.1, 0.2, 0.3], "latency_mean": [10.0, 20.0, 30.0]},
            "B200": {"throughput_qps": [0.11, 0.22, 0.33], "latency_mean": [9.0, 19.0, 29.0]},
        },
    }
    run32 = rdb._build_run_params(
        params,
        num_prompts=32,
        max_concurrency=32,
        request_rate="inf",
        sweep_index=2,
    )
    assert run32["max-concurrency"] == 32
    assert run32["baseline"] == {
        "H100": {"throughput_qps": 0.3, "latency_mean": 30.0},
        "B200": {"throughput_qps": 0.33, "latency_mean": 29.0},
    }

    sweeps = rdb._iter_sweep_runs(params)
    assert len(sweeps) == 3
    assert sweeps[2]["params"]["max-concurrency"] == 32
    assert sweeps[2]["params"]["baseline"]["H100"]["throughput_qps"] == 0.3
    assert "B200" in sweeps[2]["params"]["baseline"]
    assert not isinstance(sweeps[2]["params"]["baseline"]["H100"]["throughput_qps"], list)
