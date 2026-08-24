# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the universal benchmarks/tts/bench_tts.py CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# Add benchmarks/tts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks" / "tts"))
import bench_tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def model_configs_path(tmp_path: Path) -> Path:
    cfg = {
        "models": {
            "test/ModelA": {
                "stage_config": "model_a.yaml",
                "supported_tasks": ["voice_clone", "default_voice"],
                "backend": "openai-audio-speech",
                "endpoint": "/v1/audio/speech",
                "task_extra_body": {
                    "voice_clone": {"task_type": "Base"},
                    "default_voice": {"voice": "Vivian", "task_type": "CustomVoice"},
                },
            },
            "test/ModelB": {
                "stage_config": "model_b.yaml",
                "supported_tasks": ["voice_clone"],
                "backend": "openai-audio-speech",
                "endpoint": "/v1/audio/speech",
                "task_extra_body": {"voice_clone": {}},
            },
        }
    }
    p = tmp_path / "model_configs.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


def test_load_model_configs(model_configs_path: Path) -> None:
    configs = bench_tts.load_model_configs(model_configs_path)
    assert "test/ModelA" in configs
    assert "test/ModelB" in configs
    assert configs["test/ModelA"]["supported_tasks"] == ["voice_clone", "default_voice"]


def test_indextts25_is_registered_in_shared_model_configs() -> None:
    config_path = Path(bench_tts.__file__).with_name("model_configs.yaml")
    config = bench_tts.load_model_configs(config_path)["IndexTeam/IndexTTS-2.5"]

    assert config["supported_tasks"] == ["voice_clone"]
    assert config["task_extra_body"]["voice_clone"]["extra_params"]["lang"] == "en"


def test_build_bench_args_voice_clone(model_configs_path: Path) -> None:
    configs = bench_tts.load_model_configs(model_configs_path)
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model="test/ModelA",
        task="voice_clone",
        model_cfg=configs["test/ModelA"],
        locale="en",
        num_prompts=10,
        concurrency=1,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
    )
    assert "--dataset-name" in cmd
    idx = cmd.index("--dataset-name")
    assert cmd[idx + 1] == "seed-tts"
    assert "--max-concurrency" in cmd
    assert "--extra-body" in cmd
    extra_body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert extra_body.get("task_type") == "Base"


def test_build_bench_args_default_voice_has_voice_param(
    model_configs_path: Path,
) -> None:
    configs = bench_tts.load_model_configs(model_configs_path)
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model="test/ModelA",
        task="default_voice",
        model_cfg=configs["test/ModelA"],
        locale="en",
        num_prompts=10,
        concurrency=1,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
    )
    idx = cmd.index("--dataset-name")
    assert cmd[idx + 1] == "seed-tts-text"
    extra_body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert extra_body.get("voice") == "Vivian"


def test_build_bench_args_wer_eval_adds_flag(model_configs_path: Path) -> None:
    configs = bench_tts.load_model_configs(model_configs_path)
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model="test/ModelA",
        task="voice_clone",
        model_cfg=configs["test/ModelA"],
        locale="en",
        num_prompts=10,
        concurrency=1,
        dataset_path="/data/seed-tts",
        wer_eval=True,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
    )
    assert "--seed-tts-wer-eval" in cmd


def test_build_bench_args_supports_local_model_and_shared_sweep_options(
    model_configs_path: Path,
) -> None:
    configs = bench_tts.load_model_configs(model_configs_path)
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8092,
        model="test/ModelA",
        served_model_name="/models/indextts25",
        task="voice_clone",
        model_cfg=configs["test/ModelA"],
        locale="en",
        num_prompts=500,
        num_warmups=5,
        request_seed=42,
        concurrency=8,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=["--", "--tokenizer", "/models/indextts25/qwen0.6bemo4-merge"],
    )

    assert cmd[cmd.index("--model") + 1] == "/models/indextts25"
    assert cmd[cmd.index("--num-warmups") + 1] == "5"
    assert cmd[-2:] == ["--tokenizer", "/models/indextts25/qwen0.6bemo4-merge"]
    extra_body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert extra_body == {"task_type": "Base", "seed": 42}


def test_build_bench_args_requests_median_and_p99_by_default(
    model_configs_path: Path,
) -> None:
    """Per-concurrency first-audio latency needs median + P99 in each result JSON."""
    configs = bench_tts.load_model_configs(model_configs_path)
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model="test/ModelA",
        task="voice_clone",
        model_cfg=configs["test/ModelA"],
        locale="en",
        num_prompts=10,
        concurrency=1,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
    )
    assert "--metric-percentiles" in cmd
    assert cmd[cmd.index("--metric-percentiles") + 1] == "50,99"
    # audio_ttfp must stay in the percentile-metrics set.
    pct_metrics = cmd[cmd.index("--percentile-metrics") + 1]
    assert "audio_ttfp" in pct_metrics.split(",")


def test_build_bench_args_custom_metric_percentiles(model_configs_path: Path) -> None:
    configs = bench_tts.load_model_configs(model_configs_path)
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model="test/ModelA",
        task="voice_clone",
        model_cfg=configs["test/ModelA"],
        locale="en",
        num_prompts=10,
        concurrency=1,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
        metric_percentiles="25,50,75,99",
    )
    assert cmd[cmd.index("--metric-percentiles") + 1] == "25,50,75,99"


def test_percentile_value_extracts_target_row() -> None:
    rows = [[50.0, 123.4], [99.0, 987.6]]
    assert bench_tts._percentile_value(rows, 50.0) == 123.4
    assert bench_tts._percentile_value(rows, 99.0) == 987.6


def test_percentile_value_is_nan_for_absent_rows() -> None:
    import math

    assert math.isnan(bench_tts._percentile_value(None, 99.0))
    assert math.isnan(bench_tts._percentile_value([[50.0, 1.0]], 99.0))
    assert math.isnan(bench_tts._percentile_value("garbage", 99.0))
    assert math.isnan(bench_tts._percentile_value([[99.0, "bad"]], 99.0))


def test_print_summary_table_reports_median_and_p99(
    capsys: pytest.CaptureFixture,
) -> None:
    """vllm-omni flattens percentile rows into per-percentile JSON keys."""
    results = [
        {
            "_task": "voice_clone",
            "_concurrency": 1,
            "mean_audio_ttfp_ms": 100.0,
            "median_audio_ttfp_ms": 90.0,
            "p50_audio_ttfp_ms": 90.0,
            "p99_audio_ttfp_ms": 150.0,
            "mean_audio_rtf": 0.5,
            "audio_throughput": 10.0,
        },
        {
            "_task": "voice_clone",
            "_concurrency": 4,
            "mean_audio_ttfp_ms": 200.0,
            "median_audio_ttfp_ms": 180.0,
            "p50_audio_ttfp_ms": 180.0,
            "p99_audio_ttfp_ms": 350.0,
            "mean_audio_rtf": 0.6,
            "audio_throughput": 12.0,
        },
        # Legacy run without percentile data: median/P99 fall back to n/a.
        {
            "_task": "voice_clone",
            "_concurrency": 8,
            "mean_audio_ttfp_ms": float("nan"),
            "mean_audio_rtf": float("nan"),
            "audio_throughput": float("nan"),
        },
    ]
    bench_tts.print_summary_table(results)
    out = capsys.readouterr().out

    assert "TTFP p50" in out
    assert "TTFP p99" in out
    # Per-concurrency median + P99 are both shown for concurrency 1 and 4.
    assert "90" in out  # median @ c1
    assert "150" in out  # p99 @ c1
    assert "180" in out  # median @ c4
    assert "350" in out  # p99 @ c4
    # Missing percentile data degrades to n/a instead of crashing.
    assert "n/a" in out


def test_first_audio_latency_rows_flat_keys() -> None:
    """Flat per-percentile keys are read directly from the result JSON."""
    flat = {
        "median_audio_ttfp_ms": 90.0,
        "p50_audio_ttfp_ms": 90.0,
        "p99_audio_ttfp_ms": 150.0,
    }
    assert bench_tts._first_audio_latency_rows(flat) == (90.0, 150.0)


def test_first_audio_latency_rows_falls_back_to_row_list() -> None:
    """Raw percentiles_audio_ttfp_ms row lists are understood as fallback."""
    row_list = {"percentiles_audio_ttfp_ms": [[50.0, 90.0], [99.0, 150.0]]}
    assert bench_tts._first_audio_latency_rows(row_list) == (90.0, 150.0)

    empty: dict = {}
    p50, p99 = bench_tts._first_audio_latency_rows(empty)
    import math

    assert math.isnan(p50) and math.isnan(p99)


def test_unsupported_task_exits(
    model_configs_path: Path, capsys: pytest.CaptureFixture, mocker
) -> None:
    # ModelB does not support voice_design
    mocker.patch.object(
        sys,
        "argv",
        [
            "bench_tts.py",
            "--model",
            "test/ModelB",
            "--task",
            "voice_design",
            "--model-configs",
            str(model_configs_path),
        ],
    )
    with pytest.raises(SystemExit):
        bench_tts.main()
