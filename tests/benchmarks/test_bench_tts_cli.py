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


@pytest.mark.parametrize("model", ["tencent/AuK", "tencent/AuK-Flash"])
@pytest.mark.parametrize("task", ["default_voice", "voice_clone"])
def test_auk_benchmark_duration_and_local_bundle(model, task):
    config = bench_tts.load_model_configs(bench_tts._DEFAULT_MODEL_CONFIGS)[model]
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model=model,
        task=task,
        model_cfg=config,
        locale="en",
        num_prompts=20,
        concurrency=8,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
        served_model_name="/models/auk",
        duration_seconds=3.5,
        request_seed=7,
    )
    assert cmd[cmd.index("--model") + 1] == "/models/auk"
    body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert body == {
        "voice": "default",
        "duration_seconds": 3.5,
        "seed": 7,
        "task_type": "CustomVoice" if task == "default_voice" else "Base",
    }
    assert config["task_extra_body"][task]["duration_seconds"] == 5.0


@pytest.mark.parametrize(
    ("task", "expected_task_type"),
    [
        ("default_voice", "CustomVoice"),
        ("voice_clone", "Base"),
    ],
)
def test_auk_task_type_survives_served_model_alias(task, expected_task_type):
    model = "tencent/AuK-Flash"
    served_model = "models/flash"
    config = bench_tts.load_model_configs(bench_tts._DEFAULT_MODEL_CONFIGS)[model]
    cmd = bench_tts.build_bench_args(
        host="localhost",
        port=8000,
        model=model,
        task=task,
        model_cfg=config,
        locale="en",
        num_prompts=1,
        concurrency=1,
        dataset_path="/data/seed-tts",
        wer_eval=False,
        output_dir=None,
        result_filename=None,
        extra_cli_args=[],
        served_model_name=served_model,
    )

    assert cmd[cmd.index("--model") + 1] == served_model
    extra_body = json.loads(cmd[cmd.index("--extra-body") + 1])
    assert extra_body["task_type"] == expected_task_type
    assert "_vllm_omni_benchmark_model" not in extra_body


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


def test_build_bench_args_default_voice_has_voice_param(model_configs_path: Path) -> None:
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


def test_build_bench_args_supports_local_model_and_shared_sweep_options(model_configs_path: Path) -> None:
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


def test_unsupported_task_exits(model_configs_path: Path, capsys: pytest.CaptureFixture, mocker) -> None:
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


def test_summary_table_includes_task_type(capsys: pytest.CaptureFixture) -> None:
    bench_tts.print_summary_table(
        [
            {
                "_task": "voice_clone",
                "_task_type": "Base",
                "_concurrency": 1,
            }
        ]
    )

    output = capsys.readouterr().out
    assert "Task Type" in output
    assert "voice_clone" in output
    assert "Base" in output
