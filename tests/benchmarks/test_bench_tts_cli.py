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
