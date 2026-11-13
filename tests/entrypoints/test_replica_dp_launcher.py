# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/online_serving/replica_data_parallel"
SCRIPT = EXAMPLE / "run_server.sh"


@pytest.fixture
def launch(monkeypatch, tmp_path):
    binary = tmp_path / "vllm"
    capture = tmp_path / "invocation.json"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "config = Path(sys.argv[sys.argv.index('--deploy-config') + 1])\n"
        "Path(os.environ['CAPTURE_PATH']).write_text(json.dumps(\n"
        "    {'argv': sys.argv[1:], 'config_path': str(config), 'yaml': config.read_text()}))\n"
        "sys.exit(int(os.environ.get('FAKE_VLLM_EXIT', '0')))\n"
    )
    binary.chmod(0o755)
    for key in ("MODEL", "HOST", "PORT", "NUM_REPLICAS", "DEVICES", "DEPLOY_TEMPLATE", "FAKE_VLLM_EXIT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CAPTURE_PATH", str(capture))
    monkeypatch.setenv("TMPDIR", str(scratch))

    def run(*args, **env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        result = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, timeout=15)
        invocation = json.loads(capture.read_text()) if capture.exists() else None
        assert not list(scratch.iterdir()), "Rendered deploy config leaked after launcher exit"
        return result, invocation

    return run


def test_default_wan_launch_is_preserved(launch):
    result, invocation = launch()
    assert result.returncode == 0, result.stderr
    assert invocation["argv"][:3] == ["serve", "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "--omni"]
    assert invocation["argv"][3:7] == ["--host", "127.0.0.1", "--port", "8098"]
    stage = yaml.safe_load(invocation["yaml"])["stages"][0]
    assert (stage["num_replicas"], stage["devices"], stage["tensor_parallel_size"]) == (1, "0", 1)
    assert not Path(invocation["config_path"]).exists()


def test_h3_template_and_arguments_with_spaces(launch):
    extra = ["--served-model-name", "H3 image", "--additional-config", '{"text_encoder_cudnn_sdp":false}']
    result, invocation = launch(
        *extra,
        MODEL="MiniMaxAI/MiniMax-H3-Image",
        DEPLOY_TEMPLATE=str(EXAMPLE / "minimax_h3_image_dp.yaml"),
        NUM_REPLICAS="4",
        DEVICES="4,5,6,7",
    )
    assert result.returncode == 0, result.stderr
    assert invocation["argv"][1] == "MiniMaxAI/MiniMax-H3-Image"
    assert invocation["argv"][-len(extra) :] == extra
    stage = yaml.safe_load(invocation["yaml"])["stages"][0]
    assert stage["num_replicas"] == 4
    assert stage["devices"].split(",") == ["4", "5", "6", "7"]
    for field in ("tensor_parallel_size", "ulysses_degree", "ring_degree", "text_encoder_tp_size"):
        assert stage[field] == 1


@pytest.mark.parametrize(
    "env",
    [
        {"NUM_REPLICAS": ""},
        {"NUM_REPLICAS": "0"},
        {"NUM_REPLICAS": "-1"},
        {"NUM_REPLICAS": "1.5"},
        {"DEVICES": ""},
        {"DEVICES": "0,"},
        {"DEVICES": "0,,1", "NUM_REPLICAS": "2"},
        {"DEVICES": "-1"},
        {"DEVICES": "0,0", "NUM_REPLICAS": "2"},
        {"DEVICES": "0,1", "NUM_REPLICAS": "4"},
        {"DEVICES": "0;false"},
    ],
)
def test_invalid_replica_placement_does_not_start_vllm(launch, env):
    result, invocation = launch(**env)
    assert result.returncode != 0
    assert result.stderr
    assert invocation is None


def test_rendered_config_is_removed_when_server_fails(launch):
    result, invocation = launch(FAKE_VLLM_EXIT="7")
    assert result.returncode == 7
    assert invocation is not None
    assert not Path(invocation["config_path"]).exists()
