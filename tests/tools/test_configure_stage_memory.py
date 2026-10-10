# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for tools/configure_stage_memory.py against the deploy YAML schema."""

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
FAKE_GPUS = [
    {
        "id": 0,
        "name": "fake-gpu",
        "total_gib": 80.0,
        "free_gib": 79.0,
        "used_gib": 1.0,
        "compute_capability": "9.0",
    }
]


def _load_tool():
    module_path = REPO_ROOT / "tools" / "configure_stage_memory.py"
    spec = importlib.util.spec_from_file_location("configure_stage_memory", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _auto_configure_and_save(tool, config: dict, out_path: Path) -> dict:
    stages = tool.extract_stages(config)
    config, stages = tool.auto_configure(config, stages, FAKE_GPUS)
    updated = tool.apply_to_config(config, stages)
    OmegaConf.save(OmegaConf.create(updated), out_path)
    return updated


def test_auto_configure_updates_shipped_deploy_yaml(tmp_path: Path):
    tool = _load_tool()
    config = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "vllm_omni" / "deploy" / "qwen3_tts.yaml"))

    stages = tool.extract_stages(config)
    assert [(s["stage_id"], s["device"], s["gpu_mem"]) for s in stages] == [(0, "0", 0.3), (1, "0", 0.3)]

    out_path = tmp_path / "qwen3_tts.yaml"
    updated = _auto_configure_and_save(tool, config, out_path)

    # Two stages share GPU 0: (79 GiB free - 1.5 GiB headroom) / 2 / 80 GiB.
    assert [s["gpu_memory_utilization"] for s in updated["stages"]] == [0.484, 0.484]
    # Knobs the stage leaves unset stay unset instead of being pinned.
    assert "enforce_eager" not in updated["stages"][0]
    assert updated["stages"][0]["max_num_batched_tokens"] == 512

    deploy = load_deploy_config(out_path)
    assert [s.gpu_memory_utilization for s in deploy.stages] == [0.484, 0.484]
    assert [s.devices for s in deploy.stages] == ["0", "0"]


def test_apply_writes_nested_stage_knobs_where_the_loader_reads_them(tmp_path: Path):
    tool = _load_tool()
    config = {
        "stages": [
            {
                "stage_id": 0,
                "runtime": {"devices": "0"},
                "engine_args": {"gpu_memory_utilization": 0.5, "max_num_seqs": 8},
            }
        ]
    }

    out_path = tmp_path / "nested.yaml"
    updated = _auto_configure_and_save(tool, config, out_path)

    assert updated["stages"][0]["engine_args"]["gpu_memory_utilization"] == 0.95
    assert "gpu_memory_utilization" not in updated["stages"][0]
    deploy = load_deploy_config(out_path)
    assert deploy.stages[0].gpu_memory_utilization == 0.95
    assert deploy.stages[0].max_num_seqs == 8


def test_removed_stage_args_schema_is_rejected():
    tool = _load_tool()

    with pytest.raises(ValueError, match="stage_args"):
        tool.extract_stages({"stage_args": [{"stage_id": 0, "engine_args": {}}]})
