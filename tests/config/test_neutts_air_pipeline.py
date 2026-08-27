# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.config import load_deploy_config
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.model_executor.models.neutts_air.pipeline import (
    NEUTTS_AIR_PIPELINE,
    is_neutts_air_config,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _neutts_config(**overrides):
    values = {
        "model_type": "qwen2",
        "vocab_size": 217652,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_neutts_air_config_fingerprint_accepts_the_backbone():
    assert is_neutts_air_config(_neutts_config())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen2_audio"),
        ("vocab_size", 151936),
        ("hidden_size", 4096),
        ("num_hidden_layers", 32),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 8),
    ],
)
def test_neutts_air_config_fingerprint_rejects_other_qwen2_models(
    field,
    value,
):
    assert not is_neutts_air_config(_neutts_config(**{field: value}))


def test_neutts_air_pipeline_and_decoder_are_registered():
    assert OMNI_PIPELINES["neutts_air"] is NEUTTS_AIR_PIPELINE
    assert _OMNI_MODELS["NeuTTSAirForCausalLM"] == (
        "neutts_air",
        "neutts_air_talker",
        "NeuTTSAirForCausalLM",
    )
    assert _OMNI_MODELS["NeuTTSAirCode2Wav"] == (
        "neutts_air",
        "neutts_air_code2wav",
        "NeuTTSAirCode2Wav",
    )
    assert NEUTTS_AIR_PIPELINE.model_arch == "NeuTTSAirForCausalLM"
    assert NEUTTS_AIR_PIPELINE.validate() == []


def test_neutts_air_default_deploy_enables_async_chunking():
    deploy_path = Path(__file__).parents[2] / "vllm_omni" / "deploy" / "neutts_air.yaml"

    deploy = load_deploy_config(deploy_path)

    assert deploy.async_chunk is True
    assert len(deploy.stages) == 2


def test_neutts_air_deploy_capacity_defaults():
    from pathlib import Path

    import yaml

    deploy_path = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy" / "neutts_air.yaml"
    config = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    stages = {stage["stage_id"]: stage for stage in config["stages"]}

    stage0 = stages[0]
    stage1 = stages[1]

    assert stage0["max_num_seqs"] == 4
    assert stage0["kv_cache_memory_bytes"] == 512 * 1024 * 1024
    assert "gpu_memory_utilization" not in stage0
    assert stage1["max_num_seqs"] == 4
