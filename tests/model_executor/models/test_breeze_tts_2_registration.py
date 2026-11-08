# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The accelerated runtime preserves Breeze-TTS-2's model identifiers."""

from pathlib import Path

import pytest
import yaml

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.breeze_tts_2 import BreezeTTS2Adapter
from vllm_omni.model_executor.models.breeze_tts_2.pipeline import BREEZE_TTS_2_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_breeze_tts_2_pipeline_keeps_checkpoint_and_stage_identifiers():
    assert OMNI_PIPELINES["breeze"] is BREEZE_TTS_2_PIPELINE
    assert BREEZE_TTS_2_PIPELINE.model_arch == "BreezeForConditionalGeneration"
    assert BREEZE_TTS_2_PIPELINE.default_deploy_config_name == "breeze_tts_2.yaml"
    assert [stage.model_stage for stage in BREEZE_TTS_2_PIPELINE.stages] == [
        "breeze_tts_2",
        "breeze_tts_2_codec",
    ]


def test_breeze_tts_2_adapter_remains_registered():
    assert resolve_adapter("breeze_tts_2") is BreezeTTS2Adapter


@pytest.mark.parametrize(
    "model_stage,model_arch",
    [
        ("breeze_tts_2", None),
        ("breeze_tts_2_codec", None),
        ("breeze_tts_2", "BreezeForConditionalGeneration"),
        (None, "BreezeForConditionalGeneration"),
    ],
)
def test_breeze_tts_2_detection_keeps_existing_identifiers(model_stage, model_arch):
    assert detect_tts_model_type(model_stage, model_arch) == "breeze_tts_2"


@pytest.mark.parametrize("suffix", ["", "_throughput"])
def test_breeze_deploy_aliases_match_canonical_configuration(suffix):
    deploy_dir = Path(__file__).parents[3] / "vllm_omni" / "deploy"
    canonical = yaml.safe_load((deploy_dir / f"breeze_tts_2{suffix}.yaml").read_text())
    alias = yaml.safe_load((deploy_dir / f"breeze_tts{suffix}.yaml").read_text())
    assert canonical == alias
