# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Activation variant: pipeline registry and startup matrix."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_DUPLEX_PIPELINE,
    QWEN3_OMNI_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duplex_variant_shares_stock_topology() -> None:
    assert QWEN3_OMNI_PIPELINE.duplex_plugin is None
    assert QWEN3_OMNI_DUPLEX_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.qwen3_omni.duplex.plugin.Qwen3OmniDuplexPlugin"
    )
    assert QWEN3_OMNI_DUPLEX_PIPELINE.stages is QWEN3_OMNI_PIPELINE.stages
    assert QWEN3_OMNI_DUPLEX_PIPELINE.model_type == "qwen3_omni_moe_duplex"


def test_registry_resolves_duplex_variant_without_touching_stock() -> None:
    assert "qwen3_omni_moe_duplex" in OMNI_PIPELINES
    duplex = resolve_pipeline_config("qwen3_omni_moe_duplex")
    thinker_only = resolve_pipeline_config("qwen3_omni_moe_thinker_only")
    assert duplex is QWEN3_OMNI_DUPLEX_PIPELINE
    assert QWEN3_OMNI_PIPELINE.duplex_plugin is None
    assert QWEN3_OMNI_PIPELINE is not duplex
    assert thinker_only is OMNI_PIPELINES["qwen3_omni_moe_thinker_only"]


def test_duplex_deploy_yaml_opts_into_variant() -> None:
    deploy = load_deploy_config(Path(get_deploy_config_path("qwen3_omni_moe_duplex.yaml")))
    assert deploy.pipeline == "qwen3_omni_moe_duplex"
    assert deploy.session_mode == "duplex"


def test_stock_deploy_yaml_stays_turn_based() -> None:
    deploy = load_deploy_config(Path(get_deploy_config_path("qwen3_omni_moe.yaml")))
    assert deploy.pipeline is None or deploy.pipeline == "qwen3_omni_moe"
    assert deploy.session_mode == "turn"


def test_duplex_engine_rejects_turn_session_mode() -> None:
    stub = SimpleNamespace(
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        pipeline_config=QWEN3_OMNI_DUPLEX_PIPELINE,
        deploy_config=SimpleNamespace(session_mode="turn", duplex_session=None),
        _audio_encoder=lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="session_mode: duplex"):
        DuplexOmniEngine._validate_deployment(stub)


def test_stock_pipeline_is_not_a_duplex_model() -> None:
    stub = SimpleNamespace(
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        pipeline_config=QWEN3_OMNI_PIPELINE,
        deploy_config=SimpleNamespace(session_mode="duplex", duplex_session=None),
        _audio_encoder=lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="declares no duplex_plugin"):
        DuplexOmniEngine._validate_deployment(stub)


def test_duplex_engine_loads_plugin_when_session_mode_is_duplex() -> None:
    from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

    stub = SimpleNamespace(
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        pipeline_config=QWEN3_OMNI_DUPLEX_PIPELINE,
        deploy_config=SimpleNamespace(session_mode="duplex", duplex_session=None),
        _audio_encoder=lambda *args, **kwargs: None,
        plugin=None,
        duplex_session_config=None,
    )
    DuplexOmniEngine._validate_deployment(stub)
    assert isinstance(stub.plugin, Qwen3OmniDuplexPlugin)


def test_duplex_deploy_config_selects_variant_pipeline() -> None:
    hf_config = SimpleNamespace(
        model_type="qwen3_omni_moe",
        enable_audio_output=True,
        architectures=["Qwen3OmniMoeForConditionalGeneration"],
    )
    deploy_path = Path(get_deploy_config_path("qwen3_omni_moe_duplex.yaml"))
    with (
        patch("vllm_omni.config.config_factory.get_config", return_value=hf_config),
        patch.object(StageConfigFactory, "try_infer_model_type", return_value="qwen3_omni_moe"),
        patch.object(StageConfigFactory, "get_hf_config", return_value=hf_config),
    ):
        pipeline = StageConfigFactory.get_pipeline_config(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            trust_remote_code=False,
            deploy_config_path=str(deploy_path),
        )
    assert pipeline is not None
    assert pipeline.model_type == "qwen3_omni_moe_duplex"
    assert pipeline.duplex_plugin == QWEN3_OMNI_DUPLEX_PIPELINE.duplex_plugin
