# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deploy config and pipeline auto-detection tests for InternLM Sage-Interaction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.sage_interaction.pipeline import (
    SAGE_INTERACTION_PIPELINE,
    SAGE_INTERACTION_VOCAB_SIZE,
    is_sage_interaction_config,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy"


def test_pipeline_registry_order_and_disambiguation(monkeypatch: pytest.MonkeyPatch) -> None:
    keys = list(OMNI_PIPELINES)
    assert "sage_interaction" in keys
    assert OMNI_PIPELINES["sage_interaction"] is SAGE_INTERACTION_PIPELINE

    sage_cfg = SimpleNamespace(
        model_type="minicpmo",
        architectures=["MiniCPMO"],
        version="4.5",
        vocab_size=SAGE_INTERACTION_VOCAB_SIZE,
    )
    stock_cfg = SimpleNamespace(
        model_type="minicpmo",
        architectures=["MiniCPMO"],
        version="4.5",
        vocab_size=151748,
    )
    assert is_sage_interaction_config(sage_cfg) is True
    assert is_sage_interaction_config(stock_cfg) is False

    monkeypatch.setattr(StageConfigFactory, "try_infer_model_type", classmethod(lambda cls, **kw: "minicpmo"))
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kw: sage_cfg))
    assert StageConfigFactory.get_pipeline_config("local/checkpoint", trust_remote_code=True).model_type == (
        "sage_interaction"
    )
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kw: stock_cfg))
    assert StageConfigFactory.get_pipeline_config("local/checkpoint", trust_remote_code=True).model_type == (
        "minicpmo_4_5"
    )


def test_sage_interaction_deploy_yaml_resolves_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    deploy_cfg = _apply_platform_overrides(
        load_deploy_config(str(_DEPLOY_DIR / "sage_interaction.yaml")),
        platform="cuda",
    )
    assert deploy_cfg.pipeline == "sage_interaction"
    assert deploy_cfg.session_mode == "duplex"
    assert deploy_cfg.duplex_session is not None
    assert deploy_cfg.duplex_session.max_sessions == 16

    stages = merge_pipeline_deploy(SAGE_INTERACTION_PIPELINE, deploy_cfg)
    assert len(stages) == 3
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, True, True]
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [True, True, True]
    assert SAGE_INTERACTION_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.sage_interaction.duplex.plugin.SageInteractionDuplexPlugin"
    )
