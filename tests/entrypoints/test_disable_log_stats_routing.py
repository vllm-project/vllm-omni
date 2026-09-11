# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.config.omni_config import (
    _NON_STAGE_ENGINE_CLI_FIELDS,
    VllmOmniConfig,
    _global_stage_cli_fields,
)
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.resolver import OmniConfigResolution
from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import run_headless
from vllm_omni.utils.tracking_parser import TrackingNamespace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_disable_log_stats_is_not_a_stage_config_field() -> None:
    assert "disable_log_stats" in _NON_STAGE_ENGINE_CLI_FIELDS
    assert "disable_log_stats" not in _global_stage_cli_fields()


def test_structured_tts_config_accepts_disable_log_stats() -> None:
    pipeline = OMNI_PIPELINES["qwen3_tts"]
    assert isinstance(pipeline, PipelineConfig)

    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        cli_overrides={"disable_log_stats": True},
    )

    assert len(config.stage_configs) == len(pipeline.stages)


def test_async_engine_consumes_disable_log_stats_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_resolve(model: str, kwargs: dict[str, Any], **unused: Any) -> OmniConfigResolution:
        captured["model"] = model
        captured["kwargs"] = dict(kwargs)
        return OmniConfigResolution(
            config_path="config.yaml",
            stage_configs=(),
        )

    monkeypatch.setattr(
        "vllm_omni.engine.async_omni_engine.load_and_resolve_stage_configs",
        fake_resolve,
    )
    engine = SimpleNamespace(_apply_strategy_lb_policy=lambda *args, **kwargs: None)

    config_path, stage_configs = AsyncOmniEngine._resolve_stage_configs(
        engine,
        "fake-model",
        {"disable_log_stats": True},
        trust_remote_code=False,
    )

    assert captured["model"] == "fake-model"
    assert "disable_log_stats" not in captured["kwargs"]
    assert config_path == "config.yaml"
    assert stage_configs == []


def test_headless_consumes_disable_log_stats_before_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_resolve(*args: Any, **kwargs: Any) -> OmniConfigResolution:
        captured.update(kwargs)
        return OmniConfigResolution(
            config_path=None,
            stage_configs=(SimpleNamespace(stage_id=99),),
        )

    monkeypatch.setattr("vllm_omni.config.resolver.resolve_omni_config", fake_resolve)
    namespace = argparse.Namespace(
        model="fake-model",
        stage_id=0,
        omni_master_address="127.0.0.1",
        omni_master_port=26000,
        worker_backend="multi_process",
        omni_replica_address=None,
        omni_dp_size_local=1,
        disable_log_stats=True,
        log_stats=False,
    )
    args = TrackingNamespace(
        unfiltered_ns=namespace,
        explicit_keys=frozenset({"disable_log_stats"}),
    )

    with pytest.raises(ValueError, match="No stage config found"):
        run_headless(args)

    assert "disable_log_stats" not in captured["cli_overrides"]
