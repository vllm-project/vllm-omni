# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve NPU defaults without changing other platforms or explicit opt-outs."""

from types import SimpleNamespace

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
    Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PREDICTOR_OPTIONS = (
    "code_predictor_fused_sampling",
    "code_predictor_frame_graph",
    "code_predictor_kv_cache",
    "code_predictor_npu_native_gqa",
    "code_predictor_npu_fused_swiglu",
)


@pytest.mark.parametrize("platform", ["npu", "cuda", "rocm", "xpu"])
def test_resolved_defaults_are_npu_only(monkeypatch, platform):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: platform == "npu")
    deploy = _apply_platform_overrides(load_deploy_config(get_deploy_config_path("qwen3_tts.yaml")), platform=platform)
    connector = deploy.connectors["connector_of_shared_memory"]
    config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config=connector),
        additional_config=deploy.stages[0].engine_extras.get("additional_config", {}),
    )
    options = Predictor._stage_connector_extra_config(config)
    assert all(options.get(key, False) is (platform == "npu") for key in PREDICTOR_OPTIONS)
    # Defaults belong to the model/loader. A YAML copy could mask a change
    # to those defaults and give custom deployments different behavior.
    npu_options = {
        *PREDICTOR_OPTIONS,
        "npu_fused_snake",
        "npu_incremental_codec",
        "npu_codec_streams",
        "npu_conv_graph_shapes",
        "code_predictor_kv_cache",
    }
    assert npu_options.isdisjoint(connector.get("extra", {}))
    for stage in deploy.stages:
        assert npu_options.isdisjoint(stage.engine_extras.get("additional_config", {}))


@pytest.mark.parametrize("source", ["implicit", "additional", "connector"])
def test_npu_predictor_defaults_preserve_explicit_opt_out(monkeypatch, source):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    config = SimpleNamespace(model_config=SimpleNamespace(stage_connector_config={"extra": {}}), additional_config={})
    if source == "additional":
        config.additional_config = dict.fromkeys(PREDICTOR_OPTIONS, False)
    elif source == "connector":
        config.additional_config = dict.fromkeys(PREDICTOR_OPTIONS, True)
        config.model_config.stage_connector_config["extra"] = dict.fromkeys(PREDICTOR_OPTIONS, False)
    options = Predictor._stage_connector_extra_config(config)
    assert all(options[key] is (source == "implicit") for key in PREDICTOR_OPTIONS)


def test_disabling_frame_graph_disables_implicit_npu_kv(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config={"extra": {"code_predictor_frame_graph": False}}),
        additional_config={},
    )
    options = Predictor._stage_connector_extra_config(config)
    assert options["code_predictor_kv_cache"] is False
