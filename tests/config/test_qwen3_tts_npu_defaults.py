# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve NPU optimization defaults and preserve explicit option precedence."""

from types import SimpleNamespace

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
    Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM as Predictor,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PREDICTOR_DEFAULTS = {
    "code_predictor_fused_sampling": False,
    "code_predictor_frame_graph": True,
    "code_predictor_kv_cache": True,
    "code_predictor_npu_native_gqa": False,
    "code_predictor_npu_fused_swiglu": True,
}
PREDICTOR_OPTIONS = tuple(PREDICTOR_DEFAULTS)


@pytest.mark.parametrize("platform", ["npu", "cuda", "rocm", "xpu", "musa"])
def test_resolved_optimization_defaults_are_npu_only(monkeypatch, platform):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: platform == "npu")
    deploy = _apply_platform_overrides(load_deploy_config(get_deploy_config_path("qwen3_tts.yaml")), platform=platform)
    assert deploy.model_runner == ("v2" if platform in ("cuda", "npu") else "v1")
    if platform == "npu":
        for stage in deploy.stages:
            assert stage.compilation_config is not None
            assert stage.compilation_config["cudagraph_mode"] == (
                "FULL_AND_PIECEWISE" if stage.stage_id == 0 else "PIECEWISE"
            )
    connector = deploy.connectors["connector_of_shared_memory"]
    config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config=connector),
        additional_config=deploy.stages[0].engine_extras.get("additional_config", {}),
    )
    options = Predictor._stage_connector_extra_config(config)
    assert all(
        options.get(key, False) is (enabled and platform == "npu") for key, enabled in PREDICTOR_DEFAULTS.items()
    )
    # Defaults belong to the model/loader. A YAML copy could mask a change
    # to those defaults and give custom deployments different behavior.
    npu_options = {
        *PREDICTOR_OPTIONS,
        "npu_incremental_codec",
        "code_predictor_kv_cache",
    }
    assert npu_options.isdisjoint(connector.get("extra", {}))
    for stage in deploy.stages:
        assert npu_options.isdisjoint(stage.engine_extras.get("additional_config", {}))


@pytest.mark.parametrize("source", ["implicit", "additional", "connector", "connector_opt_in"])
def test_npu_predictor_defaults_preserve_explicit_overrides(monkeypatch, source):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    config = SimpleNamespace(model_config=SimpleNamespace(stage_connector_config={"extra": {}}), additional_config={})
    if source == "additional":
        config.additional_config = dict.fromkeys(PREDICTOR_OPTIONS, True)
    elif source == "connector":
        config.additional_config = dict.fromkeys(PREDICTOR_OPTIONS, True)
        config.model_config.stage_connector_config["extra"] = dict.fromkeys(PREDICTOR_OPTIONS, False)
    elif source == "connector_opt_in":
        config.additional_config = dict.fromkeys(PREDICTOR_OPTIONS, False)
        config.model_config.stage_connector_config["extra"] = dict.fromkeys(PREDICTOR_OPTIONS, True)
    options = Predictor._stage_connector_extra_config(config)
    assert all(
        options[key] is (enabled if source == "implicit" else source in ("additional", "connector_opt_in"))
        for key, enabled in PREDICTOR_DEFAULTS.items()
    )


@pytest.mark.parametrize("frame_graph", [False, True])
def test_frame_graph_opt_in_controls_implicit_npu_kv(monkeypatch, frame_graph):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config={"extra": {"code_predictor_frame_graph": frame_graph}}),
        additional_config={},
    )
    options = Predictor._stage_connector_extra_config(config)
    assert options["code_predictor_kv_cache"] is frame_graph


def test_custom_npu_v1_overlay_overrides_default_runner_and_full_graphs(tmp_path):
    overlay = tmp_path / "qwen3_tts_npu_v1.yaml"
    overlay.write_text(
        f"base_config: {get_deploy_config_path('qwen3_tts.yaml')}\n"
        "platforms:\n"
        "  npu:\n"
        "    model_runner: v1\n"
        "    stages:\n"
        "      - stage_id: 0\n"
        "        compilation_config:\n"
        "          cudagraph_mode: PIECEWISE\n"
    )
    deploy = _apply_platform_overrides(load_deploy_config(overlay), platform="npu")
    assert deploy.model_runner == "v1"
    talker = next(stage for stage in deploy.stages if stage.stage_id == 0)
    assert talker.enforce_eager is False
    assert talker.compilation_config is not None
    assert talker.compilation_config["cudagraph_mode"] == "PIECEWISE"
