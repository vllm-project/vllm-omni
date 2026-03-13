# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vllm_omni.engine.arg_utils import _resolve_bundled_hf_config_path
from vllm_omni.entrypoints import utils as entrypoint_utils
from vllm_omni.entrypoints.omni import OmniBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_bundled_hf_config_path_uses_cosyvoice3_bundle_by_default():
    resolved = _resolve_bundled_hf_config_path("FunAudioChatCosyVoice3Code2Wav", None)

    assert resolved is not None
    assert resolved.endswith("vllm_omni/model_executor/models/cosyvoice3/hf_config")
    assert (Path(resolved) / "config.json").is_file()


def test_resolve_bundled_hf_config_path_preserves_explicit_override():
    resolved = _resolve_bundled_hf_config_path("FunAudioChatCosyVoice3Code2Wav", "/tmp/custom-hf-config")

    assert resolved == "/tmp/custom-hf-config"


def test_get_stage_model_prefers_stage_override():
    stage = SimpleNamespace(engine_args=SimpleNamespace(model="stage-specific-model"))

    assert OmniBase._get_stage_model(stage, "fallback-model") == "stage-specific-model"


def test_get_stage_model_falls_back_when_stage_override_missing():
    stage = SimpleNamespace(engine_args=SimpleNamespace())

    assert OmniBase._get_stage_model(stage, "fallback-model") == "fallback-model"


def test_resolve_model_config_path_detects_funaudiochat_default_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        entrypoint_utils,
        "get_config",
        lambda model, trust_remote_code=True: SimpleNamespace(model_type="funaudiochat"),
    )

    resolved = entrypoint_utils.resolve_model_config_path("dummy-funaudiochat-model")

    assert resolved is not None
    assert resolved.endswith("vllm_omni/model_executor/stage_configs/funaudiochat.yaml")


def test_funaudiochat_default_stage_config_limits_audio_profile_and_keeps_audio_towers():
    config_path = (
        Path(__file__).resolve().parents[2] / "vllm_omni" / "model_executor" / "stage_configs" / "funaudiochat.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    stage0_engine_args = config["stage_args"][0]["engine_args"]

    assert "language_model_only" not in stage0_engine_args
    assert stage0_engine_args["hf_overrides"]["audio_config"]["max_source_positions"] == 100
    assert stage0_engine_args["limit_mm_per_prompt"]["audio"] == 1
