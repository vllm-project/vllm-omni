# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vllm_omni.engine.arg_utils import _ARCH_TO_MODEL_TYPE
from vllm_omni.entrypoints import utils as entrypoint_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_funaudiochat_arches_register_model_types_for_hf_config_patching():
    assert _ARCH_TO_MODEL_TYPE["FunAudioChatForConditionalGeneration"] == "funaudiochat"
    assert _ARCH_TO_MODEL_TYPE["FunAudioChatCosyVoice3Code2Wav"] == "cosyvoice3"


def test_resolve_model_config_path_detects_funaudiochat_default_yaml(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        entrypoint_utils,
        "get_config",
        lambda model, trust_remote_code=True: SimpleNamespace(model_type="funaudiochat"),
    )

    resolved = entrypoint_utils.resolve_model_config_path("dummy-funaudiochat-model")

    assert resolved is not None
    assert resolved.endswith("vllm_omni/model_executor/stage_configs/funaudiochat.yaml")


def test_funaudiochat_default_stage_config_keeps_audio_profile_and_audio_towers():
    config_path = (
        Path(__file__).resolve().parents[2] / "vllm_omni" / "model_executor" / "stage_configs" / "funaudiochat.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    stage0_engine_args = config["stage_args"][0]["engine_args"]

    assert "language_model_only" not in stage0_engine_args
    assert stage0_engine_args["hf_overrides"]["audio_config"]["max_source_positions"] == 1500
    assert stage0_engine_args["limit_mm_per_prompt"]["audio"] == 1
