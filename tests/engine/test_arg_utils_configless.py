# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
import os

import pytest
from transformers import PretrainedConfig

from vllm_omni.engine.arg_utils import _ARCH_TO_MODEL_TYPE, _CONFIG_LESS_MODEL_TYPES, OmniEngineArgs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chatterbox_arch_maps_to_model_type_001() -> None:
    assert _ARCH_TO_MODEL_TYPE["ChatterboxForConditionalGeneration"] == "chatterbox"
    assert {"indextts2_5", "chatterbox"} <= _CONFIG_LESS_MODEL_TYPES


def test_a_hub_repo_without_config_gets_one_written_001(monkeypatch) -> None:
    def no_config(*args, **kwargs):
        raise OSError("no config.json")

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", no_config)
    args = OmniEngineArgs(model="ResembleAI/chatterbox-turbo", model_arch="ChatterboxForConditionalGeneration")
    args.hf_config_path = None

    args._patch_empty_hf_config("chatterbox")

    assert args.hf_config_path is not None
    with open(os.path.join(args.hf_config_path, "config.json")) as handle:
        assert json.load(handle) == {
            "model_type": "chatterbox",
            "architectures": ["ChatterboxForConditionalGeneration"],
        }


def test_an_unlisted_model_type_is_left_alone_001(monkeypatch) -> None:
    def no_config(*args, **kwargs):
        raise OSError("no config.json")

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", no_config)
    args = OmniEngineArgs(model="some/other-model", model_arch="CosyVoice3Model")
    args.hf_config_path = None

    args._patch_empty_hf_config("cosyvoice3")

    assert args.hf_config_path is None
