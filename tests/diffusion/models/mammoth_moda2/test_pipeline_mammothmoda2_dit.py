# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _root_weight_source,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _raw_config() -> dict:
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 8,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [8, 8]},
        "gen_dit_config": {"hidden_size": 8, "in_channels": 4},
    }


def _od_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_raw_config()),
    )


def test_build_mammoth_config_uses_shared_transformer_projection() -> None:
    config = _build_mammoth_config(_od_config())
    assert config.model_type == "mammothmoda2"
    assert config.get_text_config().hidden_size == 8
    assert config.gen_dit_config == {"hidden_size": 8, "in_channels": 4}


def test_build_mammoth_config_rejects_empty_shared_projection() -> None:
    config = _od_config()
    config.tf_model_config = TransformerConfig()
    with pytest.raises(ValueError, match="root checkpoint config"):
        _build_mammoth_config(config)


def test_root_weight_source_loads_combined_checkpoint_once() -> None:
    source = _root_weight_source(_od_config())
    assert source.model_or_path == "/models/MammothModa2-Preview"
    assert source.subfolder is None
    assert source.prefix == ""
    assert source.fall_back_to_pt is True


def test_root_weight_source_forwards_revision() -> None:
    config = SimpleNamespace(model="/models/MammothModa2-Preview", revision="rev-7")
    assert _root_weight_source(config).revision == "rev-7"


def test_pipeline_declares_native_components_and_single_request_mode_only() -> None:
    assert MammothModa2DiTPipeline._dit_modules == ["gen_transformer"]
    assert MammothModa2DiTPipeline._encoder_modules == ["gen_image_condition_refiner"]
    assert MammothModa2DiTPipeline._vae_modules == ["gen_vae"]
    assert MammothModa2DiTPipeline.supports_request_batch is False
    assert MammothModa2DiTPipeline.supports_step_execution is False


def test_root_weight_source_rejects_missing_model_path() -> None:
    config = SimpleNamespace(model=None, revision=None)
    with pytest.raises(ValueError, match="model path"):
        _root_weight_source(config)
