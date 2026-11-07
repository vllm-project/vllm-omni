# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression for the pinned SGLang comparison image's component validator."""

import json

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
pytest.importorskip("sglang")


@pytest.fixture
def loader():
    from sglang.multimodal_gen.runtime.utils import hf_diffusers_utils

    return hf_diffusers_utils


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanVideo15Pipeline",
                "guider": ["diffusers", "ClassifierFreeGuidance"],
                "transformer": ["diffusers", "HunyuanVideo15Transformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLHunyuanVideo15"],
            }
        )
    )
    (tmp_path / "guider").mkdir()
    (tmp_path / "guider/guider_config.json").write_text('{"guidance_scale": 6.0}')
    for component in ("transformer", "vae"):
        (tmp_path / component).mkdir()
        (tmp_path / component / "diffusion_pytorch_model.safetensors").touch()
    return tmp_path


def test_config_only_guider_is_a_complete_snapshot(loader, snapshot):
    assert loader._get_missing_declared_weight_components(str(snapshot)) == []
    assert loader._verify_diffusers_model_complete(str(snapshot))
    assert loader._check_index_files_for_missing_shards(str(snapshot))[0]


def test_missing_transformer_still_rejected(loader, snapshot):
    (snapshot / "transformer/diffusion_pytorch_model.safetensors").unlink()
    assert loader._get_missing_declared_weight_components(str(snapshot)) == ["transformer/<weights>"]
    assert not loader._verify_diffusers_model_complete(str(snapshot))
    assert not loader._check_index_files_for_missing_shards(str(snapshot))[0]


def test_custom_guider_still_requires_weights(loader, snapshot):
    path = snapshot / "model_index.json"
    config = json.loads(path.read_text())
    config["guider"] = ["custom_library", "ClassifierFreeGuidance"]
    path.write_text(json.dumps(config))
    assert loader._get_missing_declared_weight_components(str(snapshot)) == ["guider/<weights>"]
    assert not loader._verify_diffusers_model_complete(str(snapshot))
