# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Solar's nested bundle must resolve before invoking its model loader."""

from __future__ import annotations

import json

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("subfolder", ["SolarWM-h3-33B-base/transformer", "transformer"])
def test_explicit_pipeline_enriches_bundle_without_any_root_config(tmp_path, monkeypatch, subfolder):
    # Omni constructs/enriches this config before invoking a pipeline loader.
    # The actual publisher bundles base components under a directory, with no
    # top-level HF config or Diffusers index. An explicit class must reach it.
    transformer = tmp_path / subfolder
    transformer.mkdir(parents=True)
    descriptor = {"num_attention_heads": 56, "attention_head_dim": 128, "num_layers": 50}
    (transformer / "config.json").write_text(json.dumps(descriptor))
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)
    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name="SolarWMH3Pipeline")
    config.enrich_config()
    assert config.model_class_name == "SolarWMH3Pipeline"
    assert config.tf_model_config.to_dict() == descriptor
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 1
