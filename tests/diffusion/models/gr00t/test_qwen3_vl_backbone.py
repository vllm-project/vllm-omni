# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for F4: Qwen3VLBackbone with select_layer=16 truncation.

Validates the layer-truncation contract and the forward-time image_mask /
backbone_attention_mask shape.  Numerical parity vs Isaac-GR00T is F10.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_qwen3vl_config(num_hidden_layers: int = 4, image_token_id: int = 7):
    """A tiny Qwen3VLConfig that instantiates and can run a forward on CPU.

    Vocab is bumped to include `image_token_id` so input IDs containing the
    image marker token don't trip the embedding bounds check.
    """
    from transformers import Qwen3VLConfig, Qwen3VLTextConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )

    tc = Qwen3VLTextConfig(
        vocab_size=max(64, image_token_id + 1),
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 1]},
    )
    vc = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=24,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=24,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    return Qwen3VLConfig(
        text_config=tc.to_dict(),
        vision_config=vc.to_dict(),
        image_token_id=image_token_id,
    )


@pytest.fixture
def gr00t_config():
    """A Gr00tN1d7Config with a select_layer that we can verify against the
    tiny Qwen3VL config (which has 4 layers)."""
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    return Gr00tN1d7Config(model_name="", select_layer=2)


def test_select_layer_truncation(gr00t_config):
    from vllm_omni.diffusion.models.gr00t.adapter_qwen3_vl import Qwen3VLBackbone

    hf_cfg = _tiny_qwen3vl_config(num_hidden_layers=4)
    backbone = Qwen3VLBackbone(gr00t_config, hf_config=hf_cfg)
    lm_layers = backbone.model.model.language_model.layers
    assert len(lm_layers) == gr00t_config.select_layer == 2


def test_truncation_rejects_too_few_layers(gr00t_config):
    """When the backbone has fewer layers than select_layer, we should raise
    rather than silently keep all layers."""
    from vllm_omni.diffusion.models.gr00t.adapter_qwen3_vl import Qwen3VLBackbone

    gr00t_config.select_layer = 8
    hf_cfg = _tiny_qwen3vl_config(num_hidden_layers=4)
    with pytest.raises(ValueError, match="select_layer"):
        Qwen3VLBackbone(gr00t_config, hf_config=hf_cfg)


def test_image_token_id_property(gr00t_config):
    from vllm_omni.diffusion.models.gr00t.adapter_qwen3_vl import Qwen3VLBackbone

    hf_cfg = _tiny_qwen3vl_config(num_hidden_layers=4, image_token_id=11)
    backbone = Qwen3VLBackbone(gr00t_config, hf_config=hf_cfg)
    assert backbone.image_token_id == 11


def test_forward_returns_expected_dict_and_shapes(gr00t_config):
    """Run a text-only forward through the tiny truncated backbone and check
    the output dict.  Pixel inputs are exercised by F10."""
    from vllm_omni.diffusion.models.gr00t.adapter_qwen3_vl import Qwen3VLBackbone

    img_tok = 7
    hf_cfg = _tiny_qwen3vl_config(num_hidden_layers=4, image_token_id=img_tok)
    backbone = Qwen3VLBackbone(gr00t_config, hf_config=hf_cfg)
    backbone.eval()

    B, S = 2, 6
    input_ids = torch.tensor(
        [
            [1, 2, img_tok, img_tok, 5, 6],
            [3, 4, 5, img_tok, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert set(out.keys()) == {
        "backbone_features",
        "backbone_attention_mask",
        "image_mask",
    }
    assert out["backbone_features"].shape == (B, S, hf_cfg.text_config.hidden_size)
    assert out["image_mask"].shape == (B, S)
    assert out["image_mask"].dtype == torch.bool
    assert out["backbone_attention_mask"].shape == (B, S)
    assert out["backbone_attention_mask"].dtype == torch.bool

    expected_image_mask = input_ids == img_tok
    torch.testing.assert_close(
        out["image_mask"], expected_image_mask, rtol=0, atol=0
    )
    expected_attn_mask = attention_mask == 1
    torch.testing.assert_close(
        out["backbone_attention_mask"], expected_attn_mask, rtol=0, atol=0
    )


def test_missing_overlay_raises_when_no_hf_config():
    """Without text_config/vision_config on the GR00T config and without an
    explicit hf_config, Qwen3VLBackbone must raise — it has nothing to build
    the inner HF model from."""
    from vllm_omni.diffusion.models.gr00t.adapter_qwen3_vl import Qwen3VLBackbone
    from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

    cfg = Gr00tN1d7Config(model_name="")
    assert getattr(cfg, "text_config", None) is None
    with pytest.raises(ValueError, match="text_config"):
        Qwen3VLBackbone(cfg)
