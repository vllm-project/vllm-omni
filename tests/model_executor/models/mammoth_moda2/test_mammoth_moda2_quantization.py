# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only structural unit tests for MammothModa2 AR quantization wiring.

Covers the pure logic that backs the FP8 quantization path, without requiring a
GPU or a real checkpoint:

* Generation-expert routing (``moe_enable`` / ``moe_forward``).
* Extra generation vocabulary/head weight-name mapping (``hf_to_vllm_mapper``).
* Base + generation vocabulary sizing from the Dev config.
* Quantization-scale name remapping (``maybe_remap_kv_scale_name``).

The GPU end-to-end FP8 A/B gates (understanding + t2i) live in
``tests/e2e/offline_inference/test_mammoth_moda2_fp8_quantization.py``.
"""

from __future__ import annotations

import pytest
import torch
from vllm.model_executor.model_loader.weight_utils import maybe_remap_kv_scale_name

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2Qwen3ARForConditionalGeneration,
    moe_enable,
    moe_forward,
)
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestMoeEnable:
    """Generation-expert routing switch (``moe_type`` range parsing)."""

    def test_range_inclusive_start_exclusive_end(self):
        assert moe_enable("ffn-18:36", "ffn", 18) is True
        assert moe_enable("ffn-18:36", "ffn", 35) is True

    def test_before_start_disabled(self):
        assert moe_enable("ffn-18:36", "ffn", 17) is False

    def test_at_end_disabled(self):
        assert moe_enable("ffn-18:36", "ffn", 36) is False

    def test_wrong_layer_type_disabled(self):
        assert moe_enable("ffn-18:36", "attention", 20) is False

    def test_unbounded_range_enables_all_ffn(self):
        assert moe_enable("ffn", "ffn", 0) is True

    def test_none_disables(self):
        assert moe_enable("none", "ffn", 20) is False


class TestMoeForward:
    """``gen_mlp`` vs ``mlp`` expert routing under different token masks."""

    @staticmethod
    def _experts():
        und = lambda x: x * 0.0  # noqa: E731
        gen = lambda x: x + 100.0  # noqa: E731
        return und, gen

    def test_no_gen_expert_routes_to_und(self):
        hidden = torch.arange(8.0).reshape(2, 4)
        und, _ = self._experts()
        out = moe_forward(hidden, und, None, torch.tensor([True, False]))
        assert torch.equal(out, und(hidden))

    def test_none_mask_routes_to_und(self):
        hidden = torch.arange(8.0).reshape(2, 4)
        und, gen = self._experts()
        out = moe_forward(hidden, und, gen, None)
        assert torch.equal(out, und(hidden))

    def test_all_false_mask_routes_to_und(self):
        hidden = torch.arange(8.0).reshape(2, 4)
        und, gen = self._experts()
        out = moe_forward(hidden, und, gen, torch.tensor([False, False]))
        assert torch.equal(out, und(hidden))

    def test_all_true_mask_routes_to_gen(self):
        hidden = torch.arange(8.0).reshape(2, 4)
        und, gen = self._experts()
        out = moe_forward(hidden, und, gen, torch.tensor([True, True]))
        assert torch.equal(out, gen(hidden))

    def test_mixed_mask_reorders_correctly(self):
        hidden = torch.arange(12.0).reshape(3, 4)
        und, gen = self._experts()
        mask = torch.tensor([True, False, True])
        out = moe_forward(hidden, und, gen, mask)

        expected = torch.zeros_like(hidden)
        expected[0] = gen(hidden[0])
        expected[1] = und(hidden[1])
        expected[2] = gen(hidden[2])
        assert torch.equal(out, expected)

    def test_mask_shape_mismatch_raises(self):
        hidden = torch.arange(12.0).reshape(3, 4)
        und, gen = self._experts()
        with pytest.raises(ValueError):
            moe_forward(hidden, und, gen, torch.tensor([True, False]))


class TestCheckpointWeightsMapper:
    """Checkpoint prefix remap: extra vocab/head + generation-side filtering."""

    _CHECKPOINT_NAMES = [
        "llm_model.model.visual.blocks.0.attn.q.weight",
        "llm_model.lm_head.weight",
        "llm_model.model.language_model.gen_embed_tokens.weight",
        "llm_model.model.language_model.layers.0.self_attn.q_proj.weight",
        "llm_model.gen_head.weight",
        # Generation-side (DiT / VAE / tokenizer) weights must be skipped.
        "gen_vae.decoder.conv_out.weight",
        "gen_transformer.layers.0.weight",
        "gen_tokenizer.image_tokenizer.quant_conv.weight",
        "gen_image_condition_refiner.layers.0.weight",
    ]

    def test_prefix_remap_and_generation_side_filtering(self):
        mapper = MammothModa2Qwen3ARForConditionalGeneration.hf_to_vllm_mapper
        mapped = mapper.apply_list(self._CHECKPOINT_NAMES)

        assert mapped == [
            "visual.blocks.0.attn.q.weight",
            "language_model.lm_head.weight",
            "language_model.gen_embed_tokens.weight",
            "language_model.layers.0.self_attn.q_proj.weight",
            "language_model.gen_head.weight",
        ]

        # None of the generation-side sub-modules may survive the remap.
        assert all(not name.startswith("gen_") for name in mapped)


class TestExtraVocabHeadConfig:
    """Extra generation vocabulary / head sizes derived from the Dev config."""

    def test_dev_config_exposes_base_plus_gen_vocab(self):
        config = Mammothmoda2Config(
            llm_config={
                "model_type": "mammothmoda2_qwen3_vl",
                "text_config": {
                    "model_type": "mammothmoda2_qwen3_vl_text",
                    "vocab_size": 151936,
                    "gen_vocab_size": 32800,
                    "gen_vocab_start_index": 152064,
                },
            }
        )

        text_cfg = config.get_text_config()
        # vLLM validates sampling against base + generation vocabulary.
        assert text_cfg.vocab_size == 152064 + 32800
        assert text_cfg.gen_vocab_start_index == 152064


class TestQuantizationScaleRemap:
    """Quantization-scale name remapping used by ``load_weights``.

    Covers the ``maybe_remap_kv_scale_name`` path in
    ``MammothModa2Qwen2ForCausalLM.load_weights`` (FP8 ``*_scale`` loading).
    """

    def test_existing_name_returns_unchanged(self):
        params = {"model.layers.0.self_attn.attn.k_scale": None}
        assert (
            maybe_remap_kv_scale_name("model.layers.0.self_attn.attn.k_scale", params)
            == "model.layers.0.self_attn.attn.k_scale"
        )

    def test_deprecated_kv_scale_remapped_to_k_scale(self):
        params = {"model.layers.0.self_attn.attn.k_scale": None}
        assert (
            maybe_remap_kv_scale_name("model.layers.0.self_attn.kv_scale", params)
            == "model.layers.0.self_attn.attn.k_scale"
        )

    def test_qkv_proj_scale_remapped(self):
        params = {
            "model.layers.0.self_attn.attn.k_scale": None,
            "model.layers.0.self_attn.attn.v_scale": None,
        }
        assert (
            maybe_remap_kv_scale_name("model.layers.0.self_attn.qkv_proj.k_scale", params)
            == "model.layers.0.self_attn.attn.k_scale"
        )
        assert (
            maybe_remap_kv_scale_name("model.layers.0.self_attn.qkv_proj.v_scale", params)
            == "model.layers.0.self_attn.attn.v_scale"
        )

    def test_missing_scale_returns_none(self):
        assert maybe_remap_kv_scale_name("model.layers.0.self_attn.qkv_proj.k_scale", {}) is None
