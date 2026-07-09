# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the NVFP4 blockwise W4A16 target-inclusion quant config.

Verifies the wiring that serves the weight-only nvfp4_blockwise_mixed_v1
artifact FP4-resident: the config must select vLLM's W4A16 method (Marlin, bf16
activations — no input_scale) and quantize ONLY the MLP / MLP-MoE-gen projections.
"""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# The pure prefix predicate does not need vLLM; the config build does.
from vllm_omni.quantization import nvfp4_blockwise as nb  # noqa: E402

_HAS_W4A16 = True
try:
    from vllm.model_executor.layers.quantization.modelopt import (  # noqa: E402
        ModelOptNvFp4W4A16LinearMethod,
    )
except Exception:  # pragma: no cover - depends on installed vLLM
    _HAS_W4A16 = False
    ModelOptNvFp4W4A16LinearMethod = None  # bound for skip-guarded references

needs_w4a16 = pytest.mark.skipif(
    not _HAS_W4A16, reason="vLLM ModelOptNvFp4W4A16LinearMethod not available in this env"
)


# --- pure prefix predicate --------------------------------------------------

@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.layers.0.mlp.gate_proj",
        "language_model.layers.35.mlp.up_proj",
        "language_model.layers.7.mlp.down_proj",
        "gen_layers.0.mlp.gate_proj",
        "gen_layers.35.mlp.down_proj",
        "gen_layers.0.mlp_moe_gen.gate_proj",
        "gen_layers.10.mlp_moe_gen.up_proj",
        "gen_layers.35.mlp_moe_gen.down_proj",
    ],
)
def test_target_prefixes_included(prefix):
    assert nb.is_target_prefix(prefix) is True


@pytest.mark.parametrize(
    "prefix",
    [
        "language_model.layers.0.self_attn.to_q",
        "language_model.layers.0.self_attn.to_out",
        "gen_layers.0.self_attn.add_q_proj",
        "proj_in",
        "proj_out",
        "language_model.embed_tokens",
        "lm_head",
        "time_embedder.linear_1",
        "action_proj_in",
        "audio_proj_out",
        "language_model.layers.0.mlp",  # the container, not a projection
    ],
)
def test_non_target_prefixes_excluded(prefix):
    assert nb.is_target_prefix(prefix) is False


# --- config build -----------------------------------------------------------

@needs_w4a16
def test_build_selects_w4a16_method():
    cfg = nb.build_nvfp4_blockwise_w4a16_config()
    assert cfg.get_name() == "modelopt_fp4"
    assert cfg.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod
    assert cfg.group_size == 16


@needs_w4a16
def test_config_quantizes_only_targets():
    cfg = nb.build_nvfp4_blockwise_w4a16_config()
    # targets NOT excluded (get the W4A16 FP4 method)
    assert cfg.is_layer_excluded("language_model.layers.0.mlp.gate_proj") is False
    assert cfg.is_layer_excluded("gen_layers.10.mlp.down_proj") is False
    assert cfg.is_layer_excluded("gen_layers.10.mlp_moe_gen.down_proj") is False
    # non-targets excluded (stay BF16 / UnquantizedLinearMethod)
    assert cfg.is_layer_excluded("language_model.layers.0.self_attn.to_q") is True
    assert cfg.is_layer_excluded("lm_head") is True
    assert cfg.is_layer_excluded("proj_out") is True


# --- resolution helper ------------------------------------------------------

def test_maybe_build_returns_active_for_non_recipe():
    sentinel = object()
    assert nb.maybe_build_nvfp4_blockwise_config(None, sentinel) is sentinel
    assert nb.maybe_build_nvfp4_blockwise_config("some_other_recipe", sentinel) is sentinel


@needs_w4a16
def test_maybe_build_builds_for_recipe_when_no_active():
    cfg = nb.maybe_build_nvfp4_blockwise_config(nb.RECIPE, None)
    assert cfg is not None
    assert cfg.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod


@needs_w4a16
def test_transformer_config_quant_recipe_builds_w4a16_config():
    from vllm_omni.diffusion.data import TransformerConfig

    tf_config = TransformerConfig.from_dict({"quant_recipe": nb.RECIPE})

    assert tf_config.quant_config is not None
    assert tf_config.quant_config.LinearMethodCls is ModelOptNvFp4W4A16LinearMethod
    assert tf_config.quant_method == nb.RECIPE
