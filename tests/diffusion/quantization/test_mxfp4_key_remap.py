# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for merge_mxfp4_dualscale_checkpoint.py key-remapping helpers.

These are pure-Python unit tests that exercise the transformation functions
without loading any actual checkpoint files or requiring NPU hardware.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# SUPPORTED_MODEL_TYPES
# ---------------------------------------------------------------------------


def test_supported_model_types_includes_a14b():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-T2V-A14B" in SUPPORTED_MODEL_TYPES
    assert "Wan2.2-I2V-A14B" in SUPPORTED_MODEL_TYPES


def test_supported_model_types_excludes_ti2v_5b():
    """TI2V-5B is explicitly NOT supported under W4A4 quantization."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-TI2V-5B" not in SUPPORTED_MODEL_TYPES


# ---------------------------------------------------------------------------
# _apply_rename_dict
# ---------------------------------------------------------------------------


def test_apply_rename_dict_self_attn_qkvo():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.0.self_attn.q.weight") == "blocks.0.attn1.to_q.weight"
    assert _apply_rename_dict("blocks.0.self_attn.k.weight") == "blocks.0.attn1.to_k.weight"
    assert _apply_rename_dict("blocks.0.self_attn.v.weight") == "blocks.0.attn1.to_v.weight"
    assert _apply_rename_dict("blocks.0.self_attn.o.weight") == "blocks.0.attn1.to_out.0.weight"


def test_apply_rename_dict_ffn():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.1.ffn.0.weight") == "blocks.1.ffn.net.0.proj.weight"
    assert _apply_rename_dict("blocks.1.ffn.2.weight") == "blocks.1.ffn.net.2.weight"


def test_apply_rename_dict_norm_order_swap():
    """norm2↔norm3 swap: quant tool uses norm1/norm3/norm2 order,
    Diffusers uses norm1/norm2/norm3."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.0.norm2.weight") == "blocks.0.norm3.weight"
    assert _apply_rename_dict("blocks.0.norm3.weight") == "blocks.0.norm2.weight"


def test_apply_rename_dict_cross_attn():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("blocks.0.cross_attn.q.weight") == "blocks.0.attn2.to_q.weight"
    assert _apply_rename_dict("blocks.0.cross_attn.k.weight") == "blocks.0.attn2.to_k.weight"
    assert _apply_rename_dict("blocks.0.cross_attn.v.weight") == "blocks.0.attn2.to_v.weight"
    assert _apply_rename_dict("blocks.0.cross_attn.o.weight") == "blocks.0.attn2.to_out.0.weight"


def test_apply_rename_dict_head_and_modulation():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _apply_rename_dict

    assert _apply_rename_dict("head.head.weight") == "proj_out.weight"


# ---------------------------------------------------------------------------
# _strip_mxfp4_wrapper
# ---------------------------------------------------------------------------


def test_strip_mxfp4_wrapper_linear_weight():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.weight") == "blocks.0.attn1.to_q.weight"


def test_strip_mxfp4_wrapper_linear_weight_scale():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.weight_scale") == "blocks.0.attn1.to_q.weight_scale"


def test_strip_mxfp4_wrapper_linear_weight_dual_scale():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert (
        _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.weight_dual_scale") == "blocks.0.attn1.to_q.weight_dual_scale"
    )


def test_strip_mxfp4_wrapper_linear_bias():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.linear.bias") == "blocks.0.attn1.to_q.bias"


def test_strip_mxfp4_wrapper_div_mul_scale():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.div.mul_scale") == "blocks.0.attn1.to_q.mul_scale"


def test_strip_mxfp4_wrapper_noop_for_plain_weight():
    """MXFP8 / FLOAT tensors have no wrapper — must be returned unchanged."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _strip_mxfp4_wrapper

    assert _strip_mxfp4_wrapper("blocks.0.attn1.to_q.weight") == "blocks.0.attn1.to_q.weight"
    assert _strip_mxfp4_wrapper("blocks.0.norm_q.weight") == "blocks.0.norm_q.weight"
    assert _strip_mxfp4_wrapper("condition_embedder.time_embedder.linear_1.weight") == (
        "condition_embedder.time_embedder.linear_1.weight"
    )


# ---------------------------------------------------------------------------
# _classify_blocks
# ---------------------------------------------------------------------------


def test_classify_blocks_mixed_mxfp8_and_mxfp4():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _classify_blocks

    quant_meta = {
        "blocks.0.attn1.to_q.weight": "W8A8_MXFP8",
        "blocks.1.attn1.to_q.weight": "W8A8_MXFP8",
        "blocks.2.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "blocks.3.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
        "condition_embedder.time_embedder.linear_1.weight": "FLOAT",
    }
    block_types = _classify_blocks(quant_meta)
    assert block_types[0] == "mxfp8"
    assert block_types[1] == "mxfp8"
    assert block_types[2] == "mxfp4_dualscale"
    assert block_types[3] == "mxfp4_dualscale"
    # Non-block key must not produce an entry
    assert None not in block_types


def test_classify_blocks_float_entries_skipped():
    """FLOAT-typed tensors must not contribute to block classification."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _classify_blocks

    quant_meta = {
        "blocks.0.bias": "FLOAT",
        "blocks.1.attn1.to_q.weight": "W4A4_MXFP4_DUALSCALE",
    }
    block_types = _classify_blocks(quant_meta)
    assert 0 not in block_types
    assert block_types[1] == "mxfp4_dualscale"


def test_classify_blocks_empty():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _classify_blocks

    assert _classify_blocks({}) == {}


# ---------------------------------------------------------------------------
# _is_mxfp4_tensor
# ---------------------------------------------------------------------------


def test_is_mxfp4_tensor_quantized_weight():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _is_mxfp4_tensor

    assert _is_mxfp4_tensor("blocks.0.attn1.to_q.linear.weight", "W4A4_MXFP4_DUALSCALE") is True
    assert _is_mxfp4_tensor("blocks.0.attn1.to_q.linear.weight_scale", "W4A4_MXFP4_DUALSCALE") is True
    assert _is_mxfp4_tensor("blocks.0.attn1.to_q.linear.weight_dual_scale", "W4A4_MXFP4_DUALSCALE") is True


def test_is_mxfp4_tensor_companion_bias_and_mul_scale():
    """Companion tensors (bias, mul_scale) are FLOAT but belong to MXFP4 layers
    and must be included in the merged checkpoint."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _is_mxfp4_tensor

    # bias inside .linear. wrapper → must be included
    assert _is_mxfp4_tensor("blocks.0.attn1.to_q.linear.bias", "FLOAT") is True
    # mul_scale inside .div. wrapper → must be included
    assert _is_mxfp4_tensor("blocks.0.attn1.to_q.div.mul_scale", "FLOAT") is True


def test_is_mxfp4_tensor_bf16_fallback_weight():
    """BF16 fallback linear layers have plain .weight keys (no wrapper) → must NOT be included."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _is_mxfp4_tensor

    assert _is_mxfp4_tensor("blocks.0.attn1.to_q.weight", "FLOAT") is False
    assert _is_mxfp4_tensor("condition_embedder.time_embedder.linear_1.weight", "FLOAT") is False


def test_is_mxfp4_tensor_norm_weight():
    """Norm layers are always FLOAT with plain .weight → not MXFP4 related."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _is_mxfp4_tensor

    assert _is_mxfp4_tensor("blocks.0.norm1.weight", "FLOAT") is False


# ---------------------------------------------------------------------------
# _collect_ignored_layers
# ---------------------------------------------------------------------------


def test_collect_ignored_layers_basic():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _collect_ignored_layers

    # block 0: all three of Q/K/V are BF16 → fused as to_qkv in ignored_layers
    # block 1: to_qkv is MXFP4 → not in ignored_layers
    merged = {
        "blocks.0.attn1.to_q.weight": torch.zeros(4, 4),
        "blocks.0.attn1.to_k.weight": torch.zeros(4, 4),
        "blocks.0.attn1.to_v.weight": torch.zeros(4, 4),
        "blocks.1.attn1.to_q.weight": torch.zeros(4, 4),
        "blocks.1.attn1.to_q.weight_scale": torch.zeros(4, 1),
    }
    mxfp4_prefixes = {"blocks.1.attn1.to_q"}

    ignored = _collect_ignored_layers(merged, mxfp4_prefixes)
    assert "blocks.0.attn1.to_qkv" in ignored  # fused vllm-omni name
    assert "blocks.0.attn1.to_q" not in ignored  # diffusers name must not appear
    assert "blocks.0.attn1.to_k" not in ignored
    assert "blocks.1.attn1.to_qkv" not in ignored  # MXFP4, not ignored


def test_collect_ignored_layers_empty_mxfp4():
    """When no layers are MXFP4 (all BF16), all .weight prefixes become ignored_layers."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _collect_ignored_layers

    # Use non-attn layers to avoid the Q/K/V completeness requirement.
    merged = {
        "blocks.0.ffn.net.0.proj.weight": torch.zeros(4, 4),
        "proj_out.weight": torch.zeros(4, 4),
    }
    ignored = _collect_ignored_layers(merged, set())
    assert sorted(ignored) == ["blocks.0.ffn.net_0.proj", "proj_out"]


def test_collect_ignored_layers_returns_sorted():
    """ignored_layers must be sorted for deterministic config.json output."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _collect_ignored_layers

    # Use FFN layers (no Q/K/V fusion concern) to isolate the sort guarantee.
    merged = {
        "blocks.2.ffn.net.0.proj.weight": torch.zeros(1),
        "blocks.0.ffn.net.0.proj.weight": torch.zeros(1),
        "blocks.1.ffn.net.0.proj.weight": torch.zeros(1),
    }
    ignored = _collect_ignored_layers(merged, set())
    assert ignored == sorted(ignored)


# ---------------------------------------------------------------------------
# _build_quant_config (new mxfp4_dualscale format)
# ---------------------------------------------------------------------------


def test_build_quant_config_new_format():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _build_quant_config

    config = _build_quant_config(["blocks.0.attn1.to_q", "proj_out"])
    assert config["quant_method"] == "mxfp4_dualscale"
    assert config["is_checkpoint_serialized"] is True
    assert config["ignored_layers"] == ["blocks.0.attn1.to_q", "proj_out"]
    assert "num_mxfp8_blocks" not in config


def test_build_quant_config_empty_ignored_layers():
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _build_quant_config

    config = _build_quant_config([])
    assert config["ignored_layers"] == []


# ---------------------------------------------------------------------------
# _diffusers_to_vllm_ignored
# ---------------------------------------------------------------------------


def test_remap_self_attn_qkv_fusion():
    """All three of attn1.to_q/k/v in BF16 → fused attn1.to_qkv."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    result = _diffusers_to_vllm_ignored(["blocks.0.attn1.to_k", "blocks.0.attn1.to_q", "blocks.0.attn1.to_v"])
    assert result == ["blocks.0.attn1.to_qkv"]


def test_remap_self_attn_qkv_partial_raises():
    """Partial Q/K/V in ignored_layers is invalid — must raise ValueError.

    Self-attention Q/K/V are fused into a single to_qkv layer at runtime;
    partial precision (some BF16, some MXFP4) cannot be expressed and must
    be caught early in the merge script.
    """
    import pytest

    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    with pytest.raises(ValueError, match="Partial BF16 fallback"):
        _diffusers_to_vllm_ignored(["blocks.0.attn1.to_q", "blocks.0.attn1.to_k"])


def test_remap_cross_attn_kept_separate():
    """Cross-attention (attn2) to_q/k/v must NOT be fused."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    result = _diffusers_to_vllm_ignored(["blocks.0.attn2.to_k", "blocks.0.attn2.to_q", "blocks.0.attn2.to_v"])
    assert "blocks.0.attn2.to_qkv" not in result
    assert "blocks.0.attn2.to_q" in result
    assert "blocks.0.attn2.to_k" in result
    assert "blocks.0.attn2.to_v" in result


def test_remap_ffn_dot_to_underscore():
    """ffn.net.0.proj → ffn.net_0.proj; ffn.net.2 → ffn.net_2."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    result = _diffusers_to_vllm_ignored(["blocks.0.ffn.net.0.proj", "blocks.0.ffn.net.2"])
    assert "blocks.0.ffn.net_0.proj" in result
    assert "blocks.0.ffn.net_2" in result
    assert "blocks.0.ffn.net.0.proj" not in result
    assert "blocks.0.ffn.net.2" not in result


def test_remap_to_out_strips_index():
    """attn1.to_out.0 and attn2.to_out.0 must have the .0 suffix removed."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    result = _diffusers_to_vllm_ignored(["blocks.0.attn1.to_out.0", "blocks.0.attn2.to_out.0"])
    assert "blocks.0.attn1.to_out" in result
    assert "blocks.0.attn2.to_out" in result
    assert "blocks.0.attn1.to_out.0" not in result
    assert "blocks.0.attn2.to_out.0" not in result


def test_remap_noop_for_non_block_layers():
    """Layers outside 'blocks.N.*' (condition_embedder, proj_out) pass through unchanged."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    inputs = [
        "condition_embedder.time_embedder.linear_1",
        "condition_embedder.text_embedder.linear_1",
        "proj_out",
    ]
    result = _diffusers_to_vllm_ignored(inputs)
    assert result == sorted(inputs)


def test_remap_mixed_block_layers():
    """Typical mixed block: QKV fused, to_out stripped, FFN renamed, cross-attn separate."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    inputs = [
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_out.0",
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_v",
        "blocks.0.attn2.to_k",
        "blocks.0.attn2.to_out.0",
        "blocks.0.attn2.to_q",
        "blocks.0.attn2.to_v",
        "blocks.0.ffn.net.0.proj",
        "blocks.0.ffn.net.2",
    ]
    result = _diffusers_to_vllm_ignored(inputs)
    assert "blocks.0.attn1.to_qkv" in result
    assert "blocks.0.attn1.to_out" in result
    assert "blocks.0.attn2.to_q" in result
    assert "blocks.0.attn2.to_k" in result
    assert "blocks.0.attn2.to_v" in result
    assert "blocks.0.attn2.to_out" in result
    assert "blocks.0.ffn.net_0.proj" in result
    assert "blocks.0.ffn.net_2" in result
    # Old names must be gone
    assert "blocks.0.attn1.to_q" not in result
    assert "blocks.0.ffn.net.0.proj" not in result


def test_remap_returns_sorted():
    """Output of _diffusers_to_vllm_ignored must always be sorted."""
    from vllm_omni.quantization.tools.merge_mxfp4_dualscale_checkpoint import _diffusers_to_vllm_ignored

    inputs = [
        "blocks.2.ffn.net.0.proj",
        "blocks.0.attn1.to_q",
        "blocks.0.attn1.to_k",
        "blocks.0.attn1.to_v",
        "blocks.1.attn2.to_out.0",
    ]
    result = _diffusers_to_vllm_ignored(inputs)
    assert result == sorted(result)
