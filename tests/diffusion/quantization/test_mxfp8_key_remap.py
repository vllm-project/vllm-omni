# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for merge_mxfp8_checkpoint.py key-remapping helpers and model metadata.

These are pure-Python unit tests that exercise the transformation functions
without loading any actual checkpoint files or requiring NPU hardware.

Key contracts verified:
- SUPPORTED_MODEL_TYPES includes Wan2.2-TI2V-5B (MXFP8 supports it; MXFP4 does not)
- MXFP8_QUANT_CONFIG structure matches what TransformerConfig.from_dict() reads
- _remap_keys correctly translates msModelSlim naming → Diffusers naming
- _get_transformer_dirs routes cascade vs single-transformer models
- _get_quant_subdir maps high/low noise subdirs for cascade models
"""

import pathlib

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# SUPPORTED_MODEL_TYPES and CASCADE_MODEL_TYPES
# ---------------------------------------------------------------------------


def test_supported_model_types_includes_all_wan22():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-T2V-A14B" in SUPPORTED_MODEL_TYPES
    assert "Wan2.2-I2V-A14B" in SUPPORTED_MODEL_TYPES


def test_supported_model_types_includes_ti2v_5b():
    """MXFP8 supports TI2V-5B (contrast: MXFP4 explicitly excludes it)."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-TI2V-5B" in SUPPORTED_MODEL_TYPES


def test_cascade_model_types_excludes_ti2v():
    """TI2V-5B is a single-transformer model — must not be in CASCADE_MODEL_TYPES."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import CASCADE_MODEL_TYPES

    assert "Wan2.2-TI2V-5B" not in CASCADE_MODEL_TYPES
    assert "Wan2.2-T2V-A14B" in CASCADE_MODEL_TYPES
    assert "Wan2.2-I2V-A14B" in CASCADE_MODEL_TYPES


# ---------------------------------------------------------------------------
# MXFP8_QUANT_CONFIG — auto-detection contract
# ---------------------------------------------------------------------------


def test_mxfp8_quant_config_has_required_keys():
    """MXFP8_QUANT_CONFIG must carry exactly the keys that auto-detection reads:
    quant_method (selects DiffusionMXFP8Config) and is_checkpoint_mxfp8_serialized
    (selects NPUMxfp8LinearMethod over the online path)."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import MXFP8_QUANT_CONFIG

    assert MXFP8_QUANT_CONFIG["quant_method"] == "mxfp8"
    assert MXFP8_QUANT_CONFIG["is_checkpoint_mxfp8_serialized"] is True


# ---------------------------------------------------------------------------
# _get_transformer_dirs
# ---------------------------------------------------------------------------


def test_get_transformer_dirs_cascade():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_transformer_dirs

    assert _get_transformer_dirs("Wan2.2-T2V-A14B") == ["transformer", "transformer_2"]
    assert _get_transformer_dirs("Wan2.2-I2V-A14B") == ["transformer", "transformer_2"]


def test_get_transformer_dirs_single():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_transformer_dirs

    assert _get_transformer_dirs("Wan2.2-TI2V-5B") == ["transformer"]


# ---------------------------------------------------------------------------
# _get_quant_subdir
# ---------------------------------------------------------------------------


def test_get_quant_subdir_cascade_high_noise():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_quant_subdir

    base = pathlib.Path("/quant")
    result = _get_quant_subdir("Wan2.2-T2V-A14B", base, "transformer")
    assert result == base / "high_noise_model"


def test_get_quant_subdir_cascade_low_noise():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_quant_subdir

    base = pathlib.Path("/quant")
    result = _get_quant_subdir("Wan2.2-T2V-A14B", base, "transformer_2")
    assert result == base / "low_noise_model"


def test_get_quant_subdir_non_cascade():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_quant_subdir

    base = pathlib.Path("/quant")
    result = _get_quant_subdir("Wan2.2-TI2V-5B", base, "transformer")
    assert result == base


# ---------------------------------------------------------------------------
# _remap_keys — msModelSlim naming → Diffusers naming
# ---------------------------------------------------------------------------


def test_remap_keys_self_attn_q():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.self_attn.q.weight": torch.zeros(1)}
    meta = {"blocks.0.self_attn.q.weight": "W8A8_MXFP8"}
    new_state, new_meta = _remap_keys(state, meta)
    assert "blocks.0.attn1.to_q.weight" in new_state
    assert new_meta.get("blocks.0.attn1.to_q.weight") == "W8A8_MXFP8"


def test_remap_keys_self_attn_all_heads():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    pairs = {
        "self_attn.q": "attn1.to_q",
        "self_attn.k": "attn1.to_k",
        "self_attn.v": "attn1.to_v",
        "self_attn.o": "attn1.to_out.0",
    }
    for src_part, dst_part in pairs.items():
        src_key = f"blocks.0.{src_part}.weight"
        state = {src_key: torch.zeros(1)}
        new_state, _ = _remap_keys(state, {})
        expected = f"blocks.0.{dst_part}.weight"
        assert expected in new_state, f"{src_key} → expected {expected}, got {list(new_state)}"


def test_remap_keys_ffn():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {
        "blocks.1.ffn.0.weight": torch.zeros(1),
        "blocks.1.ffn.2.weight": torch.zeros(1),
    }
    new_state, _ = _remap_keys(state, {})
    assert "blocks.1.ffn.net.0.proj.weight" in new_state
    assert "blocks.1.ffn.net.2.weight" in new_state


def test_remap_keys_norm_order_swap():
    """norm2↔norm3 swap: msModelSlim uses norm1/norm3/norm2 order,
    Diffusers uses norm1/norm2/norm3."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {
        "blocks.0.norm2.weight": torch.zeros(1),
        "blocks.0.norm3.weight": torch.zeros(1),
    }
    new_state, _ = _remap_keys(state, {})
    # norm2 → norm3 and norm3 → norm2
    assert "blocks.0.norm3.weight" in new_state
    assert "blocks.0.norm2.weight" in new_state
    # Both must be present (swap, not collapse)
    assert len([k for k in new_state if "norm" in k and "norm_q" not in k and "norm_k" not in k]) == 2


def test_remap_keys_head():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"head.head.weight": torch.zeros(1)}
    new_state, _ = _remap_keys(state, {})
    assert "proj_out.weight" in new_state


def test_remap_keys_cross_attn():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.cross_attn.q.weight": torch.zeros(1)}
    new_state, _ = _remap_keys(state, {})
    assert "blocks.0.attn2.to_q.weight" in new_state


def test_remap_keys_meta_only_mapped_for_existing_state_keys():
    """quant_meta entries are only emitted for keys present in state_dict."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.self_attn.q.weight": torch.zeros(1)}
    # meta has an extra key not in state_dict
    meta = {
        "blocks.0.self_attn.q.weight": "W8A8_MXFP8",
        "blocks.0.self_attn.q.weight_scale": "W8A8_MXFP8",
    }
    _, new_meta = _remap_keys(state, meta)
    assert "blocks.0.attn1.to_q.weight" in new_meta
    # weight_scale was in meta but not state_dict → must be absent
    assert "blocks.0.attn1.to_q.weight_scale" not in new_meta


def test_remap_keys_preserves_tensors():
    """Tensor values must survive the key rename unchanged."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    t = torch.randn(4, 8)
    state = {"blocks.0.self_attn.q.weight": t}
    new_state, _ = _remap_keys(state, {})
    assert torch.equal(new_state["blocks.0.attn1.to_q.weight"], t)
