# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.attention.backends.trtllm_calibration import (
    is_ignored,
    layer_match_names,
    parse_sparse_attention_config,
    resolve_layer_calibration,
    select_expert,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_layer_match_names_strips_expert_prefix():
    names = layer_match_names("transformer_2.blocks.5.attn1")
    assert "transformer_2.blocks.5.attn1" in names
    assert "blocks.5.attn1" in names


def test_layer_match_names_strips_orig_mod():
    names = layer_match_names("transformer.blocks.5._orig_mod.attn1")
    assert "transformer.blocks.5.attn1" in names or "blocks.5.attn1" in names


def test_is_ignored_matches_relative_pattern():
    assert is_ignored("transformer.blocks.5.attn2", ["blocks.*.attn2"])
    assert not is_ignored("transformer.blocks.5.attn1", ["blocks.*.attn2"])


def test_is_ignored_empty_patterns():
    assert not is_ignored("transformer.blocks.5.attn1", [])
    assert not is_ignored("transformer.blocks.5.attn1", None)


def test_is_ignored_ancestor_match_with_attn_suffix():
    assert is_ignored("transformer.blocks.0.attn2.attn", ["blocks.0.attn2"])

    assert not is_ignored("transformer.blocks.11.attn2.attn", ["blocks.1.attn2"])

    assert not is_ignored("transformer.blocks.5.attn1.attn", ["blocks.0.attn2", "blocks.1.attn2"])


def test_select_expert_longest_prefix_wins():
    keys = ("transformer", "transformer_2")
    assert select_expert("transformer_2.blocks.5.attn1", keys) == "transformer_2"
    assert select_expert("transformer.blocks.5.attn1", keys) == "transformer"


def test_select_expert_no_match():
    assert select_expert("blocks.5.attn1", ("transformer", "transformer_2")) is None


def _two_expert_calib():
    return {
        "by_expert": {
            "transformer": {
                "a": 2142.7,
                "b": 4.28,
                "target_sparsity": 0.45,
                "formula": "a*exp(b*target_sparsity)",
                "ignore": ["blocks.*.attn2"],
            },
            "transformer_2": {
                "a": 314.68,
                "b": 6.17,
                "target_sparsity": 0.45,
                "formula": "a*exp(b*target_sparsity)",
                "ignore": ["blocks.*.attn2"],
            },
        }
    }


def test_resolve_routes_to_correct_expert():
    calib = _two_expert_calib()
    hi = resolve_layer_calibration("transformer.blocks.5.attn1", calib)
    lo = resolve_layer_calibration("transformer_2.blocks.5.attn1", calib)
    assert hi["a"] == pytest.approx(2142.7)
    assert lo["a"] == pytest.approx(314.68)


def test_resolve_ignored_layer_is_dense():
    assert resolve_layer_calibration("transformer.blocks.5.attn2", _two_expert_calib()) is None


def test_resolve_single_transformer():
    calib = {
        "a": 104.76,
        "b": 7.81,
        "target_sparsity": 0.45,
        "formula": "a*exp(b*target_sparsity)",
        "ignore": ["refiner_blocks.*"],
    }
    assert resolve_layer_calibration("transformer_blocks.5.attn", calib)["a"] == pytest.approx(104.76)
    assert resolve_layer_calibration("refiner_blocks.0.attn", calib) is None


def test_resolve_none_calibration():
    assert resolve_layer_calibration("transformer.blocks.5.attn1", None) is None


def test_resolve_unmatched_component_is_uncalibrated():
    calib = {"by_expert": {"transformer": {"a": 1.0, "b": 2.0, "ignore": []}}}
    assert resolve_layer_calibration("transformer.blocks.5.attn1", calib)["a"] == pytest.approx(1.0)
    assert resolve_layer_calibration("transformer_2.blocks.5.attn1", calib) is None
    assert resolve_layer_calibration("blocks.5.attn1", calib) is None


def test_parse_045_config_groups_schema():
    sac = {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "ignore": ["blocks.0.attn1"],
                "target_sparsity": 0.45,
                "threshold_scale_factor": {
                    "formula": "a * exp(b * target_sparsity)",
                    "coefficients": {"a": 2142.7, "b": 4.28},
                },
            }
        }
    }
    parsed = parse_sparse_attention_config(sac)
    assert parsed["a"] == pytest.approx(2142.7)
    assert parsed["b"] == pytest.approx(4.28)
    assert parsed["ignore"] == ["blocks.0.attn1"]


def test_parse_unsupported_formula_raises():
    sac = {
        "config_groups": {
            "group_0": {
                "algorithm": "skip_softmax",
                "threshold_scale_factor": {"formula": "a * b ** target_sparsity", "coefficients": {"a": 1.0, "b": 2.0}},
            }
        }
    }
    with pytest.raises(ValueError, match="unsupported skip-softmax formula"):
        parse_sparse_attention_config(sac)


def test_parse_empty_returns_none():
    assert parse_sparse_attention_config(None) is None
    assert parse_sparse_attention_config({}) is None


class _FakeImpl:
    def __init__(self):
        self.stamped = None

    def set_layer_calibration(self, a, b):
        self.stamped = (a, b)


def _fake_pipeline():
    import torch.nn as nn

    def _attn_layer():
        layer = nn.Module()
        layer.attention = _FakeImpl()
        return layer

    def _wan_attn():
        wa = nn.Module()
        wa.attn = _attn_layer()
        return wa

    def _transformer():
        tf = nn.Module()
        blocks = nn.ModuleList()
        for _ in range(3):
            blk = nn.Module()
            blk.attn1 = _wan_attn()
            blk.attn2 = _wan_attn()
            blocks.append(blk)
        tf.blocks = blocks
        return tf

    pipe = nn.Module()
    pipe.transformer = _transformer()
    pipe.transformer_2 = _transformer()
    return pipe


def test_apply_to_pipeline_routes_and_ignores():
    from vllm_omni.diffusion.attention.backends.trtllm_calibration import apply_to_pipeline

    pipe = _fake_pipeline()

    calib = {
        "by_expert": {
            "transformer": {
                "a": 2142.7,
                "b": 4.28,
                "target_sparsity": 0.5,
                "formula": "a*exp(b*target_sparsity)",
                "ignore": ["blocks.*.attn2", "blocks.0.attn1"],
            },
            "transformer_2": {
                "a": 314.68,
                "b": 6.17,
                "target_sparsity": 0.5,
                "formula": "a*exp(b*target_sparsity)",
                "ignore": ["blocks.*.attn2", "blocks.0.attn1"],
            },
        }
    }
    stamped = apply_to_pipeline(pipe, calib)

    assert stamped == 4

    def impl(expert, blk, attn):
        return getattr(getattr(pipe, expert).blocks[blk], attn).attn.attention

    assert impl("transformer", 1, "attn1").stamped == pytest.approx((2142.7, 4.28))

    assert impl("transformer_2", 1, "attn1").stamped == pytest.approx((314.68, 6.17))

    assert impl("transformer", 1, "attn2").stamped is None

    assert impl("transformer", 0, "attn1").stamped is None
