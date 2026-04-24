# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniDiffusionConfig.is_moe (fix is_moe type and threshold, 6663c0b)."""

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniDiffusionConfigIsMoE:
    """Tests for OmniDiffusionConfig.is_moe property.

    Covers commit 6663c0b: fix is_moe type and threshold
    - num_experts must be (list, tuple, int); otherwise return False.
    - Threshold: is_moe is True when num_experts > 0 (not > 1).
    """

    def test_is_moe_missing_num_experts_returns_false(self):
        """When num_experts is absent, is_moe should be False."""
        tf_config = TransformerConfig.from_dict({})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is False

    def test_is_moe_none_num_experts_returns_false(self):
        """When num_experts is explicitly None (e.g. in params), is_moe should be False."""
        tf_config = TransformerConfig.from_dict({"num_experts": None})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is False

    def test_is_moe_non_allowed_type_returns_false(self):
        """When num_experts is not int/list/tuple (e.g. str), is_moe should be False."""
        tf_config = TransformerConfig.from_dict({"num_experts": "2"})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is False

    def test_is_moe_int_zero_returns_false(self):
        """num_experts int 0 should yield is_moe False (threshold > 0)."""
        tf_config = TransformerConfig.from_dict({"num_experts": 0})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is False

    def test_is_moe_int_one_returns_true(self):
        """num_experts int 1 should yield is_moe True (threshold > 0, not > 1)."""
        tf_config = TransformerConfig.from_dict({"num_experts": 1})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is True

    def test_is_moe_int_gt_one_returns_true(self):
        """num_experts int > 1 should yield is_moe True."""
        tf_config = TransformerConfig.from_dict({"num_experts": 2})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is True

    def test_is_moe_list_all_zero_returns_false(self):
        """num_experts list with all <= 0 should yield is_moe False."""
        tf_config = TransformerConfig.from_dict({"num_experts": [0]})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is False

    def test_is_moe_list_has_positive_returns_true(self):
        """num_experts list with any int > 0 should yield is_moe True."""
        tf_config = TransformerConfig.from_dict({"num_experts": [0, 1]})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is True

    def test_is_moe_tuple_has_positive_returns_true(self):
        """num_experts tuple with any int > 0 should yield is_moe True."""
        tf_config = TransformerConfig.from_dict({"num_experts": (0, 2)})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is True

    def test_is_moe_list_non_int_ignored(self):
        """num_experts list with only non-int entries should yield is_moe False."""
        tf_config = TransformerConfig.from_dict({"num_experts": ["a", 0.0]})
        config = OmniDiffusionConfig(model="test", tf_model_config=tf_config)
        assert config.is_moe is False
