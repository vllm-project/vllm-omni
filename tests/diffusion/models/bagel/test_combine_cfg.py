# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Bagel._combine_cfg logic."""

import pytest
import torch

from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestCombineCfg:
    """Tests for the _combine_cfg static method."""

    def _make_tensors(self, shape=(10, 64), seed=42):
        """Create deterministic test tensors."""
        gen = torch.Generator().manual_seed(seed)
        v_t = torch.randn(shape, generator=gen)
        cfg_text_v_t = torch.randn(shape, generator=gen)
        cfg_img_v_t = torch.randn(shape, generator=gen)
        return v_t, cfg_text_v_t, cfg_img_v_t

    def test_text_channel_renorm_preserves_direction(self):
        """text_channel renorm should change direction but constrain magnitude."""
        v_t, cfg_text_v_t, _ = self._make_tensors()

        result = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            None,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        # Result norm per token should be <= original v_t norm (clamp max=1.0)
        result_norm = torch.norm(result, dim=-1)
        v_t_norm = torch.norm(v_t, dim=-1)
        assert torch.all(result_norm <= v_t_norm + 1e-6), "text_channel renorm should not increase per-token norm"

    def test_scale_1_returns_v_t(self):
        """cfg_text_scale=1.0 means no CFG: result should equal v_t."""
        v_t, cfg_text_v_t, _ = self._make_tensors()

        result = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            None,
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        # scale=1 → v_t_text_ = cfg_text + 1*(v_t - cfg_text) = v_t
        # renorm scale = norm(v_t)/norm(v_t) = 1.0, so result = v_t
        assert torch.allclose(result, v_t, atol=1e-6)

    def test_img_cfg_applied_when_scale_gt_1(self):
        """When cfg_img_scale > 1.0, result should differ from text-only CFG."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        text_only = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            None,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        with_img = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            cfg_img_v_t,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        assert not torch.allclose(text_only, with_img, atol=1e-6), (
            "Image CFG should produce different result from text-only CFG"
        )

    def test_img_cfg_none_ignored(self):
        """cfg_img_v_t=None should be equivalent to cfg_img_scale <= 1.0."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        result_none = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            None,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        result_low_scale = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            cfg_img_v_t,
            cfg_text_scale=4.0,
            cfg_img_scale=0.5,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        assert torch.allclose(result_none, result_low_scale, atol=1e-6), (
            "cfg_img_v_t=None and cfg_img_scale<=1.0 should give same result"
        )

    def test_global_renorm(self):
        """global renorm should produce valid output without error."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        result = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            cfg_img_v_t,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        assert result.shape == v_t.shape
        assert not torch.any(torch.isnan(result))

    def test_channel_renorm(self):
        """channel renorm should produce valid output without error."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        result = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            cfg_img_v_t,
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_renorm_type="channel",
            cfg_renorm_min=0.0,
        )

        assert result.shape == v_t.shape
        assert not torch.any(torch.isnan(result))

    def test_invalid_renorm_type_raises(self):
        """Unknown renorm type should raise NotImplementedError."""
        v_t, cfg_text_v_t, _ = self._make_tensors()

        with pytest.raises(NotImplementedError):
            Bagel._combine_cfg(
                v_t,
                cfg_text_v_t,
                None,
                cfg_text_scale=4.0,
                cfg_img_scale=1.0,
                cfg_renorm_type="unknown",
                cfg_renorm_min=0.0,
            )

    def test_renorm_min_clamps_scale(self):
        """cfg_renorm_min should prevent scale from going too low."""
        v_t = torch.ones(10, 64)
        # Make cfg_text_v_t very different so CFG amplifies heavily
        cfg_text_v_t = torch.zeros(10, 64)

        result_no_min = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            None,
            cfg_text_scale=100.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        result_with_min = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            None,
            cfg_text_scale=100.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.5,
        )

        # With higher renorm_min, result magnitude should be larger
        # (scale is clamped to at least 0.5 instead of going near 0)
        norm_no_min = torch.norm(result_no_min)
        norm_with_min = torch.norm(result_with_min)
        assert norm_with_min >= norm_no_min - 1e-6, "Higher cfg_renorm_min should preserve more magnitude"

    def test_global_renorm_with_img_cfg(self):
        """global renorm + img CFG should produce valid, different output."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        text_only = Bagel._combine_cfg(
            v_t.clone(),
            cfg_text_v_t.clone(),
            None,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        with_img = Bagel._combine_cfg(
            v_t.clone(),
            cfg_text_v_t.clone(),
            cfg_img_v_t.clone(),
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_renorm_type="global",
            cfg_renorm_min=0.0,
        )

        assert not torch.allclose(text_only, with_img, atol=1e-6), (
            "global renorm + img CFG should differ from text-only"
        )
        assert not torch.any(torch.isnan(with_img))

    def test_channel_renorm_with_img_cfg(self):
        """channel renorm + img CFG should produce valid, different output."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        text_only = Bagel._combine_cfg(
            v_t.clone(),
            cfg_text_v_t.clone(),
            None,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_renorm_type="channel",
            cfg_renorm_min=0.0,
        )

        with_img = Bagel._combine_cfg(
            v_t.clone(),
            cfg_text_v_t.clone(),
            cfg_img_v_t.clone(),
            cfg_text_scale=4.0,
            cfg_img_scale=1.5,
            cfg_renorm_type="channel",
            cfg_renorm_min=0.0,
        )

        assert not torch.allclose(text_only, with_img, atol=1e-6), (
            "channel renorm + img CFG should differ from text-only"
        )
        assert not torch.any(torch.isnan(with_img))

    def test_global_channel_renorm_constrains_norm(self):
        """global and channel renorm should not increase overall norm."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        for renorm_type in ["global", "channel"]:
            result = Bagel._combine_cfg(
                v_t.clone(),
                cfg_text_v_t.clone(),
                cfg_img_v_t.clone(),
                cfg_text_scale=4.0,
                cfg_img_scale=1.5,
                cfg_renorm_type=renorm_type,
                cfg_renorm_min=0.0,
            )
            # Global norm of result should be <= global norm of v_t (clamp max=1.0)
            assert torch.norm(result) <= torch.norm(v_t) + 1e-5, f"{renorm_type} renorm should not increase global norm"

    def test_text_channel_img_cfg_no_second_renorm(self):
        """text_channel mode: img CFG is applied AFTER renorm, without a second renorm.
        So the result norm can exceed v_t norm when img_scale > 1."""
        v_t, cfg_text_v_t, cfg_img_v_t = self._make_tensors()

        result = Bagel._combine_cfg(
            v_t,
            cfg_text_v_t,
            cfg_img_v_t,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_renorm_type="text_channel",
            cfg_renorm_min=0.0,
        )

        # text_channel renorms after text CFG, then applies img CFG without renorm
        # So result norm CAN exceed v_t norm — this is expected behavior
        assert result.shape == v_t.shape
        assert not torch.any(torch.isnan(result))

    def test_all_renorm_types_consistent_direction(self):
        """All renorm types should guide in the same general direction."""
        v_t, cfg_text_v_t, _ = self._make_tensors()

        results = {}
        for renorm_type in ["text_channel", "global", "channel"]:
            results[renorm_type] = Bagel._combine_cfg(
                v_t.clone(),
                cfg_text_v_t.clone(),
                None,
                cfg_text_scale=4.0,
                cfg_img_scale=1.0,
                cfg_renorm_type=renorm_type,
                cfg_renorm_min=0.0,
            )

        # All results should have positive cosine similarity with each other
        for a_name, a in results.items():
            for b_name, b in results.items():
                cos_sim = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0))
                assert cos_sim > 0.5, (
                    f"{a_name} and {b_name} should point in similar direction, "
                    f"but cosine similarity = {cos_sim.item():.4f}"
                )
