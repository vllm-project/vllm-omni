# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the shared Helios CFG-Zero* / transformer-kwargs helpers.

These tests pin the deduplication refactor that collapsed four copies of the
CFG-Zero* alpha/blend computation and the ``transformer_kwargs`` dict
(stepwise stage1/stage2, request-mode stage1/stage2) into the module-level
helpers ``cfg_zero_star_alpha`` / ``cfg_zero_star_blend`` /
``build_transformer_kwargs`` / ``stepwise_transformer_kwargs``.

The ``TestRefactorParity`` class re-implements the *old* inline code and
asserts the helpers produce byte-identical results, proving the refactor is
behaviour-preserving.  The remaining classes cover the helpers' own
contracts (shape/dtype, formula, key set, ``extra`` resolution).
"""

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import (
    build_transformer_kwargs,
    cfg_zero_star_alpha,
    cfg_zero_star_blend,
    optimized_scale,
    stepwise_transformer_kwargs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# cfg_zero_star_alpha
# ---------------------------------------------------------------------------


class TestCfgZeroStarAlpha:
    def test_shape_broadcasts_over_noise_pred(self):
        noise_pred = torch.randn(2, 3, 4, 5)
        noise_uncond = torch.randn(2, 3, 4, 5)
        alpha = cfg_zero_star_alpha(noise_pred, noise_uncond)
        # [B, 1, 1, 1] so it broadcasts over [B, C, H, W]
        assert alpha.shape == (2, 1, 1, 1)

    def test_shape_for_2d_noise_pred(self):
        noise_pred = torch.randn(4, 8)
        noise_uncond = torch.randn(4, 8)
        alpha = cfg_zero_star_alpha(noise_pred, noise_uncond)
        assert alpha.shape == (4, 1)

    def test_dtype_matches_noise_pred(self):
        for dt in (torch.float32, torch.float16, torch.bfloat16):
            noise_pred = torch.randn(2, 3, 4, dtype=dt)
            noise_uncond = torch.randn(2, 3, 4, dtype=dt)
            assert cfg_zero_star_alpha(noise_pred, noise_uncond).dtype is dt

    def test_value_matches_reference(self):
        torch.manual_seed(0)
        noise_pred = torch.randn(2, 3, 4, 5)
        noise_uncond = torch.randn(2, 3, 4, 5)
        alpha = cfg_zero_star_alpha(noise_pred, noise_uncond)
        # Reference: the exact computation the helper encapsulates.
        bs = noise_pred.shape[0]
        pos = noise_pred.view(bs, -1)
        neg = noise_uncond.view(bs, -1)
        ref = optimized_scale(pos, neg).view(bs, *([1] * (noise_pred.dim() - 1))).to(noise_pred.dtype)
        torch.testing.assert_close(alpha, ref)


# ---------------------------------------------------------------------------
# cfg_zero_star_blend
# ---------------------------------------------------------------------------


class TestCfgZeroStarBlend:
    def test_formula_matches_reference(self):
        torch.manual_seed(1)
        noise_pred = torch.randn(2, 3, 4, 5)
        noise_uncond = torch.randn(2, 3, 4, 5)
        alpha = torch.rand(2, 1, 1, 1)
        scale = 5.0
        out = cfg_zero_star_blend(noise_pred, noise_uncond, alpha, scale)
        ref = noise_uncond * alpha + scale * (noise_pred - noise_uncond * alpha)
        torch.testing.assert_close(out, ref)

    def test_alpha_one_reduces_to_standard_cfg(self):
        # alpha == 1 -> blend == uncond + scale * (cond - uncond)  (standard CFG)
        noise_pred = torch.randn(2, 3, 4)
        noise_uncond = torch.randn(2, 3, 4)
        scale = 7.0
        alpha = torch.ones(2, 1, 1)
        out = cfg_zero_star_blend(noise_pred, noise_uncond, alpha, scale)
        ref = noise_uncond + scale * (noise_pred - noise_uncond)
        torch.testing.assert_close(out, ref)

    def test_alpha_zero_returns_scaled_cond(self):
        # alpha == 0 -> blend == scale * cond
        noise_pred = torch.randn(2, 4)
        noise_uncond = torch.randn(2, 4)
        scale = 3.0
        alpha = torch.zeros(2, 1)
        out = cfg_zero_star_blend(noise_pred, noise_uncond, alpha, scale)
        torch.testing.assert_close(out, scale * noise_pred)


# ---------------------------------------------------------------------------
# build_transformer_kwargs
# ---------------------------------------------------------------------------


_EXPECTED_KEYS = {
    "hidden_states",
    "timestep",
    "indices_hidden_states",
    "indices_latents_history_short",
    "indices_latents_history_mid",
    "indices_latents_history_long",
    "latents_history_short",
    "latents_history_mid",
    "latents_history_long",
    "attention_kwargs",
    "return_dict",
}


class TestBuildTransformerKwargs:
    def _inputs(self, dtype=torch.float32):
        return dict(
            latents=torch.randn(2, 4, 8, 8, 8),
            timestep=torch.tensor([10, 10]),
            indices_hidden_states=torch.arange(8),
            indices_latents_history_short=torch.arange(4),
            indices_latents_history_mid=torch.arange(2),
            indices_latents_history_long=torch.arange(1),
            latents_history_short=torch.randn(2, 4, 4, 4, 4),
            latents_history_mid=torch.randn(2, 4, 2, 4, 4),
            latents_history_long=torch.randn(2, 4, 1, 4, 4),
            attention_kwargs={"foo": 1},
            dtype=dtype,
        )

    def test_returns_exactly_expected_keys(self):
        kw = build_transformer_kwargs(**self._inputs())
        assert set(kw) == _EXPECTED_KEYS

    def test_return_dict_is_false(self):
        kw = build_transformer_kwargs(**self._inputs())
        assert kw["return_dict"] is False

    def test_hidden_states_cast_to_dtype(self):
        kw = build_transformer_kwargs(**self._inputs(dtype=torch.float16))
        assert kw["hidden_states"].dtype is torch.float16

    def test_history_tensors_cast_to_dtype(self):
        kw = build_transformer_kwargs(**self._inputs(dtype=torch.float16))
        for k in ("latents_history_short", "latents_history_mid", "latents_history_long"):
            assert kw[k].dtype is torch.float16

    def test_indices_and_attention_kwargs_passed_through(self):
        inputs = self._inputs()
        kw = build_transformer_kwargs(**inputs)
        assert torch.equal(kw["indices_hidden_states"], inputs["indices_hidden_states"])
        assert kw["attention_kwargs"] is inputs["attention_kwargs"]
        assert torch.equal(kw["timestep"], inputs["timestep"])

    def test_dtype_already_matches_is_noop(self):
        inputs = self._inputs(dtype=torch.float32)
        kw = build_transformer_kwargs(**inputs)
        # latents already fp32 -> .to(fp32) returns the same tensor
        assert kw["hidden_states"] is inputs["latents"]


# ---------------------------------------------------------------------------
# stepwise_transformer_kwargs (resolves state.extra)
# ---------------------------------------------------------------------------


class TestStepwiseTransformerKwargs:
    def _extra(self, dtype=torch.float32):
        return {
            "indices_hidden_states": torch.arange(8),
            "indices_latents_history_short": torch.arange(4),
            "indices_latents_history_mid": torch.arange(2),
            "indices_latents_history_long": torch.arange(1),
            "latents_history_short": torch.randn(2, 4, 4, 4, 4),
            "latents_history_mid": torch.randn(2, 4, 2, 4, 4),
            "latents_history_long": torch.randn(2, 4, 1, 4, 4),
            "attention_kwargs": {"bar": 2},
            "dtype": dtype,
        }

    def test_resolves_extra_into_build_kwargs(self):
        extra = self._extra(dtype=torch.float16)
        latents = torch.randn(2, 4, 8, 8, 8)
        timestep = torch.tensor([10, 10])
        kw = stepwise_transformer_kwargs(extra, latents, timestep)
        # Matches build_transformer_kwargs with the same resolved values.
        ref = build_transformer_kwargs(
            latents=latents,
            timestep=timestep,
            indices_hidden_states=extra["indices_hidden_states"],
            indices_latents_history_short=extra["indices_latents_history_short"],
            indices_latents_history_mid=extra["indices_latents_history_mid"],
            indices_latents_history_long=extra["indices_latents_history_long"],
            latents_history_short=extra["latents_history_short"],
            latents_history_mid=extra["latents_history_mid"],
            latents_history_long=extra["latents_history_long"],
            attention_kwargs=extra["attention_kwargs"],
            dtype=extra["dtype"],
        )
        assert set(kw) == set(ref)
        for k in ref:
            if isinstance(ref[k], torch.Tensor):
                assert torch.equal(kw[k], ref[k]), f"mismatch for {k}"
            else:
                assert kw[k] == ref[k], f"mismatch for {k}"

    def test_dtype_taken_from_extra(self):
        extra = self._extra(dtype=torch.bfloat16)
        latents = torch.randn(1, 4, 4, 4, 4)
        kw = stepwise_transformer_kwargs(extra, latents, torch.tensor([5]))
        assert kw["hidden_states"].dtype is torch.bfloat16

    def test_missing_extra_key_raises(self):
        extra = self._extra()
        del extra["indices_hidden_states"]
        with pytest.raises(KeyError):
            stepwise_transformer_kwargs(extra, torch.randn(1, 4, 4, 4, 4), torch.tensor([5]))


# ---------------------------------------------------------------------------
# Refactor parity: helpers must reproduce the OLD inline code exactly
# ---------------------------------------------------------------------------


class TestRefactorParity:
    """The helpers must be byte-identical to the pre-refactor inline code."""

    @staticmethod
    def _old_alpha(noise_pred: torch.Tensor, noise_uncond: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Verbatim copy of the old inline alpha computation (stage1 stepwise)."""
        positive_flat = noise_pred.view(batch_size, -1)
        negative_flat = noise_uncond.view(batch_size, -1)
        alpha_cfg = optimized_scale(positive_flat, negative_flat)
        alpha_cfg = alpha_cfg.view(batch_size, *([1] * (len(noise_pred.shape) - 1))).to(noise_pred.dtype)
        return alpha_cfg

    @staticmethod
    def _old_blend(noise_pred, noise_uncond, alpha_cfg, guidance_scale):
        """Verbatim copy of the old inline blend formula."""
        return noise_uncond * alpha_cfg + guidance_scale * (noise_pred - noise_uncond * alpha_cfg)

    def test_alpha_matches_old_inline(self):
        torch.manual_seed(42)
        for shape in [(2, 3, 4, 5), (4, 8), (1, 3, 4, 5, 6)]:
            noise_pred = torch.randn(*shape)
            noise_uncond = torch.randn(*shape)
            batch_size = shape[0]
            new = cfg_zero_star_alpha(noise_pred, noise_uncond)
            old = self._old_alpha(noise_pred, noise_uncond, batch_size)
            torch.testing.assert_close(new, old, msg=f"shape {shape}")

    def test_blend_matches_old_inline(self):
        torch.manual_seed(43)
        noise_pred = torch.randn(2, 3, 4, 5)
        noise_uncond = torch.randn(2, 3, 4, 5)
        alpha = torch.rand(2, 1, 1, 1)
        scale = 5.5
        torch.testing.assert_close(
            cfg_zero_star_blend(noise_pred, noise_uncond, alpha, scale),
            self._old_blend(noise_pred, noise_uncond, alpha, scale),
        )

    def test_alpha_old_3line_split_form_matches(self):
        # The request-mode sites split the alpha into 3 lines
        # (.view then .to as a separate statement).  The helper does the
        # same in two lines; result must be identical.
        torch.manual_seed(44)
        noise_pred = torch.randn(2, 3, 4, dtype=torch.float16)
        noise_uncond = torch.randn(2, 3, 4, dtype=torch.float16)
        batch_size = 2
        positive_flat = noise_pred.view(batch_size, -1)
        negative_flat = noise_uncond.view(batch_size, -1)
        alpha_cfg = optimized_scale(positive_flat, negative_flat)
        alpha_cfg = alpha_cfg.view(batch_size, *([1] * (len(noise_pred.shape) - 1)))
        alpha_cfg = alpha_cfg.to(noise_pred.dtype)  # 3-line form
        torch.testing.assert_close(cfg_zero_star_alpha(noise_pred, noise_uncond), alpha_cfg)

    def test_stepwise_kwargs_match_old_inline_dict(self):
        """stepwise_transformer_kwargs == the old inline dict literal."""
        dtype = torch.float32
        extra = {
            "indices_hidden_states": torch.arange(8),
            "indices_latents_history_short": torch.arange(4),
            "indices_latents_history_mid": torch.arange(2),
            "indices_latents_history_long": torch.arange(1),
            "latents_history_short": torch.randn(2, 4, 4, 4, 4),
            "latents_history_mid": torch.randn(2, 4, 2, 4, 4),
            "latents_history_long": torch.randn(2, 4, 1, 4, 4),
            "attention_kwargs": {"k": "v"},
            "dtype": dtype,
        }
        latents = torch.randn(2, 4, 8, 8, 8)
        timestep = torch.tensor([10, 10])
        # The old inline dict (verbatim from pre-refactor _denoise_stage1_step).
        old = {
            "hidden_states": latents.to(extra["dtype"]),
            "timestep": timestep,
            "indices_hidden_states": extra["indices_hidden_states"],
            "indices_latents_history_short": extra["indices_latents_history_short"],
            "indices_latents_history_mid": extra["indices_latents_history_mid"],
            "indices_latents_history_long": extra["indices_latents_history_long"],
            "latents_history_short": extra["latents_history_short"].to(extra["dtype"]),
            "latents_history_mid": extra["latents_history_mid"].to(extra["dtype"]),
            "latents_history_long": extra["latents_history_long"].to(extra["dtype"]),
            "attention_kwargs": extra["attention_kwargs"],
            "return_dict": False,
        }
        new = stepwise_transformer_kwargs(extra, latents, timestep)
        assert set(new) == set(old)
        for k in old:
            if isinstance(old[k], torch.Tensor):
                assert torch.equal(new[k], old[k]), f"mismatch for {k}"
                assert new[k].dtype == old[k].dtype, f"dtype mismatch for {k}"
            else:
                assert new[k] == old[k], f"mismatch for {k}"
