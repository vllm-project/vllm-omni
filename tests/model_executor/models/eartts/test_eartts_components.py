# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimalistic CPU unit tests for the EarTTS model components.

The outer ``EarTTSForCausalLM`` wraps a Gemma3 backbone whose
construction requires a real :class:`VllmConfig`, so the tests here
focus on the leaf modules that operate on plain tensors and on the
top-level class helpers that can be exercised via the
``object.__new__`` shortcut.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.eartts.eartts import (
    MLP,
    EarTTSForCausalLM,
    EarTTSInputEmbedding,
    GatedProjectedSumRMSNorm,
    MLPLayer,
    MaskGITSampler,
    MoGHead,
    PrecomputedSubwordEmbedding,
    RMSNorm,
    batch_matmul,
    gumbel_like,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Tiny config builder shared by EarTTSInputEmbedding / MaskGITSampler
# ---------------------------------------------------------------------------


def _tiny_config(
    *,
    hidden_size: int = 16,
    intermediate_size: int = 32,
    latent_size: int = 8,
    num_quantizers: int = 3,
    codebook_size: int = 4,
    emb_vocab_size: int = 20,
    use_gated_fusion_for_text_audio: bool = True,
    use_audio_prompt_frozen_projection: bool = False,
    num_iter: int = 2,
    exponent: float = 3.0,
    noise_scale: float = 0.0,
    top_p_or_k: float = 1.0,
    mog_low_rank: int | None = 4,
    mog_num_layers: int = 1,
    mog_num_predictions: int = 2,
    mog_min_log_std: float = -4.0,
    mog_eps: float = 1e-6,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        latent_size=latent_size,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        emb_vocab_size=emb_vocab_size,
        use_gated_fusion_for_text_audio=use_gated_fusion_for_text_audio,
        use_audio_prompt_frozen_projection=use_audio_prompt_frozen_projection,
        num_iter=num_iter,
        exponent=exponent,
        noise_scale=noise_scale,
        top_p_or_k=top_p_or_k,
        mog_low_rank=mog_low_rank,
        mog_num_layers=mog_num_layers,
        mog_num_predictions=mog_num_predictions,
        mog_min_log_std=mog_min_log_std,
        mog_eps=mog_eps,
    )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_forward_preserves_shape_and_dtype(self) -> None:
        layer = RMSNorm(dim=8)
        x = torch.randn(2, 4, 8)
        out = layer(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_zero_weight_gives_unit_norm(self) -> None:
        """With the default zero ``weight`` (init), each row has unit RMS norm."""
        layer = RMSNorm(dim=4, eps=1e-12)
        x = torch.randn(3, 4) * 10.0  # large to make eps negligible
        out = layer(x)
        # RMS of each row should be ~1 because weight multiplier is (1 + 0) = 1.
        rms = out.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# ---------------------------------------------------------------------------
# MLP / MLPLayer
# ---------------------------------------------------------------------------


class TestMLP:
    def test_forward_shape(self) -> None:
        mlp = MLP(hidden_size=8, intermediate_size=16)
        x = torch.randn(5, 8)
        assert mlp(x).shape == x.shape


class TestMLPLayer:
    def test_forward_shape(self) -> None:
        layer = MLPLayer(hidden_size=8, intermediate_size=16)
        x = torch.randn(5, 8)
        assert layer(x).shape == x.shape

    def test_residual_is_applied(self) -> None:
        """Zero out the MLP's ``down_proj`` so MLPLayer is exactly identity."""
        layer = MLPLayer(hidden_size=8, intermediate_size=16)
        with torch.no_grad():
            layer.mlp.down_proj.weight.zero_()
        x = torch.randn(3, 8)
        out = layer(x)
        assert torch.allclose(out, x, atol=1e-6)


# ---------------------------------------------------------------------------
# PrecomputedSubwordEmbedding
# ---------------------------------------------------------------------------


class TestPrecomputedSubwordEmbedding:
    def test_forward_lookup(self) -> None:
        emb = PrecomputedSubwordEmbedding(vocab_size=10, hidden_size=4)
        ids = torch.tensor([0, 3, 9])
        out = emb(ids)
        assert out.shape == (3, 4)
        # The lookup is exactly ``embed_subwords``'s output.
        assert torch.equal(out, emb.embed_subwords(ids))


# ---------------------------------------------------------------------------
# GatedProjectedSumRMSNorm
# ---------------------------------------------------------------------------


class TestGatedProjectedSumRMSNorm:
    def test_forward_shape(self) -> None:
        fusion = GatedProjectedSumRMSNorm(
            audio_dim=8, text_dim=8, hidden_dim=8, num_codebooks=4
        )
        bt = 5
        audio = torch.randn(bt, 8)
        text = torch.randn(bt, 8)
        out = fusion(audio, text)
        assert out.shape == (bt, 8)
        assert torch.isfinite(out).all()

    def test_final_norm_disabled_is_identity_on_norm(self) -> None:
        """``final_norm=False`` swaps the final RMSNorm for an Identity."""
        fusion = GatedProjectedSumRMSNorm(
            audio_dim=4, text_dim=4, hidden_dim=4, final_norm=False
        )
        # Identity replaces RMSNorm.
        assert isinstance(fusion.final_norm, torch.nn.Identity)


# ---------------------------------------------------------------------------
# EarTTSInputEmbedding
# ---------------------------------------------------------------------------


class TestEarTTSInputEmbedding:
    def test_forward_shape_without_speaker_latent(self) -> None:
        config = _tiny_config(use_gated_fusion_for_text_audio=False)
        emb = EarTTSInputEmbedding(config)

        bt = 4
        acoustic = torch.zeros(bt, config.num_quantizers, dtype=torch.long)
        text_tokens = torch.zeros(bt, dtype=torch.long)
        text_mask = torch.ones(bt, dtype=torch.float32)
        bos_mask = torch.zeros(bt, dtype=torch.float32)

        out = emb(acoustic, text_tokens, text_mask, bos_mask, speaker_latent=None)
        assert out.shape == (bt, config.hidden_size)
        assert torch.isfinite(out).all()

    def test_forward_shape_with_gated_fusion(self) -> None:
        config = _tiny_config(use_gated_fusion_for_text_audio=True)
        emb = EarTTSInputEmbedding(config)
        assert isinstance(emb.gated_fusion_audio_text, GatedProjectedSumRMSNorm)

        bt = 3
        acoustic = torch.zeros(bt, config.num_quantizers, dtype=torch.long)
        text_tokens = torch.tensor([1, 2, 3], dtype=torch.long)
        text_mask = torch.ones(bt, dtype=torch.float32)
        bos_mask = torch.zeros(bt, dtype=torch.float32)
        speaker_latent = torch.zeros(bt, config.hidden_size)

        out = emb(acoustic, text_tokens, text_mask, bos_mask, speaker_latent)
        assert out.shape == (bt, config.hidden_size)

    def test_text_mask_zeros_out_text_branch(self) -> None:
        """When ``use_gated_fusion`` is off, the combined embedding is
        ``audio_emb + text_emb * text_mask``; zeroing ``text_mask``
        and ``bos_mask`` should produce purely audio-driven output.
        """
        config = _tiny_config(use_gated_fusion_for_text_audio=False)
        emb = EarTTSInputEmbedding(config)

        bt = 2
        acoustic = torch.zeros(bt, config.num_quantizers, dtype=torch.long)
        text_tokens_a = torch.tensor([1, 2], dtype=torch.long)
        text_tokens_b = torch.tensor([5, 7], dtype=torch.long)
        text_mask = torch.zeros(bt, dtype=torch.float32)
        bos_mask = torch.zeros(bt, dtype=torch.float32)

        with torch.no_grad():
            out_a = emb(acoustic, text_tokens_a, text_mask, bos_mask, None)
            out_b = emb(acoustic, text_tokens_b, text_mask, bos_mask, None)

        # Different text tokens but text_mask=0 -> outputs identical.
        assert torch.allclose(out_a, out_b)

    def test_bos_mask_injects_bos_embedding(self) -> None:
        """``bos_mask`` rows add ``self.bos_emb`` once to the audio branch."""
        config = _tiny_config(use_gated_fusion_for_text_audio=False)
        emb = EarTTSInputEmbedding(config)
        with torch.no_grad():
            # Make audio branch zero by zeroing ``embed_code``.
            emb.embed_code.weight.zero_()
            # Make text branch zero by zeroing the table.
            emb.embed_subword.embed_subwords.weight.zero_()
            # Make ``bos_emb`` a known constant.
            emb.bos_emb.fill_(0.5)

        bt = 3
        acoustic = torch.zeros(bt, config.num_quantizers, dtype=torch.long)
        text_tokens = torch.zeros(bt, dtype=torch.long)
        text_mask = torch.zeros(bt, dtype=torch.float32)
        bos_mask = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        out = emb(acoustic, text_tokens, text_mask, bos_mask, None)
        assert torch.allclose(out[0], torch.zeros(config.hidden_size))
        assert torch.allclose(out[1], torch.full((config.hidden_size,), 0.5))
        assert torch.allclose(out[2], torch.zeros(config.hidden_size))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_gumbel_like_shape_and_dtype(self) -> None:
        x = torch.randn(2, 3, 4)
        g = gumbel_like(x)
        assert g.shape == x.shape
        assert g.dtype == x.dtype

    def test_batch_matmul_shape_and_selection(self) -> None:
        bsz, d_in, d_out, num_weights = 3, 4, 5, 2
        x = torch.randn(bsz, d_in)
        w = torch.randn(num_weights, d_out, d_in)
        y = torch.tensor([0, 1, 0])
        out = batch_matmul(x, w, y)
        assert out.shape == (bsz, d_out)

        expected = torch.stack(
            [w[int(y[i])] @ x[i] for i in range(bsz)], dim=0
        )
        assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# MoGHead
# ---------------------------------------------------------------------------


class TestMoGHead:
    def test_forward_shape_with_low_rank(self) -> None:
        head = MoGHead(
            hidden_size=8,
            intermediate_size=16,
            out_size=8,
            num_layers=1,
            num_predictions=2,
            low_rank=4,
            top_p_or_k=1.0,
        )
        bt = 5
        x = torch.randn(bt, 8)
        mu, logs = head(x)
        assert mu.shape == (bt, 8)
        assert logs.shape == (bt, 1)
        assert torch.isfinite(mu).all()
        assert (logs >= head.min_log_std - 1e-6).all()

    def test_forward_shape_without_low_rank(self) -> None:
        head = MoGHead(
            hidden_size=4,
            intermediate_size=8,
            out_size=4,
            num_layers=1,
            num_predictions=2,
            low_rank=None,
            top_p_or_k=1,
        )
        x = torch.randn(2, 4)
        mu, logs = head(x)
        assert mu.shape == (2, 4)
        assert logs.shape == (2, 1)


# ---------------------------------------------------------------------------
# MaskGITSampler
# ---------------------------------------------------------------------------


class TestMaskGITSampler:
    def test_init_precomputes_num_to_sample(self) -> None:
        config = _tiny_config(num_iter=4, exponent=3.0, num_quantizers=3)
        sampler = MaskGITSampler(config)
        # Each iteration unmasks some tokens; the total over all iters
        # must equal ``num_quantizers``.
        assert sum(sampler.num_to_sample) == config.num_quantizers
        assert all(x > 0 for x in sampler.num_to_sample)

    def test_depthsum_embedding_shape(self) -> None:
        config = _tiny_config()
        sampler = MaskGITSampler(config)
        bt = 4
        # ``codebook_size`` is the trailing "pad" row, so all-pad code
        # input should yield an all-zero embedding (the pad row is
        # appended with ``F.pad`` and is zero).
        code = torch.full(
            (config.num_quantizers, bt), config.codebook_size, dtype=torch.long
        )
        out = sampler._depthsum_embedding(code)
        assert out.shape == (bt, config.latent_size)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_forward_returns_valid_codes(self) -> None:
        config = _tiny_config(noise_scale=0.0)
        sampler = MaskGITSampler(config)
        bt = 3
        hidden = torch.randn(bt, config.hidden_size)
        codes = sampler(hidden)
        assert codes.shape == (bt, config.num_quantizers)
        assert codes.dtype == torch.long
        # Sampled codes must be valid indices into the codebook
        # (the pad index ``codebook_size`` is the initial value and
        # should have been fully replaced after sampling).
        assert (codes >= 0).all()
        assert (codes < config.codebook_size).all()


# ---------------------------------------------------------------------------
# EarTTSForCausalLM helpers (use object.__new__ to avoid the heavy
# Gemma3 backbone construction in __init__)
# ---------------------------------------------------------------------------


class TestEarTTSForCausalLMHelpers:
    def test_unwrap_singleton_list(self) -> None:
        x = torch.zeros(3)
        assert EarTTSForCausalLM._unwrap_singleton([x]) is x
        assert EarTTSForCausalLM._unwrap_singleton([]) is None
        assert EarTTSForCausalLM._unwrap_singleton(x) is x

    @staticmethod
    def _make_instance(*, hidden_size: int = 4) -> EarTTSForCausalLM:
        """Build an instance without invoking ``__init__`` (the real init
        constructs a Gemma3 backbone, requiring a full ``VllmConfig``).
        """
        model = object.__new__(EarTTSForCausalLM)
        torch.nn.Module.__init__(model)
        model._hidden_size = hidden_size
        model._text_pad_id = 12
        model._eos_token_id = 2
        model._speaker_latent = torch.zeros(1, hidden_size, dtype=torch.float32)
        return model

    def test_validate_speaker_latent_accepts_correct_shape(self) -> None:
        model = self._make_instance(hidden_size=4)
        latent = torch.randn(5, 4)
        out = model._validate_speaker_latent(latent)
        assert out.shape == (5, 4)
        assert out.is_contiguous()
        assert out.dtype == model._speaker_latent.dtype

    def test_validate_speaker_latent_unwraps_list(self) -> None:
        model = self._make_instance(hidden_size=4)
        latent = torch.randn(2, 4)
        out = model._validate_speaker_latent([latent])
        assert torch.equal(out, latent.to(model._speaker_latent.dtype))

    def test_validate_speaker_latent_rejects_bad_shape(self) -> None:
        model = self._make_instance(hidden_size=4)
        with pytest.raises(AssertionError, match="speaker_latent must have shape"):
            model._validate_speaker_latent(torch.randn(3, 5))
        with pytest.raises(AssertionError, match="speaker_latent must be a torch.Tensor"):
            model._validate_speaker_latent("not a tensor")

    def test_build_prefill_tensors_layout(self) -> None:
        model = self._make_instance(hidden_size=4)
        latent = torch.randn(5, 4)
        text_tokens, text_mask, bos_mask, sl = model._build_prefill_tensors(
            latent, device=torch.device("cpu")
        )

        # ``text_tokens = [PAD]*(n-1) + [EOS]``
        assert text_tokens.tolist() == [12, 12, 12, 12, 2]
        # ``text_mask = [0]*(n-2) + [1, 1]``
        assert text_mask.tolist() == [0, 0, 0, 1, 1]
        # ``bos_mask = [0]*(n-1) + [1]``
        assert bos_mask.tolist() == [0, 0, 0, 0, 1]
        # ``speaker_latent`` is forwarded with model dtype.
        assert sl.shape == (5, 4)
        assert sl.dtype == model._speaker_latent.dtype
        assert sl.is_contiguous()

    def test_build_prefill_tensors_rejects_empty_latent(self) -> None:
        model = self._make_instance(hidden_size=4)
        latent = torch.zeros(0, 4)
        with pytest.raises(AssertionError, match="at least one frame"):
            model._build_prefill_tensors(latent, device=torch.device("cpu"))
