# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the MOSS-TTS depth whole-loop NPU adapter.

Tests the tensor-only helper functions that the adapter contributes:
``_apply_topk_topp_mask``, ``_make_gumbel_noise``, ``_whole_loop_compute``,
and the dispatch gating in ``_patched_generate_frame``.

These run on CPU — no NPU hardware required.  The NPUGraph capture/replay
path itself is exercised on-device during E2E testing.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
    MossTTSLocalDepthTransformer,
)
from vllm_omni.platforms.npu.models import moss_tts_local_depth as adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def _test_config() -> GPT2Config:
    return GPT2Config(
        n_embd=32,
        n_head=4,
        n_inner=64,
        layer_norm_epsilon=1e-6,
        rope_base=10000.0,
    )


def _make_depth_model() -> MossTTSLocalDepthTransformer:
    model = MossTTSLocalDepthTransformer(_test_config()).eval()
    model.h[0].attn.prepare_rope_cache(12, torch.device("cpu"), torch.float32)
    return model


def _make_heads(n_vq: int, vocab: int, hidden: int) -> tuple[nn.ModuleList, nn.ModuleList, nn.Module]:
    audio_lm_heads = nn.ModuleList([nn.Linear(hidden, vocab, bias=False) for _ in range(n_vq)])
    audio_embeddings = nn.ModuleList([nn.Embedding(vocab, hidden) for _ in range(n_vq)])
    local_text_lm_head = nn.Linear(hidden, 2, bias=False)
    for m in [*audio_lm_heads, *audio_embeddings, local_text_lm_head]:
        nn.init.normal_(m.weight, std=0.02)
    return audio_lm_heads, audio_embeddings, local_text_lm_head


# ---------------------------------------------------------------------------
# _apply_topk_topp_mask
# ---------------------------------------------------------------------------


class TestApplyTopkToppMask:
    def test_no_masking_when_topk_zero_and_topp_one(self):
        logits = torch.randn(2, 10)
        out = adapter._apply_topk_topp_mask(logits.clone(), 0, 1.0)
        torch.testing.assert_close(out, logits)

    def test_topk_masks_below_threshold(self):
        logits = torch.tensor([[3.0, 1.0, 2.0, 0.5, 4.0]])
        out = adapter._apply_topk_topp_mask(logits, top_k=2, top_p=1.0)
        kept = torch.isfinite(out)
        assert kept.sum().item() == 2
        assert kept[0, 0].item() is True  # 3.0 is 2nd highest
        assert kept[0, 4].item() is True  # 4.0 is highest

    def test_topp_drops_cumsum_above_threshold(self):
        logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        out = adapter._apply_topk_topp_mask(logits, top_k=0, top_p=0.5)
        assert torch.isfinite(out[0, 0]).item() is True
        assert torch.isinf(out[0, 1]).item() is True
        assert torch.isinf(out[0, 2]).item() is True

    def test_topk_and_topp_combined(self):
        logits = torch.randn(3, 20)
        out = adapter._apply_topk_topp_mask(logits, top_k=5, top_p=0.9)
        per_row_finite = torch.isfinite(out).sum(dim=-1)
        assert (per_row_finite <= 5).all()
        assert (per_row_finite >= 1).all()

    def test_preserves_finite_values(self):
        logits = torch.randn(4, 50)
        out = adapter._apply_topk_topp_mask(logits, top_k=10, top_p=0.8)
        finite_mask = torch.isfinite(out)
        original_at_finite = logits[finite_mask]
        result_at_finite = out[finite_mask]
        torch.testing.assert_close(result_at_finite, original_at_finite)

    def test_topk_keeps_exact_k_candidates_on_ties(self):
        """top-k must keep exactly ``k`` candidates even when the kth value
        ties, matching ``_sample_token`` (which operates on the compact top-k
        indices). Threshold masking would retain every token tied with the
        kth value -- e.g. ``[3, 2, 2, 2]`` with ``top_k=2`` keeps four, not
        two."""
        logits = torch.tensor([[3.0, 2.0, 2.0, 2.0]])
        out = adapter._apply_topk_topp_mask(logits, top_k=2, top_p=1.0)
        kept = torch.isfinite(out)
        assert kept.sum().item() == 2
        # The strictly-largest value (3.0) is always retained.
        assert kept[0, 0].item() is True
        assert out[0, 0].item() == 3.0
        # Exactly one of the tied 2.0 values is retained (not all three).
        assert int((out[0, 1:] == 2.0).sum()) == 1

    def test_topk_tie_matches_sample_token_candidate_set(self):
        """The retained candidate set must equal what ``_sample_token`` keeps
        via ``torch.topk`` for the same tied logits."""
        torch.manual_seed(0)
        logits = torch.tensor([[5.0, 4.0, 4.0, 4.0, 1.0]])
        masked = adapter._apply_topk_topp_mask(logits.clone(), top_k=2, top_p=1.0)
        adapter_kept = set(torch.isfinite(masked[0]).nonzero(as_tuple=True)[0].tolist())

        # _sample_token's top-k branch gathers over the topk indices; the set
        # of reachable tokens is exactly those indices.
        _, top_indices = torch.topk(logits[0], 2)
        reference_kept = set(top_indices.tolist())

        assert adapter_kept == reference_kept


# ---------------------------------------------------------------------------
# _make_gumbel_noise
# ---------------------------------------------------------------------------


class TestMakeGumbelNoise:
    def test_shape_and_dtype(self):
        gc, gb = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 4, 12, 100, generator=None)
        assert gc.shape == (4, 12, 100)
        assert gb.shape == (4, 2)
        assert gc.dtype == torch.float32
        assert gb.dtype == torch.float32

    def test_reproducible_with_same_generator(self):
        g1 = torch.Generator(device="cpu").manual_seed(42)
        g2 = torch.Generator(device="cpu").manual_seed(42)
        gc1, gb1 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 2, 8, 50, g1)
        gc2, gb2 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 2, 8, 50, g2)
        torch.testing.assert_close(gc1, gc2)
        torch.testing.assert_close(gb1, gb2)

    def test_different_with_different_seed(self):
        g1 = torch.Generator(device="cpu").manual_seed(42)
        g2 = torch.Generator(device="cpu").manual_seed(99)
        gc1, _ = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 2, 8, 50, g1)
        gc2, _ = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 2, 8, 50, g2)
        assert not torch.allclose(gc1, gc2)

    def test_gumbel_values_are_reasonable(self):
        gc, _ = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 64, 12, 100, None)
        assert torch.isfinite(gc).all()
        assert gc.mean().abs() < 2.0


# ---------------------------------------------------------------------------
# _whole_loop_compute (greedy / do_sample=False path on CPU)
# ---------------------------------------------------------------------------


class TestWholeLoopCompute:
    def test_greedy_matches_eager_generate_frame(self):
        """The adapter's _whole_loop_compute (greedy) must match the shared
        model's eager generate_frame bit-for-bit when do_sample=False."""
        torch.manual_seed(42)
        model = _make_depth_model()
        n_vq, vocab, hidden = 12, 50, model.hidden_size
        audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(n_vq, vocab, hidden)

        backbone = torch.randn(1, hidden)

        with torch.inference_mode():
            eager_out = model.generate_frame(
                backbone,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                n_vq=n_vq,
                do_sample=False,
            )
            adapter_cont, adapter_codes = adapter._whole_loop_compute(
                model,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
            )

        torch.testing.assert_close(adapter_codes, eager_out[1])
        torch.testing.assert_close(adapter_cont.eq(0), eager_out[0])

    def test_greedy_batch_2_matches_eager(self):
        torch.manual_seed(7)
        model = _make_depth_model()
        n_vq, vocab, hidden = 8, 40, model.hidden_size
        audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(n_vq, vocab, hidden)

        backbone = torch.randn(2, hidden)

        with torch.inference_mode():
            eager_out = model.generate_frame(
                backbone,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                n_vq=n_vq,
                do_sample=False,
            )
            adapter_cont, adapter_codes = adapter._whole_loop_compute(
                model,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
            )

        torch.testing.assert_close(adapter_codes, eager_out[1])
        torch.testing.assert_close(adapter_cont.eq(0), eager_out[0])

    def test_sampled_output_changes_with_different_noise(self):
        torch.manual_seed(123)
        model = _make_depth_model()
        n_vq, vocab, hidden = 12, 50, model.hidden_size
        audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(n_vq, vocab, hidden)
        backbone = torch.randn(1, hidden)

        g1 = torch.Generator(device="cpu").manual_seed(1)
        g2 = torch.Generator(device="cpu").manual_seed(2)
        gc1, gb1 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g1)
        gc2, gb2 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g2)

        with torch.inference_mode():
            _, codes1 = adapter._whole_loop_compute(
                model,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
                gumb_codes=gc1,
                gumb_bin=gb1,
            )
            _, codes2 = adapter._whole_loop_compute(
                model,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
                gumb_codes=gc2,
                gumb_bin=gb2,
            )
        assert not torch.equal(codes1, codes2)

    def test_sampled_reproducible_with_same_noise(self):
        torch.manual_seed(999)
        model = _make_depth_model()
        n_vq, vocab, hidden = 12, 50, model.hidden_size
        audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(n_vq, vocab, hidden)
        backbone = torch.randn(1, hidden)

        g = torch.Generator(device="cpu").manual_seed(42)
        gc, gb = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g)

        with torch.inference_mode():
            _, codes1 = adapter._whole_loop_compute(
                model,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
                gumb_codes=gc,
                gumb_bin=gb,
            )
            _, codes2 = adapter._whole_loop_compute(
                model,
                audio_lm_heads,
                audio_embeddings,
                local_text_lm_head,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
                gumb_codes=gc,
                gumb_bin=gb,
            )
        torch.testing.assert_close(codes1, codes2)


# ---------------------------------------------------------------------------
# Per-head temperature guard (matches _sample_token)
# ---------------------------------------------------------------------------


class TestSamplingTemperatureGuard:
    """``_sample_token`` treats ``temperature<=0`` as greedy even when
    ``do_sample=True`` and clamps positive temperatures to ``>= 1e-6``. The
    adapter must mirror this *independently* for the text and audio heads;
    otherwise ``temperature=0`` divides by zero (``[1, 2]`` -> ``[inf, inf]``
    -> picks "continue" instead of the greedy "stop") and negative
    temperatures reverse the logits."""

    def _setup(self):
        torch.manual_seed(42)
        model = _make_depth_model()
        n_vq, vocab, hidden = 12, 50, model.hidden_size
        audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(n_vq, vocab, hidden)
        backbone = torch.randn(1, hidden)
        return model, audio_lm_heads, audio_embeddings, local_text_lm_head, backbone, n_vq, vocab

    def test_temperature_zero_is_greedy_under_do_sample(self):
        """``do_sample=True`` with ``temperature=0`` must behave greedily
        (matching ``do_sample=False``), not divide by zero into inf/nan."""
        model, heads, embs, text_head, backbone, n_vq, vocab = self._setup()
        g = torch.Generator(device="cpu").manual_seed(42)
        gc, gb = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g)

        with torch.inference_mode():
            greedy_cont, greedy_codes = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
            )
            zero_cont, zero_codes = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                do_sample=True,
                temperature=0.0,
                top_k=50,
                top_p=1.0,
                text_temperature=0.0,
                text_top_k=50,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
                gumb_codes=gc,
                gumb_bin=gb,
            )

        # No inf/nan leaked through a division by zero.
        assert torch.isfinite(zero_codes.float()).all()
        assert torch.isfinite(zero_cont.float()).all()
        # temperature<=0 -> greedy, identical to do_sample=False.
        torch.testing.assert_close(zero_codes, greedy_codes)
        torch.testing.assert_close(zero_cont.eq(0), greedy_cont.eq(0))

    def test_negative_temperature_is_greedy_not_reversed(self):
        """A negative temperature must be greedy (not reverse the logits)."""
        model, heads, embs, text_head, backbone, n_vq, vocab = self._setup()
        g = torch.Generator(device="cpu").manual_seed(7)
        gc, gb = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g)

        with torch.inference_mode():
            greedy_cont, greedy_codes = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
            )
            neg_cont, neg_codes = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                do_sample=True,
                temperature=-2.0,
                top_k=50,
                top_p=1.0,
                text_temperature=-2.0,
                text_top_k=50,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
                gumb_codes=gc,
                gumb_bin=gb,
            )

        torch.testing.assert_close(neg_codes, greedy_codes)
        torch.testing.assert_close(neg_cont.eq(0), greedy_cont.eq(0))

    def test_mixed_text_greedy_audio_sampled(self):
        """``do_sample=True`` with ``text_temperature<=0`` (text greedy) and
        ``temperature>0`` (audio sampled): the continue/stop output must match
        the all-greedy reference (text greedy in both) and be invariant to the
        audio Gumbel noise, while the audio codes change with the noise
        (proving audio is actually sampled, not greedy)."""
        model, heads, embs, text_head, backbone, n_vq, vocab = self._setup()

        g1 = torch.Generator(device="cpu").manual_seed(1)
        g2 = torch.Generator(device="cpu").manual_seed(2)
        gc1, gb1 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g1)
        gc2, gb2 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g2)

        common = dict(
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            text_temperature=0.0,
            text_top_k=0,
            text_top_p=1.0,
            n_vq=n_vq,
            backbone_last_hidden=backbone,
        )
        with torch.inference_mode():
            cont1, codes1 = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                gumb_codes=gc1,
                gumb_bin=gb1,
                **common,
            )
            cont2, codes2 = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                gumb_codes=gc2,
                gumb_bin=gb2,
                **common,
            )
            greedy_cont, _ = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
            )

        # Text head is greedy (text_temperature=0) -> continue is independent
        # of the audio Gumbel noise and matches the all-greedy reference.
        torch.testing.assert_close(cont1.eq(0), cont2.eq(0))
        torch.testing.assert_close(cont1.eq(0), greedy_cont.eq(0))
        # Audio head is sampled (temperature>0) -> codes differ with noise.
        assert not torch.equal(codes1, codes2)

    def test_mixed_text_sampled_audio_greedy(self):
        """The symmetric mixed mode: text sampled + audio greedy. Audio codes
        must match the all-greedy reference and be invariant to the text
        Gumbel noise, proving the audio greedy guard holds regardless of the
        text head's sampling state. (The binary text head has only two
        tokens, so its per-seed choice is not asserted against the greedy
        reference here; ``test_mixed_text_greedy_audio_sampled`` proves the
        audio-sampled direction, and this proves the mirrored independence.)"""
        model, heads, embs, text_head, backbone, n_vq, vocab = self._setup()

        g1 = torch.Generator(device="cpu").manual_seed(3)
        g2 = torch.Generator(device="cpu").manual_seed(4)
        gc1, gb1 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g1)
        gc2, gb2 = adapter._make_gumbel_noise(torch.device("cpu"), torch.float32, 1, n_vq, vocab, g2)

        common = dict(
            do_sample=True,
            temperature=0.0,
            top_k=50,
            top_p=1.0,
            text_temperature=1.0,
            text_top_k=0,
            text_top_p=1.0,
            n_vq=n_vq,
            backbone_last_hidden=backbone,
        )
        with torch.inference_mode():
            cont1, codes1 = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                gumb_codes=gc1,
                gumb_bin=gb1,
                **common,
            )
            cont2, codes2 = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                gumb_codes=gc2,
                gumb_bin=gb2,
                **common,
            )
            _, greedy_codes = adapter._whole_loop_compute(
                model,
                heads,
                embs,
                text_head,
                do_sample=False,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                text_temperature=1.0,
                text_top_k=0,
                text_top_p=1.0,
                n_vq=n_vq,
                backbone_last_hidden=backbone,
            )

        # No inf/nan leaked through the text head's sampling path.
        assert torch.isfinite(cont1.float()).all()
        assert torch.isfinite(cont2.float()).all()
        # Audio head is greedy (temperature=0) -> codes invariant to text noise
        # and match the all-greedy reference.
        torch.testing.assert_close(codes1, codes2)
        torch.testing.assert_close(codes1, greedy_codes)


# ---------------------------------------------------------------------------
# _patched_generate_frame dispatch gating
# ---------------------------------------------------------------------------


class TestPatchedGenerateFrameDispatch:
    """Verify the adapter falls back to eager when the NPU fast path
    conditions are not met (CPU device, repetition_penalty != 1, etc.)."""

    def _setup(self):
        torch.manual_seed(42)
        model = _make_depth_model()
        n_vq, vocab, hidden = 12, 50, model.hidden_size
        audio_lm_heads, audio_embeddings, local_text_lm_head = _make_heads(n_vq, vocab, hidden)
        backbone = torch.randn(1, hidden)
        return model, audio_lm_heads, audio_embeddings, local_text_lm_head, backbone, n_vq

    def test_cpu_falls_back_to_eager(self):
        """On CPU (not NPU), the adapter must delegate to the original
        generate_frame even if a runner is registered."""
        model, heads, embs, text_head, backbone, n_vq = self._setup()

        # Register a dummy runner so the first gate (runner is not None) passes,
        # but the device check (== "npu") should still force fallback.
        dummy = cast(adapter.NPUExactGraphRunner, SimpleNamespace())
        adapter._depth_graph_runners[model] = dummy

        # Store the *unbound* class method so _patched_generate_frame can
        # call it with explicit self.
        cls = type(model)
        adapter._original_setup_compile = cls.setup_compile
        adapter._original_generate_frame = cls.generate_frame
        try:
            with torch.inference_mode():
                out = adapter._patched_generate_frame(
                    model, backbone, heads, embs, text_head, n_vq=n_vq, do_sample=False
                )
            assert out[0].shape == (1,)
            assert out[1].shape == (1, n_vq)
        finally:
            del adapter._depth_graph_runners[model]

    def test_repetition_penalty_falls_back_to_eager(self):
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        dummy = cast(adapter.NPUExactGraphRunner, SimpleNamespace())
        adapter._depth_graph_runners[model] = dummy
        cls = type(model)
        adapter._original_generate_frame = cls.generate_frame
        try:
            with torch.inference_mode():
                out = adapter._patched_generate_frame(
                    model,
                    backbone,
                    heads,
                    embs,
                    text_head,
                    n_vq=n_vq,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            assert out[1].shape == (1, n_vq)
        finally:
            del adapter._depth_graph_runners[model]

    def test_history_per_codebook_falls_back_to_eager(self):
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        dummy = cast(adapter.NPUExactGraphRunner, SimpleNamespace())
        adapter._depth_graph_runners[model] = dummy
        cls = type(model)
        adapter._original_generate_frame = cls.generate_frame
        try:
            with torch.inference_mode():
                out = adapter._patched_generate_frame(
                    model,
                    backbone,
                    heads,
                    embs,
                    text_head,
                    n_vq=n_vq,
                    do_sample=False,
                    history_per_codebook=[[1, 2]],
                )
            assert out[1].shape == (1, n_vq)
        finally:
            del adapter._depth_graph_runners[model]

    def test_per_row_generators_fall_back_to_eager(self, monkeypatch):
        """Per-row generators bypass the graph and are forwarded to eager.

        The graph draws its Gumbel noise from a single generator, so it cannot
        reproduce per-request streams; dispatch must defer and pass the
        ``generators`` list through to the original generate_frame.
        """
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        monkeypatch.setattr(adapter, "_MAX_GRAPH_BATCH", 4)

        seen: list[str] = []
        dummy = cast(adapter.NPUExactGraphRunner, SimpleNamespace())

        def fake_run(name, inputs, constants, compute):
            seen.append(name)
            batch = inputs[0].shape[0]
            return torch.zeros(batch, dtype=torch.long), torch.zeros(batch, constants[0], dtype=torch.long)

        dummy.run = fake_run  # type: ignore[attr-defined]
        adapter._depth_graph_runners[model] = dummy

        forwarded: list[object] = []

        def _record_orig(_self, hidden, *_args, **kwargs):
            forwarded.append(kwargs.get("generators"))
            batch = hidden.shape[0]
            return torch.ones(batch, dtype=torch.bool), torch.zeros(batch, n_vq, dtype=torch.long)

        adapter._original_generate_frame = _record_orig
        try:

            class _FakeHidden:
                device = SimpleNamespace(type="npu")
                shape = (1, model.hidden_size)

                def to(self, dtype):
                    return self

            gen = torch.Generator()
            with torch.inference_mode(), patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
                adapter._patched_generate_frame(
                    model, _FakeHidden(), heads, embs, text_head, n_vq=n_vq, do_sample=False, generators=[gen]
                )
            assert seen == [], "graph path was taken despite per-row generators"
            assert forwarded == [[gen]]
        finally:
            del adapter._depth_graph_runners[model]

    def test_no_runner_falls_back_to_eager(self):
        """When no runner is registered, must use eager path."""
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        assert model not in adapter._depth_graph_runners
        cls = type(model)
        adapter._original_generate_frame = cls.generate_frame
        with torch.inference_mode():
            out = adapter._patched_generate_frame(model, backbone, heads, embs, text_head, n_vq=n_vq, do_sample=False)
        assert out[1].shape == (1, n_vq)

    def test_batch_over_max_falls_back_to_eager(self, monkeypatch):
        """Batches above _MAX_GRAPH_BATCH must not be captured/replayed.

        The whole-loop graph only wins at small batch, where per-step launch
        overhead dominates; above the measured crossover replay is slower than
        eager, so dispatch must defer to the original generate_frame.
        """
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        monkeypatch.setattr(adapter, "_MAX_GRAPH_BATCH", 1)
        calls: list[tuple[object, ...]] = []

        def _record_noise(*args: object, **_: object) -> None:
            calls.append(("noise", args[2]))

        monkeypatch.setattr(adapter, "_make_gumbel_noise", _record_noise)

        cls = type(model)
        adapter._original_generate_frame = cls.generate_frame
        big = torch.randn(2, model.hidden_size)
        with torch.inference_mode():
            out = adapter._patched_generate_frame(model, big, heads, embs, text_head, n_vq=n_vq, do_sample=False)
        assert out[1].shape == (2, n_vq)
        assert calls == [], "graph path was taken for a batch above _MAX_GRAPH_BATCH"

    def test_batch_at_max_still_uses_graph(self, monkeypatch):
        """A batch exactly at _MAX_GRAPH_BATCH must still reach the graph path."""
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        monkeypatch.setattr(adapter, "_MAX_GRAPH_BATCH", 2)

        seen: list[str] = []
        dummy = cast(adapter.NPUExactGraphRunner, SimpleNamespace())

        def fake_run(name, inputs, constants, compute):
            seen.append(name)
            batch = inputs[0].shape[0]
            cont = torch.zeros(batch, dtype=torch.long)
            codes = torch.zeros(batch, constants[0], dtype=torch.long)
            return cont, codes

        dummy.run = fake_run  # type: ignore[attr-defined]
        adapter._depth_graph_runners[model] = dummy
        cls = type(model)
        adapter._original_generate_frame = cls.generate_frame
        monkeypatch.setattr(adapter, "_make_gumbel_noise", lambda *a, **k: (None, None))
        monkeypatch.setattr(adapter, "_whole_loop_compute", lambda *a, **k: None)
        try:
            # Stand-in for an NPU-resident hidden state: the graph dispatch gate
            # only reads .device.type, .shape[0] and calls .to(dtype).
            class _FakeHidden:
                device = SimpleNamespace(type="npu")
                shape = (2, model.hidden_size)

                def to(self, dtype):
                    return self

            with torch.inference_mode(), patch.object(torch.npu, "is_current_stream_capturing", return_value=False):
                monkeypatch.setattr(model.h[0].attn, "prepare_rope_cache", lambda *a, **k: None)
                adapter._patched_generate_frame(model, _FakeHidden(), heads, embs, text_head, n_vq=n_vq, do_sample=True)
            assert seen == ["depth_whole_loop"]
        finally:
            del adapter._depth_graph_runners[model]

    def test_max_graph_batch_default_is_sixteen(self):
        """The default gate is B<=16: the range where the graph wins in both modes.

        Measured on Ascend 910B2C (this branch, in-process A/B, no RNG, 8 warm
        groups of median-of-30, seed fixed): greedy 1.42x/1.16x/1.11x at
        B=1/8/16 then 1.02x/1.01x/1.00x/0.97x/0.98x at B=20/24/32/48/64;
        sampling 1.65x/1.27x/1.28x at B=1/8/16 then 1.08x/0.99x/0.99x/0.91x/
        0.92x at B=20/24/32/48/64. The sampling crossover is the earlier of the
        two and is the path real requests take, so gate at the largest batch
        that still wins in both; past it the paths are at parity and the graph
        would only add fixed-shape capture and its extra buffers. Raise it with
        MOSS_TTS_LOCAL_DEPTH_GRAPH_MAX_BATCH.
        """
        assert adapter._MAX_GRAPH_BATCH == 16


# ---------------------------------------------------------------------------
# AR worker startup registration (Fix A: patch wired into the AR init path)
# ---------------------------------------------------------------------------


class TestArWorkerPatchRegistration:
    """Verify the depth patch is installed through the AR worker startup path.

    MOSS stage 0 runs on ``NPUARWorker``, whose ``init_device`` inherits
    vllm-ascend's ``_init_device`` and never reaches ``NPUOmniPlatform.set_device``.
    Registration therefore hangs off ``init_ar_worker_runtime``, invoked from
    ``NPUARWorker.init_device`` before model loading.
    """

    @pytest.fixture
    def restored_patch(self):
        """Save/restore the adapter's global patch state around each test."""
        from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
            MossTTSLocalDepthTransformer,
        )

        cls = MossTTSLocalDepthTransformer
        orig_patched = adapter._PATCHED
        orig_setup = cls.setup_compile
        orig_gen = cls.generate_frame
        try:
            yield cls
        finally:
            cls.setup_compile = orig_setup
            cls.generate_frame = orig_gen
            adapter._PATCHED = orig_patched
            adapter._original_setup_compile = None
            adapter._original_generate_frame = None

    def test_init_ar_worker_runtime_installs_depth_patch(self, restored_patch):
        """``NPUOmniPlatform.init_ar_worker_runtime`` installs the patched
        ``setup_compile`` / ``generate_frame`` on the depth transformer class."""
        pytest.importorskip("vllm_ascend")
        from vllm_omni.platforms.npu.platform import NPUOmniPlatform

        cls = restored_patch
        adapter._PATCHED = False

        NPUOmniPlatform.init_ar_worker_runtime(vllm_config=SimpleNamespace(), device=torch.device("cpu"))

        assert adapter._PATCHED is True
        assert cls.setup_compile is adapter._patched_setup_compile
        assert cls.generate_frame is adapter._patched_generate_frame

    def test_init_ar_worker_runtime_is_idempotent(self, restored_patch):
        """A second call must not re-swap (the ``_PATCHED`` guard holds), so
        the original methods captured on the first call stay consistent."""
        pytest.importorskip("vllm_ascend")
        from vllm_omni.platforms.npu.platform import NPUOmniPlatform

        cls = restored_patch
        adapter._PATCHED = False

        NPUOmniPlatform.init_ar_worker_runtime(vllm_config=SimpleNamespace(), device=torch.device("cpu"))
        first_setup = cls.setup_compile
        first_gen = cls.generate_frame
        assert adapter._original_setup_compile is not None

        NPUOmniPlatform.init_ar_worker_runtime(vllm_config=SimpleNamespace(), device=torch.device("cpu"))
        assert cls.setup_compile is first_setup
        assert cls.generate_frame is first_gen

    def test_ar_worker_init_device_invokes_ar_runtime_hook(self, monkeypatch):
        """``NPUARWorker.init_device`` must call
        ``current_omni_platform.init_ar_worker_runtime(vllm_config, device)``
        before constructing the model runner."""
        pytest.importorskip("vllm_ascend")
        from vllm_omni.platforms import current_omni_platform
        from vllm_omni.platforms.npu.worker import npu_ar_worker as mod
        from vllm_omni.platforms.npu.worker.npu_ar_worker import NPUARWorker

        # Build a worker instance without running the real __init__ (which
        # needs a fully configured vllm_config + distributed env).
        worker = NPUARWorker.__new__(NPUARWorker)
        fake_device = torch.device("cpu")
        fake_config = SimpleNamespace()
        worker._init_device = lambda: fake_device  # type: ignore[method-assign]
        worker.vllm_config = fake_config
        worker.model_runner_cls = lambda *a, **k: None  # type: ignore[method-assign]
        monkeypatch.setattr(mod, "init_workspace_manager", lambda *a, **k: None)

        calls: list[tuple[object, object]] = []

        # An instance attribute shadows the classmethod, so the call resolves
        # to spy(vllm_config, device) without an implicit cls/self binding.
        def spy(vllm_config, device):
            calls.append((vllm_config, device))

        monkeypatch.setattr(current_omni_platform, "init_ar_worker_runtime", spy)

        worker.init_device()

        assert len(calls) == 1
        assert calls[0][0] is fake_config
        assert calls[0][1] is fake_device

    def test_ar_worker_init_device_installs_patch_through_hook(self, restored_patch, monkeypatch):
        """End-to-end-on-CPU: ``NPUARWorker.init_device`` drives the real
        ``init_ar_worker_runtime`` so the depth methods are patched before the
        model runner is constructed."""
        pytest.importorskip("vllm_ascend")
        from vllm_omni.platforms import current_omni_platform
        from vllm_omni.platforms.npu.worker import npu_ar_worker as mod
        from vllm_omni.platforms.npu.worker.npu_ar_worker import NPUARWorker

        # Only run where the active platform is the NPU platform (its
        # init_ar_worker_runtime actually applies the patch); elsewhere the
        # default no-op would make this assertion meaningless.
        if not current_omni_platform.is_npu():
            pytest.skip("requires NPUOmniPlatform as the active platform")

        cls = restored_patch
        adapter._PATCHED = False

        worker = NPUARWorker.__new__(NPUARWorker)
        worker._init_device = lambda: torch.device("cpu")  # type: ignore[method-assign]
        worker.vllm_config = SimpleNamespace()
        runner_built = []
        worker.model_runner_cls = lambda *a, **k: runner_built.append(True)  # type: ignore[method-assign]
        monkeypatch.setattr(mod, "init_workspace_manager", lambda *a, **k: None)

        worker.init_device()

        # The patch was installed before the model runner was constructed.
        assert adapter._PATCHED is True
        assert cls.setup_compile is adapter._patched_setup_compile
        assert runner_built == [True]
