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

    def test_no_runner_falls_back_to_eager(self):
        """When no runner is registered, must use eager path."""
        model, heads, embs, text_head, backbone, n_vq = self._setup()
        assert model not in adapter._depth_graph_runners
        cls = type(model)
        adapter._original_generate_frame = cls.generate_frame
        with torch.inference_mode():
            out = adapter._patched_generate_frame(model, backbone, heads, embs, text_head, n_vq=n_vq, do_sample=False)
        assert out[1].shape == (1, n_vq)
