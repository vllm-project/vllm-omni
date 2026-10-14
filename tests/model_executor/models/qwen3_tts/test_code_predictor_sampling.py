# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Supplied-uniform samples must match the existing ATen path exactly."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


@pytest.mark.parametrize("top_k", [0, 1, 50, 2048])
@torch.inference_mode()
def test_fused_sample_kernel_preserves_signed_zero_cutoff_ties(top_k):
    from vllm_omni.model_executor.models.qwen3_tts.fused_code_predictor import _cp_sample_kernel

    vocab, hidden, groups = 2048, 64, 4
    logits = torch.zeros(2, vocab, device="cuda", dtype=torch.bfloat16)
    logits[0, 1] = -0.0
    logits[1, ::2] = -0.0
    uniforms = torch.full((2, groups - 1, vocab), 0.01, device="cuda")[:, 1, :]
    uniforms[0, 1] = uniforms[1, 2] = 0.99
    codes = torch.full((2, groups), -1, device="cuda", dtype=torch.int64)
    table = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16)
    next_input = torch.empty(2, hidden, device="cuda", dtype=table.dtype)

    def sample():
        _cp_sample_kernel[(2,)](
            logits, uniforms, uniforms.stride(0), codes, 1, groups, table, next_input, 1.0,
            V=vocab, TOPK=top_k, HID=hidden, HAS_NEXT=True, num_warps=4,
        )  # fmt: skip

    scaled = logits.float()
    if top_k > 0:
        scaled = scaled.masked_fill(scaled < scaled.topk(top_k).values[:, -1:], -torch.inf)
    expected = (scaled - torch.log(-torch.log(uniforms))).argmax(-1)
    assert expected.tolist() == [1, 2]
    sample()
    torch.testing.assert_close(codes[:, 1], expected)
    torch.testing.assert_close(next_input, table[expected])
    assert (codes[:, [0, 2, 3]] == -1).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sample()
    codes.fill_(-1)
    graph.replay()
    torch.testing.assert_close(codes[:, 1], expected)
    torch.testing.assert_close(next_input, table[expected])


@torch.inference_mode()
def test_fused_predictor_warmup_with_capacity_one():
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
        Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
    )

    cp = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_code_groups=4,
    )  # fmt: skip
    talker = Qwen3TTSTalkerConfig(hidden_size=64, num_code_groups=4)
    config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config={}),
        additional_config={"code_predictor_kv_cache": True, "code_predictor_fused": True},
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
        vllm_config=config, config=cp, talker_config=talker
    ).to(device="cuda", dtype=torch.bfloat16)
    predictor._setup_compile()
    assert predictor._fused is not None
    hidden = torch.randn(1, 1, 64, device="cuda", dtype=torch.bfloat16)
    result = predictor(
        torch.zeros(1, 1, device="cuda", dtype=torch.long), hidden, hidden,
        sample_uniforms=torch.full((1, 3, 64), 0.5, device="cuda"),
    )  # fmt: skip
    assert result.shape == (1, 4)
    assert ((result >= 0) & (result < 64)).all()


@pytest.mark.parametrize("batch", [1, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("top_k", [-1, 1, 50, 2048])
def test_topk_gumbel_exact_samples_and_graph(batch, dtype, top_k):
    from vllm_omni.model_executor.models.qwen3_tts.code_predictor_sampling import sample_code_topk_gumbel

    with torch.inference_mode():
        torch.manual_seed(42)
        logits = torch.randn(batch, 2048, device="cuda", dtype=dtype) * 5
        uniforms = torch.rand(batch, 15, 2048, device="cuda")[:, 7, :]
        uniforms.clamp_(1e-6, 1 - 1e-6)
        for case in ("random", "ties", "all_masked", "part_masked", "uniform_zero", "uniform_one", "nan"):
            if case == "ties":
                logits.zero_()
                uniforms.fill_(0.5)
            elif case == "all_masked":
                logits.fill_(-torch.inf)
            elif case == "part_masked":
                logits[:, 4:9] = 1
            elif case == "uniform_zero":
                logits.normal_()
                uniforms.zero_()
            elif case == "uniform_one":
                uniforms.fill_(1)
            elif case == "nan":
                uniforms.fill_(0.5)
                logits[:, 7] = torch.nan
            scaled = logits * (1 / 0.9)
            if top_k > 0:
                threshold = scaled.topk(top_k, dim=-1).values[:, -1:]
                scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
            expected = (scaled.float() - torch.log(-torch.log(uniforms))).argmax(-1, keepdim=True)
            actual = sample_code_topk_gumbel(logits, uniforms, top_k, 1 / 0.9)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = sample_code_topk_gumbel(logits, uniforms, top_k, 1 / 0.9)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)


@pytest.mark.parametrize("top_k", [0, 1, 50, 97])
def test_topk_gumbel_accepts_strided_supplied_uniforms(top_k):
    from vllm_omni.model_executor.models.qwen3_tts.code_predictor_sampling import sample_code_topk_gumbel

    torch.manual_seed(81)
    logits = torch.randn(3, 194, device="cuda", dtype=torch.bfloat16)[:, ::2]
    uniforms = torch.rand(3, 15, 194, device="cuda")[:, 3, ::2].clamp_(1e-6, 1 - 1e-6)
    scaled = logits * (1 / 0.9)
    if top_k > 0:
        scaled = scaled.masked_fill(scaled < scaled.topk(top_k).values[:, -1:], -torch.inf)
    expected = (scaled.float() - torch.log(-torch.log(uniforms))).argmax(-1, keepdim=True)
    actual = sample_code_topk_gumbel(logits, uniforms, top_k, 1 / 0.9)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("top_k", [1, 50])
@torch.inference_mode()
def test_fused_predictor_matches_full_reference_with_fixed_uniforms(mocker, batch, top_k):
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm import (
        Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
    )

    torch.manual_seed(42)
    cp = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_code_groups=4,
    )  # fmt: skip
    config = mocker.Mock(
        spec=VllmConfig,
        model_config=mocker.Mock(stage_connector_config={}),
        additional_config={"code_predictor_kv_cache": True, "code_predictor_fused": True},
        scheduler_config=mocker.Mock(max_num_seqs=4),
    )
    predictor = Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(
        vllm_config=config, config=cp, talker_config=Qwen3TTSTalkerConfig(hidden_size=64, num_code_groups=4)
    ).to(device="cuda", dtype=torch.bfloat16)
    predictor._setup_compile()
    assert predictor._fused is not None
    ids = torch.randint(0, 64, (batch, 1), device="cuda")
    embeds = torch.randn(batch, 1, 64, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn_like(embeds)
    uniforms = torch.rand(batch, 3, 64, device="cuda").clamp_(1e-6, 1 - 1e-6)
    actual = predictor(ids, embeds, hidden, top_k=top_k, sample_uniforms=uniforms).clone()
    # Use identical parameters through the full re-prefill and ATen sampling
    # path, bypassing both the fused predictor and its frame-local KV cache.
    predictor._fused_requested = False
    predictor._fused = None
    predictor._frame_cache = None
    expected = predictor(ids, embeds, hidden, top_k=top_k, sample_uniforms=uniforms)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
