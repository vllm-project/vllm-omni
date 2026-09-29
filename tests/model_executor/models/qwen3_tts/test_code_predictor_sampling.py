# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Supplied-uniform samples must match the existing ATen path exactly."""

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


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
