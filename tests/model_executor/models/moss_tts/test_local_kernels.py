# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA numerical, state-isolation, sampling and graph tests for local kernels."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_lookup_attention_every_position(batch, dtype):
    from vllm_omni.model_executor.models.moss_tts.local_kernels import lookup_attention

    torch.manual_seed(17)
    heads, dim, depth, vocab = 32, 80, 12, 67
    hidden = heads * dim
    table = torch.randn(vocab, 3 * hidden, device="cuda", dtype=dtype)
    embedding = torch.randn(vocab, hidden, device="cuda", dtype=dtype)
    key = torch.full((batch, heads, depth, dim), float("nan"), device="cuda", dtype=dtype)
    value = torch.full_like(key, float("nan"))
    expected_k, expected_v = key.clone(), value.clone()
    for frame in range(2):
        for pos in range(depth):
            tokens = torch.randint(vocab, (batch,), device="cuda")
            qkv = table[tokens]
            q, k, v = (x.reshape(batch, heads, 1, dim) for x in qkv.split(hidden, -1))
            expected_k[:, :, pos : pos + 1] = k
            expected_v[:, :, pos : pos + 1] = v
            if pos:
                actual, residual = lookup_attention(table, key, value, pos, tokens=tokens, embedding=embedding)
                torch.testing.assert_close(residual, embedding[tokens], rtol=0, atol=0)
            else:
                actual, _ = lookup_attention(qkv, key, value, pos)
            expected = F.scaled_dot_product_attention(q, expected_k[:, :, : pos + 1], expected_v[:, :, : pos + 1])
            torch.testing.assert_close(actual, expected.reshape(batch, hidden), atol=0.02, rtol=0.02)
            torch.testing.assert_close(key[:, :, : pos + 1], expected_k[:, :, : pos + 1], atol=0, rtol=0)
            torch.testing.assert_close(value[:, :, : pos + 1], expected_v[:, :, : pos + 1], atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 8, 64])
@pytest.mark.parametrize("epilogue", ["bias", "silu", "residual"])
@torch.inference_mode()
def test_fused_linear_epilogues(batch, epilogue):
    from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_linear

    torch.manual_seed(8)
    # Non-tile-aligned shapes exercise masks, including the final K iteration.
    x = torch.randn(2 * batch, 257, device="cuda", dtype=torch.bfloat16)[::2]
    weight = (torch.randn(262, 257, device="cuda", dtype=x.dtype) * 0.1)[::2]
    bias = torch.randn(131, device="cuda", dtype=x.dtype)
    residual = torch.randn(2 * batch, 131, device="cuda", dtype=x.dtype)[::2]
    expected = F.linear(x, weight, bias)
    if epilogue == "silu":
        expected = F.silu(expected)
    if epilogue == "residual":
        expected = expected + residual
    actual = fused_linear(x, weight, bias, residual if epilogue == "residual" else None, activation=epilogue == "silu")
    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)


def _reference_sample(logits, uniforms, top_k, temperature, top_p):
    values, ids = torch.sort(logits.float(), descending=True, stable=True)
    values, ids = values[:, :top_k], ids[:, :top_k]
    probs = (values / temperature).softmax(-1)
    drop = probs.cumsum(-1) > top_p
    drop[:, 1:] = drop[:, :-1].clone()
    drop[:, 0] = False
    probs = probs.masked_fill(drop, 0)
    cdf = probs.cumsum(-1)
    indices = (cdf <= uniforms[:, None] * probs.sum(-1, keepdim=True)).sum(-1).clamp_max(top_k - 1)
    return ids.gather(1, indices[:, None]).squeeze(1)


@pytest.mark.parametrize("top_k,top_p", [(1, 0.8), (25, 0.8), (32, 1.0)])
@torch.inference_mode()
def test_fused_sampling_distribution_and_ties(top_k, top_p):
    from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_sample

    torch.manual_seed(12)
    # All-tied rows plus arbitrary logits, and deterministic quantiles spanning
    # [0,1): test inverse CDF rather than comparing different RNG algorithms.
    logits = torch.randn(512, 1024, device="cuda")
    logits[:256] = 0
    uniforms = (torch.arange(512, device="cuda", dtype=torch.float32) + 0.5) / 512
    actual = fused_sample(logits, uniforms, top_k=top_k, temperature=1.7, top_p=top_p)
    expected = _reference_sample(logits, uniforms, top_k, 1.7, top_p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_sampling_graph_rng():
    from vllm_omni.model_executor.models.moss_tts.local_kernels import fused_sample

    # Replay must consume fresh PyTorch RNG state, not a captured host seed.
    flat_logits = torch.zeros(128, 1024, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fused_sample(flat_logits, torch.rand(128, device="cuda"))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fused_sample(flat_logits, torch.rand(128, device="cuda"))
    graph.replay()
    first = result.clone()
    graph.replay()
    assert not torch.equal(first, result)
    assert ((result >= 0) & (result < 25)).all()


@pytest.mark.parametrize("all_kernels", [False, True])
@torch.inference_mode()
def test_fused_frame_compilation_sampling_and_replay(all_kernels):
    from tests.model_executor.models.moss_tts.test_local_depth_qkv_lookup import _models

    model, embeddings, heads, stop_head = _models(12, "cuda", torch.bfloat16)
    model.prepare_qkv_lookup(embeddings, 12)
    model._fused_attention = True
    model._fused_sampling = True
    model._fused_linear = all_kernels
    model.setup_compile()
    graphs = []
    for batch in (1, 4):
        hidden = torch.randn(batch, 32, device="cuda", dtype=torch.bfloat16)

        def run():
            return model.generate_frame(
                hidden, heads, embeddings, stop_head, n_vq=12, top_k=16, top_p=0.8, temperature=1.7
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        graphs.append((graph, hidden, output))
    for graph, hidden, (keep, codes) in graphs * 2:
        hidden.normal_()
        graph.replay()
        assert keep.dtype == torch.bool
        assert codes.shape == (hidden.shape[0], 12)
        assert ((codes >= 0) & (codes < 19)).all()
