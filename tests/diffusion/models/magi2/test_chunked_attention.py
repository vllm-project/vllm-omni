# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.magi2.attention import torch_varlen_attention_with_sink

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _inputs(dtype=torch.float32, kv_heads=2, sinks=2):
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(11, 4, 32, generator=generator).to(dtype)[..., ::2]
    k = torch.randn(13, kv_heads, 32, generator=generator).to(dtype)[..., ::2]
    v = torch.randn(13, kv_heads, 32, generator=generator).to(dtype)[..., ::2]
    sink = torch.randn(sinks, 4, generator=generator) if sinks else None
    return q, k, v, sink, [0, 4, 4, 11], [0, 6, 6, 13]


def _dense_reference(q, k, v, sink, cq, ck, softcap):
    output = torch.empty_like(q)
    for qs, qe, ks, ke in zip(cq[:-1], cq[1:], ck[:-1], ck[1:]):
        qq = q[qs:qe].float().transpose(0, 1)
        kk = k[ks:ke].float().repeat_interleave(q.shape[1] // k.shape[1], dim=1).transpose(0, 1)
        vv = v[ks:ke].float().repeat_interleave(q.shape[1] // v.shape[1], dim=1).transpose(0, 1)
        scores = (qq @ kk.transpose(-1, -2)) * q.shape[-1] ** -0.5
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        if sink is not None and sink.numel():
            scores = torch.cat((scores, sink.float().t().unsqueeze(1).expand(-1, qe - qs, -1)), dim=-1)
        probabilities = scores.softmax(-1)[..., : ke - ks]
        output[qs:qe] = (probabilities @ vv).transpose(0, 1).to(output.dtype)
    return output


def _run(q, k, v, sink, cq, ck, *, softcap=2.0, chunk=3):
    return torch_varlen_attention_with_sink(
        q,
        k,
        v,
        sink=sink,
        cu_seqlens_q=torch.tensor(cq, dtype=torch.int32),
        cu_seqlens_k=torch.tensor(ck, dtype=torch.int32),
        softcap=softcap,
        query_chunk_size=chunk,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_heads,sinks,softcap", [(4, 0, -1.0), (2, 1, -1.0), (1, 2, 2.0)])
@pytest.mark.parametrize("chunk", [1, 3, 512])
def test_dense_oracle_parity(dtype, kv_heads, sinks, softcap, chunk):
    q, k, v, sink, cq, ck = _inputs(dtype, kv_heads, sinks)
    actual = _run(q, k, v, sink, cq, ck, softcap=softcap, chunk=chunk)
    expected = _dense_reference(q, k, v, sink, cq, ck, softcap)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.cpu
@pytest.mark.parametrize("chunk", [0, -1, True, 2.5])
def test_invalid_chunk_size(chunk):
    with pytest.raises(ValueError, match="positive integer"):
        _run(*_inputs(), chunk=chunk)


@pytest.mark.cpu
@pytest.mark.parametrize("empty_query", [True, False])
@pytest.mark.parametrize("sinks", [0, 2])
def test_empty_sequences(empty_query, sinks):
    q, k, v, sink, _, _ = _inputs(sinks=sinks)
    if empty_query:
        q, cq, ck = q[:0], [0, 0], [0, k.shape[0]]
    else:
        k, v, cq, ck = k[:0], v[:0], [0, q.shape[0]], [0, 0]
    actual = _run(q, k, v, sink, cq, ck)
    torch.testing.assert_close(actual, torch.zeros_like(q), rtol=0, atol=0)


@pytest.mark.cpu
def test_scores_are_bounded_but_all_keys_are_visible():
    original = torch.einsum
    shapes = []

    def record(equation, *tensors):
        if equation == "qhd,khd->hqk":
            shapes.append((tensors[0].shape[0], tensors[1].shape[0]))
        return original(equation, *tensors)

    with patch.object(torch, "einsum", side_effect=record):
        _run(*_inputs(), chunk=3)
    assert shapes == [(3, 6), (1, 6), (0, 0), (3, 7), (3, 7), (1, 7)]


@pytest.mark.cpu
def test_default_chunk_size_applies_to_long_queries():
    q, k, v = torch.ones(1025, 1, 4), torch.ones(3, 1, 4), torch.ones(3, 1, 4)
    original = torch.einsum
    rows = []

    def record(equation, *tensors):
        if equation == "qhd,khd->hqk":
            rows.append(tensors[0].shape[0])
        return original(equation, *tensors)

    with torch.inference_mode(), patch.object(torch, "einsum", side_effect=record):
        result = torch_varlen_attention_with_sink(
            q,
            k,
            v,
            cu_seqlens_q=torch.tensor([0, 1025]),
            cu_seqlens_k=torch.tensor([0, 3]),
        )
    assert rows == [512, 512, 1]
    torch.testing.assert_close(result, q, rtol=0, atol=0)


@pytest.mark.cpu
def test_gradients_match_dense_reference():
    q, k, v, sink, cq, ck = _inputs()
    inputs = (
        q.clone().requires_grad_(),
        k.clone().requires_grad_(),
        v.clone().requires_grad_(),
        sink.clone().requires_grad_(),
    )
    expected = _dense_reference(*inputs, cq, ck, 2.0)
    reference_grads = torch.autograd.grad(expected.square().sum(), inputs)
    actual = _run(*inputs, cq, ck)
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    for grad, reference in zip(actual_grads, reference_grads):
        torch.testing.assert_close(grad, reference, rtol=2e-5, atol=2e-6)


@pytest.mark.cpu
def test_empty_query_keeps_autograd_dependencies():
    q, k, v, sink, _, _ = _inputs()
    inputs = (
        q[:0].clone().requires_grad_(),
        k.clone().requires_grad_(),
        v.clone().requires_grad_(),
        sink.clone().requires_grad_(),
    )
    cq, ck = [0, 0], [0, k.shape[0]]
    expected = _dense_reference(*inputs, cq, ck, 2.0)
    actual = _run(*inputs, cq, ck)
    assert actual.requires_grad == expected.requires_grad
    for left, right in zip(torch.autograd.grad(actual.sum(), inputs), torch.autograd.grad(expected.sum(), inputs)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.musa
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_musa_chunked_matches_unchunked(dtype):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    q, k, v, sink, cq, ck = _inputs(dtype)
    q, k, v, sink = (tensor.to("musa") for tensor in (q, k, v, sink))
    for cap in (-1.0, 2.0):
        expected = _run(q, k, v, sink, cq, ck, softcap=cap, chunk=512)
        for chunk in (1, 3):
            actual = _run(q, k, v, sink, cq, ck, softcap=cap, chunk=chunk)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
