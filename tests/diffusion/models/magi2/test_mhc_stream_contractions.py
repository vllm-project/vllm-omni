# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MAGI-2 mHC stream contractions: unchanged off MUSA, FP32 forms on MUSA."""

import pytest
import torch

from vllm_omni.diffusion.layers.mhc import MHCMix
from vllm_omni.diffusion.models.magi2 import layers

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

STREAMS, HIDDEN = 4, 64
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
U32 = 2.0**-24


def _handler(monkeypatch, musa):
    monkeypatch.setattr(layers.current_omni_platform, "is_musa", lambda: musa)
    return layers.MHCHandler(STREAMS, HIDDEN)


def _pre_inputs(dtype, tokens=37, seed=5):
    generator = torch.Generator().manual_seed(seed)
    streams = torch.randn(tokens, STREAMS, HIDDEN, generator=generator).to(dtype)
    bias = torch.randn(STREAMS, generator=generator)
    # A view with the row stride of the fused pre/post/residual projection.
    logits = torch.randn(tokens, STREAMS * (STREAMS + 2), generator=generator)[:, :STREAMS]
    return streams, (torch.tensor(0.7), bias, logits)


def _mix_inputs(dtype, tokens=37, seed=6):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(tokens, STREAMS, HIDDEN, generator=generator).to(dtype),
        torch.randn(tokens, HIDDEN, generator=generator).to(dtype),
        torch.rand(tokens, STREAMS, generator=generator).to(dtype),
        torch.randn(tokens, STREAMS, STREAMS, generator=generator).to(dtype),
    )


def _projection_inputs(tokens=37, seed=7):
    generator = torch.Generator().manual_seed(seed)
    flattened = torch.randn(tokens, STREAMS * HIDDEN, generator=generator)
    phi = torch.randn(STREAMS * HIDDEN, 2 * STREAMS + STREAMS**2, generator=generator) * 0.05
    return flattened, phi


def _coefficients(handler, alpha_bias_logits, dtype):
    alpha, bias, logits = alpha_bias_logits
    return torch.sigmoid(alpha * handler.matmul_scale * logits + bias.unsqueeze(0)).to(dtype)


def _assert_within(actual, reference, magnitude, terms, dtype):
    # One output rounding plus an FP32 sum of ``terms`` products.
    bound = torch.finfo(dtype).eps * reference.abs() + 2 * terms * U32 * magnitude + torch.finfo(dtype).tiny
    error = (actual.double() - reference).abs()
    assert bool((error <= bound).all()), f"max excess {(error - bound).max().item():.3e}"


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_contractions_are_unchanged_off_musa(monkeypatch, dtype):
    handler = _handler(monkeypatch, musa=False)
    assert not handler.fp32_stream_contractions

    streams, alpha_bias_logits = _pre_inputs(dtype)
    expected = torch.einsum("tn,tnc->tc", _coefficients(handler, alpha_bias_logits, dtype), streams)
    torch.testing.assert_close(handler.apply_pre(streams, alpha_bias_logits), expected, rtol=0, atol=0)

    flattened, phi = _projection_inputs()
    pre, post, residual = handler.compute_logits(flattened, lambda tensor: tensor, phi)
    expected_pre, expected_post, expected_residual = torch.split(flattened @ phi, (4, 4, 16), dim=-1)
    torch.testing.assert_close(pre, expected_pre, rtol=0, atol=0)
    torch.testing.assert_close(post, expected_post, rtol=0, atol=0)
    torch.testing.assert_close(residual, expected_residual.view(-1, 4, 4), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("musa", [False, True])
def test_compiled_mix_selection(monkeypatch, musa):
    handler = _handler(monkeypatch, musa=musa)
    calls = []

    def record(name, function):
        def wrapped(*args):
            calls.append(name)
            return function(*args)

        return wrapped

    monkeypatch.setattr(layers, "_mhc_mix_fp32", record("fp32", layers._mhc_mix_fp32))
    monkeypatch.setattr(layers._mhc_mix, "forward_native", record("native", MHCMix.forward_native))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    streams, branch_output, post, residual = _mix_inputs(torch.float32)
    handler.hyper_connect(streams, branch_output, post, residual)
    assert calls == ["fp32" if musa else "native"]


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_musa_apply_pre_is_an_fp32_contraction(monkeypatch, dtype):
    handler = _handler(monkeypatch, musa=True)
    streams, alpha_bias_logits = _pre_inputs(dtype)
    actual = handler.apply_pre(streams, alpha_bias_logits)
    assert actual.dtype == dtype

    coefficients = _coefficients(handler, alpha_bias_logits, dtype)
    products = coefficients.double().unsqueeze(-1) * streams.double()
    _assert_within(actual, products.sum(1), products.abs().sum(1), STREAMS, dtype)
    if dtype != torch.float32:
        # Products are formed in FP32, not rounded to the input dtype before the sum.
        rounded_products = (coefficients.unsqueeze(-1) * streams).sum(1)
        assert not torch.equal(actual, rounded_products)


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["float64", "mixed_dtype", "broadcast_tokens", "single_stream", "extra_rank"])
def test_musa_apply_pre_keeps_the_einsum_contract_elsewhere(monkeypatch, case):
    monkeypatch.setattr(layers.current_omni_platform, "is_musa", lambda: True)
    handler = layers.MHCHandler(1 if case == "single_stream" else STREAMS, HIDDEN)
    streams, (alpha, bias, logits) = _pre_inputs(torch.float64 if case == "float64" else torch.float32)
    if case == "single_stream":
        streams, bias, logits = streams[:, :1], bias[:1], logits[:, :1]
    elif case == "broadcast_tokens":
        logits = logits[:1]
    elif case == "extra_rank":
        logits = logits.unsqueeze(0)
    out_dtype = torch.bfloat16 if case == "mixed_dtype" else None
    coefficients = _coefficients(handler, (alpha, bias, logits), out_dtype or streams.dtype)
    if case in ("mixed_dtype", "extra_rank"):
        with pytest.raises(RuntimeError):
            torch.einsum("tn,tnc->tc", coefficients, streams)
        with pytest.raises(RuntimeError):
            handler.apply_pre(streams, (alpha, bias, logits), out_dtype=out_dtype)
    else:
        expected = torch.einsum("tn,tnc->tc", coefficients, streams)
        actual = handler.apply_pre(streams, (alpha, bias, logits), out_dtype=out_dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_musa_mix_keeps_float64(monkeypatch):
    streams, branch_output, post, residual = _mix_inputs(torch.float64)
    mixed = layers._mhc_mix_fp32(streams, branch_output, torch.zeros_like(post), residual)
    assert mixed.dtype == torch.float64
    torch.testing.assert_close(mixed, torch.einsum("tij,tjc->tic", residual, streams), rtol=1e-12, atol=1e-12)


@pytest.mark.cpu
def test_musa_compute_logits_splits_k_by_stream(monkeypatch):
    handler = _handler(monkeypatch, musa=True)
    flattened, phi = _projection_inputs()
    pre, post, residual = handler.compute_logits(flattened, lambda tensor: tensor, phi)
    assert (pre.shape, post.shape, residual.shape) == ((37, 4), (37, 4), (37, 4, 4))

    products = flattened.double().unsqueeze(-1) * phi.double().unsqueeze(0)
    reference, magnitude = products.sum(1), products.abs().sum(1)
    actual = torch.cat((pre, post, residual.flatten(1)), dim=-1)
    _assert_within(actual, reference, magnitude, HIDDEN + STREAMS, torch.float32)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_musa_mix_is_an_fp32_contraction(dtype):
    streams, branch_output, post, residual = _mix_inputs(dtype)
    # Zero post coefficients isolate the stream mix; a zero matrix isolates the branch term.
    mixed = layers._mhc_mix_fp32(streams, branch_output, torch.zeros_like(post), residual)
    assert mixed.dtype == dtype
    products = residual.double().unsqueeze(-1) * streams.double().unsqueeze(1)
    _assert_within(mixed, products.sum(2), products.abs().sum(2), STREAMS, dtype)

    branch = layers._mhc_mix_fp32(streams, branch_output, post, torch.zeros_like(residual))
    torch.testing.assert_close(branch, torch.einsum("tn,tc->tnc", post, branch_output), rtol=0, atol=0)


@pytest.mark.musa
@pytest.mark.parametrize("dtype", DTYPES)
def test_musa_device_contractions_are_as_accurate_as_einsum(dtype):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    handler = layers.MHCHandler(STREAMS, HIDDEN)
    assert handler.fp32_stream_contractions

    def max_error(actual, coefficients, values):
        reference = (coefficients.double().cpu().unsqueeze(-1) * values.double().cpu()).sum(1)
        return (actual.double().cpu() - reference).abs().max().item()

    streams, alpha_bias_logits = _pre_inputs(dtype, tokens=1031)
    device_streams = streams.to("musa")
    device_able = tuple(tensor.to("musa") for tensor in alpha_bias_logits)
    with torch.no_grad():
        exact = torch.sigmoid(device_able[0] * handler.matmul_scale * device_able[2] + device_able[1].unsqueeze(0))
        rounded = exact.to(dtype)
        einsum_error = max_error(torch.einsum("tn,tnc->tc", rounded, device_streams), rounded, streams)
        eager = handler.apply_pre(device_streams, device_able)
        compiled = torch.compile(handler.apply_pre, fullgraph=True)(device_streams, device_able)
    assert max_error(eager, rounded, streams) <= 2 * einsum_error
    # Inductor may keep the fused coefficients in FP32 instead of rounding them to the stream dtype.
    compiled_error = min(max_error(compiled, rounded, streams), max_error(compiled, exact, streams))
    assert compiled_error <= 2 * einsum_error

    flattened, phi = _projection_inputs(tokens=1031)
    reference = flattened.double() @ phi.double()
    with torch.no_grad():
        pre, post, residual = handler.compute_logits(flattened.to("musa"), lambda tensor: tensor, phi.to("musa"))
        single = (flattened.to("musa") @ phi.to("musa")).cpu()
    split = torch.cat((pre, post, residual.flatten(1)), dim=-1).cpu()
    assert (split.double() - reference).abs().max() <= 2 * (single.double() - reference).abs().max()

    streams, branch_output, post, residual = (tensor.to("musa") for tensor in _mix_inputs(dtype, tokens=1031))
    with torch.no_grad():
        mixed = layers._mhc_mix_fp32(streams, branch_output, torch.zeros_like(post), residual)
        einsum_mixed = torch.einsum("tij,tjc->tic", residual, streams)
    reference = (residual.double().cpu().unsqueeze(-1) * streams.double().cpu().unsqueeze(1)).sum(2)
    einsum_mix_error = (einsum_mixed.double().cpu() - reference).abs().max()
    assert (mixed.double().cpu() - reference).abs().max() <= 2 * einsum_mix_error
    torch._dynamo.reset()
