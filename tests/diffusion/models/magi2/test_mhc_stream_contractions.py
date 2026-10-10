# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MAGI-2 mHC stream contractions: unchanged off MUSA, FP32 forms on MUSA."""

import re
from functools import partial

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.mhc import MHCMix
from vllm_omni.diffusion.models.magi2 import layers

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def _deterministic_inductor(monkeypatch):
    # Reduction configs are otherwise benchmarked per compile, which can change the bits.
    # Dynamo resets ``deterministic`` after every traced frame; the config filter stays on.
    monkeypatch.setattr(torch._inductor.config, "deterministic", True)
    monkeypatch.setattr(torch._inductor.config.test_configs, "force_filter_reduction_configs", True)


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


@hardware_test(res={"musa": "S5000"}, num_cards=1)
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


GROUP_SIZES = [(37,), (37, 0, 0), (0, 37, 0), (20, 0, 17), (25, 9, 3)]


def _norm(group_sizes, hidden=HIDDEN, seed=8, device="cpu"):
    """The mHC norm, bound to a dispatcher for the given modality group sizes."""
    norm = layers.MultiModalityRMSNorm(STREAMS * hidden, num_modality=len(group_sizes), out_dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(norm.weight.shape, generator=generator) * 0.1)
    norm = norm.to(device)
    if len(group_sizes) == 1:
        return norm
    modalities = torch.cat([torch.full((size,), index, dtype=torch.int32) for index, size in enumerate(group_sizes)])
    dispatcher = layers.ModalityDispatcher(modalities.to(device), len(group_sizes))
    return lambda tensor: norm(tensor, dispatcher)


def _strided_operand_compute_logits(handler, flattened, norm, phi_fused):
    """The MUSA ``compute_logits`` that hands bmm the batch-strided stream view of the flat norm."""
    normed = norm(flattened).to(handler.dtype)
    fused = torch.bmm(
        normed.reshape(-1, handler.num_streams, handler.hidden_size).transpose(0, 1),
        phi_fused.reshape(handler.num_streams, handler.hidden_size, -1),
    ).sum(0)
    pre, post, residual = torch.split(
        fused,
        (handler.num_streams, handler.num_streams, handler.num_streams**2),
        dim=-1,
    )
    return pre, post, residual.view(-1, handler.num_streams, handler.num_streams)


def _recording_bmm(monkeypatch):
    operands = []
    bmm = torch.bmm

    def recording_bmm(lhs, rhs):
        operands.append(lhs)
        return bmm(lhs, rhs)

    monkeypatch.setattr(torch, "bmm", recording_bmm)
    return operands


@pytest.mark.cpu
@pytest.mark.parametrize("group_sizes", GROUP_SIZES, ids=lambda sizes: "-".join(map(str, sizes)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_split_features_match_flat_features(group_sizes, dtype):
    tokens = sum(group_sizes)
    streams = torch.randn(tokens, STREAMS, HIDDEN, generator=torch.Generator().manual_seed(9)).to(dtype)
    norm = _norm(group_sizes)
    split = norm(streams)
    flat = norm(streams.flatten(1))
    assert split.shape == streams.shape and split.dtype == flat.dtype == torch.float32
    torch.testing.assert_close(split.flatten(1), flat, rtol=2e-6, atol=1e-6)


@pytest.mark.cpu
def test_rmsnorm_rejects_features_that_do_not_split_dim():
    norm = layers.MultiModalityRMSNorm(STREAMS * HIDDEN)
    with pytest.raises(ValueError):
        norm(torch.randn(5, STREAMS, HIDDEN + 1))
    with pytest.raises(ValueError):
        norm(torch.randn(STREAMS * HIDDEN + 1))
    with pytest.raises(ValueError):
        layers.MultiModalityRMSNorm(STREAMS * HIDDEN, num_patterns=2)(torch.randn(5, STREAMS, HIDDEN))
    # A matching last dimension keeps the single-dimension norm.
    assert norm(torch.randn(5, STREAMS * HIDDEN)).shape == (5, STREAMS * HIDDEN)


@pytest.mark.cpu
@pytest.mark.parametrize("group_sizes", GROUP_SIZES, ids=lambda sizes: "-".join(map(str, sizes)))
@pytest.mark.parametrize("dtype", DTYPES)
def test_musa_eager_compute_logits_hands_bmm_a_dense_operand(monkeypatch, group_sizes, dtype):
    handler = _handler(monkeypatch, musa=True)
    tokens = sum(group_sizes)
    flattened, phi = _projection_inputs(tokens=tokens)
    flattened = flattened.to(dtype)
    norm = _norm(group_sizes)
    bmm = torch.bmm
    operands = _recording_bmm(monkeypatch)
    expected = _strided_operand_compute_logits(handler, flattened, norm, phi)
    actual = handler.compute_logits(flattened, norm, phi)

    strided, dense = operands
    assert dense.shape == (STREAMS, tokens, HIDDEN)
    assert dense.is_contiguous()
    assert torch.equal(dense, strided)
    # MUSA bmm makes a strided operand contiguous before its GEMM, so that copy is the reference input.
    reference = torch.split(bmm(strided.contiguous(), phi.reshape(STREAMS, HIDDEN, -1)).sum(0), (4, 4, 16), dim=-1)
    for actual_part, reference_part, expected_part in zip(actual, reference, expected, strict=True):
        assert actual_part.shape == expected_part.shape and actual_part.dtype == expected_part.dtype
        torch.testing.assert_close(actual_part, reference_part.view(expected_part.shape), rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [0, 1, 37])
def test_musa_compiling_compute_logits_writes_the_stream_major_operand(monkeypatch, tokens):
    handler = _handler(monkeypatch, musa=True)
    flattened, phi = _projection_inputs(tokens=tokens)
    expected = handler.compute_logits(flattened, lambda tensor: tensor, phi)
    seen = []
    operands = _recording_bmm(monkeypatch)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def norm(tensor):
        seen.append(tensor.shape)
        return tensor

    actual = handler.compute_logits(flattened, norm, phi)
    # Under compilation the norm sees the [tokens, streams, hidden] view.
    assert seen == [(tokens, STREAMS, HIDDEN)]
    assert len(operands) == 1
    assert operands[0].shape == (STREAMS, tokens, HIDDEN)
    assert operands[0].stride() == (tokens * HIDDEN, HIDDEN, 1)
    assert torch.equal(operands[0], flattened.view(tokens, STREAMS, HIDDEN).transpose(0, 1))
    for actual_part, expected_part in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)


def _region_inputs(tokens, hidden, device, seed=10):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(tokens, STREAMS, hidden, generator=generator).to(torch.bfloat16).to(device),
        torch.randn(tokens, hidden, generator=generator).to(torch.bfloat16).to(device),
        torch.rand(tokens, STREAMS, generator=generator).to(torch.bfloat16).to(device),
        torch.rand(tokens, STREAMS, STREAMS, generator=generator).to(torch.bfloat16).to(device),
        (torch.randn(STREAMS * hidden, 2 * STREAMS + STREAMS**2, generator=generator) * 0.05).to(device),
    )


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@pytest.mark.parametrize("tokens", [1031, 3702])
@pytest.mark.parametrize(
    "group_sizes",
    [(1,), (1, 0, 0), (0, 1, 0), (0.8, 0.15, 0.05)],
    ids=["single-modality", "first-group", "second-group", "three-groups"],
)
@pytest.mark.parametrize("producer", ["input", "mix"])
def test_musa_compiled_compute_logits_is_bitwise_unchanged(tokens, group_sizes, producer):
    """The compiled stream-major norm matches the strided-operand form bit for bit, with the mix fused in."""
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    from torch._inductor.utils import run_and_get_code

    hidden = 3072
    sizes = [int(tokens * share) for share in group_sizes]
    sizes[0] += tokens - sum(sizes)
    handler = layers.MHCHandler(STREAMS, hidden)
    assert handler.fp32_stream_contractions
    norm = _norm(tuple(sizes), hidden=hidden, device="musa")
    streams, branch_output, post, residual, phi = _region_inputs(tokens, hidden, "musa")

    def region(compute_logits):
        def run(streams, branch_output, post, residual, phi):
            if producer == "mix":
                streams = handler.hyper_connect(streams, branch_output, post, residual)
            return streams, *compute_logits(handler.flatten(streams), norm, phi)

        return torch.compile(run, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True})

    with torch.no_grad():
        expected = region(partial(_strided_operand_compute_logits, handler))(
            streams, branch_output, post, residual, phi
        )
        actual, codes = run_and_get_code(region(handler.compute_logits), streams, branch_output, post, residual, phi)
    code = "\n".join(codes)
    dense = rf"extern_kernels\.bmm\(reinterpret_tensor\(buf\d+, \({STREAMS}, {tokens}, {hidden}\), \({tokens * hidden}, {hidden}, 1\)"
    assert re.search(dense, code), "bmm does not receive the dense stream-major operand"
    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert actual_part.dtype == expected_part.dtype
        # Compare bit patterns so signed zeros and NaN payloads also count.
        integer = torch.int32 if actual_part.dtype == torch.float32 else torch.int16
        assert torch.equal(actual_part.cpu().view(integer), expected_part.cpu().view(integer))
    torch._dynamo.reset()
