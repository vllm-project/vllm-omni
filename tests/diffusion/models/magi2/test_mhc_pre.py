# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import patch

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.diffusion.models.magi2.test_native_preview import _initialize_tiny_model, _tiny_config
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.layers import MHCHandler
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer, Modality

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _reference(handler, streams, alpha_bias_logits, *, out_dtype=None):
    handler._check_multi(streams)
    alpha, bias, logits = alpha_bias_logits
    coefficients = torch.sigmoid(alpha * handler.matmul_scale * logits + bias.unsqueeze(0))
    return torch.einsum("tn,tnc->tc", coefficients.to(out_dtype or streams.dtype), streams)


def _inputs(dtype=torch.float32, tokens=7, streams=4, hidden=64, seed=31):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, streams, hidden * 2, generator=generator).to(dtype)[..., ::2]
    alpha = torch.tensor(0.7)
    bias = torch.randn(streams * 2, generator=generator)[::2]
    # A view with the row stride of a fused pre/post/residual projection.
    logits = torch.randn(tokens, streams * (streams + 2), generator=generator)[:, :streams]
    return MHCHandler(streams, hidden), x, (alpha, bias, logits)


def _check(handler, x, coefficients, out_dtype=None):
    snapshots = [tensor.clone() for tensor in (x, *coefficients)]
    expected = _reference(handler, x, coefficients, out_dtype=out_dtype)
    actual = handler.apply_pre(x, coefficients, out_dtype=out_dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    finite = torch.isfinite(expected)
    assert torch.equal(torch.signbit(actual[finite]), torch.signbit(expected[finite]))
    for tensor, snapshot in zip((x, *coefficients), snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0, equal_nan=True)
    return actual


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("tokens,streams,hidden", [(7, 4, 64), (1, 3, 128), (0, 4, 64), (7, 1, 64), (3, 4, 1)])
def test_pre_exact_parity(dtype, tokens, streams, hidden):
    handler, x, coefficients = _inputs(dtype, tokens, streams, hidden)
    _check(handler, x, coefficients)
    _check(handler, x, coefficients, out_dtype=dtype)


class _RecordBMM(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operands = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.bmm.default:
            self.operands.append([(tuple(tensor.shape), tensor.dtype, tuple(tensor.stride())) for tensor in args[:2]])
        return func(*args, **(kwargs or {}))


@pytest.mark.cpu
def test_same_matrix_contraction_without_einsum_dispatch():
    handler, x, coefficients = _inputs(torch.bfloat16)
    with _RecordBMM() as original:
        _reference(handler, x, coefficients)
    with _RecordBMM() as explicit, patch.object(torch, "einsum", side_effect=AssertionError("einsum called")):
        handler.apply_pre(x, coefficients)
    assert len(original.operands) == len(explicit.operands) == 1
    assert original.operands == explicit.operands


@pytest.mark.cpu
def test_does_not_round_products_before_reduction():
    handler, x, coefficients = _inputs(torch.bfloat16)
    result = _check(handler, x, coefficients)
    alpha, bias, logits = coefficients
    pre = torch.sigmoid(alpha * handler.matmul_scale * logits + bias.unsqueeze(0)).to(x.dtype)
    rounded_products = (x * pre.unsqueeze(-1)).sum(dim=1)
    assert not torch.equal(result, rounded_products)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "case",
    [
        "single_stream",
        "broadcast_tokens",
        "broadcast_streams",
        "extra_rank",
        "wrong_batch",
        "wrong_hidden",
        "mixed_dtype",
    ],
)
def test_broadcast_and_error_contracts(case):
    handler, x, (alpha, bias, logits) = _inputs(streams=1 if case == "single_stream" else 4)
    if case == "broadcast_tokens":
        logits = logits[:1]
    elif case == "broadcast_streams":
        logits, bias = logits[:, :1], bias[:1]
    elif case == "extra_rank":
        logits = logits.unsqueeze(0)
    elif case == "wrong_batch":
        logits = logits[:2]
    elif case == "wrong_hidden":
        x = x[..., :1]
    if case in ("extra_rank", "wrong_batch", "wrong_hidden", "mixed_dtype"):
        dtype = torch.bfloat16 if case == "mixed_dtype" else None
        with pytest.raises((RuntimeError, ValueError)) as reference_error:
            _reference(handler, x, (alpha, bias, logits), out_dtype=dtype)
        with pytest.raises(type(reference_error.value)):
            handler.apply_pre(x, (alpha, bias, logits), out_dtype=dtype)
    else:
        with patch.object(torch, "einsum", wraps=torch.einsum) as fallback:
            _check(handler, x, (alpha, bias, logits))
        assert fallback.call_count == 2  # Oracle and compatibility path.


@pytest.mark.cpu
def test_nonfinite_and_signed_zero():
    for streams in (1, 4):
        handler, x, coefficients = _inputs(streams=streams)
        x[0] = -0.0
        x[1, 0, 0], x[2, 0, 0], x[3, 0, 0] = torch.inf, -torch.inf, torch.nan
        _check(handler, x, coefficients)


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [0, 7])
def test_gradients(tokens):
    handler, x, (alpha, bias, logits) = _inputs(tokens=tokens)
    x, alpha, bias, logits = (value.clone().requires_grad_() for value in (x, alpha, bias, logits))
    expected = _reference(handler, x, (alpha, bias, logits))
    actual = handler.apply_pre(x, (alpha, bias, logits))
    variables = (x, alpha, bias, logits)
    for grad, reference in zip(
        torch.autograd.grad(actual.square().sum(), variables), torch.autograd.grad(expected.square().sum(), variables)
    ):
        torch.testing.assert_close(grad, reference, rtol=0, atol=0)


@pytest.mark.cpu
def test_compile_frontend_and_changed_coefficients():
    handler, x, (alpha, bias, logits) = _inputs()
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(handler.apply_pre, backend=backend, fullgraph=True)
    try:
        for tokens in (1, 7):
            for shift in (0.0, 0.2):
                values = (alpha, bias + shift, logits[:tokens])
                torch.testing.assert_close(
                    compiled(x[:tokens], values), _reference(handler, x[:tokens], values), rtol=0, atol=0
                )
        assert graphs
    finally:
        torch._dynamo.reset()


@pytest.mark.cpu
def test_two_layer_native_model_parity():
    model = Magi2PreviewTransformer(_tiny_config(num_layers=2)).eval()
    _initialize_tiny_model(model, seed=7)
    generator = torch.Generator().manual_seed(11)
    packed, coords = torch.randn(6, 4, generator=generator), torch.ones(6, 9)
    modalities = torch.tensor(
        [Modality.VIDEO, Modality.VIDEO, Modality.AUDIO, Modality.AUDIO, Modality.TEXT, Modality.TEXT]
    )
    cu = torch.tensor([0, 6], dtype=torch.int32)
    with torch.no_grad(), patch.object(MHCHandler, "apply_pre", _reference):
        expected = model(packed, coords, modalities, VarlenHandler(cu, cu, 6, 6))
    original, calls = MHCHandler.apply_pre, []

    def observed(handler, *args, **kwargs):
        calls.append(1)
        return original(handler, *args, **kwargs)

    with torch.no_grad(), patch.object(MHCHandler, "apply_pre", observed):
        actual = model(packed, coords, modalities, VarlenHandler(cu, cu, 6, 6))
    assert len(calls) == 4
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.musa
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_musa_pre_matches_matrix_reference(dtype):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    for seed in (31, 32):
        for tokens, streams, hidden in ((7, 4, 6144), (1, 3, 128), (0, 4, 128), (7, 1, 128)):
            handler, x, coefficients = _inputs(dtype, tokens, streams, hidden, seed)
            x = x.to("musa").repeat_interleave(2, dim=-1)[..., ::2]
            alpha, bias, logits = (tensor.to("musa") for tensor in coefficients)
            logits = logits.repeat_interleave(2, dim=-1)[..., ::2]
            with torch.no_grad():
                _check(handler, x, (alpha, bias, logits))
                _check(handler, x, (alpha, bias, logits), out_dtype=dtype)


@pytest.mark.musa
def test_musa_autocast_preserves_pre_behavior():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    handler, x, coefficients = _inputs()
    x = x.to("musa")
    coefficients = tuple(tensor.to("musa") for tensor in coefficients)
    for dtype in (torch.float16, torch.bfloat16):
        with torch.no_grad(), torch.autocast("musa", dtype=dtype):
            assert torch.is_autocast_enabled("musa")
            actual = _check(handler, x, coefficients, out_dtype=torch.float32)
            assert actual.dtype == dtype
