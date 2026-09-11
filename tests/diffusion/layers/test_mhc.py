# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers import mhc
from vllm_omni.diffusion.models.magi2 import layers as model_layers

pytestmark = [pytest.mark.diffusion, pytest.mark.local_model]


def coefficients(tokens=7, streams=4):
    torch.manual_seed(71)
    # Actual model split views: post and residual share a wider FP32 row.
    packed = torch.randn(tokens, streams * (streams + 2))
    post = packed[:, streams : 2 * streams]
    residual = packed[:, 2 * streams :].view(tokens, streams, streams)
    return post, residual, torch.tensor([0.8]), torch.randn(streams), torch.tensor([1.1]), torch.randn(streams, streams)


def reference(args, *, scale=0.125, iterations=20, epsilon=1e-12, out_dtype=torch.float32):
    post, residual, ap, bp, ar, br = args
    post = 2.0 * torch.sigmoid(ap * scale * post + bp.unsqueeze(0))
    residual = ar * scale * residual.float() + br.unsqueeze(0).float()
    residual = torch.exp(residual - residual.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iterations):
        residual = residual / (residual.sum(-2, keepdim=True) + epsilon)
        residual = residual / (residual.sum(-1, keepdim=True) + epsilon)
    return post.to(out_dtype), residual.to(out_dtype)


def mix_reference(args):
    streams, output, post, matrix = args
    branch = torch.einsum("tn,tc->tnc", post, output)
    mixed = torch.einsum("tij,tjc->tic", matrix, streams)
    return mixed + branch


def mix_inputs(tokens=7, hidden=70, dtype=torch.float32, streams=4):
    torch.manual_seed(41)
    return (
        torch.randn(tokens, streams, hidden).to(dtype),
        torch.randn(tokens, hidden).to(dtype),
        torch.rand(tokens, streams).to(dtype),
        torch.randn(tokens, streams, streams).to(dtype),
    )


@pytest.mark.cpu
@pytest.mark.parametrize("streams", [3, 4])
@pytest.mark.parametrize("iterations", [-1, 0, 1, 20, 65])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_coefficient_contract(streams, iterations, dtype):
    args = coefficients(streams=streams)
    kwargs = dict(scale=0.125, iterations=iterations, out_dtype=dtype)
    actual = mhc.MHCPostResidual().forward_cuda(*args, **kwargs)
    for a, b in zip(actual, reference(args, **kwargs)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.cpu
def test_native_gradients_and_reused_parameters():
    op = mhc.MHCPostResidual()
    args = tuple(x.detach().clone().requires_grad_() for x in coefficients())
    expected_args = tuple(x.detach().clone().requires_grad_() for x in args)
    result = op.forward_cuda(*args, scale=0.125, out_dtype=torch.float32)
    expected = reference(expected_args)
    sum(x.square().sum() for x in result).backward()
    sum(x.square().sum() for x in expected).backward()
    for a, b in zip(args, expected_args):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    with torch.no_grad():
        args[3].add_(2)
        result = op.forward_cuda(*args, scale=0.125, out_dtype=torch.float32)
        for a, b in zip(result, reference(args)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_mix_dtype_and_noncontiguous(dtype):
    args = mix_inputs(dtype=dtype)
    args = (args[0][..., ::2], args[1][..., ::2], *args[2:])
    torch.testing.assert_close(mhc.MHCMix().forward_cuda(*args), mix_reference(args), rtol=0, atol=0)


@pytest.mark.cpu
def test_meta_shapes_and_empty():
    args = tuple(x.to("meta") for x in coefficients())
    post, matrix = mhc.MHCPostResidual().forward_cuda(*args, scale=0.125, out_dtype=torch.bfloat16)
    assert post.shape == (7, 4) and matrix.shape == (7, 4, 4)
    assert post.dtype == matrix.dtype == torch.bfloat16
    mix = mhc.MHCMix().forward_cuda(*[x.to("meta") for x in mix_inputs()])
    assert mix.shape == (7, 4, 70) and mix.device.type == "meta"
    args = coefficients(tokens=0)
    post, matrix = mhc.MHCPostResidual().forward_cuda(*args, scale=0.125, out_dtype=torch.float32)
    assert post.shape == (0, 4) and matrix.shape == (0, 4, 4)
    assert mhc.MHCMix().forward_cuda(*mix_inputs(tokens=0)).shape == (0, 4, 70)


@pytest.mark.cpu
def test_sinkhorn_alias_and_rectangular_native_contract():
    assert model_layers.sinkhorn_knopp is mhc.sinkhorn_knopp
    logits = torch.randn(2, 3, 4)
    expected = torch.exp(logits - logits.amax((-2, -1), keepdim=True))
    torch.testing.assert_close(mhc.sinkhorn_knopp(logits, 0, 1e-12), expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_model_wiring_is_opt_in(monkeypatch):
    calls = []

    class Prepare:
        def __call__(self, *args, **kwargs):
            calls.append("dispatch")
            return reference(args, **kwargs)

        def forward_native(self, *args, **kwargs):
            calls.append("native")
            return reference(args, **kwargs)

    monkeypatch.setattr(model_layers, "_mhc_post_residual", Prepare())
    handler = model_layers.MHCHandler(4, 16)
    post, residual, ap, bp, ar, br = coefficients()
    monkeypatch.delenv("MAGI2_USE_FUSED_MHC", raising=False)
    handler.compute_post_residual((ap, bp, post), (ar, br, residual), out_dtype=torch.float32)
    monkeypatch.setenv("MAGI2_USE_FUSED_MHC", "1")
    handler.compute_post_residual((ap, bp, post), (ar, br, residual), out_dtype=torch.float32)
    assert calls == ["native", "dispatch"]


@pytest.mark.cpu
def test_compile_keeps_native_graph():
    torch._dynamo.reset()
    targets: list[str] = []

    def backend(gm, _inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    op = torch.compile(mhc.MHCPostResidual(), fullgraph=True, backend=backend)
    args = coefficients()
    actual = op(*args, scale=0.125, iterations=2, out_dtype=torch.float32)
    for a, b in zip(actual, reference(args, iterations=2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any("sigmoid" in x for x in targets) and any("sum" in x for x in targets)
    assert not any("triton" in x for x in targets)


DEVICES = [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("musa", marks=pytest.mark.musa)]


def device(name):
    api = getattr(torch, name, None)
    if api is None or not api.is_available():
        pytest.skip(f"requires {name}")
    return torch.device(name)


def spy(monkeypatch, name):
    original = getattr(mhc, name)
    launches = []

    class Spy:
        def __getitem__(self, grid):
            launches.append(grid)
            return original[grid]

    monkeypatch.setattr(mhc, name, Spy())
    return launches


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,iterations", [(0, 20), (1, 0), (17, 20)])
def test_coefficients_gpu(monkeypatch, device_name, dtype, tokens, iterations):
    dev = device(device_name)
    # Preserve the model's strided split views on the GPU too.
    args = coefficients(tokens)
    packed = torch.zeros(tokens, 24, device=dev)
    packed[:, 4:8] = args[0].to(dev)
    packed[:, 8:] = args[1].reshape(tokens, 16).to(dev)
    args = (packed[:, 4:8], packed[:, 8:].view(tokens, 4, 4), *(x.to(dev) for x in args[2:]))
    launches = spy(monkeypatch, "_mhc_post_residual_kernel")
    actual = mhc.MHCPostResidual()(*args, scale=0.125, iterations=iterations, out_dtype=dtype)
    getattr(torch, device_name).synchronize()
    assert len(launches) == bool(tokens)
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    for a, b in zip(actual, reference(args, iterations=iterations, out_dtype=dtype)):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=1e-6)


@pytest.mark.parametrize("device_name", DEVICES)
def test_sinkhorn_gpu_shift_and_extremes(device_name):
    dev = device(device_name)
    args = tuple(x.to(dev) for x in coefficients())
    op = mhc.MHCPostResidual()
    post, matrix = op(*args, scale=1.0, out_dtype=torch.float32)
    shifted = (args[0], args[1] + 16.0, *args[2:])
    shifted_post, shifted_matrix = op(*shifted, scale=1.0, out_dtype=torch.float32)
    torch.testing.assert_close(shifted_matrix, matrix, rtol=1e-5, atol=3e-6)
    torch.testing.assert_close(shifted_post, post, rtol=0, atol=0)
    torch.testing.assert_close(matrix.sum(-1), torch.ones_like(post), rtol=1e-5, atol=1e-5)
    # A well-conditioned case converges in both dimensions at the requested
    # iteration count. Extreme matrices below need not converge in 20 steps.
    balanced_args = list(args)
    balanced_args[1] = args[1] * 0.125
    balanced_args[5] = torch.zeros_like(args[5])
    _, balanced = op(*balanced_args, scale=1.0, out_dtype=torch.float32)
    for dim in (-2, -1):
        torch.testing.assert_close(balanced.sum(dim), torch.ones_like(post), rtol=1e-5, atol=1e-5)
    extreme = list(args)
    extreme[1] = torch.eye(4, device=dev).expand(7, -1, -1).contiguous() * 1000
    extreme[1][0, 0, 0] = float("nan")
    extreme[1][1, 0, 0] = float("inf")
    for iters in (0, 20):
        actual = op(*extreme, scale=1.0, iterations=iters, out_dtype=torch.float32)
        expected = reference(extreme, scale=1.0, iterations=iters)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,hidden", [(0, 70), (2, 70), (17, 3072)])
def test_mix_gpu(monkeypatch, device_name, dtype, tokens, hidden):
    dev = device(device_name)
    args = tuple(x.to(dev) for x in mix_inputs(tokens, hidden, dtype))
    launches = spy(monkeypatch, "_mhc_mix_kernel")
    actual = mhc.MHCMix()(*args)
    getattr(torch, device_name).synchronize()
    assert len(launches) == bool(tokens)
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}[dtype]
    torch.testing.assert_close(actual, mix_reference(args), rtol=tolerance, atol=1e-5)


@pytest.mark.parametrize("device_name", DEVICES)
def test_mix_preserves_separate_bf16_rounding(device_name):
    dev = device(device_name)
    streams = torch.ones(1, 4, 1, device=dev, dtype=torch.bfloat16)
    branch = torch.full((1, 1), 1.0078125, device=dev, dtype=torch.bfloat16)
    post = torch.full((1, 4), 1.03125, device=dev, dtype=torch.bfloat16)
    matrix = torch.eye(4, device=dev, dtype=torch.bfloat16).unsqueeze(0)
    args = (streams, branch, post, matrix)
    expected = mix_reference(args)
    once = (
        torch.einsum("tij,tjc->tic", matrix.float(), streams.float())
        + torch.einsum("tn,tc->tnc", post.float(), branch.float())
    ).bfloat16()
    assert not torch.equal(expected, once), "fixture must distinguish the cast boundaries"
    torch.testing.assert_close(mhc.MHCMix()(*args), expected, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", DEVICES)
def test_gpu_unsupported_layout_falls_back(monkeypatch, device_name):
    dev = device(device_name)

    class Forbidden:
        def __getitem__(self, _grid):
            pytest.fail("unsupported layouts must not launch the kernel")

    monkeypatch.setattr(mhc, "_mhc_post_residual_kernel", Forbidden())
    monkeypatch.setattr(mhc, "_mhc_mix_kernel", Forbidden())
    args = tuple(x.to(dev) for x in coefficients(streams=3))
    actual = mhc.MHCPostResidual()(*args, scale=0.125, out_dtype=torch.float32)
    for a, b in zip(actual, reference(args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    args = tuple(x.to(dev) for x in mix_inputs())
    args = (args[0][..., ::2], args[1][..., ::2], *args[2:])
    torch.testing.assert_close(mhc.MHCMix()(*args), mix_reference(args), rtol=0, atol=0)
    # A singleton alpha with extra rank changes broadcasting, so it cannot be
    # treated like the scalar alpha loaded by the eager kernel.
    broadcast_args = list(x.to(dev) for x in coefficients())
    broadcast_args[2] = broadcast_args[2].view(1, 1, 1)
    actual = mhc.MHCPostResidual()(*broadcast_args, scale=0.125, out_dtype=torch.float32)
    for a, b in zip(actual, reference(broadcast_args)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device_name", DEVICES)
def test_gpu_compiled_parameters_do_not_get_cached(monkeypatch, device_name):
    dev = device(device_name)

    class Forbidden:
        def __getitem__(self, _grid):
            pytest.fail("compilation must retain the native expression")

    monkeypatch.setattr(mhc, "_mhc_post_residual_kernel", Forbidden())
    op = torch.compile(mhc.MHCPostResidual(), backend="eager", fullgraph=True)
    args = list(x.to(dev) for x in coefficients())
    for bias in (0.0, 2.0, -1.0):
        args[3].fill_(bias)
        actual = op(*args, scale=0.125, out_dtype=torch.float32)
        for a, b in zip(actual, reference(args)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
