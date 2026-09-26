# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import gated_residual_adaln as fusion
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.gated_residual_adaln import try_fused_gated_residual_adaln

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def kernel_correctness(monkeypatch):
    # Preserve broad kernel/rounding coverage without enabling unmeasured
    # layouts or CI GPUs in the production dispatcher.
    monkeypatch.setattr(fusion, "_has_measured_speedup", lambda *args: True)


@pytest.mark.cpu
@pytest.mark.parametrize("seq_len", [12, 29, 4096])
@pytest.mark.parametrize(
    "case",
    ["contiguous", "sliced", "other_gpu", "fp32", "batch", "sequence", "stride", "compile", "grad", "native_off"],
)
def test_performance_dispatch(monkeypatch, seq_len, case):
    # CPU FakeTensors keep metadata operations independent of a CUDA build.
    # Only availability and the device name are simulated; tensor layout ops
    # and production selection logic execute unchanged.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    calls = []
    if case == "native_off":
        monkeypatch.setenv("VLLM_OMNI_QWEN_ADALN_V2", "off")

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append(grid)

    monkeypatch.setattr(fusion, "HAS_TRITON", True)
    monkeypatch.setattr(fusion, "triton", SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
    monkeypatch.setattr(fusion.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        fusion.current_platform,
        "get_device_name",
        lambda index: "NVIDIA H100 80GB HBM3" if case == "other_gpu" else "NVIDIA A100-SXM4-40GB",
    )
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: case == "compile")
    monkeypatch.setattr(fusion, "_gated_residual_cast_kernel", Kernel(), raising=False)
    monkeypatch.setattr(fusion, "_cast_modulate_kernel", Kernel(), raising=False)
    fusion._is_benchmarked_device.cache_clear()
    try:
        with FakeTensorMode():
            b = 2 if case == "batch" else 1
            s = 17 if case == "sequence" else seq_len
            d = 3072
            dtype = torch.float32 if case == "fp32" else torch.bfloat16
            residual = torch.empty(b, s, d, dtype=dtype, device="cpu", requires_grad=case == "grad")
            stride_factor = 3 if case == "stride" else 2 if case == "sliced" else 1
            branch = torch.empty(b, s, stride_factor * d, dtype=dtype, device="cpu")[..., :d]
            modulation = torch.empty(b, 6 * d, dtype=dtype, device="cpu")
            gate, scale, shift = [t.unsqueeze(1) for t in modulation.chunk(6, -1)[:3]]
            result = try_fused_gated_residual_adaln(residual, branch, gate, scale, shift, 1e-6)
            enabled = case in ("contiguous", "sliced")
            assert (result is not None) == enabled
            assert len(calls) == 2 * enabled
            if result is not None:
                assert all(t.shape == residual.shape and t.dtype == dtype and t.is_contiguous() for t in result)
    finally:
        fusion._is_benchmarked_device.cache_clear()


def _inputs(b, s, d, dtype, device, seed=7382):
    generator = torch.Generator(device=device).manual_seed(seed)
    residual = torch.randn(b, s, d, dtype=dtype, device=device, generator=generator)
    # Attention output and modulation both have realistic slice strides.
    branch = torch.randn(b, s, 2 * d, dtype=dtype, device=device, generator=generator)[..., :d]
    modulation = torch.randn(b, 6 * d, dtype=dtype, device=device, generator=generator)
    shift, scale, gate, _, _, _ = [t.unsqueeze(1) for t in modulation.chunk(6, -1)]
    return residual, branch, gate, scale, shift


def _reference(residual, branch, gate, scale, shift, eps=1e-6):
    r = residual + gate * branch
    return r, AdaLayerNorm(r.shape[-1], eps=eps)(r, scale, shift)


def _checked_call(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], eps=1e-6):
    before = [x.clone() for x in inputs]
    expected = _reference(*inputs, eps=eps)
    result = try_fused_gated_residual_adaln(*inputs, eps)
    assert result is not None
    r, y = result
    # Residual arithmetic has no reduction and preserves every rounding step.
    torch.testing.assert_close(r, expected[0], rtol=0, atol=0, equal_nan=True)
    # A one-ULP BF16 difference can accumulate across Qwen denoising steps.
    # Native LayerNorm and the pointwise rounding boundaries must match exactly.
    torch.testing.assert_close(y, expected[1], atol=0, rtol=0, equal_nan=True)
    assert torch.equal(torch.isnan(y), torch.isnan(expected[1]))
    assert torch.equal(torch.isinf(y), torch.isinf(expected[1]))
    assert y.shape == r.shape == inputs[0].shape
    assert r.is_contiguous() and y.is_contiguous()
    assert r.data_ptr() != y.data_ptr()
    for x, original in zip(inputs, before):
        torch.testing.assert_close(x, original, atol=0, rtol=0, equal_nan=True)
        assert r.data_ptr() != x.data_ptr() and y.data_ptr() != x.data_ptr()
    return result


@pytest.mark.cpu
def test_cpu_declines_without_mutation():
    inputs = _inputs(2, 3, 128, torch.float32, "cpu")
    before = [x.clone() for x in inputs]
    assert try_fused_gated_residual_adaln(*inputs, 1e-6) is None
    for x, original in zip(inputs, before):
        assert torch.equal(x, original)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("b,s,d", [(1, 1, 3072), (2, 3, 3072), (2, 129, 3072), (2, 5, 257), (1, 7, 8192)])
def test_two_outputs_and_strided_modulation(dtype, b, s, d, kernel_correctness):
    _checked_call(_inputs(b, s, d, dtype, "cuda"))


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("seed", [142, 143, 144])
@pytest.mark.parametrize("seq_len", [12, 29, 4096])
def test_qwen_layer_norm_rounding(seed, seq_len, kernel_correctness):
    # Qwen T2I's image/text shapes. The former two-pass Triton reduction
    # differed from native LayerNorm by one FP32 ULP, occasionally crossing a
    # BF16 midpoint. Small tensors with a one-BF16-ULP tolerance missed it.
    inputs = _inputs(1, seq_len, 3072, torch.bfloat16, "cuda", seed=seed)
    _checked_call(inputs)
    residual, branch, gate, scale, shift = inputs
    # Also inspect the normalization boundary before scale/shift can hide it.
    scale.zero_()
    shift.zero_()
    _checked_call((residual, branch, gate, scale, shift))


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "case", ["zero_gate", "minus_one_scale", "constant", "tiny_variance", "small", "large", "nan", "inf"]
)
def test_numerical_boundaries(dtype, case, kernel_correctness):
    inputs = _inputs(2, 3, 3072, dtype, "cuda")
    residual, branch, gate, scale, shift = inputs
    if case == "zero_gate":
        gate.zero_()
    elif case == "minus_one_scale":
        scale.fill_(-1)
    elif case in ("constant", "tiny_variance"):
        residual.fill_(1)
        branch.zero_()
        if case == "tiny_variance":
            residual[..., 0] += torch.finfo(dtype).eps
    elif case == "small":
        residual.mul_(1e-15)
        branch.mul_(1e-15)
    elif case == "large":
        residual.mul_(1e8)
        branch.mul_(1e8)
    elif case == "nan":
        residual[0, 0, 0] = float("nan")
    elif case == "inf":
        residual[0, 0, 0] = float("inf")
    _checked_call(inputs)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fallback_and_compile(dtype, kernel_correctness):
    inputs = _inputs(2, 3, 3072, dtype, "cuda")
    r, branch, gate, scale, shift = inputs
    assert try_fused_gated_residual_adaln(r.requires_grad_(), branch, gate, scale, shift, 1e-6) is None
    r.requires_grad_(False)
    assert try_fused_gated_residual_adaln(r, branch, gate.double(), scale, shift, 1e-6) is None
    assert (
        try_fused_gated_residual_adaln(
            r[..., ::2], branch[..., ::2], gate[..., ::2], scale[..., ::2], shift[..., ::2], 1e-6
        )
        is None
    )
    assert try_fused_gated_residual_adaln(r, branch, gate.expand_as(r), scale, shift, 1e-6) is None

    def caller(r, branch, gate, scale, shift):
        fused = try_fused_gated_residual_adaln(r, branch, gate, scale, shift, 1e-6)
        if fused is not None:
            return fused
        residual = r + gate * branch
        normed = torch.nn.functional.layer_norm(residual.float(), (r.shape[-1],), eps=1e-6).to(r.dtype)
        return residual, normed * (1 + scale) + shift

    expected = torch.compile(caller, fullgraph=True)(*inputs)
    # fullgraph traces both pointwise kernels and preserves native LayerNorm.
    reference = _reference(*inputs)
    torch.testing.assert_close(expected[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(expected[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(expected, caller(*inputs), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"})
def test_cuda_graph_replays_new_inputs(kernel_correctness):
    inputs = _inputs(2, 5, 3072, torch.bfloat16, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _checked_call(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = try_fused_gated_residual_adaln(*inputs, 1e-6)
    inputs[0].add_(0.5)
    graph.replay()
    assert result is not None
    expected = _reference(*inputs)
    torch.testing.assert_close(result[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(result[1], expected[1], rtol=0, atol=0)
