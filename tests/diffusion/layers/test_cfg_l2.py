# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers.cfg_l2 import try_fused_cfg_l2

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _reference(p, n, w):
    combined = n + w * (p - n)
    return combined * (torch.norm(p, dim=-1, keepdim=True) / torch.norm(combined, dim=-1, keepdim=True))


def _check(p, n, w):
    originals = (p.clone(), n.clone())
    expected = _reference(p, n, w)
    out = try_fused_cfg_l2(p, n, w)
    assert out is not None
    atol, rtol = (0.015625, 0.016) if p.dtype == torch.bfloat16 else (3e-6, 3e-6)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol, equal_nan=True)
    assert torch.equal(torch.isnan(out), torch.isnan(expected))
    assert torch.equal(torch.isinf(out), torch.isinf(expected))
    assert out.shape == p.shape and out.dtype == p.dtype and out.is_contiguous()
    for x, before in zip((p, n), originals):
        torch.testing.assert_close(x, before, rtol=0, atol=0, equal_nan=True)
        assert out.data_ptr() != x.data_ptr()
    return out


@pytest.mark.cpu
def test_cpu_fallback():
    p, n = torch.randn(2, 3, 64), torch.randn(2, 3, 64)
    assert try_fused_cfg_l2(p, n, 4.0) is None


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("b,s,d", [(1, 1, 64), (2, 3, 64), (2, 129, 64), (2, 7, 63), (1, 3, 4096)])
@pytest.mark.parametrize("w", [0.0, 1.0, 3.7, 4.0, 20.0])
def test_cfg_l2(dtype, b, s, d, w):
    generator = torch.Generator(device="cuda").manual_seed(7382)
    p = torch.randn(b, s + 3, d, dtype=dtype, device="cuda", generator=generator)[:, :s]
    n = torch.randn(b, s + 5, d, dtype=dtype, device="cuda", generator=generator)[:, :s]
    _check(p, n, w)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["equal", "zero", "cancellation", "tiny", "large", "nan", "inf"])
def test_cfg_l2_special_values(dtype, case):
    p = torch.randn(2, 5, 64, dtype=dtype, device="cuda")
    n = torch.randn_like(p)
    w = 4.0
    if case == "equal":
        n.copy_(p)
    elif case == "zero":
        p.zero_()
        n.zero_()
    elif case == "cancellation":
        p.fill_(1)
        n.fill_(2)
        w = 2.0
    elif case == "tiny":
        p.mul_(1e-20)
        n.mul_(1e-20)
    elif case == "large":
        p.mul_(1e20)
        n.mul_(1e20)
    else:
        p[0, 0, 0] = float(case)
    result = _check(p, n, w)
    if case == "tiny":
        torch.testing.assert_close(result, _reference(p, n, w), rtol=0.02, atol=1e-23)


@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unsupported_inputs_and_compile(dtype):
    p = torch.randn(2, 5, 64, device="cuda", dtype=dtype)
    n = torch.randn_like(p)
    assert try_fused_cfg_l2(p[..., ::2], n[..., ::2], 4.0) is None
    assert try_fused_cfg_l2(p.requires_grad_(), n, 4.0) is None
    p.requires_grad_(False)
    assert try_fused_cfg_l2(p, n.double(), 4.0) is None
    assert try_fused_cfg_l2(p[:, :0], n[:, :0], 4.0) is None

    def caller(p, n):
        out = try_fused_cfg_l2(p, n, 4.0)
        return _reference(p, n, 4.0) if out is None else out

    compiled = torch.compile(caller, fullgraph=True)
    actual = compiled(p, n)
    atol, rtol = (0.015625, 0.016) if dtype == torch.bfloat16 else (3e-6, 3e-6)
    torch.testing.assert_close(actual, _reference(p, n, 4.0), rtol=rtol, atol=atol)
    torch.testing.assert_close(actual, caller(p, n), rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"})
def test_cuda_graph():
    p = torch.randn(2, 7, 64, device="cuda", dtype=torch.bfloat16)
    n = torch.randn_like(p)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            try_fused_cfg_l2(p, n, 4.0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = try_fused_cfg_l2(p, n, 4.0)
    p.add_(0.25)
    graph.replay()
    torch.testing.assert_close(out, _reference(p, n, 4.0), rtol=0.016, atol=0.015625)
