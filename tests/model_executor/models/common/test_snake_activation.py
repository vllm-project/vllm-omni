# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Snake activation layout, copy-elision, and CUDA Graph contracts."""

import pytest
import torch

from vllm_omni.model_executor.models.common.alias_free_activation import AliasFreeActivation1d
from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("activation", [Snake, SnakeBeta])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 4, 1), (2, 8, 127), (1, 4, 4097)])
@pytest.mark.parametrize("layout", ["contiguous", "cropped", "transposed", "stepped"])
@torch.inference_mode()
def test_snake_layouts(activation, dtype, shape, layout):
    torch.manual_seed(42)
    batch, channels, length = shape
    module = activation(channels, alpha_logscale=True).to(device="cuda", dtype=dtype).eval()
    module.alpha.uniform_(-0.2, 0.2)
    if activation is SnakeBeta:
        module.beta.uniform_(-0.2, 0.2)
    if layout == "cropped":
        backing = torch.randn(batch, channels, length + 30, device="cuda", dtype=dtype)
        x = backing[..., 15:-15]
    elif layout == "transposed":
        backing = torch.randn(channels, batch, length, device="cuda", dtype=dtype)
        x = backing.transpose(0, 1)
    elif layout == "stepped":
        backing = torch.randn(batch, channels, length * 2, device="cuda", dtype=dtype)
        x = backing[..., ::2]
    else:
        backing = torch.randn(shape, device="cuda", dtype=dtype)
        x = backing
    before = backing.clone()
    assert module._init_triton()
    expected = module._eager_forward(x)
    copied = module._triton_forward(x.contiguous())
    actual = module._triton_forward(x)
    # The optimization changes addresses, not arithmetic or rounding.
    torch.testing.assert_close(actual, copied, atol=0, rtol=0)
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(backing, before, atol=0, rtol=0)
    assert actual.is_contiguous()
    assert actual.dtype == dtype


@torch.inference_mode()
def test_snake_upsample_view_does_not_copy():
    module = AliasFreeActivation1d(SnakeBeta(8)).to("cuda").eval()
    x = module.upsample(torch.randn(2, 8, 127, device="cuda"))
    assert x.stride(-1) == 1 and not x.is_contiguous()
    assert module.act._init_triton()
    module.act._triton_forward(x)  # warm caches and JIT before profiling
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        module.act._triton_forward(x)
    assert "aten::contiguous" not in {event.key for event in profile.key_averages()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_alias_free_activation_cuda_graph(dtype):
    module = AliasFreeActivation1d(SnakeBeta(8)).to(device="cuda", dtype=dtype).eval()
    x = torch.randn(2, 8, 127, device="cuda", dtype=dtype)
    # Warm cuDNN and the Triton kernel on a side stream before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(x)
    torch.cuda.current_stream().wait_stream(stream)
    assert module.act._triton_kernel is not False
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = module(x)
    for _ in range(2):
        x.normal_()
        upsampled = module.upsample(x)
        expected = module.downsample(module.act._triton_forward(upsampled.contiguous()))
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
