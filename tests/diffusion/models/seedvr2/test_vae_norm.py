# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent arithmetic and CUDA graph replay checks for VAE affine fusion."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.seedvr2.vae_norm import apply_spatial_norm

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,groups", [((2, 12, 3, 7, 11), 3), ((1, 128, 5, 120, 848), 32)])
def test_fused_affine_matches_eager_fp32_rounding_and_replay(dtype, shape, groups):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    generator = torch.Generator(device="cuda").manual_seed(1234)
    b, c, t, h, w = shape
    norm = torch.nn.GroupNorm(groups, c).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(c, device="cuda", dtype=dtype, generator=generator))
        norm.bias.copy_(torch.randn(c, device="cuda", dtype=dtype, generator=generator))
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    grouped = x.to(torch.float32, copy=True).reshape(b, groups, c // groups, t, h, w)
    variance, mean = torch.var_mean(grouped, dim=(2, 4, 5), correction=0, keepdim=True)
    inverse = torch.rsqrt(variance + norm.eps)

    def reference():
        normalized = grouped.clone().sub_(mean).mul_(inverse).reshape(shape)
        normalized.mul_(norm.weight.float().view(1, -1, 1, 1, 1))
        normalized.add_(norm.bias.float().view(1, -1, 1, 1, 1))
        return normalized.to(dtype)

    original = grouped.clone()
    expected = reference()
    actual = apply_spatial_norm(grouped, mean, inverse, norm, shape, dtype)
    assert torch.equal(actual, expected)
    assert torch.equal(grouped, original)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            apply_spatial_norm(grouped, mean, inverse, norm, shape, dtype)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = apply_spatial_norm(grouped, mean, inverse, norm, shape, dtype)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    # Replay with changed values at the same addresses, then restore A.
    grouped.mul_(0.5)
    expected_b = reference()
    graph.replay()
    torch.testing.assert_close(captured, expected_b, rtol=0, atol=0)
    grouped.copy_(original)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
    graph.reset()
