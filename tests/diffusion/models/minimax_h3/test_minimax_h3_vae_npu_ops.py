# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU numerical smoke tests for MiniMax H3 VAE fused operators."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.minimax_h3.ops.vae.qk_norm_rope import try_rms_norm_npu, try_rotary_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu import is_a5

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture
def a5_device():
    if not current_omni_platform.is_npu():
        pytest.skip("requires an NPU")
    device = current_omni_platform.get_torch_device()
    if not is_a5(device):
        pytest.skip("requires an Ascend A5")
    return device


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_npu_rms_norm_matches_reference(a5_device, dtype):
    generator = torch.Generator().manual_seed(17)
    # Q and K are strided views of the packed QKV tensor.
    x = torch.randn(1, 195, 2, 192, generator=generator, dtype=dtype).to(a5_device).chunk(3, dim=-1)[0]
    gamma = torch.ones(64, device=a5_device, dtype=torch.float32)
    with torch.inference_mode():
        expected = F.rms_norm(x, (64,), None, 1e-6)
        actual = try_rms_norm_npu(x, gamma, 1e-6, unit_gamma=True)
    assert actual is not None
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)


@hardware_test(res={"npu": "A5"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_npu_rotary_matches_reference_and_preserves_suffix(a5_device, dtype):
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(1, 195, 2, 192, generator=generator, dtype=dtype).to(a5_device).chunk(3, dim=-1)[0]
    angles = torch.randn(1, 195, 1, 24, generator=generator, dtype=torch.float32)
    angles = torch.cat((angles, angles), dim=-1).to(a5_device)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    first, second = x[..., :48].chunk(2, dim=-1)
    expected = x[..., :48] * cos + torch.cat((-second, first), dim=-1) * sin
    with torch.inference_mode():
        actual = try_rotary_npu(x, (cos, sin))
    assert actual is not None
    assert actual.dtype == dtype
    torch.testing.assert_close(actual[..., :48], expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual[..., 48:], x[..., 48:], rtol=0, atol=0)
