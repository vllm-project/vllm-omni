# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Ascend checks for H3's FP32 per-head Q/K normalization path."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3 import vae_ops
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,sequence", [(1, 1797), (2, 7)])
def test_non_affine_qk_rmsnorm_matches_native_on_npu(mocker, dtype, batch, sequence):
    if not current_omni_platform.is_npu() or not current_omni_platform.is_available():
        pytest.skip("Requires an Ascend NPU")
    import torch_npu

    device = current_omni_platform.get_torch_device()
    owner = nn.Module()
    owner.projection = nn.Linear(1, 1, device=device)
    owner.norm = nn.RMSNorm(64, eps=1e-5, elementwise_affine=False)
    assert vae_ops._replace_rms_norm(owner, "norm")
    gamma = owner.norm._unit_weight
    assert gamma.device == owner.projection.weight.device
    assert gamma.dtype == torch.float32
    kernel = mocker.spy(torch_npu, "npu_rms_norm")
    torch.manual_seed(5985)
    # FP32 chunks exercise a strided kernel boundary; FP16/BF16 chunks also
    # exercise H3's conversion to FP32 before normalization and cast back.
    value = torch.randn(batch, sequence, 32, 192, device=device, dtype=dtype).chunk(3, dim=-1)[1]
    assert not value.is_contiguous()
    with torch.inference_mode():
        expected_fp32 = F.rms_norm(value.float(), (64,), None, 1e-5)
        for _ in range(2):
            actual_fp32 = owner.norm(value.float())
            torch.testing.assert_close(actual_fp32, expected_fp32, rtol=1e-5, atol=1e-6)
            tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
            torch.testing.assert_close(actual_fp32.to(dtype), expected_fp32.to(dtype), rtol=tolerance, atol=tolerance)
    assert kernel.call_count == 2
    assert all(call.args[0].dtype == torch.float32 for call in kernel.call_args_list)
    assert all(call.kwargs["gamma"].data_ptr() == gamma.data_ptr() for call in kernel.call_args_list)
