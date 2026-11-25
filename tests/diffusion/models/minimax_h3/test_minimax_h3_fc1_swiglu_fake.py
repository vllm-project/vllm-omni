# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused fc1 + SwiGLU op's fake implementation gives torch.compile the output shape for 2D and 3D inputs."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_omni.diffusion.models.minimax_h3.ops.fc1_swiglu import fc1_swiglu

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("lead", [(777,), (2, 33)])
def test_fake_impl_keeps_leading_dims(lead: tuple[int, ...]) -> None:
    with FakeTensorMode():
        x = torch.empty(*lead, 512, dtype=torch.bfloat16)
        weight = torch.empty(768, 512, dtype=torch.bfloat16)
        out = fc1_swiglu(x, weight)
    assert out.shape == (*lead, 384)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("lead", [(777,), (2, 33)])
def test_meta_dispatch_keeps_leading_dims(lead: tuple[int, ...]) -> None:
    out = fc1_swiglu(
        torch.empty(*lead, 512, dtype=torch.bfloat16, device="meta"),
        torch.empty(768, 512, dtype=torch.bfloat16, device="meta"),
    )
    assert out.shape == (*lead, 384)
