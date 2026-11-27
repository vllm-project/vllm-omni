# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.helios.scheduling_helios import _solve_rho_system

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_solve(order, seed):
    g = torch.Generator().manual_seed(seed)
    R = torch.rand(order, order, generator=g) + 0.5
    b = torch.rand(order, generator=g)
    expected = torch.linalg.solve(R, b)
    result = _solve_rho_system(R, b)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_preserves_device_and_dtype():
    R = torch.eye(3) * 2
    b = torch.ones(3)
    result = _solve_rho_system(R, b)
    assert result.device == R.device
    assert result.dtype == R.dtype


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("order", [2, 3])
def test_npu_matches_explicit_cpu_reference(order):
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")
    matrix = 2 * torch.eye(order) + 0.25
    rhs = torch.arange(1, order + 1, dtype=matrix.dtype)
    expected = torch.linalg.solve(matrix.double(), rhs.double()).to(matrix.dtype)
    result = _solve_rho_system(matrix.to("npu"), rhs.to("npu"))
    assert result.device.type == "npu"
    assert result.dtype == matrix.dtype
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
