# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ring accumulation needs a real LSE, and XPU's efficient aten op does not give one."""

import pytest
import torch

from vllm_omni.diffusion.attention.backends.ring.ring_kernels import _ring_op_type

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_xpu_efficient_switches_to_flash(dtype):
    """aten's XPU efficient op returns LSE = 0, so the merge must use the flash op."""
    assert _ring_op_type("efficient", torch.device("xpu"), dtype) == "flash"


def test_xpu_float32_keeps_efficient():
    """The flash aten path rejects float32, so fp32 must keep the requested op."""
    assert _ring_op_type("efficient", torch.device("xpu"), torch.float32) == "efficient"


def test_xpu_explicit_flash_is_unchanged():
    assert _ring_op_type("flash", torch.device("xpu"), torch.bfloat16) == "flash"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_other_devices_keep_the_requested_op(device):
    """Only XPU's efficient op is known to return a zero LSE."""
    assert _ring_op_type("efficient", torch.device(device), torch.bfloat16) == "efficient"
