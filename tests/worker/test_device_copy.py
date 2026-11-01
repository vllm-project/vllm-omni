# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.utils.device_copy import to_device_nonblocking

pytestmark = pytest.mark.core_model


@pytest.mark.cpu
def test_tensor_copy_cpu_preserves_view_and_dtype():
    source = torch.arange(12, dtype=torch.float64).reshape(3, 4).T
    result = to_device_nonblocking(source, "cpu")
    assert result is source
    assert result.dtype == torch.float64


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_tensor_copy_cuda_handles_pageable_and_pinned_sources():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    for host in (source, source.pin_memory()):
        result = to_device_nonblocking(host, "cuda")
        torch.testing.assert_close(result.cpu(), host)
        assert to_device_nonblocking(result, result.device) is result
