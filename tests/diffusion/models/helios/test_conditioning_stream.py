# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.helios.conditioning_cache import _record_stream

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_record_stream_is_a_noop_on_cpu():
    tensor = torch.zeros(3, device="cpu")
    # Must not raise (and must not touch any stream API).
    _record_stream(tensor)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_record_stream_covers_npu_devices():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")
    tensor = torch.zeros(3, device="npu")
    # Regression: previously skipped because is_cuda is False on NPU.
    _record_stream(tensor)
    torch.npu.synchronize()
